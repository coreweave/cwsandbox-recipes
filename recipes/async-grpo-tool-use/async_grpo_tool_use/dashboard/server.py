"""The live dashboard: page, event hub and log parser.

``async_grpo_tool_use.run`` feeds every launcher, Gym and Slurm job log line to ``LineParser``,
which turns them into page events in a ``Hub``. The page (``index.html``) draws the trainer
(SUNK), the sampler (Dedicated Inference) and the environments (CoreWeave Sandboxes) from them.
Every number comes from the run:

* the launcher's own output and the Slurm job log (steps, phases, NeMo-RL step metrics)
* ``@@metric`` lines that the recipe prints when ``DASHBOARD_METRICS=1``: GPU utilization on the
  trainer node, requests and tokens seen by Gym's model server, the rollout sandbox pool, a
  stream of finished rollouts for the examples, and the start and end of each hot-load
* the sandbox API, polled every 3 seconds for the run's sandboxes
* the inference deployment, polled every 10 seconds for its ready replicas

The same handler serves the page locally and, read-only, in the viewer sandbox (``viewer.py``).
"""

from __future__ import annotations

import json
import queue
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
METRIC_LINE = re.compile(r"@@metric (\{.*\})")
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Launcher and trainer lines that mark a bring-up stage.
STAGES = [
    (re.compile(r"\[grpo\] run tag (\S+)"), "run"),
    (re.compile(r"\[grpo\] storage: temporary access key"), "storage"),
    (re.compile(r"\[grpo\] inference: gateway (\S+) ready"), "gateway"),
    (re.compile(r"\[grpo\] gym: sandbox (\S+) requested"), "gym"),
    (re.compile(r"\[grpo\] sunk: job (\S+) submitted"), "job"),
    (re.compile(r"\[grpo\] sunk: node (\S+?),"), "node"),
    (re.compile(r"\[grpo\] sunk: importing"), "image"),
    (re.compile(r"\[grpo\] trainer: model staged"), "staged"),
    (re.compile(r"\[grpo\] inference: deployment (\S+) requested"), "deployment"),
    (re.compile(r"\[grpo\] inference: deployment ready"), "deployed"),
    (re.compile(r"\[math_tir_sandbox\] starting (\d+) episode sandboxes"), "pool"),
    (re.compile(r"\[grpo\] calibration: (\w+)\s+(\d+)% graded"), "calibration"),
    (re.compile(r"\[grpo\] trainer: starting GRPO"), "grpo"),
    (re.compile(r"\[grpo\] sunk: job \S+ (COMPLETED|FAILED|CANCELLED\S*|TIMEOUT)"), "job_end"),
    (re.compile(r"\[grpo\] trainer exited with (\d+)"), "exit"),
    (re.compile(r"\[grpo\] inference: gateway \S+ deleted"), "cleaned"),
]
STEP = re.compile(r"=+ Step (\d+)/(\d+) =+")
PHASES = [
    (re.compile(r"\[grpo\] hot-load v\d+: exporting weights"), "export"),
    # Only a blocking hot-load is a trainer phase; a background one ends "while training continues".
    (re.compile(r"\[grpo\] hot-load v\d+: uploading and loading$"), "hotload"),
    (re.compile(r"▶ Generating responses"), "generation"),
    # async GRPO: the trainer waits here until the replay buffer holds a full batch of rollouts
    (re.compile(r"📦 Sampling from replay buffer"), "generation"),
    (re.compile(r"▶ Processing rewards"), "rewards"),
    (re.compile(r"▶ (Preparing for logprob inference|Computing logprobs)"), "logprobs"),
    (re.compile(r"▶ (Preparing for training|Training policy)"), "training"),
    (re.compile(r"▶ Starting validation"), "validation"),
]
HOTLOAD_DONE = re.compile(
    r"\[grpo\] hot-load v(\d+): (\d+) tensors, ([\d.]+) GiB; export (\d+)s, upload and load (\d+)s"
)
METRICS = {
    "loss": re.compile(r"• Loss: (-?[\d.]+)"),
    "kl_error": re.compile(r"• Generation KL Error: (-?[\d.]+)"),
    "reward": re.compile(r"• Avg (?:Filtered )?Reward: (-?[\d.]+)"),
    "total_reward": re.compile(r"• Avg Total Reward: (-?[\d.]+)"),
    "gen_length": re.compile(r"• Mean Generation Length: ([\d.]+)"),
    "step_time": re.compile(r"• Total step time: ([\d.]+)s"),
    "e2e_tokens_s": re.compile(r"- E2E \(Tokens/sec\): ([\d.]+)"),
    "train_tokens_s": re.compile(r"- Training Worker Group \(Tokens/sec\): ([\d.]+)"),
    "gen_tokens_s": re.compile(r"- Generation Worker Group \(Tokens/sec\): ([\d.]+)"),
    "accuracy": re.compile(r"• Accuracy: ([\d.]+)"),
}
TIMING = re.compile(r"• ([\w/]+): ([\d.]+)s \(([\d.]+)%\)")
VALIDATION_TIME = re.compile(r"• Total validation time: ([\d.]+)s")
READY_REPLICAS = re.compile(r"Ready replicas: (\d+)/(\d+)")
NOTABLE = re.compile(r"\[grpo\]|=+ Step|▶|Accuracy|\[math_sandbox_eval\]")


class Hub:
    """Holds every event of the run and fans them out to the page's event streams."""

    def __init__(self):
        self.lock = threading.Lock()
        self.events: list[dict] = []
        self.listeners: list[queue.Queue] = []

    def publish(self, event: dict) -> None:
        event = {"t": time.time(), **event}
        with self.lock:
            self.events.append(event)
            for listener in self.listeners:
                listener.put(event)

    def subscribe(self) -> tuple[list[dict], queue.Queue]:
        listener: queue.Queue = queue.Queue()
        with self.lock:
            self.listeners.append(listener)
            return list(self.events), listener

    def unsubscribe(self, listener: queue.Queue) -> None:
        with self.lock:
            self.listeners.remove(listener)


class LineParser:
    """Turns launcher, Gym and Slurm job log lines into page events."""

    def __init__(self, hub: Hub):
        self.hub = hub
        self.step = 0
        self.phase = None
        self.metrics: dict[str, float] = {}
        self.timings: dict[str, float] = {}

    def feed(self, raw: str) -> None:
        line = ANSI.sub("", raw).rstrip()
        metric = METRIC_LINE.search(line)
        if metric:
            try:
                self.hub.publish({"type": "metric", **json.loads(metric.group(1))})
            except json.JSONDecodeError:
                pass
            return
        if NOTABLE.search(line):
            self.hub.publish({"type": "log", "line": line[:240]})
        for pattern, stage in STAGES:
            match = pattern.search(line)
            if match:
                self.hub.publish({"type": "stage", "stage": stage, "args": list(match.groups())})
        match = STEP.search(line)
        if match:
            self.flush_step()
            self.step = int(match.group(1))
            self.hub.publish({"type": "step", "step": self.step, "of": int(match.group(2))})
        for pattern, phase in PHASES:
            if pattern.search(line):
                self.phase = phase
                self.hub.publish({"type": "phase", "phase": phase, "step": self.step})
        match = HOTLOAD_DONE.search(line)
        if match:
            version, tensors, gib, export_s, load_s = match.groups()
            self.hub.publish({"type": "hotload", "version": int(version), "gib": float(gib),
                              "export_s": int(export_s), "load_s": int(load_s)})  # fmt: skip
            # A hot-load in the background ends while the trainer is already on the next step.
            if self.phase == "hotload":
                self.phase = "generation"
                self.hub.publish({"type": "phase", "phase": "generation", "step": self.step})
        for name, pattern in METRICS.items():
            match = pattern.search(line)
            if match:
                self.metrics[name] = float(match.group(1))
                if name == "accuracy":
                    self.hub.publish({"type": "accuracy", "step": self.step, "accuracy": float(match.group(1))})
        match = TIMING.search(line)
        if match:
            self.timings[match.group(1)] = float(match.group(2))
        match = VALIDATION_TIME.search(line)
        if match:
            self.timings["validation"] = float(match.group(1))
        if "Generation Worker Group (Tokens/sec):" in line and "/gpu" not in line:
            self.flush_step()  # the last line of a step's summary

    def flush_step(self) -> None:
        if self.metrics.get("step_time"):
            self.hub.publish({"type": "step_done", "step": self.step, "metrics": self.metrics,
                              "timings": self.timings})  # fmt: skip
        self.metrics, self.timings = {}, {}


def poll_sandboxes(hub: Hub, sdk, run_tag: str, stop: threading.Event) -> None:
    """Count the run's sandboxes in the sandbox API every 3 seconds."""
    last = None
    while not stop.wait(3):
        try:
            sandboxes = sdk.Sandbox.list(tags=[run_tag]).result()
        except Exception:  # noqa: BLE001 - a failed poll only leaves a gap in the chart
            continue
        states: dict[str, int] = {}
        for sandbox in sandboxes:
            state = str(getattr(sandbox.status, "value", sandbox.status)).lower()
            states[state] = states.get(state, 0) + 1
        row = {"type": "metric", "k": "sandboxes", "ts": time.time(), "total": len(sandboxes), "states": states}
        if row["states"] != (last or {}).get("states"):
            hub.publish(row)
            last = row
        elif int(time.time()) % 15 < 3:
            hub.publish(row)  # a heartbeat keeps the chart's line going


def cwic_json(*args: str) -> dict:
    done = subprocess.run(["cwic", *args, "-o", "json"], capture_output=True, text=True, timeout=60)
    if done.returncode != 0:
        raise RuntimeError(done.stderr.strip() or done.stdout.strip())
    return json.loads(done.stdout)


def poll_deployment(hub: Hub, launcher, gpus_per_replica: int, stop: threading.Event) -> None:
    """Report the deployment's ready replicas every 10 seconds, so the page shows its GPUs."""
    last = None
    while not stop.wait(10):
        if not launcher.deployment_id:
            continue
        try:
            found = cwic_json("inference", "deployment", "get", launcher.deployment_id)
        except (RuntimeError, ValueError, subprocess.TimeoutExpired):
            continue
        ready, desired = 0, found["spec"]["autoscaling"]["min"]
        for condition in found.get("status", {}).get("conditions", []):
            match = READY_REPLICAS.search(condition.get("message", ""))
            if match:
                ready = int(match.group(1))
        if found.get("status", {}).get("status") == "STATUS_READY" and not ready:
            ready = desired
        row = (ready, desired)
        if row != last or int(time.time()) % 60 < 10:
            hub.publish({"type": "metric", "k": "deployment", "ts": time.time(), "ready": ready, "desired": desired,
                         "gpus": ready * gpus_per_replica})  # fmt: skip
            last = row


def make_handler(hub: Hub, meta: dict):
    page = (HERE / "index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(page)
                return
            if self.path == "/meta":
                body = json.dumps(meta).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path != "/events":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            backlog, listener = hub.subscribe()
            # Each event's id is its position in the run. A reconnecting page (the shared page's
            # HTTPS endpoint closes requests after a while) resumes after the last one it saw.
            resume = self.headers.get("Last-Event-ID", "")
            try:
                if resume.isdigit() and int(resume) <= len(backlog):
                    seen = int(resume)
                    for position, event in enumerate(backlog[seen:], start=seen + 1):
                        self.wfile.write(f"id: {position}\ndata: {json.dumps(event)}\n\n".encode())
                else:
                    self.wfile.write(f"id: {len(backlog)}\nevent: backlog\ndata: {json.dumps(backlog)}\n\n".encode())
                self.wfile.flush()
                position = len(backlog)
                while True:
                    try:
                        event = listener.get(timeout=10)
                        position += 1
                        self.wfile.write(f"id: {position}\ndata: {json.dumps(event)}\n\n".encode())
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                hub.unsubscribe(listener)

    return Handler


def serve(hub: Hub, meta: dict, port: int) -> str:
    """Serve the page on localhost in the background, on ``port`` or, if it is taken, any free
    port. Return its URL."""
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(hub, meta))
    except OSError:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(hub, meta))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://localhost:{server.server_address[1]}"
