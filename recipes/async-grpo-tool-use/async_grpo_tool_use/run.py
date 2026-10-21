"""Run the recipe with its live dashboard: one command, one link.

1. Read ``.env`` (values already in the environment win).
2. Start the viewer sandbox and print the dashboard's public URL first, about 2 s after the
   start. The page answers a few seconds later and shows the bring-up as it happens.
3. Run the launcher (``launch.py``): storage key, gateway, Gym sandbox, Slurm job, deployment,
   training, cleanup. Every log line also feeds the dashboard.
4. When the run ends, the dashboard stays up with the results until you press Ctrl+C, which
   stops the viewer sandbox. It also stops when its lifetime (``DASHBOARD_HOURS``) runs out.

The header links the W&B run (trainer metrics), the Agent Lens traces of the rollouts and the
Weave evaluations, when ``WANDB_API_KEY`` is set. What the page draws also goes to a second W&B
run in the same group, ``<run tag>-dashboard`` (``dashboard/wandb_log.py``).

Usage: async-grpo [--max-num-steps N] [--no-share] [-- hydra overrides...]
   or: python -m async_grpo_tool_use.run ...
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
import traceback
from urllib.parse import quote

from async_grpo_tool_use import env_file, launch
from async_grpo_tool_use.dashboard import server, share, wandb_log

FORGE_URL = "https://forge.coreweave.com/wandb"


def run_links(env: dict[str, str], run_tag: str) -> list[dict[str, str]]:
    """Header links into CoreWeave Forge. Each opens with its own login."""
    if not (env.get("WANDB_API_KEY") and env.get("WANDB_ENTITY")):
        return []
    project = f"{FORGE_URL}/{quote(env['WANDB_ENTITY'])}/{quote(env['WANDB_PROJECT'])}"
    return [
        {"label": "W&B run", "url": f"{project}/runs/{quote(run_tag)}"},
        {"label": "W&B: page metrics", "url": f"{project}/runs/{quote(run_tag)}-dashboard"},
        {"label": "Agent Lens", "url": f"{project}/weave/agents"},
        {"label": "Weave evals", "url": f"{project}/weave/evaluations"},
    ]


class Viewer:
    """The shared dashboard: a viewer sandbox, and the relay that feeds it once its page answers."""

    def __init__(self, sdk, hub: server.Hub, run_tag: str, hours: float, known) -> None:
        self.relay = None
        self.sandbox, self.url, token = share.create_viewer(sdk, run_tag, hours)
        threading.Thread(target=self._connect, args=(hub, token, known), daemon=True).start()

    def _connect(self, hub: server.Hub, token: str, known) -> None:
        try:
            share.wait_until_serving(self.url)
        except RuntimeError as exc:
            print(f"[dashboard] the shared dashboard did not start: {exc}", flush=True)
            return
        self.relay = share.Relay(hub, self.url, token, lambda: [token, *known()])

    def stop(self) -> None:
        if self.relay:
            self.relay.flush()
        self.sandbox.stop(missing_ok=True).result()
        print("[dashboard] viewer sandbox stopped", flush=True)


def main(argv: list[str] | None = None) -> int:
    signal.signal(signal.SIGINT, signal.default_int_handler)
    parser = argparse.ArgumentParser(description="Run async GRPO with tool use, with a live dashboard.")
    parser.add_argument("--max-num-steps", type=int, default=None)
    parser.add_argument("--no-share", action="store_true", help="serve the dashboard on localhost only")
    parser.add_argument("--port", type=int, default=8766, help="local dashboard port")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides for NeMo-RL (after --)")
    args = parser.parse_args(argv)

    env_file.load()
    if not os.environ.get("CWSANDBOX_API_KEY"):
        print("set CWSANDBOX_API_KEY (a CoreWeave API access token) in .env", file=sys.stderr)
        return 2
    os.environ.setdefault("DASHBOARD_METRICS", "1")
    # cwic runs as a subprocess next to cwsandbox's gRPC client; keep gRPC's fork notices quiet.
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    import cwsandbox

    config = launch.config_from_env(args, dict(os.environ))
    parsers: list[server.LineParser] = []

    def sink(line: str) -> None:
        print(line, end="" if line.endswith("\n") else "\n", flush=True)
        parsers[0].feed(line)

    launcher = launch.Launcher(config, cwsandbox, sink=sink)
    hub = server.Hub()
    parsers.append(server.LineParser(hub))
    # The launcher's own messages go through print(); send them through the parser too.
    launch.log = lambda message: sink(f"[grpo] {message}\n")
    pool = int(os.environ.get("SANDBOX_POOL_MAX", 256))
    on_slurm, di = config.trainer == "sunk", config.sampler == "di"
    gpu_name = "H100" if on_slurm else "RTX PRO 6000"
    meta = {
        "mode": "live", "run_tag": launcher.run_tag, "steps": int(config.env["MAX_NUM_STEPS"]),
        "trainer": f"SUNK · {config.partition}" if on_slurm else "GPU sandbox",
        "model": config.model_repo.split("/")[-1], "gpu_name": gpu_name,
        "trainer_label": f"SUNK · {gpu_name}" if on_slurm else f"CoreWeave GPU sandbox · {gpu_name}",
        "sampler_label": ("Dedicated Inference · H100" if di else
                          f"self-hosted on the trainer's node · {gpu_name}" if on_slurm
                          else f"{config.sampler_replicas} CoreWeave GPU sandbox{'es' if config.sampler_replicas > 1 else ''} · {gpu_name}"),
        "trainer_kind": config.trainer, "sampler_kind": config.sampler,
        "replicas": config.deployment.replicas if di else config.sampler_replicas,
        "pool_max": pool, "pool_min": int(os.environ.get("SANDBOX_POOL_MIN", pool)),
        "trainer_gpus": int(config.env.get("TRAINER_GPUS", 8)),
        "gpus_per_replica": config.deployment.gpus_per_replica if di else int(config.env["SAMPLER_GPUS"]),
        "tir": True, "async": True, "links": run_links(config.env, launcher.run_tag),
    }  # fmt: skip
    viewer = None
    if not args.no_share:

        def known() -> list:
            """Secrets to redact from what the shared page shows."""
            secrets = [config.api_token, config.env.get("WANDB_API_KEY"), config.env.get("HF_TOKEN")]
            return [*secrets, *(launcher.access_key or {}).values()]

        hours = float(os.environ.get("DASHBOARD_HOURS", 12))
        try:
            viewer = Viewer(cwsandbox, hub, launcher.run_tag, hours, known)
        except Exception as exc:  # noqa: BLE001 - the run goes on without a shared page
            print(f"[dashboard] could not share the dashboard: {exc}", flush=True)
        if viewer:
            meta["share_url"] = viewer.url
            print(f"[dashboard] {viewer.url}  (read-only, shareable)", flush=True)
    stop = threading.Event()
    code = 1
    metrics_run = None
    # From here on the viewer sandbox exists; the finally below stops it whatever fails.
    try:
        if meta["links"]:
            try:
                metrics_run = wandb_log.WandbLog(hub, config.env, launcher.run_tag, meta)
            except Exception as exc:  # noqa: BLE001 - the run goes on without the dashboard's W&B run
                print(f"[dashboard] could not log the dashboard to W&B: {exc}", flush=True)
        hub.publish({"type": "meta", **meta})
        print(f"[dashboard] {server.serve(hub, meta, args.port)}  (this machine)", flush=True)
        for link in meta["links"]:
            print(f"[dashboard] {link['label']}: {link['url']}", flush=True)
        threading.Thread(target=server.poll_sandboxes, args=(hub, cwsandbox, launcher.run_tag, stop),
                         daemon=True).start()  # fmt: skip
        threading.Thread(target=server.poll_deployment, args=(hub, launcher, config.deployment.gpus_per_replica, stop),
                         daemon=True).start()  # fmt: skip
        code = launcher.run()
    except Exception:
        traceback.print_exc()
        hub.publish({"type": "log", "line": "[grpo] the run failed; see the terminal"})
        hub.publish({"type": "stage", "stage": "exit", "args": ["1"]})
    finally:
        stop.set()
        try:
            if metrics_run:
                metrics_run.finish()
            if viewer:
                print(f"[dashboard] still up at {viewer.url}; Ctrl+C stops it", flush=True)
                while True:
                    time.sleep(3600)
        except KeyboardInterrupt:
            pass
        finally:
            if viewer:
                viewer.stop()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
