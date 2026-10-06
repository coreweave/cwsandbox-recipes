"""Deploy a Claude orchestrator and provision one CoreWeave sandbox per work order."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid

from cwsandbox import AuthStrategy, Sandbox, SandboxFileError
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
REMOTE = Path("/opt/recipe")
STATE = Path("/workspace/orchestrator")
AUTH = AuthStrategy.COREWEAVE_API_KEY
SECRET = "SELF_HOSTED_RUNNER_ENVIRONMENT_SECRET"


class RecipeError(Exception):
    """An operator-facing error that contains no SDK request data."""


def save(path, value):
    """Persist intent before submitting a non-idempotent sandbox creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def active(tag):
    return Sandbox.list(auth=AUTH, tags=[tag], timeout_seconds=10).result()


def get_box(sandbox_id):
    return Sandbox.from_id(sandbox_id, auth=AUTH).result()


def run_box(role, config, env, files):
    return Sandbox.run(
        "python",
        str(REMOTE / "entrypoint.py"),
        "bash",
        str(REMOTE / "bootstrap.sh"),
        role,
        auth=AUTH,
        container_image="python:3.12-bookworm",
        placement_mode="serverless",
        placement_spillover="strict",
        max_lifetime_seconds=config[role + "_hours"] * 3600,
        request_timeout_seconds=30,
        resources={
            "cpu": "1" if role == "orchestrator" else "2",
            "memory": "2Gi" if role == "orchestrator" else "4Gi",
        },
        environment_variables=env,
        mounted_files=files,
        tags=[config["tag"], config["tag"] + ":" + role],
    )


def asset(name):
    return {"mount_path": str(REMOTE / name), "file_content": (ROOT / name).read_bytes()}


def spawn(config, directory=STATE):
    order = os.environ.get("CLAUDE_RUNNER_ORDER_ID", "")
    if (
        not order
        or len(order) > 256
        or os.environ.get("CLAUDE_RUNNER_POOL_ID") != config["environment"]
    ):
        return 2
    try:
        token = Path(os.environ["CLAUDE_RUNNER_WORK_ORDER_FILE"]).read_text().strip()
        if not token or len(token) > 65536 or any(c.isspace() for c in token):
            return 2
    except (KeyError, OSError):
        return 2
    name = hashlib.sha256(order.encode()).hexdigest()
    orders = directory / "orders"
    orders.mkdir(parents=True, exist_ok=True)
    with (directory / "spawn.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record = orders / (name + ".json")
        if record.exists():
            return 0 if json.loads(record.read_text()).get("sandbox_id") else 2
        try:
            if len(active(config["tag"] + ":worker")) >= config["max_workers"]:
                print("Worker limit reached; retry later", file=sys.stderr, flush=True)
                return 1
        except Exception:
            print("Cannot check worker capacity; retry later", file=sys.stderr, flush=True)
            return 1
        save(record, {"session_id": os.environ.get("CLAUDE_RUNNER_SESSION_ID", "")})
        try:
            box = run_box(
                "worker",
                config,
                {
                    SECRET: token,
                    "WORKER_IDLE_MINUTES": str(config["idle_minutes"]),
                    "WORKER_RETIRE_AT": str(int(time.time()) + config["worker_hours"] * 3600 - 300),
                    "WORKER_LABEL": "cws-" + name[:16],
                },
                [asset("bootstrap.sh"), asset("entrypoint.py")],
            )
            data = json.loads(record.read_text())
            data["sandbox_id"] = box.sandbox_id
            save(record, data)
            print("Submitted worker " + box.sandbox_id, flush=True)
            return 0
        except Exception:
            # The API may have accepted the request. Never replay the one-use token.
            print(
                "Worker submission uncertain; inspect status and logs before retrying", flush=True
            )
            return 2


def supervise():
    STATE.mkdir(parents=True, exist_ok=True)
    hook = STATE / "hooks" / "spawn-runner"
    hook.parent.mkdir(exist_ok=True)
    hook.write_text(
        "#!/bin/sh\nexec " + shlex.join([sys.executable, str(REMOTE / "cloud.py"), "spawn"]) + "\n"
    )
    hook.chmod(0o700)
    process = None
    stopping = False

    def shutdown(signum, frame):
        nonlocal stopping
        stopping = True
        if process is not None and process.poll() is None:
            process.terminate()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    command = [
        "claude",
        "self-hosted-runner",
        "orchestrator",
        "--hooks-dir",
        str(hook.parent),
        "--hook-concurrency",
        "1",
        "--hook-timeout",
        "60",
        "--expected-spawn-seconds",
        "600",
        "--health-port",
        "8080",
    ]
    # Single supervisor preserves the hook ledger across native process restarts.
    with (STATE / "orchestrator.log").open("a") as log:
        while not stopping:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            save(STATE / "process.json", {"pid": process.pid})
            process.wait()
            if not stopping:
                print("Orchestrator exited; restarting in 5 seconds", flush=True)
                time.sleep(5)


def deployment(path):
    data = json.loads(path.read_text())
    if not data.get("sandbox_id"):
        matches = active(data["config"]["tag"] + ":orchestrator")
        if len(matches) != 1:
            raise RecipeError(
                "Deployment submission uncertain. Inspect its tag; do not redeploy blindly: "
                + data["config"]["tag"]
            )
        data["sandbox_id"] = matches[0].sandbox_id
        save(path, data)
    return data


def health(box):
    result = box.exec(
        [
            "python",
            "-c",
            "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8080/healthz',timeout=3).read().decode())",
        ],
        timeout_seconds=10,
    ).result()
    if result.returncode:
        return {"connected": False}
    return json.loads(result.stdout)


def deploy(args):
    if args.state.exists():
        raise RecipeError(
            "Deployment record already exists. Use status or stop before redeploying."
        )
    if not re.fullmatch(r"ccpool_[A-Za-z0-9]+", args.environment):
        raise RecipeError("Use the ccpool_ environment ID from Claude's admin page")
    if not (
        1 <= args.orchestrator_hours <= 720
        and 1 <= args.worker_hours <= 720
        and 1 <= args.idle_minutes < args.worker_hours * 60 - 5
        and args.max_workers >= 1
    ):
        raise RecipeError(
            "Require lifetimes 1..720 hours, max-workers >= 1, and idle-minutes below worker lifetime minus 5 minutes"
        )
    secret = os.environ.get(SECRET, "").strip()
    if not secret or any(c.isspace() for c in secret):
        raise RecipeError("Set " + SECRET)
    config = {
        key: getattr(args, key)
        for key in (
            "environment",
            "orchestrator_hours",
            "worker_hours",
            "idle_minutes",
            "max_workers",
        )
    }
    config["tag"] = "claude-cloud:" + uuid.uuid4().hex
    data = {"config": config, "created_at": time.time()}
    save(args.state, data)
    files = [
        asset(name) for name in ("cloud.py", "bootstrap.sh", "entrypoint.py", "requirements.txt")
    ]
    files.append(
        {"mount_path": str(REMOTE / "config.json"), "file_content": json.dumps(config).encode()}
    )
    box = run_box(
        "orchestrator",
        config,
        {
            "CWSANDBOX_API_KEY": os.environ["CWSANDBOX_API_KEY"],
            SECRET: secret,
            "CWSANDBOX_DISABLE_SIGNAL_HANDLERS": "1",
            "PYTHONUNBUFFERED": "1",
        },
        files,
    )
    data["sandbox_id"] = box.sandbox_id
    save(args.state, data)
    print("Orchestrator: " + box.sandbox_id, flush=True)
    box.wait(timeout=600)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        try:
            if health(box).get("connected"):
                print("Connected. Select your environment at https://claude.ai/code")
                return
        except Exception:
            pass
        time.sleep(3)
    raise RecipeError(
        "Orchestrator did not connect. Run logs and status; resources remain deployed."
    )


def select_worker(data, worker_id):
    if not worker_id:
        raise RecipeError("Supply --worker using an ID from status")
    if worker_id not in {b.sandbox_id for b in active(data["config"]["tag"] + ":worker")}:
        raise RecipeError("Worker is not active in this deployment")
    return get_box(worker_id)


def stop(args, data):
    box = get_box(data["sandbox_id"])
    box.stop(missing_ok=True).result()
    box.wait_until_complete(timeout=120, raise_on_termination=False).result()
    # Hooks may have submitted a worker during shutdown. Enumerate after the host stops.
    for worker in active(data["config"]["tag"] + ":worker"):
        worker.stop(missing_ok=True).result()
        worker.wait_until_complete(timeout=120, raise_on_termination=False).result()
    if active(data["config"]["tag"]):
        raise RecipeError("Resources are still active; rerun stop")
    args.state.unlink()
    print("Stopped orchestrator and workers; removed local deployment record")


def main():
    os.umask(0o077)
    os.environ["CWSANDBOX_DISABLE_SIGNAL_HANDLERS"] = "1"
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=ROOT / ".deployment.json")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("deploy", help="Deploy one sandbox-hosted orchestrator")
    create.add_argument("--environment", required=True)
    create.add_argument("--orchestrator-hours", type=int, default=24)
    create.add_argument("--worker-hours", type=int, default=8)
    create.add_argument("--idle-minutes", type=int, default=15)
    create.add_argument("--max-workers", type=int, default=4)
    sub.add_parser("status", help="Show connection state and active workers")
    logs = sub.add_parser("logs", help="Show recent orchestrator or worker logs")
    logs.add_argument("--worker")
    logs.add_argument("--bootstrap", action="store_true")
    read = sub.add_parser("read", help="Read a UTF-8 file from an active worker")
    read.add_argument("--worker", required=True)
    read.add_argument("--path", required=True)
    sub.add_parser("stop", help="Stop this deployment, including all its workers")
    sub.add_parser("supervise", help="Internal deployment command")
    sub.add_parser("spawn", help="Internal deployment command")
    args = parser.parse_args()
    if args.command == "supervise":
        supervise()
        return
    if args.command == "spawn":
        sys.exit(spawn(json.loads((REMOTE / "config.json").read_text())))
    if not os.environ.get("CWSANDBOX_API_KEY"):
        raise RecipeError("Set CWSANDBOX_API_KEY")
    if args.command in ("deploy", "stop"):
        args.state.parent.mkdir(parents=True, exist_ok=True)
        with args.state.with_suffix(".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RecipeError("Another deployment operation is running") from None
            if args.command == "deploy":
                deploy(args)
            else:
                stop(args, deployment(args.state))
        return
    data = deployment(args.state)
    box = get_box(data["sandbox_id"])
    if args.command == "status":
        print("Orchestrator: " + box.sandbox_id + " (" + str(box.get_status()) + ")")
        print(
            "Expires: "
            + time.strftime(
                "%Y-%m-%d %H:%M:%S UTC",
                time.gmtime(data["created_at"] + data["config"]["orchestrator_hours"] * 3600),
            )
        )
        try:
            h = health(box)
            print("Connected: " + str(h.get("connected", False)))
        except Exception:
            print("Connected: unavailable; inspect logs")
        for worker in active(data["config"]["tag"] + ":worker"):
            print("Worker: " + worker.sandbox_id + " (" + str(worker.status) + ")")
    elif args.command == "logs":
        if args.worker:
            box = select_worker(data, args.worker)
        path = (
            "/workspace/bootstrap.log"
            if args.bootstrap
            else ("/workspace/runner.log" if args.worker else str(STATE / "orchestrator.log"))
        )
        result = box.exec(["tail", "-n", "60", path], timeout_seconds=15).result()
        if result.returncode:
            raise RecipeError("Log unavailable; try --bootstrap while setup is running")
        print(result.stdout, end="")
    elif args.command == "read":
        box = select_worker(data, args.worker)
        try:
            content = box.read_file(args.path).result()
        except SandboxFileError:
            raise RecipeError(
                "Cannot read file. Check that the path exists and the worker is still running."
            ) from None
        print(content.decode(), end="")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # SDK exceptions can embed request bodies containing credentials.
        message = str(error) if isinstance(error, RecipeError) else type(error).__name__
        print("Error: " + message, file=sys.stderr)
        sys.exit(1)
