"""Run independent browser agents, collect artifacts, and stop their sandboxes."""

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

from cwsandbox import AuthStrategy, Sandbox

HERE = Path(__file__).resolve().parent
AUTH = AuthStrategy.WANDB
BOOTSTRAP = """
set -eu
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq chromium ca-certificates fonts-liberation
python -m venv /opt/browser
/opt/browser/bin/pip install --quiet -r /work/requirements.txt
useradd --create-home browser
chown -R browser:browser /work
"""


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def check_environment():
    for name in ("WANDB_API_KEY", "WANDB_ENTITY"):
        value = os.environ.get(name, "").strip()
        if not value or value.startswith("["):
            raise ValueError(f"Set {name} before running this recipe")
    # This recipe targets the hosted W&B service for both compute and inference.
    for name in ("WANDB_BASE_URL", "WANDB_HOST"):
        os.environ.pop(name, None)


def exec_logged(sandbox, command, folder, name, timeout):
    result = sandbox.exec(command, timeout_seconds=timeout).result(timeout=timeout + 60)
    key = os.environ["WANDB_API_KEY"]
    for suffix, text in (("stdout", result.stdout), ("stderr", result.stderr)):
        (folder / f"{name}.{suffix}.log").write_text(text.replace(key, "[REDACTED]"))
    return result.returncode


def run_worker(index, output):
    folder = output / f"worker-{index}"
    folder.mkdir()
    summary = {"worker": index, "passed": False, "stopped": False}
    started = time.monotonic()
    sandbox = Sandbox(
        auth=AUTH,
        placement_mode="serverless",
        container_image="python:3.12-bookworm",
        resources={
            "requests": {"cpu": "2", "memory": "4Gi"},
            "limits": {"cpu": "2", "memory": "4Gi"},
        },
        max_lifetime_seconds=1800,
        tags=["browser-use-recipe"],
        environment_variables={
            "MODEL_API_KEY": os.environ["WANDB_API_KEY"],
            "INFERENCE_PROJECT": (
                f"{os.environ['WANDB_ENTITY']}/{os.environ.get('WANDB_PROJECT', 'browser-use')}"
            ),
            "ANONYMIZED_TELEMETRY": "false",
            "BROWSER_USE_CLOUD_SYNC": "false",
        },
    )
    try:
        sandbox.start().result(timeout=180)
        save(folder / "sandbox.json", {"sandbox_id": sandbox.sandbox_id})
        sandbox.wait(timeout=180)
        summary["ready_seconds"] = round(time.monotonic() - started, 3)
        for name in ("agent.py", "verify.py", "requirements.txt"):
            sandbox.write_file(f"/work/{name}", (HERE / name).read_bytes()).result(timeout=60)
        setup_started = time.monotonic()
        if exec_logged(sandbox, ["bash", "-ec", BOOTSTRAP], folder, "setup", 900):
            raise RuntimeError("Setup failed; inspect setup logs")
        summary["setup_seconds"] = round(time.monotonic() - setup_started, 3)
        code = exec_logged(
            sandbox,
            ["runuser", "-u", "browser", "--", "/opt/browser/bin/python", "/work/agent.py"],
            folder,
            "agent",
            360,
        )
        for name in ("result.json", "history.json", "final.png"):
            try:
                data = sandbox.read_file(f"/work/output/{name}").result(timeout=60)
                # Scan textual artifacts as well as logs before saving locally.
                data = data.replace(os.environ["WANDB_API_KEY"].encode(), b"[REDACTED]")
                (folder / name).write_bytes(data)
            except Exception as exc:
                summary.setdefault("artifact_errors", {})[name] = type(exc).__name__
        if (folder / "result.json").exists():
            result = json.loads((folder / "result.json").read_text())
            summary["passed"] = code == 0 and result["passed"]
            summary["agent_seconds"] = result["agent_seconds"]
        summary["exit_code"] = code
    except Exception as exc:
        summary["error_type"] = type(exc).__name__
    finally:
        if sandbox.sandbox_id:
            save(folder / "sandbox.json", {"sandbox_id": sandbox.sandbox_id})
            try:
                sandbox.stop().result(timeout=60)
                summary["stopped"] = True
            except Exception as exc:
                summary["cleanup_error"] = type(exc).__name__
        else:
            summary["stopped"] = True
        summary["total_seconds"] = round(time.monotonic() - started, 3)
        save(folder / "summary.json", summary)
    return summary


def cleanup(output):
    state = json.loads((output / "run.json").read_text())
    if state["entity"] != os.environ["WANDB_ENTITY"]:
        raise ValueError("WANDB_ENTITY must match the team in run.json")
    failed = False
    for path in sorted(output.glob("worker-*/sandbox.json")):
        sandbox_id = json.loads(path.read_text())["sandbox_id"]
        try:
            sandbox = Sandbox.from_id(sandbox_id, auth=AUTH).result(timeout=60)
            sandbox.stop().result(timeout=60)
            print(f"{path.parent.name}: stopped")
        except Exception as exc:
            # Do not treat an ambiguous lookup or authorization failure as successful cleanup.
            print(f"{path.parent.name}: cleanup failed ({type(exc).__name__})", file=sys.stderr)
            failed = True
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--output", type=Path, default=Path("outputs/browser-use"))
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()
    check_environment()
    if args.cleanup:
        return cleanup(args.output)
    # Never overwrite the IDs needed to clean up a previous run.
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "run.json", {"entity": os.environ["WANDB_ENTITY"], "workers": args.workers})
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_worker, index, args.output) for index in range(args.workers)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    summary = {
        "workers": sorted(results, key=lambda row: row["worker"]),
        "passed": sum(row["passed"] for row in results),
        "total_seconds": round(time.monotonic() - started, 3),
    }
    save(args.output / "summary.json", summary)
    print(f"{summary['passed']}/{args.workers} passed; artifacts: {args.output}")
    return 0 if all(row["passed"] and row["stopped"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
