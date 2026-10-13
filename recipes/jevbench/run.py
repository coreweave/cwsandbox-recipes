"""Run JevBench's public original cohort in a disposable CPU sandbox."""

import argparse
import hashlib
import io
import json
import math
import os
import subprocess
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

from cwsandbox import AuthStrategy, Sandbox
from dotenv import load_dotenv

from evaluate import MODEL

ROOT = Path(__file__).resolve().parent
REPOSITORY = "https://github.com/fstandhartinger/jevbench.git"
REVISION = "bb05a335bc809e61b20c0f745d25499a82b326fc"
IMAGE = "python:3.12-slim-bookworm"
MAX_LIFETIME_SECONDS = 1200
CLEANUP_RESERVE_SECONDS = 300


def source_archive():
    """Fetch only the pinned public source; do not depend on a local checkout."""
    with tempfile.TemporaryDirectory(prefix="jevbench-") as temp:
        for command in [
            ["git", "init", "--quiet", temp],
            ["git", "-C", temp, "fetch", "--quiet", "--depth=1", REPOSITORY, REVISION],
        ]:
            subprocess.run(command, capture_output=True, check=True, timeout=120)
        return subprocess.run(
            [
                "git",
                "-C",
                temp,
                "archive",
                "--format=tar.gz",
                REVISION,
                "jevbench",
                "datasets/public/original.jsonl",
                "LICENSE",
            ],
            capture_output=True,
            check=True,
            timeout=30,
        ).stdout


def extract_results(blob, output, secrets):
    """Validate the entire export before writing any sandbox-produced artifact."""
    files = []
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        for member in archive.getmembers():
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or path.parts[:1] != ("output",):
                raise ValueError("Unsafe result path")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError("Result archive contains a non-regular file")
            data = archive.extractfile(member).read()
            if any(secret.encode() in data for secret in secrets):
                raise ValueError("Credential found in results; export rejected")
            files.append((path, data))
    for path, data in files:
        destination = output / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)


def run(args):
    secrets = tuple(
        os.environ.get(name, "").strip() for name in ("CWSANDBOX_API_KEY", "ANTHROPIC_API_KEY")
    )
    if not all(secrets):
        raise ValueError("Set CWSANDBOX_API_KEY and ANTHROPIC_API_KEY before running")

    def redact(text):
        for secret in secrets:
            text = text.replace(secret, "[REDACTED]")
        return text

    args.output.mkdir(parents=True, exist_ok=False)
    source = source_archive()
    (args.output / "source.tar.gz").write_bytes(source)
    manifest = {
        "source_repository": REPOSITORY,
        "source_revision": REVISION,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "model": MODEL,
        "image": IMAGE,
        "resources": {"cpu": "2", "memory": "4Gi"},
        "placement": "serverless",
        "max_lifetime_seconds": MAX_LIFETIME_SECONDS,
        "tasks": args.limit,
        "budget_usd": args.budget_usd,
        "scope": "public original cohort; not the official held-out leaderboard",
        "inference": "Anthropic native Messages API; schema-constrained verbalized probabilities",
        "started_utc": datetime.now(UTC).isoformat(),
    }

    def save():
        (args.output / "run.json").write_text(redact(json.dumps(manifest, indent=2)) + "\n")

    def execute(label, command, timeout=60):
        process = sandbox.exec(command, timeout_seconds=timeout)
        result = process.result(timeout=timeout + 30)
        log = redact((result.stdout_bytes + result.stderr_bytes).decode(errors="replace"))
        (args.output / f"{label}.log").write_text(log)
        if process.returncode:
            raise RuntimeError(f"{label} exited {process.returncode}; see {label}.log")

    sandbox = Sandbox(
        command="sleep",
        args=["infinity"],
        auth=AuthStrategy.COREWEAVE_API_KEY,
        container_image=IMAGE,
        resources=manifest["resources"],
        placement_mode="serverless",
        max_lifetime_seconds=MAX_LIFETIME_SECONDS,
        tags=["jevbench-recipe"],
        environment_variables={"ANTHROPIC_API_KEY": secrets[1], "PYTHONUNBUFFERED": "1"},
    )
    save()
    # Start conservatively before provisioning so startup also consumes the time budget.
    deadline = monotonic() + MAX_LIFETIME_SECONDS
    try:
        sandbox.start().result(timeout=120)
        manifest["sandbox_id"] = sandbox.sandbox_id
        save()
        print(f"Sandbox: {sandbox.sandbox_id}", flush=True)
        sandbox.wait(timeout=120)
        execute("prepare", ["mkdir", "-p", "/work/jevbench", "/work/output"])
        sandbox.write_file("/work/source.tar.gz", source).result(timeout=60)
        execute("extract", ["tar", "xzf", "/work/source.tar.gz", "-C", "/work/jevbench"])
        for name in ("transport.py", "evaluate.py"):
            contents = (ROOT / name).read_bytes()
            sandbox.write_file(f"/work/{name}", contents).result(timeout=30)
            manifest[f"{name}_sha256"] = hashlib.sha256(contents).hexdigest()
        # Leave time for the exec result, archive, download, and stop operations.
        evaluation_timeout = min(900, int(deadline - monotonic()) - CLEANUP_RESERVE_SECONDS)
        if evaluation_timeout <= 0:
            raise TimeoutError("Setup exhausted the evaluation time budget")
        manifest["evaluation_timeout_seconds"] = evaluation_timeout
        save()
        print(f"Evaluating {args.limit} public tasks with {MODEL}...", flush=True)
        execute(
            "evaluation",
            [
                "python",
                "/work/evaluate.py",
                "--limit",
                str(args.limit),
                "--budget-usd",
                str(args.budget_usd),
            ],
            timeout=evaluation_timeout,
        )
        manifest["evaluation_completed"] = True
    except (Exception, KeyboardInterrupt) as error:
        manifest["error"] = redact(str(error)) or type(error).__name__
    finally:
        if sandbox.sandbox_id:
            manifest["sandbox_id"] = sandbox.sandbox_id
            try:
                execute("archive", ["tar", "czf", "/work/results.tar.gz", "-C", "/work", "output"])
                blob = sandbox.read_file("/work/results.tar.gz").result(timeout=60)
                extract_results(blob, args.output, secrets)
                manifest["artifacts_retrieved"] = True
            except Exception as error:
                manifest["artifact_error"] = redact(str(error))
            finally:
                try:
                    sandbox.stop(missing_ok=True).result(timeout=60)
                    manifest["sandbox_stopped"] = True
                except Exception as error:
                    manifest["cleanup_error"] = redact(str(error))
        manifest["finished_utc"] = datetime.now(UTC).isoformat()
        save()
    success = all(
        manifest.get(key)
        for key in ("evaluation_completed", "artifacts_retrieved", "sandbox_stopped")
    )
    if success:
        print((args.output / "output/verification.json").read_text())
        print(f"Results: {args.output}\nSandbox stopped.")
    else:
        print("Run incomplete; inspect run.json and evaluation.log in", args.output)
    return 0 if success else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"),
    )
    parser.add_argument("--limit", type=int, default=72, help="First 1-72 public original tasks")
    parser.add_argument(
        "--budget-usd", type=float, default=2, help="Inference ledger cap; excludes compute"
    )
    args = parser.parse_args()
    if not 1 <= args.limit <= 72:
        parser.error("--limit must be between 1 and 72")
    if not math.isfinite(args.budget_usd) or args.budget_usd <= 0:
        parser.error("--budget-usd must be finite and positive")
    load_dotenv(ROOT / ".env")
    try:
        return run(args)
    except Exception as error:
        # Avoid printing SDK/subprocess exception details before redaction is available.
        print(
            f"Setup failed ({type(error).__name__}); check credentials, Git access, and output path."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
