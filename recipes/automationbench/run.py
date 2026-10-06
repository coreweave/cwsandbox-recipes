"""Harbor owns trial retries; quota failures never trigger pass-seeking retries."""

import argparse
import asyncio
import json
import os
import re
import shutil
from pathlib import Path

from harbor.job import Job
from harbor.models.job.config import JobConfig, RetryConfig
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig
from harbor.trial.hooks import TrialEvent


def retry_policy():
    return RetryConfig(
        max_retries=5,
        include_exceptions={
            "SandboxResourceExhaustedError",
            "SandboxUnavailableError",
            "SandboxRequestTimeoutError",
            "EnvironmentStartTimeoutError",
            "ApiRateLimitError",
            "ApiUsageLimitError",
        },
        exclude_exceptions=RetryConfig().exclude_exceptions - {"ApiUsageLimitError"},
        min_wait_sec=30,
        wait_multiplier=2,
        max_wait_sec=300,
    )


def build_task(root, task_name):
    path = root / task_name.replace(".", "-")
    (path / "environment").mkdir(parents=True)
    (path / "tests").mkdir()
    (path / "instruction.md").write_text(json.dumps({"task": task_name}))
    (path / "task.toml").write_text(
        """schema_version = "1.4"
[task]
name = "automationbench/"""
        + task_name.replace(".", "-")
        + """"
version = "1.0.0"
authors = []
[agent]
timeout_sec = 2700
[verifier]
timeout_sec = 60
[environment]
docker_image = "python:3.13-slim-bookworm"
cpus = 2
memory_mb = 4096
build_timeout_sec = 240
"""
    )
    (path / "tests/test.sh").write_text("""#!/bin/bash
set -euo pipefail
python - <<'PYTHON'
import json
from pathlib import Path
data = json.loads(Path('/logs/agent/automationbench.json').read_text())
assert len(data['tasks']) == 1
task = data['tasks'][0]
Path('/logs/verifier/reward.json').write_text(json.dumps({'pass': int(task['passed']), 'partial_credit': task['score']}))
PYTHON
""")
    return path


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--tasks-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inject-quota-once", action="store_true")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    names = args.task + (
        args.tasks_file.read_text().splitlines() if args.tasks_file else []
    )
    names = [name.strip() for name in names if name.strip()]
    if not names or len(names) != len(set(names)):
        parser.error("Select at least one task, without duplicates")
    if args.concurrency < 1:
        parser.error("Concurrency must be positive")
    if any(not re.fullmatch(r"[a-z]+\.[a-z0-9_]+", name) for name in names):
        parser.error("Invalid task name")
    for key in ("CWSANDBOX_API_KEY", "MODEL_API_KEY", "MODEL_NAME", "MODEL_BASE_URL"):
        if not os.environ.get(key):
            parser.error(f"Set {key} before running")
    output = args.output.resolve()
    output.mkdir(exist_ok=False)
    (output / "sandbox-ids.jsonl").touch()
    os.environ["HARBOR_SANDBOX_AUDIT"] = str(output / "sandbox-ids.jsonl")
    tasks = [build_task(output / "tasks", name) for name in names]
    config = JobConfig(
        job_name="automationbench",
        jobs_dir=output / "jobs",
        n_attempts=1,
        n_concurrent_trials=args.concurrency,
        retry=retry_policy(),
        environment=EnvironmentConfig(
            import_path="environment:QuotaOnceEnvironment"
            if args.inject_quota_once
            else "environment:CoreWeaveEnvironment",
            delete=True,
            kwargs={
                "max_lifetime_seconds": 3600,
                "request_timeout_seconds": 3100,
            },
        ),
        agents=[
            AgentConfig(
                import_path="adapter:NativeAutomationBenchAgent",
                model_name=os.environ["MODEL_NAME"],
                override_setup_timeout_sec=660,
            )
        ],
        tasks=[TaskConfig(path=p) for p in tasks],
    )
    (output / "job-config.json").write_text(config.model_dump_json(indent=2))
    archive = output / "attempts"

    async def preserve_attempt(event):
        destination = archive / event.trial_name / str(event.trial_id)
        source = config.jobs_dir / config.job_name / event.trial_name
        destination.mkdir(parents=True)
        if source.exists():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        (destination / "attempt-result.json").write_text(
            event.result.model_dump_json(indent=2)
        )

    job = await Job.create(config)
    job.add_hook(TrialEvent.END, preserve_attempt)
    result = await job.run()
    (output / "job-result.json").write_text(result.model_dump_json(indent=2))
    print("Harbor job finished; inspect job-result.json and archived attempts.")
    if result.stats.n_errored_trials or result.stats.n_completed_trials != len(tasks):
        raise SystemExit(
            "Harbor evaluation still has failed or missing trials; see job-result.json"
        )


if __name__ == "__main__":
    asyncio.run(main())
