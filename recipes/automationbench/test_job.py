"""Run Harbor's real scheduling/verifier path with a local SDK double, no cloud."""

import asyncio
import json
import os
import re
import shlex
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from harbor.job import Job

import environment
import run


class LocalSandbox:
    def __init__(self, root, **kwargs):
        self.root = root
        self.sandbox_id = "offline-sandbox"
        self.status = "running"
        self.config = kwargs
        self.model_runs = 0
        self.mapped_scripts = set()
        for directory in [
            "tmp",
            "logs/agent",
            "logs/verifier",
            "workspace/AutomationBench",
        ]:
            (root / directory).mkdir(parents=True, exist_ok=True)

    def path(self, remote):
        return self.root / str(remote).lstrip("/")

    async def start(self):
        pass

    def wait(self, **kwargs):
        return self

    async def write_file(self, name, content):
        target = self.path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    async def read_file(self, name):
        return self.path(name).read_bytes()

    async def exec(self, argv, cwd=None, timeout_seconds=None):
        command = argv[-1]
        if command.startswith("apt-get"):
            return SimpleNamespace(stdout="offline setup", stderr="", returncode=0)
        if "auto-bench" in command:
            self.model_runs += 1
            args = shlex.split(command)
            name = args[args.index("--tasks") + 1]
            await self.write_file(
                "/logs/agent/automationbench.json",
                json.dumps(
                    {"tasks": [{"name": name, "passed": False, "score": 0.25}]}
                ).encode(),
            )
            await self.write_file(
                "/logs/agent/eval.log", b"offline model assertion failure\n"
            )
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        script = self.path("/tests/test.sh")
        if (
            "/tests/test.sh" in command
            and script.exists()
            and script not in self.mapped_scripts
        ):
            script.write_text(
                script.read_text().replace("/logs/", str(self.path("/logs")) + "/")
            )
            self.mapped_scripts.add(script)
        # All remote absolute paths map to a temporary directory. Run the real
        # generated verifier and transfer commands, never installation or inference.
        command = re.sub(
            r"/(workspace|solution|tests|logs|tmp)(?=/|[\s'\"]|$)",
            lambda match: str(self.path(match.group())),
            command,
        )
        process = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            cwd=self.path(cwd) if cwd else self.root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=timeout_seconds or 30
        )
        return SimpleNamespace(
            stdout=stdout.decode(),
            stderr=stderr.decode(),
            returncode=process.returncode,
        )


class JobTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_harbor_retry_archive_verifier_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sandbox = LocalSandbox(root / "remote")
            output = root / "result"
            native_create = Job.create

            async def create(config):
                config.retry.min_wait_sec = 0
                config.retry.max_wait_sec = 0
                return await native_create(config)

            environment.QuotaOnceEnvironment.injected = False
            with (
                patch.dict(
                    os.environ,
                    {
                        "PATH": os.environ["PATH"],
                        "CWSANDBOX_API_KEY": "offline-sandbox-key",
                        "MODEL_API_KEY": "offline-model-key",
                        "MODEL_NAME": "offline-model",
                        "MODEL_BASE_URL": "https://example.invalid/v1",
                    },
                    clear=True,
                ),
                patch(
                    "sys.argv",
                    [
                        "run.py",
                        "--task",
                        "simple.email_sf_contact_phone_update",
                        "--output",
                        str(output),
                        "--concurrency",
                        "1",
                        "--inject-quota-once",
                    ],
                ),
                patch("environment.Sandbox", return_value=sandbox) as sdk,
                patch.object(run.Job, "create", side_effect=create),
            ):
                sdk.delete = AsyncMock()
                try:
                    await run.main()
                except SystemExit:
                    details = [
                        json.loads(p.read_text()).get("exception_info")
                        for p in (output / "attempts").glob("*/*/attempt-result.json")
                    ]
                    self.fail(str(details))
                sdk.delete.assert_awaited()
            result = json.loads((output / "job-result.json").read_text())
            self.assertEqual(result["stats"]["n_errored_trials"], 0)
            self.assertEqual(result["stats"]["n_completed_trials"], 1)
            attempts = [
                json.loads(p.read_text())
                for p in (output / "attempts").glob("*/*/attempt-result.json")
            ]
            self.assertEqual(len(attempts), 2)
            self.assertEqual(sum(a["exception_info"] is not None for a in attempts), 1)
            self.assertEqual(
                sandbox.model_runs, 1, "Ordinary failed assertions must not retry"
            )
            reward = json.loads(
                next((output / "jobs").glob("*/*/verifier/reward.json")).read_text()
            )
            self.assertEqual(reward, {"pass": 0, "partial_credit": 0.25})
            self.assertTrue((output / "sandbox-ids.jsonl").read_text().strip())
