"""Exercise the real Harbor environment interface against an SDK test double."""

import asyncio
import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from cwsandbox import AuthStrategy, SandboxResourceExhaustedError
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from environment import CoreWeaveEnvironment


class EnvironmentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.audit = self.root / "sandbox-ids.jsonl"
        self.env_vars = {
            "CWSANDBOX_API_KEY": "host-only-token",
            "MODEL_API_KEY": "inference-only-token",
            "MODEL_NAME": "example",
            "MODEL_BASE_URL": "https://example.invalid/v1",
            "HARBOR_SANDBOX_AUDIT": str(self.audit),
        }
        self.environment = CoreWeaveEnvironment(
            environment_dir=self.root,
            environment_name="example",
            session_id="test-session",
            trial_paths=TrialPaths(trial_dir=self.root / "trial"),
            task_env_config=EnvironmentConfig(
                docker_image="python:3.13-slim-bookworm",
                cpus=2,
                memory_mb=4096,
                build_timeout_sec=240,
            ),
        )
        self.sandbox = Mock(sandbox_id="owned-test-sandbox", status="running")
        self.sandbox.start = AsyncMock()
        self.sandbox.exec = AsyncMock(
            return_value=SimpleNamespace(stdout="", stderr="", returncode=0)
        )
        self.sandbox.read_file = AsyncMock(return_value=b"content")
        self.sandbox.write_file = AsyncMock()
        self.sandbox.stop = AsyncMock()

    async def test_start_injects_only_model_credentials_and_records_id_before_wait(
        self,
    ):
        self.sandbox.wait.side_effect = lambda **kwargs: self.assertTrue(
            self.audit.exists()
        )
        with (
            patch.dict(os.environ, self.env_vars, clear=True),
            patch("environment.Sandbox", return_value=self.sandbox) as sdk,
        ):
            await self.environment.start()
            kwargs = sdk.call_args.kwargs
            self.assertIs(kwargs["auth"], AuthStrategy.COREWEAVE_API_KEY)
            self.assertEqual(
                kwargs["environment_variables"]["MODEL_API_KEY"], "inference-only-token"
            )
            self.assertNotIn("CWSANDBOX_API_KEY", kwargs["environment_variables"])
            self.assertEqual(
                kwargs["resources"]["limits"], {"cpu": "2", "memory": "4Gi"}
            )
            self.assertEqual(
                json.loads(self.audit.read_text())["sandbox_id"], "owned-test-sandbox"
            )

    async def test_failed_start_deletes_and_preserves_original_error(self):
        self.sandbox.start.side_effect = SandboxResourceExhaustedError("quota")
        with (
            patch.dict(os.environ, self.env_vars, clear=True),
            patch("environment.Sandbox", return_value=self.sandbox) as sdk,
        ):
            sdk.delete = AsyncMock()
            with self.assertRaises(SandboxResourceExhaustedError):
                await self.environment.start()
            sdk.delete.assert_awaited_once_with(
                "owned-test-sandbox",
                auth=AuthStrategy.COREWEAVE_API_KEY,
                missing_ok=True,
            )
            self.assertTrue(self.audit.exists())

    async def test_cancelled_creation_finishes_then_deletes(self):
        started = asyncio.Event()
        release = asyncio.Event()

        async def create():
            started.set()
            await release.wait()

        self.sandbox.start.side_effect = create
        with (
            patch.dict(os.environ, self.env_vars, clear=True),
            patch("environment.Sandbox", return_value=self.sandbox) as sdk,
        ):
            sdk.delete = AsyncMock()
            task = asyncio.create_task(self.environment.start())
            await started.wait()
            task.cancel()
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            sdk.delete.assert_awaited_once()
            self.assertTrue(self.audit.exists())

    async def test_exec_and_file_transfer_use_public_sdk(self):
        self.environment.sandbox = self.sandbox
        result = await self.environment.exec(
            "printf test",
            cwd="/workspace",
            timeout_sec=12,
            env={"SAFE": "value with spaces"},
        )
        self.assertEqual(result.return_code, 0)
        self.sandbox.exec.assert_awaited_with(
            ["bash", "-lc", "export SAFE='value with spaces'; printf test"],
            cwd="/workspace",
            timeout_seconds=12,
        )
        source = self.root / "source.txt"
        source.write_bytes(b"input")
        await self.environment.upload_file(source, "/tests/source.txt")
        self.sandbox.write_file.assert_awaited_with("/tests/source.txt", b"input")
        await self.environment.download_file(
            "/logs/result.txt", self.root / "download/result.txt"
        )
        self.assertEqual((self.root / "download/result.txt").read_bytes(), b"content")

    async def test_download_rejects_archive_path_traversal(self):
        self.environment.sandbox = self.sandbox
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            member = tarfile.TarInfo("../escaped.txt")
            member.size = 3
            archive.addfile(member, io.BytesIO(b"bad"))
        self.sandbox.read_file.return_value = buffer.getvalue()
        with self.assertRaises(tarfile.OutsideDestinationError):
            await self.environment.download_dir("/logs", self.root / "download")
        self.assertFalse((self.root / "escaped.txt").exists())
        self.assertTrue(
            self.sandbox.exec.call_args.args[0][-1].startswith(
                "rm -f /tmp/automationbench-"
            )
        )

    async def test_directory_round_trip(self):
        self.environment.sandbox = self.sandbox
        source = self.root / "source"
        source.mkdir()
        (source / "test.sh").write_text("echo hello")
        await self.environment.upload_dir(source, "/tests")
        archive = self.sandbox.write_file.call_args.args[1]
        self.sandbox.read_file.return_value = archive
        await self.environment.download_dir("/tests", self.root / "download")
        self.assertEqual((self.root / "download/test.sh").read_text(), "echo hello")


if __name__ == "__main__":
    unittest.main()
