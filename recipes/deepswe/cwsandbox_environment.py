"""Pier environment using prebuilt task images on CoreWeave Sandbox."""

import asyncio
import re
import shlex
import tarfile
import tempfile
import uuid
from pathlib import Path

from cwsandbox import AuthStrategy, NetworkOptions, Sandbox
from pier.environments.base import BaseEnvironment, ExecResult
from pier.environments.capabilities import EnvironmentCapabilities


class CWSandboxEnvironment(BaseEnvironment):
    def __init__(self, *args, max_lifetime_seconds=5400, **kwargs):
        self.sandbox = None
        self.max_lifetime_seconds = max_lifetime_seconds
        super().__init__(*args, **kwargs)

    @staticmethod
    def type():
        return "cwsandbox"

    @property
    def capabilities(self):
        return EnvironmentCapabilities(disable_internet=True)

    def _validate_definition(self):
        self.verifier_files = []
        if not self.task_env_config.docker_image:
            # DeepSWE's verifier Dockerfile only adds four files to the same
            # task image. Reproduce those operations in a fresh sandbox.
            dockerfile = (self.environment_dir / "Dockerfile").read_text()
            lines = [
                s.strip()
                for s in dockerfile.splitlines()
                if s.strip() and not s.lstrip().startswith("#")
            ]
            files = ["test.sh", "test.patch", "grader.py", "config.json"]
            expected = [f"COPY {name} /tests/{name}" for name in files] + [
                "RUN chmod +x /tests/test.sh"
            ]
            if (
                not lines
                or not re.fullmatch(r"FROM \S+", lines[0])
                or lines[1:] != expected
            ):
                raise ValueError(
                    "Unsupported verifier Dockerfile; refusing to omit build steps"
                )
            self.task_env_config.docker_image = lines[0].split()[1]
            self.verifier_files = files
        if self.default_user not in (None, "root", 0, "0"):
            raise ValueError("This adapter supports root task images only")

    async def start(self, force_build=False):
        if force_build:
            raise ValueError(
                "This adapter uses prebuilt images; force_build is unsupported"
            )
        config = self.task_env_config
        # No inference credentials are injected into either task environment.
        self.sandbox = await asyncio.to_thread(
            Sandbox.run,
            container_image=config.docker_image,
            auth=AuthStrategy.COREWEAVE_API_KEY,
            placement_mode="serverless",
            resources={"cpu": str(config.cpus), "memory": f"{config.memory_mb}Mi"},
            network=NetworkOptions(deny_egress=not config.allow_internet),
            max_lifetime_seconds=self.max_lifetime_seconds,
            tags=["deepswe", self.session_id],
            environment_variables=self._persistent_env,
        )
        self.logger.info("Created sandbox %s", self.sandbox.sandbox_id)
        await asyncio.to_thread(self.sandbox.wait)
        await self.exec("mkdir -p /logs/agent /logs/verifier /logs/artifacts /tests")
        for name in self.verifier_files:
            await self.upload_file(self.environment_dir / name, f"/tests/{name}")
        if self.verifier_files:
            await self.exec("chmod +x /tests/test.sh")

    async def stop(self, delete=True):
        if self.sandbox is not None:
            await asyncio.to_thread(lambda: self.sandbox.stop(missing_ok=True).result())
            self.logger.info("Stopped sandbox %s", self.sandbox.sandbox_id)
            self.sandbox = None

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        if self._resolve_user(user) not in (None, "root", 0, "0"):
            raise ValueError("Non-root exec is unsupported")
        merged = self._merge_env(env) or {}
        argv = ["env", *[f"{k}={v}" for k, v in merged.items()], "bash", "-c", command]
        result = await asyncio.to_thread(
            lambda: self.sandbox.exec(
                argv, cwd=cwd, timeout_seconds=timeout_sec
            ).result()
        )
        return ExecResult(
            stdout=result.stdout, stderr=result.stderr, return_code=result.returncode
        )

    async def upload_file(self, source_path, target_path):
        parent = shlex.quote(str(Path(target_path).parent))
        await self.exec(f"mkdir -p {parent}")
        await asyncio.to_thread(
            lambda: self.sandbox.write_file(
                target_path, Path(source_path).read_bytes()
            ).result()
        )

    async def download_file(self, source_path, target_path):
        data = await asyncio.to_thread(
            lambda: self.sandbox.read_file(source_path).result()
        )
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    async def upload_dir(self, source_dir, target_dir):
        remote = f"/tmp/pier-{uuid.uuid4().hex}.tar.gz"
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "transfer.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(source_dir, arcname=".")
            await self.upload_file(archive, remote)
            try:
                target = shlex.quote(target_dir)
                result = await self.exec(
                    f"mkdir -p {target} && tar xzf {remote} -C {target}"
                )
                if result.return_code:
                    raise RuntimeError("Failed to extract uploaded directory")
            finally:
                await self.exec(f"rm -f {remote}")

    async def download_dir(self, source_dir, target_dir):
        remote = f"/tmp/pier-{uuid.uuid4().hex}.tar.gz"
        result = await self.exec(f"tar czf {remote} -C {shlex.quote(source_dir)} .")
        if result.return_code:
            raise RuntimeError(f"Failed to archive {source_dir}")
        try:
            with tempfile.TemporaryDirectory() as temp:
                archive = Path(temp) / "transfer.tar.gz"
                await self.download_file(remote, archive)
                Path(target_dir).mkdir(parents=True, exist_ok=True)
                with tarfile.open(archive) as tar:
                    tar.extractall(target_dir, filter="data")
        finally:
            await self.exec(f"rm -f {remote}")
