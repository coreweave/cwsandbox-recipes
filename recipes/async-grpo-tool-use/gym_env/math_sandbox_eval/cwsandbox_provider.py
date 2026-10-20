"""NeMo-Gym sandbox provider for CoreWeave Sandboxes (``cwsandbox`` SDK).

It implements Gym's ``SandboxProvider`` protocol, so any Gym environment that uses
``AsyncSandbox`` can run its work in CoreWeave sandboxes. Select it in a server config:

    sandbox:
      cwsandbox:
        runner_id: my-runner        # CKS runner to place sandboxes on
        tags: [my-run]              # added to every sandbox, for cleanup
        placement: cks              # cks (default), serverless, or cks_then_serverless

``serverless`` ignores the runner. ``cks_then_serverless`` tries the runner first and retries a
create on serverless when the runner cannot place it.

Authentication comes from ``CWSANDBOX_API_KEY`` in the environment.
"""

from __future__ import annotations

import asyncio
import shlex
import time
from pathlib import Path
from typing import Any

from nemo_gym.sandbox import (
    SandboxCreateError,
    SandboxExecResult,
    SandboxHandle,
    SandboxSpec,
    SandboxStatus,
    register_provider,
)

_STATUS = {
    "creating": SandboxStatus.STARTING,
    "pending": SandboxStatus.STARTING,
    "preparing": SandboxStatus.STARTING,
    "running": SandboxStatus.RUNNING,
    "completed": SandboxStatus.STOPPED,
    "terminated": SandboxStatus.STOPPED,
    "terminating": SandboxStatus.STOPPED,
    "failed": SandboxStatus.ERROR,
}


PLACEMENTS = {
    "cks": {"placement_mode": "cks", "placement_spillover": "strict"},
    "cks_then_serverless": {"placement_mode": "cks", "placement_spillover": "cks_then_serverless"},
    "serverless": {"placement_mode": "serverless", "placement_spillover": "strict"},
}


def placement_kwargs(placement: str, runner_id: str) -> dict[str, Any]:
    """``cwsandbox.Sandbox.run`` placement arguments. Serverless takes no runner pin."""
    kwargs = dict(PLACEMENTS[placement])
    if placement != "serverless":
        kwargs["runner_ids"] = [runner_id]
    return kwargs


class CWSandboxProvider:
    name = "cwsandbox"

    def __init__(
        self,
        runner_id: str,
        tags: list[str] | None = None,
        default_image: str = "python:3.12-slim",
        max_lifetime_s: int = 6 * 3600,
        ready_timeout_s: float = 600.0,
        placement: str = "cks",
    ) -> None:
        if placement not in PLACEMENTS:
            raise ValueError(f"placement must be one of {sorted(PLACEMENTS)}, got {placement!r}")
        self.placement = placement
        self.runner_id = runner_id
        self.tags = [str(tag) for tag in (tags or []) if tag]
        self.default_image = default_image
        self.max_lifetime_s = max_lifetime_s
        self.ready_timeout_s = ready_timeout_s

    def _create(self, spec: SandboxSpec) -> Any:
        import cwsandbox

        res = spec.resources
        limits = {"cpu": str(res.cpu or 1), "memory": f"{res.memory_mib or 2048}Mi"}
        sandbox = cwsandbox.Sandbox.run(
            *(spec.entrypoint or ["sleep", "infinity"]),
            container_image=spec.image or self.default_image,
            resources=cwsandbox.ResourceOptions(requests=limits, limits=limits),
            environment_variables=dict(spec.env),
            tags=[*self.tags, *spec.metadata.values()],
            max_lifetime_seconds=int(spec.ttl_s or self.max_lifetime_s),
            **placement_kwargs(self.placement, self.runner_id),
        )
        try:
            sandbox.wait(timeout=float(spec.ready_timeout_s or self.ready_timeout_s))
            # "running" can come before the container accepts exec; wait for a first command.
            for _ in range(60):
                try:
                    if sandbox.exec(["true"]).result().returncode == 0:
                        return sandbox
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(5)
            raise SandboxCreateError(f"sandbox {sandbox.sandbox_id} never accepted commands")
        except BaseException:
            sandbox.stop(missing_ok=True).result()
            raise

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        try:
            sandbox = await asyncio.to_thread(self._create, spec)
        except SandboxCreateError:
            raise
        except Exception as exc:
            raise SandboxCreateError(f"cwsandbox create failed: {exc}") from exc
        return SandboxHandle(sandbox_id=sandbox.sandbox_id, provider_name=self.name, raw=sandbox)

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        user: str | int | None = None,
    ) -> SandboxExecResult:
        if user is not None:
            raise ValueError("cwsandbox provider does not support running as another user")
        prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in (env or {}).items())
        script = f"{prefix} {command}" if prefix else command
        sandbox = handle.raw

        def run() -> Any:
            return sandbox.exec(["sh", "-c", script], cwd=cwd, timeout_seconds=timeout_s).result()

        result = await asyncio.to_thread(run)
        return SandboxExecResult(stdout=result.stdout, stderr=result.stderr, return_code=result.returncode)

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        data = Path(source_path).read_bytes()
        await asyncio.to_thread(lambda: handle.raw.write_file(target_path, data).result())

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        data = await asyncio.to_thread(lambda: handle.raw.read_file(source_path).result())
        Path(target_path).write_bytes(data)

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        try:
            status = await asyncio.to_thread(handle.raw.get_status)
        except Exception:  # noqa: BLE001
            return SandboxStatus.UNKNOWN
        return _STATUS.get(str(getattr(status, "value", status)).lower(), SandboxStatus.UNKNOWN)

    async def close(self, handle: SandboxHandle) -> None:
        try:
            await asyncio.to_thread(lambda: handle.raw.stop(missing_ok=True).result())
        except Exception:  # noqa: BLE001 - close must be safe in cleanup paths
            pass

    async def aclose(self) -> None:
        return None


def register() -> None:
    try:
        register_provider(CWSandboxProvider.name, CWSandboxProvider)
    except ValueError:
        pass  # already registered
