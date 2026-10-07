"""CoreWeave Sandboxes backend for slime's agent rollouts.

Implements slime's ``slime.agent.sandbox.Sandbox`` protocol on the ``cwsandbox``
SDK. Select it with::

    SLIME_AGENT_SANDBOX_BACKEND=slime_cwsandbox.CWSandbox

and make this directory importable wherever slime builds sandboxes (for a Ray
run, on every worker's ``PYTHONPATH``).

The SDK reads its own credentials (``CWSANDBOX_API_KEY``, optional
``CWSANDBOX_BASE_URL``). Its exec takes neither a user nor an environment, so
commands run as ``[runuser -u <user> --] [env K=V ...] bash -l -c <cmd>``, the
same login shell e2b uses. Images therefore need ``bash``, a root default user,
and ``runuser`` (util-linux) for non-root calls; Debian- and Ubuntu-based
SWE-bench images have all three.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import shlex
from collections.abc import Callable, Iterator
from pathlib import Path

logger = logging.getLogger(__name__)

ExecResult = tuple[int, str, str]
FileContent = str | bytes | Path


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name, "")
    return value.strip() if value.strip() else default


def _iter_file(path: Path, chunk_bytes: int) -> Iterator[bytes]:
    with open(path, "rb") as fp:
        while chunk := fp.read(chunk_bytes):
            yield chunk


def _transient_types() -> tuple[type[BaseException], ...]:
    from cwsandbox import SandboxRequestTimeoutError, SandboxResourceExhaustedError, SandboxUnavailableError

    return (SandboxUnavailableError, SandboxRequestTimeoutError, SandboxResourceExhaustedError)


class CWSandbox:
    """Async context manager around ``cwsandbox.Sandbox``.

    Settings (constructor argument, else environment variable, else default):

    - ``timeout``: ``SLIME_AGENT_SANDBOX_LIFETIME_SEC``, server-side lifetime cap (3600).
    - ``placement_mode``: ``SLIME_AGENT_CWSANDBOX_PLACEMENT_MODE``, ``serverless`` or ``cks``.
    - ``cpu`` / ``memory``: ``SLIME_AGENT_CWSANDBOX_CPU`` / ``_MEMORY`` (``2`` / ``4Gi``).
    - ``tags``: ``SLIME_AGENT_CWSANDBOX_TAGS``, comma-separated (``slime-agent``).
    - ``request_timeout``: ``SLIME_AGENT_CWSANDBOX_REQUEST_TIMEOUT_SEC``, bounds the create
      call and the wait for RUNNING, which includes the image pull (900).
    - ``rpc_retries``: ``SLIME_AGENT_SANDBOX_RPC_RETRIES``, attempts per transient failure (6).
    """

    default_lifetime_sec = 3600
    default_rpc_retries = 6
    default_cpu = "2"
    default_memory = "4Gi"
    default_tags = "slime-agent"
    default_request_timeout_sec = 900
    # Stop waits for the backend to report a terminal state. Bound it so a
    # stuck stop can't hang the rollout; the lifetime cap reaps the sandbox.
    stop_timeout_sec = 60
    rpc_backoff_base_sec = 1.0
    rpc_backoff_cap_sec = 32.0
    # Payloads above this stream in chunks instead of one request.
    stream_threshold_bytes = 32 * 1024 * 1024
    stream_chunk_bytes = 4 * 1024 * 1024
    file_op_timeout_sec = 600

    def __init__(
        self,
        image: str,
        *,
        timeout: int | None = None,
        placement_mode: str | None = None,
        cpu: str | None = None,
        memory: str | None = None,
        tags: list[str] | None = None,
        request_timeout: float | None = None,
        rpc_retries: int | None = None,
    ) -> None:
        # Import and validate here so misconfiguration fails when slime builds
        # the backend, not deep inside a rollout.
        from cwsandbox import PlacementMode

        self.image = image
        self.timeout = int(
            timeout if timeout is not None else _env("SLIME_AGENT_SANDBOX_LIFETIME_SEC", str(self.default_lifetime_sec))
        )
        mode = placement_mode or _env("SLIME_AGENT_CWSANDBOX_PLACEMENT_MODE")
        self.placement_mode = PlacementMode(mode.lower()).value if mode else None
        self.cpu = cpu or _env("SLIME_AGENT_CWSANDBOX_CPU", self.default_cpu)
        self.memory = memory or _env("SLIME_AGENT_CWSANDBOX_MEMORY", self.default_memory)
        if tags is None:
            tags = [t.strip() for t in _env("SLIME_AGENT_CWSANDBOX_TAGS", self.default_tags).split(",") if t.strip()]
        self.tags = tags
        self.request_timeout = float(
            request_timeout
            if request_timeout is not None
            else _env("SLIME_AGENT_CWSANDBOX_REQUEST_TIMEOUT_SEC", str(self.default_request_timeout_sec))
        )
        self.rpc_retries = int(
            rpc_retries
            if rpc_retries is not None
            else _env("SLIME_AGENT_SANDBOX_RPC_RETRIES", str(self.default_rpc_retries))
        )
        self._sb = None
        self.sandbox_id = ""

    async def _rpc_retry(
        self,
        op_name: str,
        op: Callable,
        *,
        idempotent: bool = True,
        retry_if: Callable[[BaseException], bool] | None = None,
    ):
        """Await ``op()``, retrying transient SDK failures with jittered backoff.

        A transient failure can also arrive wrapped (file operations raise
        ``SandboxFileError`` from the transport error). When ``idempotent`` is
        False it is re-raised instead, because the lost response may belong to
        a command that already ran.
        """
        transient = _transient_types()
        attempts = max(1, self.rpc_retries)
        for attempt in range(attempts):
            try:
                return await op()
            except Exception as e:
                is_transient = isinstance(e, transient) or isinstance(e.__cause__, transient)
                if not is_transient or (retry_if and not retry_if(e)) or not idempotent or attempt + 1 >= attempts:
                    raise
                backoff = random.uniform(0.0, min(self.rpc_backoff_cap_sec, self.rpc_backoff_base_sec * 2**attempt))
                logger.debug(
                    "[cwsandbox] %s %s, retry %d/%d in %.1fs", op_name, type(e).__name__, attempt + 1, attempts, backoff
                )
                await asyncio.sleep(backoff)
        raise AssertionError("unreachable")

    async def _stop(self) -> None:
        sb = self._sb
        if sb is None or sb.sandbox_id is None:
            return
        try:
            await asyncio.wait_for(
                _awaitable(sb.stop(graceful_shutdown_seconds=0, missing_ok=True)), self.stop_timeout_sec
            )
        except TimeoutError:
            logger.warning(
                "[cwsandbox] stop %s still pending after %ss; the lifetime cap will reap it",
                sb.sandbox_id,
                self.stop_timeout_sec,
            )
        except Exception as e:
            logger.warning("[cwsandbox] stop %s failed: %s", sb.sandbox_id, e)

    async def __aenter__(self) -> CWSandbox:
        from cwsandbox import Sandbox as SDKSandbox
        from cwsandbox import SandboxRequestTimeoutError

        kwargs = {
            "container_image": self.image,
            "max_lifetime_seconds": self.timeout,
            "resources": {"cpu": self.cpu, "memory": self.memory},
            "tags": self.tags or None,
            "request_timeout_seconds": self.request_timeout,
        }
        if self.placement_mode:
            kwargs["placement_mode"] = self.placement_mode
        # One handle for every attempt: the SDK keeps its create request id, so
        # retrying after a lost response cannot start a second sandbox. A create
        # that already waited out the full request timeout is not retried.
        sb = self._sb = SDKSandbox(**kwargs)
        try:
            await self._rpc_retry("create", sb.start, retry_if=lambda e: not isinstance(e, SandboxRequestTimeoutError))
            await sb  # until RUNNING
        except BaseException:
            await self._stop()
            raise
        self.sandbox_id = sb.sandbox_id or ""
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._stop()

    @staticmethod
    def _argv(cmd: str, *, user: str, env: dict[str, str] | None) -> list[str]:
        argv = ["bash", "-l", "-c", cmd]
        if env:
            argv = ["env", *(f"{k}={v}" for k, v in env.items()), *argv]
        if user != "root":
            # runuser sets HOME, USER, LOGNAME and SHELL and keeps the rest of the environment.
            argv = ["runuser", "-u", user, "--", *argv]
        return argv

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
        idempotent: bool = True,
    ) -> ExecResult:
        argv = self._argv(cmd, user=user, env=env)
        res = await self._rpc_retry(
            f"exec({cmd[:60]!r})",
            lambda: self._sb.exec(argv, timeout_seconds=timeout),
            idempotent=idempotent,
        )
        if check and res.returncode != 0:
            raise RuntimeError(
                f"cwsandbox exec failed (exit={res.returncode}): {cmd[:120]}\n{(res.stderr or '')[:400]}"
            )
        return res.returncode, res.stdout or "", res.stderr or ""

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None:
        # e2b creates missing parents as the writing user; do the same first.
        parent = os.path.dirname(sandbox_path.rstrip("/")) or "/"
        if parent not in ("/", "/tmp"):
            q = shlex.quote(parent)
            await self.exec(f"test -d {q} || mkdir -p {q}", user=user, timeout=30, check=True)

        name = f"write_file({sandbox_path})"
        if isinstance(content, Path):
            host_path = content
            await self._rpc_retry(
                name,
                lambda: self._sb.write_file_streaming(
                    sandbox_path,
                    _iter_file(host_path, self.stream_chunk_bytes),
                    timeout_seconds=self.file_op_timeout_sec,
                ),
            )
        else:
            data = content.encode() if isinstance(content, str) else content
            write = self._sb.write_file_streaming if len(data) > self.stream_threshold_bytes else self._sb.write_file
            await self._rpc_retry(name, lambda: write(sandbox_path, data, timeout_seconds=self.file_op_timeout_sec))

        if user != "root":
            # e2b writes as ``user``; match ownership so the agent can chmod or
            # overwrite what it was handed (e.g. exec_and_wait's launcher).
            await self.exec(
                f"chown {shlex.quote(user)}: {shlex.quote(sandbox_path)}", user="root", timeout=30, check=True
            )

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        """Read as root (``user`` is accepted for protocol parity); ``""`` on any failure, like e2b."""
        try:
            data = await self._rpc_retry(
                f"read_file({sandbox_path})",
                lambda: self._sb.read_file(sandbox_path, timeout_seconds=self.file_op_timeout_sec),
            )
        except Exception:
            return ""
        return data.decode("utf-8", errors="replace")


async def _awaitable(op):
    return await op
