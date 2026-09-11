"""Warm pool of τ-bench environment sandboxes.

Sizing
------
In-flight episodes at steady state are ``data.train_batch_size * rollout.n``, so
that is the per-tool-process pool size needed for full concurrency. Multiple
veRL workers can each own pools, and validation can add a split-keyed pool.
Sandboxes persist across training steps; only the policy weights change at
weight sync, and the environment side never notices.

Why a pool at all
-----------------
Provisioning a sandbox costs a cold start. Leasing from a warm pool moves that
cost off the rollout critical path. Episodes are short, so a sandbox-per-rollout
create/destroy cycle would be dominated by provisioning latency, so a
sandbox per rollout is the wrong default.

Honest framing
--------------
τ-bench's environment is trusted, lightweight Python (an in-memory JSON
database), so it *could* run in-process inside the AgentLoop workers. Sandboxes
are used here because they (a) demonstrate the pattern that generalizes to
environments that genuinely need isolation -- code execution, browsers, SWE
tasks; (b) give crash containment and keep environment CPU work off the rollout
workers' event loop at high concurrency; and (c) schedule onto idle CPU on the
GPU nodes via SUNK. τ-bench does not *require* sandboxes.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import logging
import os
import secrets
import shlex
import signal
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from verl_taubench.events import log_event, short_id
from verl_taubench.sandbox import env_server
from verl_taubench.sandbox.client import SandboxEnvClient, Transport
from verl_taubench.sandbox.env_config import resolve_serverless_auth
from verl_taubench.sandbox.tags import RECIPE_TAG

logger = logging.getLogger(__name__)

# Port the in-sandbox env server binds. Fixed at sandbox creation time --
# cwsandbox has no post-hoc port exposure, so this cannot be negotiated later.
DEFAULT_ENV_PORT = 8080
_SDK_IMPORT_LOCK = threading.Lock()


@contextlib.contextmanager
def _cwsandbox_import_guard():
    """Permit cwsandbox 1.x import from a Ray async-actor thread.

    cwsandbox installs SIGINT/SIGTERM handlers unconditionally at import time.
    Python rejects that outside the main interpreter thread, and Ray executes
    both async actor construction and methods on a background event-loop thread.
    Keep cwsandbox's atexit cleanup registration, but make its two signal
    registrations read-only during this narrowly scoped import.
    """
    if threading.current_thread() is threading.main_thread():
        yield
        return

    with _SDK_IMPORT_LOCK:
        original_signal = signal.signal

        def current_handler(signum: int, _handler: Any) -> Any:
            return signal.getsignal(signum)

        signal.signal = current_handler
        try:
            yield
        finally:
            signal.signal = original_signal


@dataclass
class SandboxHandle:
    """A provisioned sandbox and the URL of the env server running inside it."""

    sandbox_id: str
    base_url: str
    token: Optional[str] = None
    native: Any = None
    created_at: float = field(default_factory=time.monotonic)


def _wait_for_health(
    base_url: str,
    token: Optional[str],
    *,
    timeout_s: float,
    poll_interval_s: float,
    probe_timeout_s: float,
) -> Optional[str]:
    """Poll ``{base_url}/health`` until 200; return the last error on timeout.

    Runs synchronously (callers are on ``asyncio.to_thread`` workers), so plain
    urllib keeps it simple. Raises RuntimeError immediately on 401: the server
    is up but the bearer token disagrees, and retrying cannot fix that.
    """
    deadline = time.monotonic() + timeout_s
    last_error = "no attempt made"
    request = urllib.request.Request(
        f"{base_url}/health",
        headers={"Authorization": f"Bearer {token}"},
        method="GET",
    )
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(request, timeout=probe_timeout_s) as resp:
                if resp.status == 200:
                    return None
                last_error = f"HTTP {resp.status}"
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                raise RuntimeError(
                    "env server rejected our bearer token (401). The server's "
                    "TAUBENCH_ENV_TOKEN does not match the pool's token."
                ) from exc
            last_error = f"HTTP {exc.code}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(poll_interval_s)
    return last_error


@runtime_checkable
class SandboxBackend(Protocol):
    """Provisioning surface the pool depends on.

    Implemented by :class:`CwSandboxBackend` in production and by an in-process
    fake in the tests, so pool semantics are verified without cwsandbox.
    """

    async def provision(self) -> SandboxHandle: ...

    async def destroy(self, handle: SandboxHandle) -> None: ...


class CwSandboxBackend:
    """Provisions τ-bench env sandboxes via the ``cwsandbox`` SDK.

    Follows the CoreWeave Python client contract (cwsandbox >= 1.12):

    * Serverless placement selects W&B or CoreWeave API-key auth. CKS placement
      retains the default ``CWSANDBOX_API_KEY`` authentication.
    * ``placement_mode="cks"`` lands on a cluster runner. Omitting it defaults
      to serverless, which is not the enabled-on-this-cluster path.
    * ``placement_spillover="cks_then_serverless"`` retries once on serverless
      when CKS cannot place the sandbox. That is SDK-level spillover, not a
      second auth mode.
    * Ports are typed ``services=``; ``NetworkOptions`` no longer takes
      ``ingress_mode`` / ``exposed_ports``.
    * Reachability is ``sandbox.service_urls`` (port, name, url).
    """

    def __init__(
        self,
        *,
        domain: str = "retail",
        task_split: str = "train",
        container_image: str = "python:3.11",
        env_port: int = DEFAULT_ENV_PORT,
        cpu: str = "4",
        memory: str = "8Gi",
        max_lifetime_seconds: Optional[int] = 12 * 3600,
        ingress_mode: str = "public",
        egress_mode: Optional[str] = "internet",
        tau_bench_spec: str = (
            "git+https://github.com/sierra-research/tau-bench.git"
            "@59a200c6d575d595120f1cb70fea53cef0632f6b"
        ),
        install_tau_bench: bool = True,
        token: Optional[str] = None,
        tags: Optional[List[str]] = None,
        startup_timeout_s: float = 300.0,
        readiness_timeout_s: float = 120.0,
        readiness_poll_interval_s: float = 0.5,
        readiness_probe_timeout_s: float = 5.0,
        placement_mode: str = "cks",
        placement_spillover: Optional[str] = "cks_then_serverless",
        partial_credit: float = 0.0,
        runner_ids: Optional[List[str]] = None,
        serverless_auth: Optional[str] = None,
        use_wandb_auth: bool = False,
        spillover_to_wandb: bool = False,
    ):
        del use_wandb_auth, spillover_to_wandb, ingress_mode, egress_mode
        self.domain = domain
        self.task_split = task_split
        self.container_image = container_image
        self.env_port = env_port
        self.cpu = cpu
        self.memory = memory
        self.max_lifetime_seconds = max_lifetime_seconds
        self.tau_bench_spec = tau_bench_spec
        self.install_tau_bench = install_tau_bench
        # Graded credit for failed train episodes (GRPO needs in-group reward
        # variance); validation pools stay strict for comparable metrics.
        self.partial_credit = float(partial_credit) if task_split == "train" else 0.0
        self.token = token or os.environ.get("TAUBENCH_ENV_TOKEN") or secrets.token_urlsafe(32)
        # Reaping matches exactly the tags asked for; creation additionally
        # stamps the stable recipe tag so a leaked sandbox stays identifiable
        # when the run id is lost (cleanup --all).
        self.tags = list(tags or ["verl-taubench-env"])
        self.startup_timeout_s = startup_timeout_s
        self.readiness_timeout_s = readiness_timeout_s
        self.readiness_poll_interval_s = readiness_poll_interval_s
        self.readiness_probe_timeout_s = readiness_probe_timeout_s
        self.placement_mode = placement_mode
        # The cks_then_serverless default only applies when placement starts on
        # CKS; the SDK rejects it under serverless placement.
        if placement_mode == "serverless" and placement_spillover == "cks_then_serverless":
            placement_spillover = None
        self.placement_spillover = placement_spillover
        self.runner_ids = list(runner_ids) if runner_ids else None
        self.serverless_auth = resolve_serverless_auth(serverless_auth)
        self._sdk_modules: Optional[tuple[Any, ...]] = None
        self._session: Any = None
        self._session_lock = threading.Lock()

    def _get_session(self) -> Any:
        with self._session_lock:
            if self._session is None:
                self.preload_sdk()
                sdk = importlib.import_module("cwsandbox")
                # Lifetime is a Session default in cwsandbox 1.12; it is not
                # accepted as an individual Session.sandbox() argument.
                self._session = sdk.Session(
                    defaults={"max_lifetime_seconds": self.max_lifetime_seconds},
                    report_to=["wandb"],
                )
            return self._session

    def _serverless_auth_strategy(self) -> Any:
        AuthStrategy = self._sdk()[-1]
        return AuthStrategy.WANDB if self.serverless_auth == "wandb" else AuthStrategy.COREWEAVE_API_KEY

    def _report_metrics(self) -> None:
        if self._session is not None:
            try:
                # SDK-owned counters only. Provisioning is concurrent, so
                # retain cumulative counters instead of resetting mid-create.
                self._session.log_metrics(reset=False)
            except Exception:
                logger.warning("Could not publish SDK sandbox metrics", exc_info=True)

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.close()

    def _import_sdk(self) -> tuple[Any, ...]:
        try:
            with _cwsandbox_import_guard():
                sdk = importlib.import_module("cwsandbox")
        except ImportError as exc:
            raise ImportError(
                "CwSandboxBackend requires cwsandbox>=1.12.0. Install the sandbox extra: "
                'uv pip install -e ".[sandbox]"'
            ) from exc
        if sdk is None or not hasattr(sdk, "Sandbox") or not hasattr(sdk, "AuthStrategy"):
            raise ImportError("cwsandbox is not a usable sandbox SDK")
        return (
            sdk.Sandbox,
            sdk.NetworkOptions,
            sdk.ResourceOptions,
            getattr(sdk, "Service", None),
            getattr(sdk, "Endpoint", None),
            sdk.AuthStrategy,
        )

    def _sdk(self):
        """Import cwsandbox lazily.

        cwsandbox installs SIGINT/SIGTERM handlers at import time, which Python
        rejects on Ray's async-actor thread. The import guard suppresses only
        those registrations; cwsandbox's atexit cleanup and this pool's explicit
        teardown remain active.
        """
        if self._sdk_modules is not None:
            return self._sdk_modules
        self._sdk_modules = self._import_sdk()
        return self._sdk_modules

    def preload_sdk(self) -> None:
        """Load and cache the SDK before concurrent provisioning begins."""
        self._sdk()

    def _server_source(self) -> bytes:
        """Read env_server.py so it can be uploaded into the sandbox verbatim."""
        path = env_server.__file__
        with open(path, "rb") as fh:
            return fh.read()

    _LOG_PATH = "/tmp/env_server.log"

    def _creation_tags(self) -> List[str]:
        tags = list(self.tags)
        return tags if RECIPE_TAG in tags else [RECIPE_TAG, *tags]

    def _env_services(self) -> Optional[list[Any]]:
        _, _, _, Service, Endpoint, _ = self._sdk()
        if Service is None:
            return None
        endpoint = None
        if Endpoint is not None:
            endpoint = Endpoint(kind="https", auth="open")
        else:
            endpoint = {"kind": "https", "auth": "open"}
        return [
            Service(
                port=self.env_port,
                name="env",
                visibility="public",
                endpoint=endpoint,
            )
        ]

    def _base_url(self, sandbox: Any) -> str:
        urls = getattr(sandbox, "service_urls", None) or ()
        for port, _name, url in urls:
            if url and int(port) == int(self.env_port):
                return str(url).rstrip("/")
        for _port, _name, url in urls:
            if url:
                return str(url).rstrip("/")
        address = getattr(sandbox, "service_address", None)
        if address:
            text = str(address)
            if text.startswith("http://") or text.startswith("https://"):
                return text.rstrip("/")
            return f"http://{text}"
        raise RuntimeError(
            "sandbox has no service_urls: declare a public HTTPS Service at create "
            "time, or the runner did not assign an ingress URL."
        )

    def _provision_blocking(self) -> SandboxHandle:
        _, NetworkOptions, ResourceOptions, _, _, _ = self._sdk()
        run_kwargs: Dict[str, Any] = {
            "container_image": self.container_image,
            "network": NetworkOptions(),
            "resources": ResourceOptions(
                requests={"cpu": self.cpu, "memory": self.memory},
            ),
            "tags": self._creation_tags(),
            "environment_variables": {"TAUBENCH_ENV_TOKEN": self.token},
            "placement_mode": self.placement_mode,
        }
        if self.placement_mode == "serverless":
            run_kwargs["auth"] = self._serverless_auth_strategy()
        if self.placement_spillover is not None:
            run_kwargs["placement_spillover"] = self.placement_spillover
        services = self._env_services()
        if services is not None:
            run_kwargs["services"] = services
        # runner_ids is a CKS pin; the SDK rejects it under serverless placement
        # ("runner_ids requires placement_mode=CKS"), and the launcher forwards
        # CWSANDBOX_RUNNER_IDS into the GPU sandbox for its own create.
        if self.runner_ids and self.placement_mode != "serverless":
            run_kwargs["runner_ids"] = list(self.runner_ids)

        created_at = time.monotonic()
        sandbox = self._get_session().sandbox(command="sleep", args=["infinity"], **run_kwargs)
        log_event(
            f"env sandbox requested "
            f"(placement={self.placement_mode}, {time.monotonic() - created_at:.1f}s)"
        )

        # Everything after session.sandbox must be unwound on failure, or a running,
        # billed sandbox is orphaned for its entire max_lifetime_seconds.
        try:
            sandbox.wait(timeout=self.startup_timeout_s)
            base_url = self._base_url(sandbox)

            if self.install_tau_bench:
                # tau-bench is not on PyPI, so install from git unless the image ships it.
                install = sandbox.exec(
                    ["python", "-m", "pip", "install", "--no-input", "-q", self.tau_bench_spec],
                    timeout_seconds=self.startup_timeout_s,
                ).result()
                if install.returncode != 0:
                    raise RuntimeError(
                        f"tau-bench install failed in sandbox (rc={install.returncode}): "
                        f"{install.stderr[-2000:]}"
                    )

            sandbox.write_file("/srv/env_server.py", self._server_source()).result()

            # Launch DETACHED. An attached `exec` inherits the SDK's request
            # timeout (300 s by default), which would kill the gRPC stream that
            # owns the process ~5 minutes into a 12-hour sandbox. nohup + & hands
            # the server to init and lets this exec return immediately.
            launch = sandbox.exec(
                [
                    "sh",
                    "-c",
                    (
                        f"nohup python /srv/env_server.py "
                        f"--domain {shlex.quote(self.domain)} "
                        f"--task-split {shlex.quote(self.task_split)} "
                        f"--partial-credit {float(self.partial_credit)} "
                        f"--host 0.0.0.0 --port {int(self.env_port)} "
                        f"> {self._LOG_PATH} 2>&1 & echo $!"
                    ),
                ],
                timeout_seconds=self.startup_timeout_s,
            ).result()
            if launch.returncode != 0:
                raise RuntimeError(
                    f"env server launch failed (rc={launch.returncode}): {launch.stderr[-2000:]}"
                )

            # The server takes ~2 s to import tau_bench before the socket binds.
            # Returning an unready handle makes the first /reset race startup.
            self._await_ready(sandbox, base_url)

            log_event(
                f"env server ready: sandbox {short_id(sandbox.sandbox_id)} at {base_url} "
                f"({time.monotonic() - created_at:.1f}s total)"
            )
            return SandboxHandle(
                sandbox_id=str(getattr(sandbox, "sandbox_id", "unknown")),
                base_url=base_url,
                token=self.token,
                native=sandbox,
            )
        except BaseException:
            with contextlib.suppress(Exception):
                sandbox.stop().result()
            raise
        finally:
            self._report_metrics()

    def _server_log(self, sandbox: Any) -> str:
        """Best-effort tail of the env server's log, for diagnosis."""
        try:
            out = sandbox.exec(["tail", "-n", "40", self._LOG_PATH], timeout_seconds=30).result()
            return (out.stdout or out.stderr or "").strip()
        except Exception as exc:  # pragma: no cover - diagnosis only
            return f"<could not read {self._LOG_PATH}: {exc}>"

    def _await_ready(self, sandbox: Any, base_url: str) -> None:
        """Block until the in-sandbox env server answers /health, or fail loudly."""
        last_error = _wait_for_health(
            base_url,
            self.token,
            timeout_s=self.readiness_timeout_s,
            poll_interval_s=self.readiness_poll_interval_s,
            probe_timeout_s=self.readiness_probe_timeout_s,
        )
        if last_error is not None:
            raise TimeoutError(
                f"env server did not become ready within {self.readiness_timeout_s}s "
                f"(last error: {last_error}). Server log:\n{self._server_log(sandbox)}"
            )

    async def provision(self) -> SandboxHandle:
        if self._sdk_modules is None:
            self.preload_sdk()
        return await asyncio.to_thread(self._provision_blocking)

    async def destroy(self, handle: SandboxHandle) -> None:
        sandbox = handle.native
        if sandbox is None:
            return

        def _stop() -> None:
            with contextlib.suppress(Exception):
                sandbox.stop().result()

        await asyncio.to_thread(_stop)

    def reap_orphans(self) -> int:
        """Stop every sandbox carrying our tags. Returns how many were stopped.

        Provisioning runs on a worker thread via ``asyncio.to_thread``.
        Cancelling the awaiting coroutine does **not** stop that thread, so it can
        go on to create a sandbox whose handle is then discarded -- an orphan that
        lives out its ``max_lifetime_seconds`` unknown to ``aclose()``.

        There is no way to cancel an in-flight thread, so this is the mitigation:
        run it after a crashed or cancelled training job to sweep up. Safe to run
        between jobs; destructive if another job is using the same tags.
        """
        Sandbox, _, _, _, _, _ = self._sdk()
        stopped = 0
        try:
            list_kwargs: Dict[str, Any] = {"tags": list(self.tags)}
            if self.placement_mode == "serverless":
                list_kwargs["auth"] = self._serverless_auth_strategy()
            listed = Sandbox.list(**list_kwargs).result()
        except Exception as exc:
            logger.warning("sandbox list failed during reap: %s", exc)
            raise
        # ``stop()`` returns an async reference. Start every request before
        # waiting so teardown time is bounded by the slowest stop rather than
        # the sum of every stop -- large rollout pools can contain 100+ items.
        pending_stops = []
        for sandbox in listed:
            with contextlib.suppress(Exception):
                pending_stops.append(sandbox.stop())
        for stop_ref in pending_stops:
            with contextlib.suppress(Exception):
                stop_ref.result()
                stopped += 1
        return stopped


@dataclass
class Lease:
    """A leased sandbox bound to one episode."""

    handle: SandboxHandle
    client: SandboxEnvClient
    task_index: int
    episode_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SandboxPool:
    """Warm pool with lease / reset / return semantics.

    Invariants:

    * A leased sandbox is never handed to a second episode concurrently.
    * ``release`` always returns capacity, even when the sandbox is unhealthy --
      a broken sandbox is destroyed and replaced rather than leaked.
    * ``size`` is a hard ceiling on live sandboxes.
    """

    def __init__(
        self,
        backend: SandboxBackend,
        transport: Transport,
        *,
        size: int = 8,
        acquire_timeout_s: float = 300.0,
        reset_timeout_s: float = 60.0,
        max_provision_retries: int = 2,
        owns_transport: bool = True,
    ):
        if size < 1:
            raise ValueError("pool size must be >= 1")
        self.backend = backend
        self.transport = transport
        self.size = size
        self.acquire_timeout_s = acquire_timeout_s
        self.reset_timeout_s = reset_timeout_s
        self.max_provision_retries = max_provision_retries
        # The simulator may share this transport; only close what we own.
        self._owns_transport = owns_transport

        self._idle: asyncio.LifoQueue = asyncio.LifoQueue()
        self._slots = asyncio.Semaphore(size)
        self._live: Dict[str, SandboxHandle] = {}
        self._lock = asyncio.Lock()
        self._closed = False
        self.stats: Dict[str, int] = {
            "provisioned": 0,
            "destroyed": 0,
            "leases": 0,
            "replacements": 0,
        }

    # -- internals ---------------------------------------------------------

    def _client_for(self, handle: SandboxHandle) -> SandboxEnvClient:
        return SandboxEnvClient(handle.base_url, self.transport, token=handle.token)

    async def _provision_one(self) -> SandboxHandle:
        last: Optional[Exception] = None
        for attempt in range(self.max_provision_retries + 1):
            if self._closed:
                raise RuntimeError("pool is closed")
            try:
                handle = await self.backend.provision()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last = exc
                logger.warning("sandbox provision attempt %d failed: %s", attempt + 1, exc)
                continue
            # Register before any await that could be cancelled, so aclose() can
            # always find and destroy it.
            async with self._lock:
                self._live[handle.sandbox_id] = handle
                self.stats["provisioned"] += 1
            if self._closed:
                # aclose() ran while we were provisioning; do not hand out a
                # sandbox it has already swept past.
                await self._destroy(handle)
                raise RuntimeError("pool is closed")
            return handle
        raise RuntimeError(f"could not provision sandbox after {self.max_provision_retries + 1} attempts") from last

    async def _destroy(self, handle: SandboxHandle) -> None:
        async with self._lock:
            self._live.pop(handle.sandbox_id, None)
        with contextlib.suppress(Exception):
            await self.backend.destroy(handle)
        self.stats["destroyed"] += 1

    async def prewarm(self) -> None:
        """Provision the full pool up front so the first step is not cold.

        Idempotent: only tops up to ``size``, so calling it twice does not
        double-provision.
        """
        if self._closed:
            raise RuntimeError("pool is closed")
        missing = self.size - len(self._live)
        if missing <= 0:
            return
        log_event(f"prewarming env pool: provisioning {missing} sandbox(es)")
        started = time.monotonic()
        handles = await asyncio.gather(
            *(self._provision_one() for _ in range(missing)), return_exceptions=True
        )
        errors = [h for h in handles if isinstance(h, BaseException)]
        for handle in handles:
            if not isinstance(handle, BaseException):
                self._idle.put_nowait(handle)
        if errors and not any(not isinstance(h, BaseException) for h in handles):
            raise RuntimeError(f"prewarm failed to provision any sandbox: {errors[0]}") from errors[0]
        if errors:
            logger.warning("prewarm: %d/%d sandboxes failed to provision", len(errors), self.size)
        log_event(
            f"env pool ready: {len(handles) - len(errors)}/{self.size} sandboxes "
            f"({time.monotonic() - started:.1f}s)"
        )

    # -- public API --------------------------------------------------------

    async def acquire(self, task_index: int, episode_id: str) -> Lease:
        """Lease a sandbox and load ``task_index`` into it.

        Blocks until a slot is free. A sandbox that fails its reset is destroyed
        and replaced rather than handed to the episode.
        """
        if self._closed:
            raise RuntimeError("pool is closed")

        await asyncio.wait_for(self._slots.acquire(), timeout=self.acquire_timeout_s)
        try:
            attempts = 0
            while True:
                attempts += 1
                try:
                    handle = self._idle.get_nowait()
                except asyncio.QueueEmpty:
                    handle = await self._provision_one()

                client = self._client_for(handle)
                try:
                    metadata = await asyncio.wait_for(
                        client.reset(task_index, episode_id=episode_id),
                        timeout=self.reset_timeout_s,
                    )
                except asyncio.CancelledError:
                    # Cancellation is not a sandbox fault, but the handle is ours
                    # and nobody else holds a reference. Destroy it rather than
                    # leaking a running sandbox, and do not retry.
                    await self._destroy(handle)
                    raise
                except Exception as exc:
                    logger.warning(
                        "sandbox %s failed reset (%s); replacing", handle.sandbox_id, exc
                    )
                    await self._destroy(handle)
                    self.stats["replacements"] += 1
                    if attempts > self.max_provision_retries + 1:
                        raise
                    continue

                self.stats["leases"] += 1
                return Lease(
                    handle=handle,
                    client=client,
                    task_index=task_index,
                    episode_id=episode_id,
                    metadata=metadata,
                )
        except BaseException:
            self._slots.release()
            raise

    async def release(self, lease: Lease, *, healthy: bool = True) -> None:
        """Return a lease to the pool.

        The slot is released in all paths. An unhealthy sandbox is destroyed and
        a replacement is provisioned lazily on the next ``acquire``.
        """
        try:
            if healthy and not self._closed:
                self._idle.put_nowait(lease.handle)
            else:
                await self._destroy(lease.handle)
        finally:
            self._slots.release()

    @contextlib.asynccontextmanager
    async def lease(self, task_index: int, episode_id: str):
        """Scoped lease. Marks the sandbox unhealthy if the episode raised."""
        acquired = await self.acquire(task_index, episode_id)
        healthy = True
        try:
            yield acquired
        except BaseException:
            healthy = False
            raise
        finally:
            await self.release(acquired, healthy=healthy)

    async def aclose(self) -> None:
        """Destroy every live sandbox. Idempotent.

        ``_closed`` is set first so any concurrent ``acquire``/``_provision_one``
        bails out instead of racing us and leaving a sandbox alive after close.
        """
        self._closed = True
        while True:
            try:
                self._idle.get_nowait()
            except asyncio.QueueEmpty:
                break
        async with self._lock:
            handles = list(self._live.values())
            self._live.clear()

        async def _destroy_quietly(handle: SandboxHandle) -> None:
            with contextlib.suppress(Exception):
                await self.backend.destroy(handle)

        # Concurrent teardown: stop() resolves only at terminal state with a
        # ~10 s grace, so destroying a 64-sandbox pool serially would take
        # minutes.
        if handles:
            await asyncio.gather(*(_destroy_quietly(h) for h in handles))
            self.stats["destroyed"] += len(handles)

        if self._owns_transport:
            with contextlib.suppress(Exception):
                await self.transport.aclose()
        close_backend = getattr(self.backend, "aclose", None)
        if close_backend is not None:
            await close_backend()

    @property
    def live_count(self) -> int:
        return len(self._live)

    @property
    def idle_count(self) -> int:
        return self._idle.qsize()
