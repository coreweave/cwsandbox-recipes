"""Keep clean sandboxes ready for concurrent workloads."""

import asyncio
import logging
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from time import monotonic

from cwsandbox import Sandbox, Session
from cwsandbox.exceptions import (
    SandboxRequestTimeoutError,
    SandboxResourceExhaustedError,
    SandboxUnavailableError,
)

LIFETIME_SECONDS = 600
MAX_CLAIM_AGE_SECONDS = 300
EXEC_TIMEOUT_SECONDS = 30
PROBE = ["python", "-c", "print('ready')"]
PREPARE_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 1.0
RETRYABLE_PREPARATION_ERRORS = (
    SandboxUnavailableError,
    SandboxRequestTimeoutError,
    SandboxResourceExhaustedError,
)
logger = logging.getLogger(__name__)


@dataclass
class Prepared:
    sandbox: Sandbox
    created_at: float
    claimed: bool = False
    wake: asyncio.Event = field(default_factory=asyncio.Event)


async def prepare(session: Session) -> Prepared:
    created_at = monotonic()
    sandbox = session.sandbox()
    try:
        # Put shared setup before this application readiness check.
        await sandbox.exec(PROBE, check=True, timeout_seconds=EXEC_TIMEOUT_SECONDS)
        return Prepared(sandbox, created_at)
    except BaseException:
        try:
            await sandbox.stop(missing_ok=True)
        except Exception as exc:
            # Do not create replacements when cleanup could not be confirmed.
            raise RuntimeError("Failed to stop sandbox after preparation failed") from exc
        raise


class WarmPool:
    """Maintain `size` idle slots and allow up to `concurrency` active claims.

    Use within a Session context and finish request tasks before closing the pool.
    """

    def __init__(self, session: Session, size: int = 2, concurrency: int = 2):
        if size < 1 or concurrency < 1:
            raise ValueError("size and concurrency must be positive")
        self.session = session
        self.size = size
        self.available: deque[Prepared] = deque()
        self.changed = asyncio.Condition()
        self.admission = asyncio.Semaphore(concurrency)
        self.workers: list[asyncio.Task] = []
        self.closing = asyncio.Event()
        self.error: Exception | None = None

    async def _prepare_slot(self):
        for attempt in range(PREPARE_ATTEMPTS):
            if self.closing.is_set():
                return None
            try:
                return await prepare(self.session)
            except RETRYABLE_PREPARATION_ERRORS as exc:
                if attempt == PREPARE_ATTEMPTS - 1:
                    raise
                delay = RETRY_BACKOFF_SECONDS * 2**attempt
                if exc.retry_delay is not None:
                    delay = max(delay, exc.retry_delay.total_seconds())
                logger.warning(
                    "Sandbox preparation failed; retrying in %.1fs (attempt %d/%d)",
                    delay,
                    attempt + 2,
                    PREPARE_ATTEMPTS,
                )
                try:
                    await asyncio.wait_for(self.closing.wait(), timeout=delay)
                except TimeoutError:
                    pass
        return None

    async def _maintain_slot(self):
        try:
            while not self.closing.is_set():
                entry = await self._prepare_slot()
                if entry is None:
                    return
                if self.closing.is_set():
                    await entry.sandbox.stop(missing_ok=True)
                    return
                if monotonic() - entry.created_at >= MAX_CLAIM_AGE_SECONDS:
                    await entry.sandbox.stop(missing_ok=True)
                    raise TimeoutError("Sandbox preparation exceeded the claim age limit")
                async with self.changed:
                    self.available.append(entry)
                    self.changed.notify_all()
                remaining = MAX_CLAIM_AGE_SECONDS - (monotonic() - entry.created_at)
                try:
                    await asyncio.wait_for(entry.wake.wait(), timeout=max(0, remaining))
                except TimeoutError:
                    pass
                # Claims and retirement run on one event loop. Only one can own the entry.
                if not entry.claimed:
                    self.available.remove(entry)
                    await entry.sandbox.stop(missing_ok=True)
                # A claimed or retired sandbox is replaced on the next iteration.
        except Exception as exc:
            async with self.changed:
                self.error = exc
                self.changed.notify_all()

    async def __aenter__(self):
        self.workers = [asyncio.create_task(self._maintain_slot()) for _ in range(self.size)]
        try:
            async with self.changed:
                await self.changed.wait_for(
                    lambda: len(self.available) == self.size or self.error is not None
                )
                if self.error:
                    raise self.error
        except BaseException:
            await self._close()
            raise
        return self

    async def _close(self):
        async with self.changed:
            self.closing.set()
            for entry in self.available:
                entry.wake.set()
            self.changed.notify_all()
        # Let in-flight creation finish before the enclosing Session cleans up.
        await asyncio.gather(*self.workers)

    async def __aexit__(self, exc_type, exc, traceback):
        await self._close()
        if exc is None and self.error:
            raise self.error

    @asynccontextmanager
    async def claim(self):
        async with self.admission:
            while True:
                async with self.changed:
                    await self.changed.wait_for(
                        lambda: self.available or self.error is not None or self.closing.is_set()
                    )
                    if self.error:
                        raise self.error
                    if self.closing.is_set():
                        raise RuntimeError("Pool is closed")
                    entry = self.available.popleft()
                    entry.claimed = True
                    entry.wake.set()
                try:
                    if monotonic() - entry.created_at >= MAX_CLAIM_AGE_SECONDS:
                        continue
                    await entry.sandbox.exec(
                        PROBE, check=True, timeout_seconds=EXEC_TIMEOUT_SECONDS
                    )
                    yield entry.sandbox
                    return
                finally:
                    # Save outputs inside the claim. Used sandboxes never re-enter the pool.
                    await entry.sandbox.stop(missing_ok=True)
