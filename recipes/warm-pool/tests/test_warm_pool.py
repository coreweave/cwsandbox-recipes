import asyncio
from types import SimpleNamespace

import pytest

import warm_pool
from warm_pool import Prepared, WarmPool, prepare


class FakeSandbox:
    def __init__(self, *, fail_exec=False):
        self.fail_exec = fail_exec
        self.stopped = False

    async def exec(self, *args, **kwargs):
        if self.fail_exec:
            raise RuntimeError("probe failed")

    async def stop(self, **kwargs):
        self.stopped = True


@pytest.fixture
def factory(monkeypatch):
    created = []

    async def fake_prepare(session):
        sandbox = FakeSandbox()
        created.append(sandbox)
        return Prepared(sandbox, warm_pool.monotonic())

    monkeypatch.setattr(warm_pool, "prepare", fake_prepare)
    return created


async def wait_until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.001)


def test_concurrent_claims_are_distinct_and_refill(factory):
    async def scenario():
        async with WarmPool(None, size=2, concurrency=2) as pool:
            both_claimed = asyncio.Event()
            active = []

            async def request():
                async with pool.claim() as sandbox:
                    active.append(sandbox)
                    if len(active) == 2:
                        both_claimed.set()
                    async with asyncio.timeout(2):
                        await both_claimed.wait()
                    await wait_until(lambda: len(pool.available) == 2)
                    assert not sandbox.stopped

            async with asyncio.TaskGroup() as group:
                group.create_task(request())
                group.create_task(request())
            assert len(set(active)) == 2
            assert all(sb.stopped for sb in active)
            assert len(factory) == 4
            async with pool.claim() as replacement:
                assert replacement not in active
        assert all(sb.stopped for sb in factory)

    asyncio.run(scenario())


def test_concurrency_limit_queues_requests(factory):
    async def scenario():
        async with WarmPool(None, size=2, concurrency=1) as pool:
            entered = asyncio.Event()

            async def second_request():
                async with pool.claim():
                    entered.set()

            async with pool.claim():
                task = asyncio.create_task(second_request())
                await asyncio.sleep(0)
                assert not entered.is_set()
            await task
            assert entered.is_set()

    asyncio.run(scenario())


def test_idle_sandboxes_rotate_without_requests(factory, monkeypatch):
    monkeypatch.setattr(warm_pool, "MAX_CLAIM_AGE_SECONDS", 0.03)

    async def scenario():
        async with WarmPool(None, size=2) as pool:
            original = list(factory)
            await wait_until(lambda: all(sb.stopped for sb in original))
            await wait_until(lambda: len(pool.available) == 2)
            assert all(entry.sandbox not in original for entry in pool.available)
            assert len(factory) >= 4
        assert all(sb.stopped for sb in factory)

    asyncio.run(scenario())


def test_rotation_does_not_stop_claimed_workload(factory, monkeypatch):
    monkeypatch.setattr(warm_pool, "MAX_CLAIM_AGE_SECONDS", 0.03)

    async def scenario():
        async with WarmPool(None, size=1) as pool:
            async with pool.claim() as active:
                await wait_until(lambda: len(factory) >= 3)
                assert not active.stopped
            assert active.stopped

    asyncio.run(scenario())


def test_workload_error_stops_sandbox(factory):
    async def scenario():
        async with WarmPool(None, size=1) as pool:
            with pytest.raises(ValueError, match="workload failed"):
                async with pool.claim() as sandbox:
                    raise ValueError("workload failed")
            assert sandbox.stopped

    asyncio.run(scenario())


def test_cancelled_workload_stops_sandbox(factory):
    async def scenario():
        async with WarmPool(None, size=1) as pool:
            entered = asyncio.Event()

            async def request():
                async with pool.claim():
                    entered.set()
                    await asyncio.Event().wait()

            task = asyncio.create_task(request())
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert factory[0].stopped
        assert all(sb.stopped for sb in factory)

    asyncio.run(scenario())


def test_preparation_that_exhausts_lifetime_reserve_fails(monkeypatch):
    async def scenario():
        sandbox = FakeSandbox()

        async def too_slow(session):
            return Prepared(sandbox, warm_pool.monotonic() - warm_pool.MAX_CLAIM_AGE_SECONDS)

        monkeypatch.setattr(warm_pool, "prepare", too_slow)
        with pytest.raises(TimeoutError, match="claim age limit"):
            async with WarmPool(None, size=1):
                pytest.fail("over-age sandbox accepted")
        assert sandbox.stopped

    asyncio.run(scenario())


def test_stale_candidate_is_never_handed_out(factory, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(warm_pool, "monotonic", lambda: clock[0])

    async def scenario():
        async with WarmPool(None, size=1) as pool:
            expired = factory[0]
            clock[0] = warm_pool.MAX_CLAIM_AGE_SECONDS
            async with pool.claim() as sandbox:
                assert sandbox is not expired
                assert expired.stopped

    asyncio.run(scenario())


def test_unhealthy_candidate_is_stopped(factory):
    async def scenario():
        async with WarmPool(None, size=1) as pool:
            factory[0].fail_exec = True
            with pytest.raises(RuntimeError, match="probe failed"):
                async with pool.claim():
                    pytest.fail("unhealthy sandbox handed out")
            assert factory[0].stopped

    asyncio.run(scenario())


def test_failed_preparation_stops_sandbox():
    async def scenario():
        sandbox = FakeSandbox(fail_exec=True)
        session = SimpleNamespace(sandbox=lambda: sandbox)
        with pytest.raises(RuntimeError, match="probe failed"):
            await prepare(session)
        assert sandbox.stopped

    asyncio.run(scenario())


def test_pool_miss_waits_for_replenishment(monkeypatch):
    async def scenario():
        gate = asyncio.Event()
        created = []

        async def slow_prepare(session):
            if created:
                await gate.wait()
            sandbox = FakeSandbox()
            created.append(sandbox)
            return Prepared(sandbox, warm_pool.monotonic())

        monkeypatch.setattr(warm_pool, "prepare", slow_prepare)
        async with WarmPool(None, size=1) as pool:
            async with pool.claim():
                pass

            async def next_request():
                async with pool.claim():
                    pass

            task = asyncio.create_task(next_request())
            await asyncio.sleep(0)
            assert not task.done()
            gate.set()
            await task
        assert all(sb.stopped for sb in created)

    asyncio.run(scenario())


def test_shutdown_drains_inflight_creation(monkeypatch):
    async def scenario():
        gate, started = asyncio.Event(), asyncio.Event()
        created = []

        async def slow_prepare(session):
            if created:
                started.set()
                await gate.wait()
            sandbox = FakeSandbox()
            created.append(sandbox)
            return Prepared(sandbox, warm_pool.monotonic())

        monkeypatch.setattr(warm_pool, "prepare", slow_prepare)
        pool = WarmPool(None, size=1)
        await pool.__aenter__()
        async with pool.claim():
            await started.wait()
        closing = asyncio.create_task(pool.__aexit__(None, None, None))
        await asyncio.sleep(0)
        assert not closing.done()
        gate.set()
        await closing
        assert all(sb.stopped for sb in created)
        assert all(task.done() for task in pool.workers)

    asyncio.run(scenario())


def test_creation_failure_wakes_waiters_and_surfaces(monkeypatch):
    async def scenario():
        calls = 0

        async def sometimes_fails(session):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise RuntimeError("creation failed")
            return Prepared(FakeSandbox(), warm_pool.monotonic())

        monkeypatch.setattr(warm_pool, "prepare", sometimes_fails)
        with pytest.raises(RuntimeError, match="creation failed"):
            async with WarmPool(None, size=1) as pool:
                async with pool.claim():
                    pass
                async with pool.claim():
                    pytest.fail("failed pool accepted a claim")
        assert all(task.done() for task in pool.workers)

    asyncio.run(scenario())


def test_initial_failure_drains_other_creations(monkeypatch):
    async def scenario():
        created = []
        calls = 0

        async def sometimes_fails(session):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("creation failed")
            await asyncio.sleep(0)
            sandbox = FakeSandbox()
            created.append(sandbox)
            return Prepared(sandbox, warm_pool.monotonic())

        monkeypatch.setattr(warm_pool, "prepare", sometimes_fails)
        pool = WarmPool(None, size=2)
        with pytest.raises(RuntimeError, match="creation failed"):
            async with pool:
                pytest.fail("partially filled pool accepted")
        assert all(task.done() for task in pool.workers)
        assert all(sb.stopped for sb in created)

    asyncio.run(scenario())


@pytest.mark.parametrize("size,concurrency", [(0, 1), (1, 0)])
def test_capacity_must_be_positive(size, concurrency):
    with pytest.raises(ValueError, match="positive"):
        WarmPool(None, size=size, concurrency=concurrency)


@pytest.mark.parametrize("error_type", warm_pool.RETRYABLE_PREPARATION_ERRORS)
def test_transient_refill_failure_recovers_without_blocking_ready_sandboxes(
    monkeypatch, error_type
):
    async def scenario():
        calls = 0
        failed = asyncio.Event()
        created = []

        async def flaky_prepare(session):
            nonlocal calls
            calls += 1
            if calls == 3:
                failed.set()
                raise error_type("temporary failure")
            sandbox = FakeSandbox()
            created.append(sandbox)
            return Prepared(sandbox, warm_pool.monotonic())

        monkeypatch.setattr(warm_pool, "prepare", flaky_prepare)
        monkeypatch.setattr(warm_pool, "RETRY_BACKOFF_SECONDS", 0.05)
        async with WarmPool(None, size=2) as pool:
            healthy = created[1]
            async with pool.claim():
                await failed.wait()
                assert pool.error is None
                async with pool.claim() as sandbox:
                    assert sandbox is healthy
                    assert calls == 3  # The healthy slot is usable before the retry runs.
            await wait_until(lambda: len(pool.available) == 2)
            for _ in range(3):
                async with pool.claim():
                    pass
            assert pool.error is None
        assert all(sb.stopped for sb in created)

    asyncio.run(scenario())


def test_repeated_transient_failure_is_bounded(monkeypatch):
    async def scenario():
        calls = 0

        async def unavailable(session):
            nonlocal calls
            calls += 1
            raise warm_pool.SandboxUnavailableError("still unavailable")

        monkeypatch.setattr(warm_pool, "prepare", unavailable)
        monkeypatch.setattr(warm_pool, "RETRY_BACKOFF_SECONDS", 0)
        pool = WarmPool(None, size=1)
        with pytest.raises(warm_pool.SandboxUnavailableError, match="still unavailable"):
            async with pool:
                pytest.fail("unavailable pool accepted")
        assert calls == warm_pool.PREPARE_ATTEMPTS
        assert all(task.done() for task in pool.workers)

    asyncio.run(scenario())


def test_shutdown_interrupts_retry_backoff(monkeypatch):
    async def scenario():
        calls = 0
        retrying = asyncio.Event()

        async def unavailable_after_first(session):
            nonlocal calls
            calls += 1
            if calls > 1:
                retrying.set()
                raise warm_pool.SandboxUnavailableError("unavailable")
            return Prepared(FakeSandbox(), warm_pool.monotonic())

        monkeypatch.setattr(warm_pool, "prepare", unavailable_after_first)
        monkeypatch.setattr(warm_pool, "RETRY_BACKOFF_SECONDS", 60)
        async with asyncio.timeout(2):
            async with WarmPool(None, size=1) as pool:
                async with pool.claim():
                    await retrying.wait()
        assert calls == 2
        assert all(task.done() for task in pool.workers)

    asyncio.run(scenario())


def test_cleanup_failure_prevents_preparation_retry():
    async def scenario():
        created = []

        class UnavailableSandbox(FakeSandbox):
            async def exec(self, *args, **kwargs):
                raise warm_pool.SandboxUnavailableError("probe unavailable")

            async def stop(self, **kwargs):
                raise warm_pool.SandboxUnavailableError("cleanup unavailable")

        def create():
            sandbox = UnavailableSandbox()
            created.append(sandbox)
            return sandbox

        session = SimpleNamespace(sandbox=create)
        with pytest.raises(RuntimeError, match="Failed to stop"):
            async with WarmPool(session, size=1):
                pytest.fail("pool accepted uncertain cleanup")
        assert len(created) == 1

    asyncio.run(scenario())
