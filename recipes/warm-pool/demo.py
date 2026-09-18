"""Compare on-demand creation with concurrent requests to a warm pool."""

import asyncio
import statistics
from time import perf_counter
from uuid import uuid4

from cwsandbox import AuthStrategy, SandboxDefaults, Session

from warm_pool import EXEC_TIMEOUT_SECONDS, LIFETIME_SECONDS, WarmPool, prepare

AUTH = AuthStrategy.COREWEAVE_API_KEY  # Use AuthStrategy.WANDB for W&B credentials.
POOL_SIZE = 2
CONCURRENCY = 2
REQUESTS = 4

WORKLOAD = """
from pathlib import Path
import time
state = Path('/tmp/request-state')
assert not state.exists(), 'state leaked from a previous workload'
state.write_text('private request data')
time.sleep(1)
print('fresh workspace')
"""


async def main():
    tag = f"warm-pool-{uuid4().hex[:12]}"
    print(f"Run tag: {tag}", flush=True)
    defaults = SandboxDefaults(
        auth=AUTH,
        container_image="python:3.11-slim",
        placement_mode="serverless",
        resources={"cpu": "2", "memory": "4Gi"},
        max_lifetime_seconds=LIFETIME_SECONDS,
        tags=(tag,),
    )
    if defaults.placement_mode == "cks" and defaults.auth == AuthStrategy.WANDB:
        raise ValueError("CKS placement requires AuthStrategy.COREWEAVE_API_KEY")
    seen = set()
    timings = {"on-demand": [], "pooled": []}

    async def execute(sandbox, mode, start):
        assert sandbox.sandbox_id not in seen, "a sandbox was assigned twice"
        seen.add(sandbox.sandbox_id)
        result = await sandbox.exec(
            ["python", "-c", WORKLOAD], check=True, timeout_seconds=EXEC_TIMEOUT_SECONDS
        )
        # Subsequent commands in the same claim run in the same live sandbox.
        await sandbox.exec(
            ["test", "-f", "/tmp/request-state"], check=True, timeout_seconds=EXEC_TIMEOUT_SECONDS
        )
        elapsed = perf_counter() - start
        timings[mode].append(elapsed)
        print(f"{mode:9s} request_to_result={elapsed:.3f}s {result.stdout.strip()}", flush=True)

    async def batch(request):
        async with asyncio.TaskGroup() as group:
            for _ in range(REQUESTS):
                group.create_task(request())

    async with Session(defaults=defaults) as session:
        admission = asyncio.Semaphore(CONCURRENCY)

        async def on_demand():
            start = perf_counter()
            async with admission:
                entry = await prepare(session)
                try:
                    await execute(entry.sandbox, "on-demand", start)
                finally:
                    await entry.sandbox.stop(missing_ok=True)

        await batch(on_demand)
        start = perf_counter()
        async with WarmPool(session, size=POOL_SIZE, concurrency=CONCURRENCY) as pool:
            print(f"Initial pool preparation: {perf_counter() - start:.3f}s", flush=True)

            async def pooled():
                start = perf_counter()
                async with pool.claim() as sandbox:
                    await execute(sandbox, "pooled", start)

            await batch(pooled)

    for mode, samples in timings.items():
        print(f"{mode:9s} median={statistics.median(samples):.3f}s n={len(samples)}")
    print(f"Verified {len(seen)} distinct workload sandboxes; all sandboxes stopped.")


if __name__ == "__main__":
    asyncio.run(main())
