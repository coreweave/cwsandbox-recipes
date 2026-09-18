# Reduce request startup latency with a warm pool

Start sandboxes ahead of incoming work so requests can execute immediately in a
running environment. This recipe maintains a configurable pool, runs workloads
concurrently, and replaces sandboxes as they are claimed or approach expiry.

Each sandbox requests **2 vCPU and 4 GiB RAM**. Pool size sets a buffer of ready, idle
sandboxes **in addition to** those running workloads. By default, two buffer
sandboxes plus up to two active workloads use up to four sandboxes at once
(8 vCPU / 16 GiB in aggregate requests, not a hard CPU execution cap).
Idle sandboxes consume compute too; pool size determines that
standing resource cost. No GPU or model API is needed.

## How the pool works

1. **Prepare:** start `size` sandboxes and run a readiness check in each. Put shared
   setup here, or include it in the container image.
2. **Keep ready:** each sandbox runs a keep-alive process between commands. No
   periodic workload or ping is needed to keep it running. A background task
   replaces each unclaimed sandbox when it reaches five minutes old.
3. **Claim:** an incoming workload takes exclusive ownership of one ready sandbox.
   The pool starts a replacement immediately. Other workloads can claim other
   sandboxes concurrently, up to `concurrency` active workloads.
4. **Execute:** every `exec()` inside that claim runs in the same sandbox, sharing
   files and state for that workload.
5. **Finish:** save outputs before leaving the claim. The used sandbox is stopped;
   subsequent workloads receive fresh sandboxes.

Here is the pattern, using the recipe's [`WarmPool`](warm_pool.py):

```python
import asyncio
from uuid import uuid4
from cwsandbox import AuthStrategy, SandboxDefaults, Session
from warm_pool import WarmPool

async def main():
    tag = f"warm-pool-{uuid4().hex[:12]}"
    print(f"Run tag: {tag}", flush=True)
    defaults = SandboxDefaults(
        auth=AuthStrategy.COREWEAVE_API_KEY,  # Or AuthStrategy.WANDB.
        container_image="python:3.11-slim",
        placement_mode="serverless",
        resources={"cpu": "2", "memory": "4Gi"},
        max_lifetime_seconds=600,
        tags=(tag,),
    )
    async with Session(defaults=defaults) as session:
        async with WarmPool(session, size=2, concurrency=2) as pool:
            async def workload():
                async with pool.claim() as sandbox:
                    await sandbox.exec(
                        ["sh", "-c", "echo hello > /tmp/result.txt"], check=True
                    )
                    result = await sandbox.exec(["cat", "/tmp/result.txt"], check=True)
                    print(result.stdout)

            async with asyncio.TaskGroup() as requests:
                for _ in range(4):
                    requests.create_task(workload())

asyncio.run(main())
```

Workloads are isolated because each gets a separate sandbox that is never assigned
to another workload. Commands within one claim intentionally share state. Provide
customer inputs and credentials after claiming a sandbox, and keep shared external
storage scoped to the appropriate customer.

**The pool continues beyond ten minutes while its Python process and context are
running.** Ten minutes is the lifetime of an individual sandbox. Unclaimed
sandboxes are retired at five minutes and replaced automatically, even without
incoming requests. The five-minute cutoff reserves time for the workload; keep
jobs comfortably under five minutes to allow for health checks and other overhead.
Claimed sandboxes retain their original expiry deadline. Active workloads are not
migrated or restarted. For longer jobs, increase `LIFETIME_SECONDS` in `warm_pool.py`
and keep a suitable reserve between that lifetime and `MAX_CLAIM_AGE_SECONDS`.

If all ready sandboxes have been claimed, a request waits for replenishment. If the
active-workload limit is reached, it waits for a workload to finish. Transient
service, request-timeout, and resource-pressure errors during preparation get
up to three attempts, with one- and two-second backoff (or a longer server-provided
delay). Ready sandboxes remain usable during retries. Exhausted retries and
other errors surface to the caller; the surrounding contexts clean up when exited.
Workload commands and failed cleanup are never retried by the pool.

## Setup

You need Python 3.11+, [uv](https://docs.astral.sh/uv/getting-started/installation/),
and serverless access. Follow
[Get started with CoreWeave sandboxes: serverless](https://docs.coreweave.com/products/sandboxes/get-started#run-a-sandbox-on-serverless-capacity)
to choose a CoreWeave API access token or W&B API key.

From this directory, install the locked dependencies:

```bash
uv sync --frozen
cp .env.example .env
```

Uncomment and fill in your credential in `.env`. Set `AUTH` near the top of
`demo.py` to the corresponding SDK strategy:

| Credential | Environment variable | Python setting |
|---|---|---|
| CoreWeave | `CWSANDBOX_API_KEY` | `AUTH = AuthStrategy.COREWEAVE_API_KEY` (default) |
| W&B | `WANDB_API_KEY`, or an existing W&B login | `AUTH = AuthStrategy.WANDB` |

The run commands below load credentials from `.env`. The W&B command also installs
the SDK's optional authentication dependency. `warm_pool.py` contains the pool
logic to copy into your application.

## Run the comparison

1. Run the example:

   ```bash
   uv run --frozen --env-file .env demo.py
   ```

   For W&B, set `AUTH = AuthStrategy.WANDB` in `demo.py` and run:

   ```bash
   uv run --frozen --extra wandb --env-file .env demo.py
   ```

   It submits four concurrent requests per mode, with at most two active
   workloads, and prints individual request times plus a median for each mode.
   Expect `fresh workspace` for every request and this final verification:

   ```text
   Verified 8 distinct workload sandboxes; all sandboxes stopped.
   ```

   Each workload writes a private file, then reads it in a second command in the
   same sandbox. A new workload must not see a previous workload's file. The demo
   also checks that no sandbox ID is assigned twice.

   `request_to_result` includes waiting for admission, acquiring a sandbox, the
   health check, and both workload commands. The workload includes a one-second
   pause so requests overlap. Initial pool preparation is reported separately.
   Both modes run one probe per request inside the timer: readiness after
   creation for on-demand execution, or health at claim time for pooled execution.
   Cleanup occurs after each result and holds its concurrency slot until complete.
   Requests beyond the initial warm capacity may wait for new sandboxes, so their
   times can approach on-demand creation. Measure with your own workload and
   arrival pattern before choosing a pool size.

2. To change the buffer size, active limit, or request count, edit the constants
   near the top of `demo.py`, then run it again:

   ```python
   POOL_SIZE = 3
   CONCURRENCY = 3
   REQUESTS = 6
   ```

   These settings allow up to six sandboxes, each requesting 2 vCPU / 4 GiB,
   including replacements being prepared.

For a ready CKS runner, use a CoreWeave token and set `placement_mode="cks"` and
`runner_ids=("YOUR_RUNNER_ID",)` in the script's `SandboxDefaults`.

## Use your own workload

Replace `WORKLOAD` in [`demo.py`](demo.py) with your command. Include its files in
the image or upload them after claiming a sandbox. Add common setup to `prepare()`
in [`warm_pool.py`](warm_pool.py), and replace the readiness probe with a check
that your application can serve work. Collect files or upload results inside the
claim, before the sandbox stops.

Use multiple `exec()` calls within a claim for a multi-step task, such as an agent
conversation. Keep that claim for the task's duration; a new claim starts with a
fresh sandbox. Set command timeouts to fit within the sandbox's remaining lifetime.
See [timeout settings](https://docs.coreweave.com/products/sandboxes/client/guides/sandbox-configuration#timeouts).

## Reduce preparation time

| Technique | Use it for |
|---|---|
| Prebuilt image | Install stable dependencies and common code once, before deploying the pool. |
| Sandbox template | Store the image, resources, and storage settings for a recurring workload. |
| Clean filesystem snapshot | Restore prepared workspace files instead of rebuilding them for every sandbox. |

Pin your image by digest for reproducibility. A
[sandbox template](https://docs.coreweave.com/products/sandboxes/profiles/templates)
can reference the image and a baseline snapshot to keep configuration consistent.

[Filesystem snapshots](https://docs.coreweave.com/products/sandboxes/file-system-snapshots)
capture the configured scratch mount. The base image and application processes
still start separately. Store dependencies under the mount if they need to be in
the snapshot; restore virtual environments at the same path with a compatible
image. Stop writers before capturing a clean baseline, exclude customer data and
credentials, and wait for it to become `READY`. Compare restore time with setup
time: larger snapshots take longer to transfer.

### Verify independent snapshot restores

With filesystem snapshots enabled for your account, set `AUTH` at the top of
`snapshot_demo.py` to the same strategy you used for the pool, then run:

```bash
uv run --frozen --env-file .env snapshot_demo.py
```

For W&B, include `--extra wandb` before `snapshot_demo.py`, as in the pool command.

The script uses three sandboxes sequentially, each requesting 2 vCPU / 4 GiB.
It captures a clean workspace, restores it twice, and checks that changes in one
restore do not appear in the other. Expect `independent restore verified` twice.
It stops its sandboxes and deletes its temporary snapshot afterward.

To initialize pool members from your own clean snapshot, add the following argument
to `SandboxDefaults` in [`demo.py`](demo.py), importing
`FileSystemSnapshotOptions` from `cwsandbox`:

```python
file_system_snapshot=FileSystemSnapshotOptions(
    mount_path="/work",
    size="1Gi",
    file_system_snapshot_id="YOUR_READY_BASELINE_SNAPSHOT_ID",
),
```

Keep the baseline while the pool uses it, and rebuild it when shared dependencies
or data change. Store workload output separately from this baseline.

## Pool size and operation

`POOL_SIZE` sets the ready-buffer target, and `CONCURRENCY` limits active
workloads: together they bound the total at `POOL_SIZE + CONCURRENCY` sandboxes.
The buffer may temporarily have fewer ready sandboxes while replacements start.
Start with two idle sandboxes and measure how often requests wait.
For sustained traffic, estimate the idle target from the arrival rate times
the replenishment time, then allow room for bursts.

The pool maintains capacity until its context closes. The demo closes it after the
comparison finishes; an application can keep the context open while accepting
requests. This implementation coordinates claims within one Python process. Each
additional process would own a separate pool and consume additional compute.

## Cleanup

A completed or failed workload stops its claimed sandbox. Exiting the pool stops
idle sandboxes and finishes outstanding preparation before the surrounding
`Session` closes. If the Python process is killed, remaining sandboxes expire at
their individual ten-minute lifetime limits; replenishment stops with the process.

To clean up an interrupted run, use the SDK directly with the tag printed at
startup and the same authentication strategy:

For W&B, add `--extra wandb` before `python` in the command and select
`AuthStrategy.WANDB` below. Include the extra on every W&B invocation because
`uv` synchronizes optional dependencies for each command.

```bash
uv run --frozen --env-file .env python - <<'PY'
from cwsandbox import AuthStrategy, Sandbox

auth = AuthStrategy.COREWEAVE_API_KEY  # Or AuthStrategy.WANDB.
tag = "warm-pool-YOUR_RUN_TAG"
for sandbox in Sandbox.list(tags=[tag], auth=auth).result():
    sandbox.stop(missing_ok=True).result()
assert not Sandbox.list(tags=[tag], auth=auth).result()
PY
```

Only that run's active sandboxes are stopped.
The pool demo creates no snapshots. If the optional snapshot demo is killed before
cleanup, remove its printed temporary snapshot ID using the
[snapshot management API](https://docs.coreweave.com/products/sandboxes/file-system-snapshots#manage-snapshots).

## Development checks

Run `uv run --frozen python -m pytest -q` for tests without cloud resources.
The examples use the SDK and Python's standard library.
