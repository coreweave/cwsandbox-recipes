"""Hot-load new weights into the Dedicated Inference deployment after each training step.

1. The policy workers export the weights as safetensors, one shard per rank, into
   ``REFIT_EXPORT_DIR`` (node memory, ``/dev/shm``, on SUNK).
2. A publisher thread uploads them with ``cwtrain.HotLoader`` as a full snapshot to the
   deployment's hot-load bucket and creates a hot-load, which swaps the weights on every replica.

``hotload_cfg.max_versions_behind`` (default 0) is how many versions the deployment may lag the
trainer when ``sync_weights`` returns:

* 0: ``sync_weights`` returns once every replica serves the new weights. In synchronous GRPO no
  rollout runs during this. In async GRPO with in-flight weight updates, rollouts keep running:
  each replica pauses its requests while it swaps weights, then resumes them. NeMo-RL raises its
  weight version when ``sync_weights`` returns, so a rollout that spans the sync carries the older
  version, which overstates its age and never understates it.
* K > 0 (async GRPO only): ``sync_weights`` returns right after the export, and the trainer trains
  the next step while the publisher uploads and hot-loads. It waits only if the deployment would
  serve weights more than K versions old. If a newer export is ready before the publisher starts
  on the previous one, the previous one is dropped, so only the newest weights get loaded.
  Rollouts started under weights up to K versions old carry the trainer's newest version, so
  their age is understated by up to K: the true bound is ``max_trajectory_age_steps + K``. The loss
  is still correct: importance sampling compares the trainer's logprobs with the logprobs the
  deployment returned for each token, whichever weights produced it; the version only schedules
  rollouts in the replay buffer.
* ``.inf`` (async GRPO only): the trainer never waits for a hot-load. One upload runs at a time
  and only the newest export waits behind it, so the lag is about 1 + (hot-load time / step
  time) versions: 1 while a hot-load is shorter than a step, more while one is slow (a replica
  change can hold it up to 15 minutes). A rollout's true age is ``max_trajectory_age_steps`` plus
  that lag; the done log line prints the lag ("trainer N ahead").

Validation (``HotLoadGeneration.prepare_for_generation``) and the end of the run (``shutdown``)
wait until the deployment serves the newest weights. A failed hot-load raises in the trainer on
the next sync, wait or shutdown.

Each hot-load prints ``@@metric {"k": "hotload", "version": N, "state": ..., "ts": ...}`` for the
dashboard: ``start`` when its upload starts, ``done`` when every replica serves it, and
``skipped`` when a newer export replaced it before its upload.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from nemo_rl.weight_sync.interfaces import WeightSynchronizer

from async_grpo_tool_use import caios
from async_grpo_tool_use.hotload import add_model_files, checkpoint_dtypes, checkpoint_sizes

# Files uploaded at once: the export has one shard per policy rank, plus config and tokenizer.
UPLOAD_WORKERS = 16
# Seconds between hot-load status polls (cwtrain's default is 5).
POLL_INTERVAL_S = 1.0
# Per file: 32 MiB parts, 16 at a time (boto3's defaults are 8 MiB and 10). With 8 shards that is
# up to 128 parts, about 4 GiB of buffers, in flight.
PART_BYTES = 32 * 2**20
PARTS_IN_FLIGHT = 16


class TunedS3:
    """The S3 client cwtrain uploads with, with larger and more concurrent multipart parts.

    cwtrain's ``upload_snapshot`` calls ``upload_file`` without a transfer config; everything
    else passes through to the client."""

    def __init__(self, s3: Any) -> None:
        from boto3.s3.transfer import TransferConfig

        self._s3 = s3
        self._config = TransferConfig(multipart_chunksize=PART_BYTES, max_concurrency=PARTS_IN_FLIGHT)

    def upload_file(self, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("Config", self._config)
        return self._s3.upload_file(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._s3, name)


def dashboard_event(version: int, state: str) -> None:
    """One timeline line for the dashboard; the job log carries it to the dashboard."""
    print("@@metric " + json.dumps({"k": "hotload", "version": version, "state": state, "ts": time.time()}), flush=True)


def publish_when_ready(publish: Any, version: int, *, timeout_s: float = 900, every_s: float = 15) -> Any:
    """Run ``publish``, waiting while the deployment rolls out a replica change: until it
    converges, Dedicated Inference rejects hot-loads with a failed precondition."""
    from cwtrain.errors import FailedPreconditionError

    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return publish()
        except FailedPreconditionError as exc:
            if "not ready" not in str(exc) or time.monotonic() > deadline:
                raise
            print(f"[grpo] hot-load v{version}: deployment is rolling out; retrying in {every_s:.0f}s", flush=True)
            time.sleep(every_s)


@dataclass
class Export:
    version: int
    path: Path
    tensors: int
    bytes: int
    export_s: float


class HotLoadWeightSynchronizer(WeightSynchronizer):
    def __init__(
        self,
        policy: Any,
        generation: Any,
        *,
        hf_checkpoint: str,
        export_dir: str | None = None,
        async_grpo: bool = False,
    ) -> None:
        cfg = generation.hotload_cfg
        token = os.environ.get(cfg.get("token_env") or "COREWEAVE_API_TOKEN")
        if not token:
            raise ValueError("COREWEAVE_API_TOKEN must hold a CoreWeave API token with inference access")
        self.max_behind = float(cfg.get("max_versions_behind") or 0)
        if self.max_behind and not async_grpo:
            # Synchronous GRPO samples right after the sync, so it must wait for the new weights.
            raise ValueError("hotload_cfg.max_versions_behind > 0 needs async GRPO (grpo.async_grpo.enabled=true)")
        self._policy = policy
        self.base = Path(hf_checkpoint)
        self.dtypes = checkpoint_dtypes(self.base)
        self.sizes = checkpoint_sizes(self.base)
        self.export_dir = Path(export_dir or os.environ.get("REFIT_EXPORT_DIR", "/work/refit"))
        self.run_id = cfg["run_id"]
        from cwtrain import HotLoader, InferenceClient

        self.loader = HotLoader(
            InferenceClient(token=token),
            cfg["deployment_id"],
            s3=TunedS3(caios.client()),
            poll_interval=POLL_INTERVAL_S,
        )
        self._stale = True
        # Shared with the publisher thread, under _cond.
        self._cond = threading.Condition()
        self._exported = 0  # newest version exported
        self._serving = 0  # newest version every replica serves (0: the base model)
        self._pending: Optional[Export] = None  # exported, upload not started yet
        self._error: Optional[BaseException] = None
        self._failed = False  # the publisher stopped on an error
        self._closing = False
        self._publisher: Optional[threading.Thread] = None
        print(
            f"[grpo] weight sync: {len(self.dtypes)} tensors -> deployment {cfg['deployment_id']} "
            f"(up to {self.max_behind:g} versions behind)",
            flush=True,
        )

    def init_communicator(self) -> None:
        return None

    @property
    def is_stale(self) -> bool:
        return self._stale

    def mark_stale(self) -> None:
        self._stale = True

    def identity(self, version: int) -> str:
        return f"{self.run_id}-v{version:06d}"

    def _raise_if_failed(self) -> None:
        """Raise the publisher's error once, then a short error on every later call. Under _cond."""
        if self._error is not None:
            error, self._error = self._error, None
            raise error
        if self._failed:
            raise RuntimeError("the hot-load publisher stopped after an earlier error")

    def _wait_until(self, behind: int) -> None:
        """Wait until the deployment serves weights at most ``behind`` versions old. Under _cond."""
        while True:
            self._raise_if_failed()
            if self._exported - self._serving <= behind:
                return
            self._cond.wait()

    def wait(self) -> None:
        """Wait until every replica serves the newest exported weights; raise a hot-load error."""
        with self._cond:
            self._wait_until(0)

    def shutdown(self) -> None:
        try:
            with self._cond:
                already_raised = self._failed and self._error is None
                if not already_raised:
                    self._wait_until(0)
        finally:
            with self._cond:
                self._closing = True
                pending, self._pending = self._pending, None
                self._cond.notify_all()
            if pending:
                shutil.rmtree(pending.path, ignore_errors=True)
            if self._publisher:
                self._publisher.join()

    def sync_weights(self, *, timer: Optional[Any] = None, kv_scales: Optional[dict[str, float]] = None) -> None:
        def timed(label: str):
            return timer.time(f"prepare_for_generation/{label}") if timer else nullcontext()

        with self._cond:
            self._raise_if_failed()
            version = self._exported + 1
        path = self.export_dir / f"v{version:06d}"
        print(f"[grpo] hot-load v{version}: exporting weights", flush=True)
        started = time.perf_counter()
        with timed("export_weights"):
            results = self._policy.run_all_workers_single_data(
                "export_hf_checkpoint", path=str(path), dtypes=self.dtypes, sizes=self.sizes
            )
            add_model_files(self.base, path)
        export = Export(
            version,
            path,
            tensors=sum(r["tensors"] for r in results),
            bytes=sum(r["bytes"] for r in results),
            export_s=time.perf_counter() - started,
        )
        if self.max_behind:
            # WHY: uploading and hot-loading took ~70 s of a ~185 s step. Overlapping them with
            # the next steps keeps the trainer GPUs busy, at the cost of up to max_versions_behind
            # versions of extra staleness for rollouts (see the module docstring).
            print(f"[grpo] hot-load v{version}: uploading and loading while training continues", flush=True)
        else:
            print(f"[grpo] hot-load v{version}: uploading and loading", flush=True)
        with self._cond:
            dropped, self._pending = self._pending, export
            self._exported = version
            self._stale = False
            if self._publisher is None:
                self._publisher = threading.Thread(target=self._publish_loop, name="hotload", daemon=True)
                self._publisher.start()
            self._cond.notify_all()
        if dropped:
            shutil.rmtree(dropped.path, ignore_errors=True)
            dashboard_event(dropped.version, "skipped")
            print(f"[grpo] hot-load v{dropped.version}: skipped, v{version} is newer", flush=True)
        with timed("hot_load"), self._cond:
            self._wait_until(self.max_behind)

    def _publish_loop(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._closing:
                    self._cond.wait()
                if self._pending is None:
                    return
                export, self._pending = self._pending, None
                previous = self._serving
            try:
                self._publish(export, previous)
            except Exception as exc:  # noqa: BLE001 - handed to the trainer thread, which raises it
                shutil.rmtree(export.path, ignore_errors=True)
                with self._cond:
                    self._error, self._failed = exc, True
                    self._cond.notify_all()
                return

    def _publish(self, export: Export, previous: int) -> None:
        from cwtrain.infra.models import PromptCachePolicy

        version, identity = export.version, self.identity(export.version)
        dashboard_event(version, "start")
        started = time.perf_counter()
        info = self.loader.upload(export.path, identity, workers=UPLOAD_WORKERS)
        uploaded = time.perf_counter()
        # Requests admitted after the swap must not reuse KV computed with the old weights:
        # sync GRPO is on-policy, and async GRPO tags them with a newer weight version.
        hot_load = publish_when_ready(
            lambda: self.loader.trigger(identity, info, prompt_cache_policy=PromptCachePolicy.RESET_ALL), version
        )
        self.loader.wait(hot_load.id)
        loaded = time.perf_counter()
        with self._cond:
            self._serving = version
            behind = self._exported - version
            self._cond.notify_all()
        dashboard_event(version, "done")
        shutil.rmtree(export.path, ignore_errors=True)
        if previous:
            # Each snapshot is full, so the replicas no longer need the previous one.
            cfg = self.loader.deployment.hot_load
            caios.delete_prefix(self.loader.s3, cfg.bucket, f"{cfg.path_prefix.strip('/')}/{self.identity(previous)}")
        print(
            f"[grpo] hot-load v{version}: {export.tensors} tensors, {export.bytes / 2**30:.1f} GiB; "
            f"export {export.export_s:.0f}s, upload and load {loaded - started:.0f}s "
            f"(upload {uploaded - started:.0f}s, swap {loaded - uploaded:.0f}s; trainer {behind} ahead)"
            f"{', in the background' if self.max_behind else ''}",
            flush=True,
        )
