"""Hot-load new weights into the Dedicated Inference deployment after each training step.

1. The policy workers export the weights as safetensors (rank 0 writes them).
2. ``cwtrain.HotLoader`` uploads them as a full snapshot to the deployment's hot-load bucket
   and creates a hot-load, which swaps the weights on every replica. The trainer waits for it.

GRPO is synchronous, so no rollout runs during this. The next rollout uses the new weights.
"""

from __future__ import annotations

import os
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

from nemo_rl.weight_sync.interfaces import WeightSynchronizer

from dapo_nemo_rl import caios
from dapo_nemo_rl.hotload import add_model_files, checkpoint_dtypes


class HotLoadWeightSynchronizer(WeightSynchronizer):
    def __init__(self, policy: Any, generation: Any, *, hf_checkpoint: str, export_dir: str | None = None) -> None:
        cfg = generation.hotload_cfg
        token = os.environ.get(cfg.get("token_env") or "COREWEAVE_API_TOKEN")
        if not token:
            raise ValueError("COREWEAVE_API_TOKEN must hold a CoreWeave API token with inference access")
        self._policy = policy
        self.base = Path(hf_checkpoint)
        self.dtypes = checkpoint_dtypes(self.base)
        self.export_dir = Path(export_dir or os.environ.get("REFIT_EXPORT_DIR", "/work/refit"))
        self.run_id = cfg["run_id"]
        from cwtrain import HotLoader, InferenceClient

        self.loader = HotLoader(InferenceClient(token=token), cfg["deployment_id"], s3=caios.client())
        self._version = 0
        self._stale = True
        print(f"[dapo] weight sync: {len(self.dtypes)} tensors -> deployment {cfg['deployment_id']}", flush=True)

    def init_communicator(self) -> None:
        return None

    @property
    def is_stale(self) -> bool:
        return self._stale

    def mark_stale(self) -> None:
        self._stale = True

    def shutdown(self) -> None:
        return None

    def identity(self, version: int) -> str:
        return f"{self.run_id}-v{version:06d}"

    def sync_weights(self, *, timer: Optional[Any] = None, kv_scales: Optional[dict[str, float]] = None) -> None:
        def timed(label: str):
            return timer.time(f"prepare_for_generation/{label}") if timer else nullcontext()

        version = self._version + 1
        path = self.export_dir / f"v{version:06d}"
        started = time.perf_counter()
        with timed("export_weights"):
            results = self._policy.run_all_workers_single_data(
                "export_hf_checkpoint", path=str(path), dtypes=self.dtypes
            )
            add_model_files(self.base, path)
        rank0 = next(r for r in results if r["rank"] == 0)
        exported = time.perf_counter()
        with timed("hot_load"):
            from cwtrain.infra.models import PromptCachePolicy

            # GRPO is on-policy: drop KV cache computed with the old weights.
            self.loader.publish(path, self.identity(version), prompt_cache_policy=PromptCachePolicy.RESET_ALL)
        loaded = time.perf_counter()
        shutil.rmtree(path, ignore_errors=True)
        if self._version:
            # Each snapshot is full, so the replicas no longer need the previous one.
            hot_load = self.loader.deployment.hot_load
            prefix = f"{hot_load.path_prefix.strip('/')}/{self.identity(self._version)}"
            caios.delete_prefix(self.loader.s3, hot_load.bucket, prefix)
        self._version = version
        self._stale = False
        print(
            f"[dapo] hot-load v{version}: {rank0['tensors']} tensors, {rank0['bytes'] / 2**30:.1f} GiB; "
            f"export {exported - started:.0f}s, upload and load {loaded - exported:.0f}s",
            flush=True,
        )
