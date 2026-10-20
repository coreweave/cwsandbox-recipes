"""NeMo-RL DTensor policy worker with an ``export_hf_checkpoint`` method.

Every rank gathers each full tensor (one collective per tensor), but writes only the tensors
that :func:`plan_shards` gives it, as one safetensors shard. So the ranks share the device-to-host
copies and the writes, instead of rank 0 doing them all while its peers wait in the next
collective. Rank 0 also writes the index. The names and dtypes are the base checkpoint's.

The ranks write into one directory, so it must be on storage they all see: node-local storage
such as ``/dev/shm`` works for a one-node policy.

This module runs in NeMo-RL's worker venv, so it imports only torch, safetensors, ray
and nemo_rl.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import torch

_DTYPES = {
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}


def target_dtype(name: str, tensor: torch.Tensor, dtypes: dict[str, str]) -> torch.dtype:
    """The base checkpoint's dtype for ``name``. Only float-to-float casts are allowed."""
    wanted = _DTYPES.get(dtypes[name])
    if wanted is None:
        raise ValueError(f"{name}: base checkpoint dtype {dtypes[name]} is not a float dtype")
    if tensor.dtype not in _DTYPES.values():
        raise ValueError(f"{name}: policy holds {tensor.dtype}, which the export cannot cast")
    return wanted


def plan_shards(sizes: dict[str, int], count: int) -> dict[str, int]:
    """Give each tensor a shard in ``range(count)``, largest tensor first onto the lightest shard.

    Every rank computes the same plan from the same sizes, so no rank has to send it."""
    count = max(1, min(count, len(sizes)))
    loads = [0] * count
    plan = {}
    for name in sorted(sizes, key=lambda n: (-sizes[n], n)):
        shard = min(range(count), key=lambda i: (loads[i], i))
        plan[name] = shard
        loads[shard] += sizes[name]
    return plan


def shard_file(shard: int, count: int) -> str:
    return f"model-{shard + 1:05d}-of-{count:05d}.safetensors"


def write_checkpoint(
    params: Iterable[tuple[str, torch.Tensor]],
    directory: Path,
    dtypes: dict[str, str],
    sizes: dict[str, int],
    *,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    """Consume ``params`` fully and write this rank's shard of them into ``directory``.

    Every rank iterates every tensor, because the generator runs a collective per tensor.
    A write error is held until iteration ends, so peers never block in a collective.
    """
    plan = plan_shards(sizes, world_size)
    count = max(plan.values()) + 1
    mine: dict[str, torch.Tensor] = {}
    seen: set[str] = set()
    error: BaseException | None = None
    for name, tensor in params:
        if name not in dtypes:
            continue
        seen.add(name)
        if plan[name] != rank or error is not None:
            continue
        try:
            mine[name] = tensor.detach().to("cpu", target_dtype(name, tensor, dtypes), copy=True).contiguous()
        except Exception as exc:  # noqa: BLE001 - keep iterating so peers do not block
            error = exc

    missing = sorted(set(dtypes) - seen)
    if error is None and missing:
        error = RuntimeError(f"export is missing {len(missing)} tensors, e.g. {missing[:5]}")
    written = sum(t.nbytes for t in mine.values())
    if error is None:
        try:
            from safetensors.torch import save_file

            directory.mkdir(parents=True, exist_ok=True)
            if mine:
                save_file(mine, str(directory / shard_file(rank, count)), metadata={"format": "pt"})
            if rank == 0:
                weight_map = {name: shard_file(shard, count) for name, shard in sorted(plan.items())}
                index = {"metadata": {"total_size": sum(sizes.values())}, "weight_map": weight_map}
                (directory / "model.safetensors.index.json").write_text(json.dumps(index, indent=1))
        except Exception as exc:  # noqa: BLE001
            error = exc
    return {
        "error": None if error is None else f"{type(error).__name__}: {error}",
        "tensors": len(mine),
        "bytes": written,
        "path": str(directory),
    }


def finish_export(path: Path, tmp: Path, failed: bool) -> None:
    """Rank 0, once every rank has written: publish ``tmp`` as ``path``, or remove it."""
    if failed:
        shutil.rmtree(tmp, ignore_errors=True)
        return
    shutil.rmtree(path, ignore_errors=True)
    tmp.rename(path)


def _any_rank_failed(failed: bool) -> bool:
    if not torch.distributed.is_initialized():
        return failed
    flag = torch.tensor([int(failed)], dtype=torch.int32, device="cuda")
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
    return bool(flag.item())


try:
    import ray
    from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
    from nemo_rl.models.policy.workers.dtensor_policy_worker_v2 import (
        DTensorPolicyWorkerV2Impl,
        dtensor_params_generator,
    )
except ImportError:  # pragma: no cover - only importable inside the NeMo-RL container
    DTensorPolicyWorkerV2Impl = None


if DTensorPolicyWorkerV2Impl is not None:  # pragma: no cover - needs GPUs
    # Ray forbids subclassing a decorated actor, so derive from the undecorated Impl and
    # apply the stock worker's runtime env.
    @ray.remote(runtime_env=get_runtime_env_for_policy_worker("dtensor_policy_worker_v2"))
    class ExportDTensorPolicyWorker(DTensorPolicyWorkerV2Impl):
        @torch.no_grad()
        def export_hf_checkpoint(self, path: str, dtypes: dict[str, str], sizes: dict[str, int]) -> dict[str, Any]:
            """Write the policy as an HF safetensors checkpoint at ``path``, one shard per rank."""
            distributed = torch.distributed.is_initialized()
            rank = torch.distributed.get_rank() if distributed else 0
            world_size = torch.distributed.get_world_size() if distributed else 1
            final = Path(path)
            tmp = final.with_name(final.name + ".tmp")
            offloaded = bool(getattr(self, "cpu_offload", False))
            if offloaded:
                self.model = self.move_to_cuda(self.model)
            try:
                # target_dtype=None leaves each tensor in the policy's dtype; write_checkpoint
                # then casts once, to the base checkpoint's dtype.
                result = write_checkpoint(
                    dtensor_params_generator(self.model, None),
                    tmp,
                    dtypes,
                    sizes,
                    rank=rank,
                    world_size=world_size,
                )
            finally:
                if offloaded:
                    self.model = self.move_to_cpu(self.model)
            # The all-reduce is also the barrier: every shard is written before rank 0 renames.
            failed = _any_rank_failed(result["error"] is not None)
            if rank == 0:
                finish_export(final, tmp, failed)
            if failed:
                raise RuntimeError(f"checkpoint export failed (rank {rank}: {result['error'] or 'peer failed'})")
            return {**result, "path": str(final), "rank": rank}
