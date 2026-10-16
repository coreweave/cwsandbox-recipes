"""NeMo-RL DTensor policy worker with an ``export_hf_checkpoint`` method.

All ranks read the full (gathered) weights. Rank 0 writes them as safetensors, in the
same tensor names and dtypes as the base checkpoint.

This module runs in NeMo-RL's worker venv, so it imports only torch, safetensors, ray
and nemo_rl.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import torch

SHARD_BYTES = 4 * 2**30
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


def write_checkpoint(
    params: Iterable[tuple[str, torch.Tensor]],
    path: Path,
    dtypes: dict[str, str],
    *,
    write: bool,
    shard_bytes: int = SHARD_BYTES,
) -> dict[str, Any]:
    """Consume ``params`` fully; when ``write`` is true, save the base checkpoint's tensors.

    Ranks that do not write still iterate, because the generator runs a collective per tensor.
    A write error is held until iteration ends, so peers never block in a collective.
    """
    tmp = path.with_name(path.name + ".tmp")
    error: BaseException | None = None
    shards: list[tuple[str, list[str]]] = []
    pending: dict[str, torch.Tensor] = {}
    pending_bytes = 0
    total_bytes = 0
    seen: set[str] = set()

    def flush() -> None:
        nonlocal pending, pending_bytes
        if not pending:
            return
        from safetensors.torch import save_file

        name = f"shard-{len(shards):05d}.safetensors"
        save_file(pending, str(tmp / name), metadata={"format": "pt"})
        shards.append((name, list(pending)))
        pending, pending_bytes = {}, 0

    if write:
        try:
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
        except Exception as exc:  # noqa: BLE001 - reported after iteration
            error = exc

    for name, tensor in params:
        if name not in dtypes:
            continue
        seen.add(name)
        if not write or error is not None:
            continue
        try:
            cpu = tensor.detach().to("cpu", target_dtype(name, tensor, dtypes), copy=True)
            if pending and pending_bytes + cpu.nbytes > shard_bytes:
                flush()
            pending[name] = cpu.contiguous()
            pending_bytes += cpu.nbytes
            total_bytes += cpu.nbytes
        except Exception as exc:  # noqa: BLE001 - keep iterating so peers do not block
            error = exc

    missing = sorted(set(dtypes) - seen)
    if write and error is None and missing:
        error = RuntimeError(f"export is missing {len(missing)} tensors, e.g. {missing[:5]}")
    if write and error is None:
        try:
            flush()
            weight_map = {}
            for i, (name, names) in enumerate(shards):
                final = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
                (tmp / name).rename(tmp / final)
                weight_map.update(dict.fromkeys(names, final))
            index = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
            (tmp / "model.safetensors.index.json").write_text(json.dumps(index, indent=1))
            shutil.rmtree(path, ignore_errors=True)
            tmp.rename(path)
        except Exception as exc:  # noqa: BLE001
            error = exc
    if write and error is not None:
        shutil.rmtree(tmp, ignore_errors=True)
    return {
        "error": None if error is None else f"{type(error).__name__}: {error}",
        "tensors": len(seen),
        "bytes": total_bytes,
        "shards": len(shards),
        "path": str(path),
    }


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
        def export_hf_checkpoint(self, path: str, dtypes: dict[str, str]) -> dict[str, Any]:
            """Write the policy as an HF safetensors checkpoint at ``path`` (rank 0 writes)."""
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            offloaded = bool(getattr(self, "cpu_offload", False))
            if offloaded:
                self.model = self.move_to_cuda(self.model)
            try:
                # target_dtype=None leaves each tensor in the policy's dtype; write_checkpoint
                # then casts once, to the base checkpoint's dtype.
                result = write_checkpoint(
                    dtensor_params_generator(self.model, None),
                    Path(path),
                    dtypes,
                    write=rank == 0,
                )
            finally:
                if offloaded:
                    self.model = self.move_to_cpu(self.model)
            if _any_rank_failed(result["error"] is not None):
                raise RuntimeError(f"checkpoint export failed (rank {rank}: {result['error'] or 'peer failed'})")
            return {**result, "rank": rank}
