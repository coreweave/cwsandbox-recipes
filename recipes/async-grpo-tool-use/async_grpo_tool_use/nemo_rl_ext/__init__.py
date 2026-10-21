"""NeMo-RL extension for ``policy.generation.backend: coreweave_hotload``.

The patched NeMo-RL imports these classes. They run in the trainer on SUNK.

* :class:`HotLoadGeneration`: points NeMo-RL at the Dedicated Inference deployment. NeMo-Gym
  does the sampling, so this class has no GPUs and no engine.
* :class:`HotLoadWeightSynchronizer`: after each step, exports the policy, uploads it to
  object storage and hot-loads it into the deployment.
* ``worker.ExportDTensorPolicyWorker``: the policy worker with the export method.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

WORKER_EXTENSION = "async_grpo_tool_use.nemo_rl_ext.worker.ExportDTensorPolicyWorker"
_STOCK_WORKER = "nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2"

__all__ = [
    "HotLoadGeneration",
    "HotLoadWeightSynchronizer",
    "WORKER_EXTENSION",
    "register_worker_extension",
]


def register_worker_extension(policy_config: dict[str, Any]) -> str:
    """Give the export worker the stock DTensor worker's Python environment.

    Ray needs a registered environment for every actor class. The container prebuilds one venv
    per stock worker under ``NEMO_RL_VENV_DIR``; reusing it avoids building a new one.
    """
    if policy_config.get("megatron_cfg", {}).get("enabled", False):
        raise ValueError("coreweave_hotload supports the DTensor (Automodel) policy backend only")
    from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY

    extension = policy_config.get("worker_extension_cls_fqn") or WORKER_EXTENSION
    if extension not in ACTOR_ENVIRONMENT_REGISTRY:
        venv_dir = os.environ.get("NEMO_RL_VENV_DIR")
        prebuilt = Path(venv_dir or "") / _STOCK_WORKER / "bin" / "python"
        ACTOR_ENVIRONMENT_REGISTRY[extension] = (
            str(prebuilt) if venv_dir and prebuilt.exists() else ACTOR_ENVIRONMENT_REGISTRY[_STOCK_WORKER]
        )
    return extension


def __getattr__(name: str) -> Any:
    if name == "HotLoadGeneration":
        from async_grpo_tool_use.nemo_rl_ext.generation import HotLoadGeneration

        return HotLoadGeneration
    if name == "HotLoadWeightSynchronizer":
        from async_grpo_tool_use.nemo_rl_ext.weight_sync import HotLoadWeightSynchronizer

        return HotLoadWeightSynchronizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
