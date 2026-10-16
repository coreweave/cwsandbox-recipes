"""``policy.generation.backend: coreweave_hotload``: generation on a Dedicated Inference deployment.

NeMo-Gym samples from the deployment, so NeMo-RL never calls ``generate``. NeMo-RL still calls
the lifecycle methods, which do nothing here.

Config, under ``policy.generation.hotload_cfg``:

* ``base_url``: the deployment's OpenAI URL (its gateway endpoint), ending in ``/v1``.
* ``model``: the served model name.
* ``deployment_id``: the deployment. Its bucket and prefix for snapshots come from the API.
* ``run_id``: prefix of every hot-load identity (``<run_id>-v000001``, ...).
* ``max_model_len``: the deployment's ``max-model-len``.
"""

from __future__ import annotations

from typing import Any

from nemo_rl.models.generation.interfaces import GenerationInterface

REQUIRED = ("base_url", "model", "deployment_id", "run_id")


class HotLoadGeneration(GenerationInterface):
    def __init__(self, config: dict[str, Any]) -> None:
        cfg = dict(config.get("hotload_cfg") or {})
        missing = [key for key in REQUIRED if not cfg.get(key)]
        if missing:
            raise ValueError(f"hotload_cfg needs {', '.join(missing)}")
        if not cfg["base_url"].rstrip("/").endswith("/v1"):
            raise ValueError(f"hotload_cfg.base_url must end in /v1, got {cfg['base_url']!r}")
        if not cfg.get("max_model_len") or int(cfg["max_model_len"]) <= 0:
            raise ValueError("hotload_cfg.max_model_len must be a positive integer")
        self.hotload_cfg = cfg
        self.base_url = cfg["base_url"].rstrip("/")
        self.max_model_len = int(cfg["max_model_len"])
        self.cfg = {**config, "max_total_sequence_length": self.max_model_len}
        self.weight_synchronizer: Any = None  # attached by grpo.setup()

    def init_collective(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    def generate(self, data: Any, greedy: bool = False) -> Any:
        raise NotImplementedError("coreweave_hotload samples through NeMo-Gym; set env.should_use_nemo_gym=true")

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        return None

    def clear_logger_metrics(self) -> None:
        return None

    def get_logger_metrics(self) -> dict[str, Any]:
        return {}

    def shutdown(self) -> bool:
        return True
