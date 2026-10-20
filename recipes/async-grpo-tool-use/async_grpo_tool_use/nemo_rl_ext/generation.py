"""``policy.generation.backend: coreweave_hotload``: generation on a sampler NeMo-RL does not run.

The sampler is a Dedicated Inference deployment (``hotload_cfg.target: di``) or the self-hosted
vLLM of ``async_grpo_tool_use.sampler`` (``target: vllm``). NeMo-Gym samples from it, so NeMo-RL
never calls ``generate``. NeMo-RL still calls
the lifecycle methods, which do nothing here.

Async GRPO sends this object to its trajectory collector actor, which reads ``cfg`` and calls
``invalidate_kv_cache``. The copy leaves out the weight synchronizer: only the trainer hot-loads,
and its API and storage clients do not pickle.

Config, under ``policy.generation.hotload_cfg``:

* ``base_url``: the deployment's OpenAI URL (its gateway endpoint), ending in ``/v1``.
* ``model``: the served model name.
* ``target``: ``di`` (default) or ``vllm``.
* ``deployment_id`` (``di``): the deployment. Its bucket and prefix for snapshots come from the API.
* ``sampler_urls`` (``vllm``): the samplers' weight receivers, comma-separated.
* ``transfer`` (``vllm``): ``shared`` when the sampler can read the trainer's export directory,
  ``bucket`` to send exports through object storage (``weights_prefix``).
* ``run_id``: prefix of every hot-load identity (``<run_id>-v000001``, ...).
* ``max_model_len``: the deployment's ``max-model-len``.
* ``max_versions_behind``: how many versions the deployment may lag the trainer (``.inf``: no
  limit); above 0, the upload and hot-load run while the trainer trains (async GRPO only, see
  ``weight_sync``). Default 0.
"""

from __future__ import annotations

from typing import Any

from nemo_rl.models.generation.interfaces import GenerationInterface

REQUIRED = {
    "di": ("base_url", "model", "deployment_id", "run_id"),
    "vllm": ("base_url", "model", "sampler_urls", "run_id"),
}


class HotLoadGeneration(GenerationInterface):
    def __init__(self, config: dict[str, Any]) -> None:
        cfg = dict(config.get("hotload_cfg") or {})
        missing = [key for key in REQUIRED.get(cfg.get("target") or "di", REQUIRED["di"]) if not cfg.get(key)]
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

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "weight_synchronizer": None}

    def init_collective(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    def generate(self, data: Any, greedy: bool = False) -> Any:
        raise NotImplementedError("coreweave_hotload samples through NeMo-Gym; set env.should_use_nemo_gym=true")

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        # Validation calls this after a sync and must see the weights that sync exported, not
        # the ones still serving while hot-loads run in the background.
        if self.weight_synchronizer is not None:
            self.weight_synchronizer.wait()
        return True

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        return None

    def invalidate_kv_cache(self) -> bool:
        """Report success: each hot-load already resets the prompt cache (``RESET_ALL``).

        The reset covers requests admitted after the swap. Requests in flight during a swap keep
        their cache, as with vLLM's in-flight updates when the cache is not recomputed.
        """
        return True

    def clear_logger_metrics(self) -> None:
        return None

    def get_logger_metrics(self) -> dict[str, Any]:
        return {}

    def shutdown(self) -> bool:
        return True
