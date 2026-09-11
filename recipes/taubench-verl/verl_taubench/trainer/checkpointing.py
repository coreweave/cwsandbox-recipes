"""veRL trainer hook for asynchronous checkpoint persistence."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from omegaconf import open_dict
from verl.trainer.ppo.v1 import PPOTrainerSync, register_trainer

from verl_taubench.checkpoint_artifacts import (
    AsyncCheckpointUploader,
    _build_client,
    upload_checkpoint_step,
)
from verl_taubench.sandbox.reporting import configure_shared_tracking

logger = logging.getLogger(__name__)


@register_trainer("taubench_sync")
class TauBenchPPOTrainer(PPOTrainerSync):
    """Sync veRL trainer that persists every scheduled checkpoint off-thread."""

    def __init__(self, config: Any):
        # The registry key selects this subclass, but veRL also uses this value
        # to select its replay-buffer semantics and mode-specific config.
        with open_dict(config):
            config.trainer.v1.trainer_mode = "sync"
        super().__init__(config=config)
        self._checkpoint_uploader: AsyncCheckpointUploader | None = None
        self._checkpoint_uploader_initialized = False

    def on_validate_begin(self) -> None:
        configure_shared_tracking(self.logger)
        super().on_validate_begin()

    def on_train_begin(self) -> None:
        configure_shared_tracking(self.logger)
        super().on_train_begin()

    def _get_checkpoint_uploader(self) -> AsyncCheckpointUploader | None:
        if self._checkpoint_uploader_initialized:
            return self._checkpoint_uploader
        self._checkpoint_uploader_initialized = True

        access_key = os.environ.get("CW_ACCESS_KEY")
        secret_key = os.environ.get("CW_SECRET_KEY")
        if not access_key or not secret_key:
            logger.info(
                "CW_ACCESS_KEY/CW_SECRET_KEY are not set; checkpoints will remain local"
            )
            return None

        import wandb

        run = wandb.run
        if run is None:
            logger.warning(
                "W&B is not active; deferring checkpoint upload/reference logging "
                "to post-training recovery"
            )
            return None

        endpoint = os.environ.get("CW_ENDPOINT", "https://cwobject.com")
        bucket = os.environ.get("CW_BUCKET", "verl-taubench-checkpoints")
        project_name = str(self.config.trainer.project_name)
        experiment_name = str(self.config.trainer.experiment_name)
        local_root = Path(str(self.config.trainer.default_local_dir))
        prefix = f"checkpoints/{project_name}/{experiment_name}"
        client = _build_client(endpoint, access_key, secret_key)

        def upload(step: int) -> None:
            result = upload_checkpoint_step(
                client=client,
                run=run,
                local_root=local_root,
                bucket=bucket,
                prefix=prefix,
                endpoint=endpoint,
                access_key=access_key,
                secret_key=secret_key,
                experiment_name=experiment_name,
                step=step,
            )
            logger.info(
                "checkpoint step %d persisted (%d uploaded, %d already present)",
                step,
                result.uploaded,
                result.skipped,
            )

        self._checkpoint_uploader = AsyncCheckpointUploader(upload)
        return self._checkpoint_uploader

    def _save_checkpoint(self) -> None:
        # veRL writes every shard and then its local latest-step pointer before
        # this returns, so the background upload never sees a partial save.
        super()._save_checkpoint()
        uploader = self._get_checkpoint_uploader()
        if uploader is not None:
            uploader.submit(int(self.global_steps))

    def fit(self, agent_loop_manager: Any):
        try:
            return super().fit(agent_loop_manager)
        finally:
            # TaskRunnerV1 calls tracking.finish only after fit returns. Drain
            # here so normal uploads use the already-active run and never need
            # wandb.init(resume=...). Failed steps are left unmarked for the
            # post-training recovery command.
            if self._checkpoint_uploader_initialized and self._checkpoint_uploader:
                for step, exc in self._checkpoint_uploader.finish():
                    logger.warning(
                        "checkpoint step %d upload/reference failed; "
                        "post-training recovery will retry it: %s",
                        step,
                        exc,
                    )
