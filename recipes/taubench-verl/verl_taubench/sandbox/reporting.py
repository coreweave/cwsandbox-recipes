"""Attach the SDK's W&B reporter to the training run in a Ray worker."""

from __future__ import annotations

import os
import threading
from typing import Any

_run_lock = threading.Lock()


class _SharedTrainingLogger:
    """Keep veRL's step axis: shared runs ignore wandb.log(step=...)."""

    def __init__(self, run: Any):
        self.run = run
        self._defined_metrics: set[str] = set()

    def log(self, data: dict[str, Any], step: int) -> None:
        for key in data:
            if key != "training/global_step" and key not in self._defined_metrics:
                self.run.define_metric(key, step_metric="training/global_step")
                self._defined_metrics.add(key)
        self.run.log(data={**data, "training/global_step": step})

    def finish(self, exit_code: int = 0) -> None:
        self.run.finish(exit_code=exit_code)


def configure_shared_tracking(tracking: Any) -> None:
    """Adapt only veRL logging; SDK reporters keep their own metric definitions."""
    backend = tracking.logger.get("wandb")
    if backend is None or isinstance(backend, _SharedTrainingLogger):
        return
    run = backend.run
    if run is not None and run.settings.mode == "shared":
        tracking.logger["wandb"] = _SharedTrainingLogger(run)


def attach_sdk_reporter(config: Any) -> None:
    """Provide an ambient run; cwsandbox Session owns collection and reporting."""
    trainer = getattr(config, "trainer", None)
    backends = getattr(trainer, "logger", ()) or ()
    if isinstance(backends, str):
        backends = [backends]
    run_id = os.environ.get("WANDB_RUN_ID")
    if "wandb" not in backends or not run_id or os.environ.get("WANDB_MODE") != "shared":
        return

    import wandb

    with _run_lock:
        if wandb.run is not None:
            if wandb.run.id != run_id:
                raise RuntimeError("Sandbox SDK reporter found a different active W&B run")
            return
        wandb.init(
            id=run_id,
            project=str(trainer.project_name),
            entity=os.environ.get("WANDB_ENTITY"),
            mode="shared",
            settings=wandb.Settings(
                x_primary=False,
                x_update_finish_state=False,
                x_label=f"sandbox-worker-{os.getpid()}",
                x_disable_stats=True,
                console="off",
            ),
        )
