"""Log what the dashboard shows to W&B, next to the trainer's own run.

The trainer's run (NeMo-RL) is ``<entity>/<project>/<run tag>``; this module writes a second run,
``<run tag>-dashboard``, in the same group (``WANDB_RUN_GROUP`` is the run tag for both). It is a
separate run because a second writer on the trainer's run renumbers its steps: W&B assigns one
``_step`` sequence per run across all writers, while NeMo-RL logs with ``step=<training step>``.

* ``live/*`` against ``live/elapsed_s`` (seconds since the run started): every ``@@metric`` the
  page draws (trainer GPUs, sampler tokens and requests, the rollout sandbox pool, sandbox
  counts, sampler GPUs), flattened to numbers.
* ``step/*`` against ``step/step``: each step's NeMo-RL metrics and timings as the dashboard
  parsed them, with ``step/off_policy``; ``step/hotload_*`` per weights version; AIME accuracy.
"""

from __future__ import annotations

import threading
from typing import Any

from async_grpo_tool_use.dashboard.server import Hub

LIVE_KINDS = ("gpu", "sampler", "env", "sandboxes", "deployment")


def flatten(prefix: str, value: Any, out: dict[str, float]) -> None:
    if isinstance(value, bool):
        out[prefix] = float(value)
    elif isinstance(value, (int, float)):
        out[prefix] = value
    elif isinstance(value, dict):
        for key, item in value.items():
            flatten(f"{prefix}_{key}", item, out)
    elif isinstance(value, list) and value and all(isinstance(v, (int, float)) for v in value):
        out[f"{prefix}_avg"] = sum(value) / len(value)


def row(event: dict, t0: float) -> dict[str, float] | None:
    """The W&B row for a page event, or None for events with nothing to chart."""
    kind = event.get("type")
    values: dict[str, float] = {}
    if kind == "metric" and event.get("k") in LIVE_KINDS:
        for key, value in event.items():
            if key not in ("type", "k", "t", "ts"):
                flatten(f"live/{event['k']}_{key}", value, values)
        return {"live/elapsed_s": event["t"] - t0, **values} if values else None
    if kind == "step_done":
        for key, value in {**event["metrics"], **{f"time_{k}": v for k, v in event["timings"].items()}}.items():
            flatten(f"step/{key.replace('/', '_')}", value, values)
        return {"step/step": event["step"], **values}
    if kind == "hotload":
        return {"step/step": event["version"], "step/hotload_gib": event["gib"],
                "step/hotload_export_s": event["export_s"], "step/hotload_load_s": event["load_s"]}  # fmt: skip
    if kind == "accuracy":
        return {"step/step": event["step"], "step/aime_accuracy": event["accuracy"]}
    return None


class WandbLog:
    """Follow the hub and log its events to the dashboard's W&B run, from a thread."""

    def __init__(self, hub: Hub, env: dict[str, str], run_tag: str, meta: dict) -> None:
        import wandb

        self.run = wandb.init(
            entity=env["WANDB_ENTITY"], project=env["WANDB_PROJECT"], id=f"{run_tag}-dashboard",
            name=f"{run_tag}-dashboard", group=run_tag, job_type="dashboard", config=meta,
            settings=wandb.Settings(silent=True, console="off"),
        )  # fmt: skip
        self.run.define_metric("live/elapsed_s")
        self.run.define_metric("live/*", step_metric="live/elapsed_s")
        self.run.define_metric("step/step")
        self.run.define_metric("step/*", step_metric="step/step")
        self.url = self.run.url
        self.events, self.listener = hub.subscribe()
        self.hub = hub
        self.thread = threading.Thread(target=self._pump, daemon=True)
        self.thread.start()

    def _pump(self) -> None:
        t0 = self.events[0]["t"] if self.events else None
        pending = list(self.events)
        while True:
            event = pending.pop(0) if pending else self.listener.get()
            if event is None:
                return
            t0 = event["t"] if t0 is None else t0
            values = row(event, t0)
            if values:
                self.run.log(values)

    def finish(self) -> None:
        self.listener.put(None)
        self.thread.join(timeout=30)
        self.hub.unsubscribe(self.listener)
        self.run.finish()
