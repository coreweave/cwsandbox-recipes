"""Worker run attachment for the cwsandbox SDK's own reporter."""

import sys
from types import SimpleNamespace

import pytest

from verl_taubench.sandbox.reporting import attach_sdk_reporter


def test_shared_training_preserves_step_axis_without_changing_sdk_metrics():
    from unittest.mock import Mock

    from verl.utils.tracking import Tracking

    from verl_taubench.sandbox.reporting import configure_shared_tracking

    run = Mock(settings=SimpleNamespace(mode="shared"))
    backend = SimpleNamespace(run=run)
    tracking = Tracking.__new__(Tracking)
    tracking.logger = {"wandb": backend}

    configure_shared_tracking(tracking)
    configure_shared_tracking(tracking)
    data = {"reward": 0.5}
    tracking.log(data, step=3)
    tracking.log(data, step=4)
    tracking.finish()

    assert data == {"reward": 0.5}
    run.define_metric.assert_called_once_with("reward", step_metric="training/global_step")
    assert run.log.call_args_list[0].kwargs == {"data": {"reward": 0.5, "training/global_step": 3}}
    assert run.log.call_args_list[1].kwargs == {"data": {"reward": 0.5, "training/global_step": 4}}
    run.finish.assert_called_once_with(exit_code=0)


def test_non_shared_training_logger_is_unchanged():
    from verl_taubench.sandbox.reporting import configure_shared_tracking

    backend = SimpleNamespace(run=SimpleNamespace(settings=SimpleNamespace(mode="offline")))
    tracking = SimpleNamespace(logger={"wandb": backend})
    configure_shared_tracking(tracking)
    assert tracking.logger["wandb"] is backend


def test_worker_joins_existing_run_once_and_cannot_finish_it(monkeypatch):
    import wandb

    calls = []
    sdk = SimpleNamespace(run=None, Settings=wandb.Settings)

    def init(**kwargs):
        calls.append(kwargs)
        sdk.run = SimpleNamespace(id=kwargs["id"])
        return sdk.run

    sdk.init = init
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    monkeypatch.setenv("WANDB_MODE", "shared")
    monkeypatch.setenv("WANDB_RUN_ID", "training-run")
    monkeypatch.setenv("WANDB_ENTITY", "team")
    config = SimpleNamespace(trainer=SimpleNamespace(logger=["console", "wandb"], project_name="project"))

    attach_sdk_reporter(config)
    attach_sdk_reporter(config)

    assert len(calls) == 1
    assert calls[0]["id"] == "training-run"
    assert calls[0]["project"] == "project"
    assert calls[0]["entity"] == "team"
    assert calls[0]["mode"] == "shared"
    assert "resume" not in calls[0]
    settings = calls[0]["settings"]
    assert settings.x_primary is False
    assert settings.x_update_finish_state is False
    assert settings.x_disable_stats is True


@pytest.mark.parametrize("mode,backends", [("offline", ["wandb"]), ("disabled", ["wandb"]), ("shared", ["console"])])
def test_non_online_or_non_wandb_training_does_not_open_a_worker_run(monkeypatch, mode, backends):
    monkeypatch.setenv("WANDB_MODE", mode)
    monkeypatch.setenv("WANDB_RUN_ID", "training-run")
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **kw: pytest.fail("unexpected run")))
    attach_sdk_reporter(SimpleNamespace(trainer=SimpleNamespace(logger=backends, project_name="project")))
