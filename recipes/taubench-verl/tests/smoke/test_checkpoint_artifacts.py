"""Checkpoint persistence and W&B reference-artifact behavior."""

from __future__ import annotations

import importlib
import sys
import threading
import types
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError


def _checkpoint_module():
    try:
        return importlib.import_module("verl_taubench.checkpoint_artifacts")
    except ModuleNotFoundError:
        return None


class FakeS3Client:
    def __init__(self, events: list[Any]):
        self.events = events

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        raise ClientError(
            {"Error": {"Code": "404", "Message": "not found"}},
            "HeadObject",
        )

    def upload_file(self, path: str, bucket: str, key: str, *, Config: Any) -> None:
        self.events.append(("upload", bucket, key, Path(path).name))

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> None:
        self.events.append(("latest", Bucket, Key, Body))


def test_sync_always_refreshes_same_size_latest_pointer(tmp_path: Path) -> None:
    """Regression: step 3 -> 4 has the same byte size but different meaning."""
    module = _checkpoint_module()
    assert module is not None
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("4", encoding="utf-8")
    events: list[Any] = []

    class SameSizeClient(FakeS3Client):
        def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
            return {"ContentLength": 1}

    uploaded, skipped = module.sync_directory(
        SameSizeClient(events), tmp_path, "bucket", "checkpoints/project/experiment"
    )

    assert (uploaded, skipped) == (1, 0)
    assert events[0][0:3] == (
        "upload",
        "bucket",
        "checkpoints/project/experiment/latest_checkpointed_iteration.txt",
    )


def test_checkpoint_reference_is_logged_only_after_step_upload_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: W&B must never publish a reference to a partial checkpoint."""
    module = _checkpoint_module()
    assert module is not None, "checkpoint artifact integration is not implemented"

    checkpoint_root = tmp_path / "checkpoints"
    actor = checkpoint_root / "global_step_3" / "actor"
    actor.mkdir(parents=True)
    (actor / "model.pt").write_bytes(b"weights")
    events: list[Any] = []

    class FakeArtifact:
        def __init__(self, name: str, type: str, description: str, metadata: dict[str, Any]):
            self.name = name
            self.type = type
            self.description = description
            self.metadata = metadata

        def add_reference(self, uri: str, *, checksum: bool) -> None:
            assert checksum is False
            assert events[:2] == [
                (
                    "upload",
                    "bucket",
                    "checkpoints/project/experiment/global_step_3/actor/model.pt",
                    "model.pt",
                ),
                (
                    "latest",
                    "bucket",
                    "checkpoints/project/experiment/latest_checkpointed_iteration.txt",
                    b"3",
                ),
            ]
            events.append(("reference", uri))

    class LoggedArtifact:
        name = "experiment-checkpoints:v3"

        def wait(self, *, timeout: int) -> "LoggedArtifact":
            events.append(("wait", timeout))
            return self

    class FakeRun:
        def log_artifact(self, artifact: FakeArtifact, *, aliases: list[str]):
            events.append(("log", artifact.name, artifact.type, aliases, artifact.metadata))
            return LoggedArtifact()

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(Artifact=FakeArtifact))
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_S3_ENDPOINT_URL", raising=False)

    result = module.upload_checkpoint_step(
        client=FakeS3Client(events),
        run=FakeRun(),
        local_root=checkpoint_root,
        bucket="bucket",
        prefix="checkpoints/project/experiment",
        endpoint="https://cwobject.com",
        access_key="cw-access",
        secret_key="cw-secret",
        experiment_name="experiment",
        step=3,
    )

    assert result.s3_uri == (
        "s3://bucket/checkpoints/project/experiment/global_step_3"
    )
    assert events[2] == (
        "reference",
        "s3://bucket/checkpoints/project/experiment/global_step_3",
    )
    assert events[3][0:4] == (
        "log",
        "experiment-checkpoints",
        "model",
        ["latest", "step-3"],
    )
    assert events[4] == ("wait", 120)
    assert module.reference_marker(checkpoint_root, 3).is_file()
    assert module.reference_marker(checkpoint_root, 3).read_text() == (
        "experiment-checkpoints:v3"
    )
    assert module.os.environ["AWS_ACCESS_KEY_ID"] == "cw-access"
    assert module.os.environ["AWS_SECRET_ACCESS_KEY"] == "cw-secret"
    assert module.os.environ["AWS_S3_ENDPOINT_URL"] == "https://cwobject.com"


def test_async_checkpoint_upload_does_not_block_checkpoint_save() -> None:
    """Regression: multi-GiB uploads must not stall the next training step."""
    module = _checkpoint_module()
    assert module is not None, "checkpoint artifact integration is not implemented"

    started = threading.Event()
    release = threading.Event()
    worker_thread_ids: list[int] = []

    def upload(step: int) -> None:
        worker_thread_ids.append(threading.get_ident())
        started.set()
        assert release.wait(timeout=2)

    uploader = module.AsyncCheckpointUploader(upload)
    uploader.submit(3)

    assert started.wait(timeout=1)
    assert worker_thread_ids != [threading.get_ident()]
    release.set()
    assert uploader.finish() == []


def test_async_upload_failure_is_returned_for_final_recovery() -> None:
    module = _checkpoint_module()
    assert module is not None, "checkpoint artifact integration is not implemented"

    def fail(step: int) -> None:
        raise RuntimeError(f"upload {step} failed")

    uploader = module.AsyncCheckpointUploader(fail)
    uploader.submit(7)

    failures = uploader.finish()

    assert len(failures) == 1
    assert failures[0][0] == 7
    assert isinstance(failures[0][1], RuntimeError)


def test_trainer_submits_upload_only_after_local_checkpoint_save(monkeypatch) -> None:
    """The async worker must not race partially written veRL checkpoint files."""
    from verl.trainer.ppo.v1 import PPOTrainerSync

    from verl_taubench.trainer.checkpointing import TauBenchPPOTrainer

    events: list[Any] = []

    class FakeUploader:
        def submit(self, step: int) -> None:
            events.append(("submit", step))

    trainer = TauBenchPPOTrainer.__new__(TauBenchPPOTrainer)
    trainer.global_steps = 11
    trainer._checkpoint_uploader = FakeUploader()
    trainer._checkpoint_uploader_initialized = True
    monkeypatch.setattr(
        PPOTrainerSync,
        "_save_checkpoint",
        lambda _self: events.append("save"),
    )

    trainer._save_checkpoint()

    assert events == ["save", ("submit", 11)]


def test_trainer_drains_uploads_before_wandb_tracking_finishes(monkeypatch) -> None:
    """TaskRunner closes W&B only after trainer.fit returns."""
    from verl.trainer.ppo.v1 import PPOTrainerSync

    from verl_taubench.trainer.checkpointing import TauBenchPPOTrainer

    events: list[str] = []

    class FakeUploader:
        def finish(self) -> list[tuple[int, BaseException]]:
            events.append("finish uploads")
            return []

    trainer = TauBenchPPOTrainer.__new__(TauBenchPPOTrainer)
    trainer._checkpoint_uploader = FakeUploader()
    trainer._checkpoint_uploader_initialized = True
    monkeypatch.setattr(
        PPOTrainerSync,
        "fit",
        lambda _self, _manager: events.append("fit") or "result",
    )

    assert trainer.fit(object()) == "result"
    assert events == ["fit", "finish uploads"]


def test_custom_trainer_is_registered() -> None:
    from verl.trainer.ppo.v1 import get_trainer_cls

    from verl_taubench.trainer.checkpointing import TauBenchPPOTrainer

    assert get_trainer_cls("taubench_sync") is TauBenchPPOTrainer


def test_recovery_does_not_resume_run_when_all_references_are_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _checkpoint_module()
    assert module is not None
    (tmp_path / "global_step_3").mkdir()
    module.reference_marker(tmp_path, 3).write_text("artifact:v0", encoding="utf-8")
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        types.SimpleNamespace(init=lambda **_kwargs: pytest.fail("must not resume")),
    )

    recovered = module.log_missing_references(
        local_root=tmp_path,
        bucket="bucket",
        prefix="checkpoints/project/experiment",
        endpoint="https://cwobject.com",
        access_key="access",
        secret_key="secret",
        project_name="project",
        experiment_name="experiment",
        run_id="run-id",
    )

    assert recovered == []


@pytest.mark.parametrize("mode", ["online", "shared"])
def test_recovery_resumes_once_only_for_missing_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setenv("WANDB_MODE", mode)
    module = _checkpoint_module()
    assert module is not None
    (tmp_path / "global_step_3").mkdir()
    events: list[Any] = []

    class FakeArtifact:
        def __init__(self, **kwargs: Any):
            self.name = kwargs["name"]

        def add_reference(self, uri: str, *, checksum: bool) -> None:
            events.append(("reference", uri, checksum))

    class FakeLogged:
        name = "experiment-checkpoints:v0"

        def wait(self, *, timeout: int) -> None:
            events.append(("wait", timeout))

    class FakeRun:
        def log_artifact(self, artifact: FakeArtifact, *, aliases: list[str]):
            events.append(("log", artifact.name, aliases))
            return FakeLogged()

        def finish(self) -> None:
            events.append("finish")

    def init(**kwargs: Any) -> FakeRun:
        events.append(("init", kwargs))
        return FakeRun()

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        types.SimpleNamespace(Artifact=FakeArtifact, init=init),
    )

    recovered = module.log_missing_references(
        local_root=tmp_path,
        bucket="bucket",
        prefix="checkpoints/project/experiment",
        endpoint="https://cwobject.com",
        access_key="access",
        secret_key="secret",
        project_name="project",
        experiment_name="experiment",
        run_id="run-id",
        entity="team",
    )

    assert recovered == [3]
    assert events[0] == (
        "init",
        {
            "project": "project",
            "entity": "team",
            "id": "run-id",
            "resume": "must",
            **({"mode": "online"} if mode == "shared" else {}),
        },
    )
    assert events[-1] == "finish"
    assert ("reference", "s3://bucket/checkpoints/project/experiment/global_step_3", False) in events
    assert ("wait", 120) in events
    assert module.reference_marker(tmp_path, 3).read_text() == (
        "experiment-checkpoints:v0"
    )


def test_recovery_reference_failure_does_not_fail_completed_byte_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A W&B metadata outage must not restart a completed GPU training job."""
    upload_script = importlib.import_module("scripts.upload_checkpoints")
    checkpoint = tmp_path / "global_step_3" / "data.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    events: list[str] = []

    monkeypatch.setattr(upload_script, "_build_client", lambda _endpoint: object())

    def sync(*_args: Any) -> tuple[int, int]:
        events.append("sync")
        return 1, 0

    def fail_reference(**_kwargs: Any) -> list[int]:
        events.append("reference")
        raise TimeoutError("artifact commit timed out")

    monkeypatch.setattr(upload_script, "sync_directory", sync)
    monkeypatch.setattr(upload_script, "log_missing_references", fail_reference)
    monkeypatch.setenv("CW_ACCESS_KEY", "access")
    monkeypatch.setenv("CW_SECRET_KEY", "secret")

    result = upload_script.main(
        [
            "--local-dir",
            str(tmp_path),
            "--bucket",
            "bucket",
            "--endpoint",
            "https://cwobject.com",
            "--prefix",
            "checkpoints/project/experiment",
            "--project",
            "project",
            "--experiment",
            "experiment",
            "--wandb-run-id",
            "run-id",
        ]
    )

    assert result == 0
    assert events == ["sync", "reference"]
    assert "checkpoint bytes are persisted" in capsys.readouterr().err


def test_recovery_byte_upload_failure_remains_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upload_script = importlib.import_module("scripts.upload_checkpoints")
    checkpoint = tmp_path / "global_step_3" / "data.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")

    monkeypatch.setattr(upload_script, "_build_client", lambda _endpoint: object())

    def fail_sync(*_args: Any) -> tuple[int, int]:
        raise RuntimeError("object upload failed")

    monkeypatch.setattr(upload_script, "sync_directory", fail_sync)

    with pytest.raises(RuntimeError, match="object upload failed"):
        upload_script.main(
            [
                "--local-dir",
                str(tmp_path),
                "--bucket",
                "bucket",
                "--endpoint",
                "https://cwobject.com",
            ]
        )
