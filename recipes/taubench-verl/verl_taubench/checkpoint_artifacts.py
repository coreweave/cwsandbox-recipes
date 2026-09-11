"""Upload veRL checkpoints and register them as W&B reference artifacts.

Checkpoint bytes stay in CoreWeave AI Object Storage.  W&B stores only an
external reference to each completed ``global_step_*`` prefix.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_REFERENCE_MARKER_PREFIX = ".wandb-reference-global_step_"
_ARTIFACT_WAIT_TIMEOUT_SECONDS = 120
_TRANSFER_CONFIG_KWARGS = dict(
    multipart_threshold=64 * 1024 * 1024,
    multipart_chunksize=64 * 1024 * 1024,
    max_concurrency=8,
)


@dataclass(frozen=True)
class CheckpointUploadResult:
    """Result of uploading and registering one checkpoint step."""

    step: int
    s3_uri: str
    uploaded: int
    skipped: int
    artifact_name: str


def reference_marker(local_root: Path, step: int) -> Path:
    """Return the local success marker for one W&B reference artifact."""

    return Path(local_root) / f"{_REFERENCE_MARKER_PREFIX}{step}"


def checkpoint_steps(local_root: Path) -> list[int]:
    """Return checkpoint step numbers present below ``local_root``."""

    steps: list[int] = []
    for path in Path(local_root).glob("global_step_*"):
        if not path.is_dir():
            continue
        try:
            steps.append(int(path.name.removeprefix("global_step_")))
        except ValueError:
            continue
    return sorted(steps)


def steps_needing_reference(local_root: Path) -> list[int]:
    """Return locally saved steps that have no successful W&B marker."""

    root = Path(local_root)
    return [step for step in checkpoint_steps(root) if not reference_marker(root, step).is_file()]


def _configure_wandb_s3(endpoint: str, access_key: str, secret_key: str) -> None:
    """Expose CoreWeave S3 credentials under names used by W&B's S3 handler."""

    os.environ["AWS_S3_ENDPOINT_URL"] = endpoint
    os.environ["AWS_ACCESS_KEY_ID"] = access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = secret_key


def _build_client(endpoint: str, access_key: str | None = None, secret_key: str | None = None):
    """Build a virtual-hosted-style client for CoreWeave object storage."""

    try:
        import boto3
        from botocore.config import Config as BotoConfig
    except ImportError as exc:  # pragma: no cover - guarded by train_grpo.sh
        raise RuntimeError("boto3 is required: python -m pip install boto3") from exc

    access_key = access_key or os.environ.get("CW_ACCESS_KEY")
    secret_key = secret_key or os.environ.get("CW_SECRET_KEY")
    if not access_key or not secret_key:
        raise ValueError(
            "CW_ACCESS_KEY and CW_SECRET_KEY must be set for object-storage upload"
        )
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=BotoConfig(
            s3={"addressing_style": "virtual"},
            retries={"max_attempts": 3, "mode": "standard"},
            connect_timeout=20,
        ),
    )


def _remote_size(client: Any, bucket: str, key: str) -> int | None:
    from botocore.exceptions import ClientError

    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in (
            "404",
            "NoSuchKey",
            "NotFound",
        ):
            return None
        raise
    return int(head["ContentLength"])


def sync_directory(
    client: Any,
    local_dir: Path,
    bucket: str,
    prefix: str,
) -> tuple[int, int]:
    """Upload files below a directory, returning ``(uploaded, skipped)``."""

    from boto3.s3.transfer import TransferConfig

    local_dir = Path(local_dir)
    transfer_config = TransferConfig(**_TRANSFER_CONFIG_KWARGS)
    uploaded = skipped = 0
    files = sorted(
        path
        for path in local_dir.rglob("*")
        if path.is_file() and not path.name.startswith(_REFERENCE_MARKER_PREFIX)
    )
    if not files:
        raise ValueError(f"no files found under {local_dir}")
    for path in files:
        rel = path.relative_to(local_dir).as_posix()
        key = f"{prefix.rstrip('/')}/{rel}" if prefix else rel
        size = path.stat().st_size
        # This pointer commonly changes from e.g. "3" to "4" without changing
        # size. A size-only idempotency check would leave recovery pointing at
        # the previous checkpoint even though the new shards were uploaded.
        is_latest_pointer = rel == "latest_checkpointed_iteration.txt"
        if not is_latest_pointer and _remote_size(client, bucket, key) == size:
            skipped += 1
            continue
        print(
            f"upload {rel} ({size / 2**30:.2f} GiB) -> s3://{bucket}/{key}",
            flush=True,
        )
        client.upload_file(str(path), bucket, key, Config=transfer_config)
        uploaded += 1
    return uploaded, skipped


def _artifact_collection_name(experiment_name: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", experiment_name).strip("-.")
    return f"{safe_name or 'training'}-checkpoints"


def _log_reference(
    *,
    run: Any,
    local_root: Path,
    bucket: str,
    prefix: str,
    experiment_name: str,
    step: int,
) -> tuple[str, str]:
    import wandb

    artifact_name = _artifact_collection_name(experiment_name)
    s3_uri = f"s3://{bucket}/{prefix.rstrip('/')}/global_step_{step}"
    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        description="veRL checkpoint stored in CoreWeave AI Object Storage",
        metadata={"global_step": step, "s3_uri": s3_uri},
    )
    # The checkpoint files were uploaded and verified by sync_directory. Tell
    # W&B to record the external prefix without recursively listing and hashing
    # multi-GiB shards through its S3 reference handler.
    artifact.add_reference(s3_uri, checksum=False)
    logged = run.log_artifact(artifact, aliases=["latest", f"step-{step}"])
    logged.wait(timeout=_ARTIFACT_WAIT_TIMEOUT_SECONDS)
    qualified_name = getattr(logged, "name", None) or artifact_name
    reference_marker(local_root, step).write_text(str(qualified_name), encoding="utf-8")
    print(
        f"[taubench] checkpoint step {step} registered as W&B reference "
        f"artifact {qualified_name} -> {s3_uri}",
        flush=True,
    )
    return s3_uri, str(qualified_name)


def upload_checkpoint_step(
    *,
    client: Any,
    run: Any,
    local_root: Path,
    bucket: str,
    prefix: str,
    endpoint: str,
    access_key: str,
    secret_key: str,
    experiment_name: str,
    step: int,
) -> CheckpointUploadResult:
    """Upload one complete step, then publish its W&B external reference."""

    local_root = Path(local_root)
    step_dir = local_root / f"global_step_{step}"
    if not step_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {step_dir}")

    _configure_wandb_s3(endpoint, access_key, secret_key)
    step_prefix = f"{prefix.rstrip('/')}/global_step_{step}"
    uploaded, skipped = sync_directory(client, step_dir, bucket, step_prefix)

    # Publish the pointer only after every shard for this step is present.
    client.put_object(
        Bucket=bucket,
        Key=f"{prefix.rstrip('/')}/latest_checkpointed_iteration.txt",
        Body=str(step).encode(),
    )
    s3_uri, artifact_name = _log_reference(
        run=run,
        local_root=local_root,
        bucket=bucket,
        prefix=prefix,
        experiment_name=experiment_name,
        step=step,
    )
    return CheckpointUploadResult(
        step=step,
        s3_uri=s3_uri,
        uploaded=uploaded,
        skipped=skipped,
        artifact_name=artifact_name,
    )


def log_missing_references(
    *,
    local_root: Path,
    bucket: str,
    prefix: str,
    endpoint: str,
    access_key: str,
    secret_key: str,
    project_name: str,
    experiment_name: str,
    run_id: str | None,
    entity: str | None = None,
) -> list[int]:
    """Repair missing W&B references after upload, resuming only if needed."""

    missing = steps_needing_reference(local_root)
    if not missing:
        return []
    if not run_id:
        raise ValueError("WANDB_RUN_ID is required to repair checkpoint references")

    import wandb

    _configure_wandb_s3(endpoint, access_key, secret_key)
    run = wandb.init(
        project=project_name,
        entity=entity,
        id=run_id,
        resume="must",
        # Recovery runs after training; shared mode would ignore resume="must".
        **({"mode": "online"} if os.environ.get("WANDB_MODE") == "shared" else {}),
    )
    try:
        for step in missing:
            _log_reference(
                run=run,
                local_root=Path(local_root),
                bucket=bucket,
                prefix=prefix,
                experiment_name=experiment_name,
                step=step,
            )
    finally:
        run.finish()
    return missing


class AsyncCheckpointUploader:
    """Serialize checkpoint uploads off the trainer's checkpoint-save thread."""

    def __init__(self, upload_callable: Callable[[int], Any]):
        self._upload_callable = upload_callable
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="taubench-checkpoint-upload",
        )
        self._futures: list[tuple[int, Future[Any]]] = []
        self._finished = False

    def submit(self, step: int) -> None:
        if self._finished:
            raise RuntimeError("checkpoint uploader is already finished")
        self._futures.append((step, self._executor.submit(self._upload_callable, step)))

    def finish(self) -> list[tuple[int, BaseException]]:
        """Wait for queued uploads and return failures for final recovery."""

        if not self._finished:
            self._executor.shutdown(wait=True)
            self._finished = True
        failures: list[tuple[int, BaseException]] = []
        for step, future in self._futures:
            try:
                future.result()
            except BaseException as exc:
                failures.append((step, exc))
        return failures
