#!/usr/bin/env python3
"""Repair checkpoint uploads and W&B references after a training run.

verl's FSDP checkpoint manager can only write plain local files
(``trainer.default_local_dir``); its sole remote hook (``default_hdfs_dir``)
speaks HDFS, not S3. So object-storage upload happens out-of-band: this script
walks the checkpoint directory and puts every file under
``s3://$CW_BUCKET/<prefix>/`` via the S3-compatible CAIOS endpoint
(``$CW_ENDPOINT``, e.g. the in-cluster LOTA endpoint http://cwlota.com).

The trainer normally uploads each save asynchronously while its W&B run is
active. This command is the idempotent recovery path: existing objects are
skipped and only missing W&B reference artifacts cause the run to be resumed.

Usage:
    python scripts/upload_checkpoints.py \
        --local-dir checkpoints/my-project/my-experiment \
        --prefix checkpoints/my-project/my-experiment
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from verl_taubench.checkpoint_artifacts import (
    _build_client,
    log_missing_references,
    sync_directory,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", required=True, help="Checkpoint directory to upload.")
    parser.add_argument(
        "--bucket",
        default=os.environ.get("CW_BUCKET"),
        help="Target bucket (default: $CW_BUCKET).",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("CW_ENDPOINT"),
        help="S3-compatible endpoint URL (default: $CW_ENDPOINT).",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Key prefix inside the bucket (default: the --local-dir path as given).",
    )
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT"))
    parser.add_argument("--experiment", default=os.environ.get("EXPERIMENT_NAME"))
    parser.add_argument("--wandb-run-id", default=os.environ.get("WANDB_RUN_ID"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    args = parser.parse_args(argv)

    if not args.bucket or not args.endpoint:
        raise SystemExit("--bucket/$CW_BUCKET and --endpoint/$CW_ENDPOINT are required")

    local_dir = Path(args.local_dir)
    if not local_dir.is_dir():
        raise SystemExit(f"{local_dir} is not a directory")
    prefix = args.prefix if args.prefix is not None else args.local_dir.strip("./")

    client = _build_client(args.endpoint)
    uploaded, skipped = sync_directory(client, local_dir, args.bucket, prefix)
    print(
        f"done: {uploaded} uploaded, {skipped} already present "
        f"at s3://{args.bucket}/{prefix}",
        flush=True,
    )
    if args.project and args.experiment:
        try:
            recovered = log_missing_references(
                local_root=local_dir,
                bucket=args.bucket,
                prefix=prefix,
                endpoint=args.endpoint,
                access_key=os.environ["CW_ACCESS_KEY"],
                secret_key=os.environ["CW_SECRET_KEY"],
                project_name=args.project,
                experiment_name=args.experiment,
                run_id=args.wandb_run_id,
                entity=args.wandb_entity,
            )
        except Exception as exc:
            # Byte persistence is the durability boundary. A metadata-only
            # W&B outage must not make the completed trainer process fail and
            # trigger a full cwsandbox Job retry.
            print(
                "warning: checkpoint bytes are persisted at "
                f"s3://{args.bucket}/{prefix}, but W&B reference recovery "
                f"failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return 0
        if recovered:
            print(
                "recovered W&B checkpoint reference(s) for step(s): "
                + ", ".join(str(step) for step in recovered),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
