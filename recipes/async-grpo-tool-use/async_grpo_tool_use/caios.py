"""CoreWeave AI Object Storage (CAIOS) helpers: upload, copy and delete under a prefix.

The client comes from ``cwtrain`` and uses the ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY``
of a CAIOS access key; ``AWS_REGION`` is the bucket's zone. The trainer and the launcher both use
this module.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

DONE_MARKER = "_COMPLETE"  # written last, so a partial upload is never taken as done


def client(region: str | None = None) -> Any:
    from cwtrain.infra.storage import s3_client

    return s3_client(region_name=region)


def keys(s3: Any, bucket: str, prefix: str) -> list[str]:
    found = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        found.extend(item["Key"] for item in page.get("Contents", []))
    return found


def is_complete(s3: Any, bucket: str, prefix: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=f"{prefix.rstrip('/')}/{DONE_MARKER}")
        return True
    except s3.exceptions.ClientError:
        return False


def upload_dir(s3: Any, directory: Path, bucket: str, prefix: str, *, marker: bool = True, workers: int = 8) -> int:
    """Upload every file directly in ``directory`` to ``prefix``. Return the bytes sent."""
    from boto3.s3.transfer import TransferConfig

    config = TransferConfig(multipart_chunksize=64 * 2**20, max_concurrency=8)
    files = sorted(p for p in directory.iterdir() if p.is_file() and not p.name.startswith("."))
    prefix = prefix.rstrip("/")

    def put(path: Path) -> int:
        s3.upload_file(str(path), bucket, f"{prefix}/{path.name}", Config=config)
        return path.stat().st_size

    with ThreadPoolExecutor(max_workers=workers) as pool:
        total = sum(pool.map(put, files))
    if marker:
        s3.put_object(Bucket=bucket, Key=f"{prefix}/{DONE_MARKER}", Body=b"")
    return total


def download_dir(s3: Any, bucket: str, prefix: str, directory: Path, *, workers: int = 8) -> int:
    """Download every object directly under ``prefix`` into ``directory``. Return the bytes received."""
    from boto3.s3.transfer import TransferConfig

    config = TransferConfig(multipart_chunksize=64 * 2**20, max_concurrency=8)
    prefix = prefix.rstrip("/")
    directory.mkdir(parents=True, exist_ok=True)

    def get(key: str) -> int:
        path = directory / key[len(prefix) + 1 :]
        s3.download_file(bucket, key, str(path), Config=config)
        return path.stat().st_size

    found = [key for key in keys(s3, bucket, prefix) if "/" not in key[len(prefix) + 1 :]]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return sum(pool.map(get, found))


def copy_prefix(s3: Any, bucket: str, source: str, target: str, *, workers: int = 8) -> None:
    """Server-side copy of every object under ``source`` to ``target``."""
    source, target = source.rstrip("/"), target.rstrip("/")

    def copy(key: str) -> None:
        s3.copy({"Bucket": bucket, "Key": key}, bucket, target + key[len(source) :])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(copy, keys(s3, bucket, source)))


def delete_prefix(s3: Any, bucket: str, prefix: str) -> int:
    """Delete every object under ``prefix`` and abort its unfinished multipart uploads."""
    found = keys(s3, bucket, prefix)
    failed = []
    for start in range(0, len(found), 1000):
        batch = [{"Key": key} for key in found[start : start + 1000]]
        response = s3.delete_objects(Bucket=bucket, Delete={"Objects": batch, "Quiet": True})
        failed += response.get("Errors", [])
    pages = s3.get_paginator("list_multipart_uploads").paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/")
    for page in pages:
        for upload in page.get("Uploads", []):
            s3.abort_multipart_upload(Bucket=bucket, Key=upload["Key"], UploadId=upload["UploadId"])
    if failed:
        first = failed[0]
        raise RuntimeError(f"{len(failed)} object(s) not deleted, first {first.get('Key')}: {first.get('Code')}")
    return len(found)
