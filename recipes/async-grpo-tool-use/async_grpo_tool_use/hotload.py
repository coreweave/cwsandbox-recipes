"""Snapshot files for checkpoint hot-load. ``cwtrain.HotLoader`` does the upload and the hot-load.

A full snapshot is an HF checkpoint: safetensors shards, ``model.safetensors.index.json`` and
``config.json``. Its tensor names, dtypes and shapes must match the deployment's base model.

Docs: https://docs.coreweave.com/products/inference/rl-rollouts
"""

from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path


def _read_header(shard: Path) -> dict[str, dict]:
    with open(shard, "rb") as fh:
        (size,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(size))
    header.pop("__metadata__", None)
    return header


def _headers(checkpoint: Path) -> dict[str, dict]:
    """Tensor name -> safetensors header entry for every tensor in the checkpoint: the tensors its
    index lists, or those of its single ``model.safetensors`` (small models have no index)."""
    index_file = checkpoint / "model.safetensors.index.json"
    if not index_file.exists():
        return _read_header(checkpoint / "model.safetensors")
    weight_map = json.loads(index_file.read_text())["weight_map"]
    headers: dict[str, dict] = {}
    for shard in sorted(set(weight_map.values())):
        for name, info in _read_header(checkpoint / shard).items():
            if name in weight_map:
                headers[name] = info
    missing = set(weight_map) - set(headers)
    if missing:
        raise ValueError(f"{len(missing)} indexed tensors are not in any shard header")
    return headers


def checkpoint_dtypes(checkpoint: Path) -> dict[str, str]:
    """Tensor name -> safetensors dtype for every tensor in the checkpoint."""
    return {name: info["dtype"] for name, info in _headers(checkpoint).items()}


def checkpoint_sizes(checkpoint: Path) -> dict[str, int]:
    """Tensor name -> bytes, the size each tensor has in an export with the base dtypes."""
    return {name: info["data_offsets"][1] - info["data_offsets"][0] for name, info in _headers(checkpoint).items()}


def add_model_files(base: Path, export: Path) -> None:
    """A full snapshot needs config.json next to the weights; copy the base model's other files."""
    for path in base.iterdir():
        is_weights = path.suffix == ".safetensors" or path.name == "model.safetensors.index.json"
        if path.is_file() and not is_weights and not path.name.startswith("."):
            shutil.copy2(path, export / path.name)
