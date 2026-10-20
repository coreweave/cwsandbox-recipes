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


def _headers(checkpoint: Path) -> dict[str, dict]:
    """Tensor name -> safetensors header entry for every tensor the checkpoint's index lists."""
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    headers: dict[str, dict] = {}
    for shard in sorted(set(index["weight_map"].values())):
        with open(checkpoint / shard, "rb") as fh:
            (size,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(size))
        for name, info in header.items():
            if name != "__metadata__" and name in index["weight_map"]:
                headers[name] = info
    missing = set(index["weight_map"]) - set(headers)
    if missing:
        raise ValueError(f"{len(missing)} indexed tensors are not in any shard header")
    return headers


def checkpoint_dtypes(checkpoint: Path) -> dict[str, str]:
    """Tensor name -> safetensors dtype for every tensor the checkpoint's index lists."""
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
