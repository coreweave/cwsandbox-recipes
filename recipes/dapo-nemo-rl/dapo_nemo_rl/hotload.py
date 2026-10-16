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


def checkpoint_dtypes(checkpoint: Path) -> dict[str, str]:
    """Tensor name -> safetensors dtype for every tensor the checkpoint's index lists."""
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    dtypes: dict[str, str] = {}
    for shard in sorted(set(index["weight_map"].values())):
        with open(checkpoint / shard, "rb") as fh:
            (size,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(size))
        for name, info in header.items():
            if name != "__metadata__" and name in index["weight_map"]:
                dtypes[name] = info["dtype"]
    missing = set(index["weight_map"]) - set(dtypes)
    if missing:
        raise ValueError(f"{len(missing)} indexed tensors are not in any shard header")
    return dtypes


def add_model_files(base: Path, export: Path) -> None:
    """A full snapshot needs config.json next to the weights; copy the base model's other files."""
    for path in base.iterdir():
        is_weights = path.suffix == ".safetensors" or path.name == "model.safetensors.index.json"
        if path.is_file() and not is_weights and not path.name.startswith("."):
            shutil.copy2(path, export / path.name)
