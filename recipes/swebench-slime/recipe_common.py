"""Shared helpers: slime checkout on sys.path, SWE-bench rows, GPU type selection."""

from __future__ import annotations

import os
import sys
import types
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SWEBENCH_DATASET = "princeton-nlp/SWE-bench_Verified"
SWEBENCH_FIELDS = (
    "instance_id",
    "repo",
    "version",
    "base_commit",
    "problem_statement",
    "hints_text",
    "test_patch",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "environment_setup_commit",
)


def use_slime_checkout() -> Path:
    """Put $SLIME_DIR on sys.path so slime and its examples package import.

    slime's coding-agent example imports transformers at module load. Nothing in
    this recipe loads a tokenizer, so a stub stands in for it, the same way
    slime's own CPU tests do.
    """
    slime_dir = Path(os.environ.get("SLIME_DIR", "")).expanduser()
    if not (slime_dir / "slime" / "agent" / "sandbox.py").is_file():
        raise SystemExit("Set SLIME_DIR to a slime checkout (see README, Setup).")
    if str(slime_dir) not in sys.path:
        sys.path.insert(0, str(slime_dir))
    if "transformers" not in sys.modules:
        stub = types.ModuleType("transformers")
        for name in ("AutoProcessor", "AutoTokenizer", "PreTrainedTokenizerBase", "ProcessorMixin"):
            setattr(stub, name, type(name, (), {}))
        sys.modules["transformers"] = stub
    return slime_dir


def swebench_image(instance_id: str) -> str:
    """Docker Hub image the SWE-bench harness publishes for an instance."""
    return "swebench/sweb.eval.x86_64." + instance_id.replace("__", "_1776_") + ":latest"


def remote_env_info(row: dict[str, Any], workdir: str = "/testbed") -> dict[str, Any]:
    """Map a SWE-bench Verified row onto slime's ``metadata.remote_env_info`` shape."""
    info = {key: row[key] for key in SWEBENCH_FIELDS}
    info.update(image=swebench_image(row["instance_id"]), workdir=workdir)
    return info


def load_swebench_row(instance_id: str) -> dict[str, Any]:
    from datasets import load_dataset

    rows = load_dataset(SWEBENCH_DATASET, split="test").filter(lambda r: r["instance_id"] == instance_id)
    if len(rows) != 1:
        raise SystemExit(f"{instance_id} is not in {SWEBENCH_DATASET}")
    return rows[0]


def pick_gpu_type(runners: Iterable[Any], preferred: str | None = None) -> str:
    """Return ``preferred`` if a runner advertises it, else the first advertised GPU type.

    GPU types are exact runner labels; an unadvertised value never places.
    """
    advertised: list[str] = []
    for runner in runners:
        for gpu_type in getattr(runner, "supported_gpu_types", None) or ():
            if gpu_type not in advertised:
                advertised.append(gpu_type)
    if preferred:
        if preferred not in advertised:
            raise SystemExit(f"GPU_TYPE={preferred!r} is not advertised; available: {advertised or 'none'}")
        return preferred
    if not advertised:
        raise SystemExit("No runner visible to this API key advertises GPUs.")
    return advertised[0]
