"""Parsed project metadata checks for reproducible training dependencies."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER_CONFIG = REPO_ROOT / "verl_taubench" / "trainer" / "grpo_trainer.yaml"


def _project_metadata() -> dict[str, Any]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _lock_metadata() -> dict[str, Any]:
    with (REPO_ROOT / "uv.lock").open("rb") as handle:
        return tomllib.load(handle)


def test_tau_bench_dependency_is_pinned_to_locked_commit() -> None:
    dependencies = _project_metadata()["project"]["dependencies"]
    assert (
        "tau_bench @ git+https://github.com/sierra-research/tau-bench.git"
        "@59a200c6d575d595120f1cb70fea53cef0632f6b"
    ) in dependencies


def test_vllm_extra_matches_verl_090_supported_range() -> None:
    vllm_dependencies = _project_metadata()["project"]["optional-dependencies"]["vllm"]
    assert "vllm>=0.24.0,<0.25.0" in vllm_dependencies


def test_verl_v1_transfer_queue_dependency_is_pinned() -> None:
    dependencies = _project_metadata()["project"]["dependencies"]
    assert "TransferQueue==0.1.8" in dependencies


def test_python_floor_matches_cwsandbox_requirement() -> None:
    metadata = _project_metadata()
    assert metadata["project"]["requires-python"] == ">=3.11,<3.13"
    assert "Programming Language :: Python :: 3.10" not in metadata["project"]["classifiers"]


def test_sandbox_extra_requires_unified_cwsandbox_client() -> None:
    dependencies = _project_metadata()["project"]["optional-dependencies"]["sandbox"]
    assert "cwsandbox>=1.12.0,<2.0.0" in dependencies


def test_lock_uses_unified_cwsandbox_client() -> None:
    package = next(
        package
        for package in _lock_metadata()["package"]
        if package["name"] == "cwsandbox"
    )
    version = tuple(int(part) for part in package["version"].split("."))
    assert version >= (1, 12, 0)


def test_grpo_trainer_inherits_verl_ppo_trainer() -> None:
    text = TRAINER_CONFIG.read_text(encoding="utf-8")
    assert "pkg://verl.trainer.config" in text


def test_grpo_uses_checkpoint_uploading_trainer() -> None:
    text = TRAINER_CONFIG.read_text(encoding="utf-8")
    assert "trainer_mode: taubench_sync" in text
    assert "- ppo_trainer" in text
    assert "lr_scheduler" not in text
    assert "lr_warmup_steps:" in text
    assert "ppo_micro_batch_size_per_gpu:" in text
    assert "log_prob_micro_batch_size_per_gpu:" in text
    assert "log_prob_micro_batch_size:" not in text
    # Reward flows through AgentLoopOutput.reward_score; the recipe must not
    # override verl's inherited reward config or wire a custom reward function.
    assert "custom_reward_function" not in text
    assert "reward_model:" not in text


def test_multi_turn_qwen_rollouts_enable_continuous_token_boundaries() -> None:
    """User/tool roles must be rendered between generated assistant turns."""
    config = yaml.safe_load(TRAINER_CONFIG.read_text(encoding="utf-8"))

    assert config["data"]["continuous_token"] == {
        "enable": True,
        "model_family": "qwen25",
    }
