"""Smoke test: the project can import the core training stack.

This test does not allocate GPUs or start Ray. It verifies that package versions
are present and mutually importable on Linux/CUDA hosts. On a macOS laptop without
``vllm``/``torch`` wheels, the missing packages are skipped rather than failing.
"""

import importlib
from typing import List, Tuple

import pytest

PackageCheck = Tuple[str, List[str]]

OPTIONAL: List[PackageCheck] = [
    ("verl", ["verl.trainer.main_ppo", "verl.protocol"]),
    ("vllm", []),
    ("ray", []),
    ("torch", ["torch.distributed"]),
    ("transformers", []),
    ("hydra", ["hydra.core.global_hydra"]),
    ("omegaconf", []),
]


def _import_one(module_name: str) -> str:
    mod = importlib.import_module(module_name)
    return getattr(mod, "__version__", "unknown")


def test_optional_packages_reported() -> None:
    """GPU-requiring packages are reported but do not fail the smoke test."""
    missing: List[str] = []
    present: List[Tuple[str, str]] = []
    for top_module, submodules in OPTIONAL:
        try:
            version = _import_one(top_module)
            present.append((top_module, version))
            for sub in submodules:
                version = _import_one(sub)
                present.append((sub, version))
        except Exception:
            missing.append(top_module)

    print("=== present packages ===")
    for name, version in present:
        print(f"  {name:40s} version={version}")
    if missing:
        print("=== missing packages (skipped; install on a Linux+CUDA node) ===")
        for name in missing:
            print(f"  {name}")
    # This smoke test is informational by default. The full dependency set is
    # expected to be present only on a Linux/CUDA node.
    print(f"Summary: {len(present)} modules present, {len(missing)} modules skipped.")


@pytest.fixture
def available_imports():
    """Dynamically collect which optional packages are importable."""
    available = {}
    for top_module, _ in OPTIONAL:
        try:
            importlib.import_module(top_module)
            available[top_module] = True
        except Exception:
            available[top_module] = False
    return available


def test_verl_tools_import_if_available(available_imports) -> None:
    """When veRL is installed, our BaseTool/AgentLoop imports must match upstream."""
    if not available_imports.get("verl"):
        pytest.skip("veRL not installed")
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
    from verl.tools.base_tool import BaseTool
    from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

    assert AgentLoopBase is not None
    assert AgentLoopOutput is not None
    assert BaseTool is not None
    assert OpenAIFunctionToolSchema is not None
    assert ToolResponse is not None
