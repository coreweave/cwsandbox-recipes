"""Tests for preflight backend construction without provisioning or network."""

from __future__ import annotations

import argparse
from typing import Any

import pytest

from verl_taubench.sandbox.env_config import TAUBENCH_ENV_IMAGE_ENV
from verl_taubench.sandbox.preflight import build_preflight_backend


def _args(**overrides: Any) -> argparse.Namespace:
    defaults = {
        "domain": "retail",
        "task_split": "train",
        "task_index": 0,
        "container_image": "python:3.11",
        "env_port": 8080,
        "cpu": "1",
        "memory": "2Gi",
        "max_lifetime_seconds": 3600,
        "timeout": 120.0,
        "probe_tool": "list_all_product_types",
        "placement_mode": None,
        "placement_spillover": None,
        "cwsandbox_auth": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.fixture
def capture_backend(monkeypatch):
    captured: list[dict[str, Any]] = []

    class RecordingBackend:
        def __init__(self, **kwargs: Any):
            self.kwargs = kwargs
            captured.append(kwargs)

        async def provision(self):
            raise NotImplementedError

        async def destroy(self, handle) -> None:
            return None

    monkeypatch.setattr(
        "verl_taubench.sandbox.preflight.CwSandboxBackend",
        RecordingBackend,
    )
    return captured


def test_preflight_backend_uses_cli_image_and_installs_without_env(
    capture_backend, monkeypatch
) -> None:
    monkeypatch.delenv(TAUBENCH_ENV_IMAGE_ENV, raising=False)
    build_preflight_backend(_args(container_image="python:3.11"))
    assert capture_backend[-1]["container_image"] == "python:3.11"
    assert capture_backend[-1]["install_tau_bench"] is True


def test_preflight_backend_env_image_overrides_cli_and_disables_install(
    capture_backend, monkeypatch
) -> None:
    monkeypatch.setenv(TAUBENCH_ENV_IMAGE_ENV, "registry.example/taubench:baked")
    build_preflight_backend(_args(container_image="python:3.11"))
    assert capture_backend[-1]["container_image"] == "registry.example/taubench:baked"
    assert capture_backend[-1]["install_tau_bench"] is False


def test_preflight_backend_custom_cli_image_installs_when_env_unset(
    capture_backend, monkeypatch
) -> None:
    monkeypatch.delenv(TAUBENCH_ENV_IMAGE_ENV, raising=False)
    build_preflight_backend(_args(container_image="registry.example/custom:tag"))
    assert capture_backend[-1]["container_image"] == "registry.example/custom:tag"
    assert capture_backend[-1]["install_tau_bench"] is True


def test_preflight_backend_defaults_to_serverless(capture_backend, monkeypatch) -> None:
    monkeypatch.delenv(TAUBENCH_ENV_IMAGE_ENV, raising=False)
    build_preflight_backend(_args())
    # Matches the tool YAML default; spillover does not apply from serverless.
    assert capture_backend[-1]["placement_mode"] == "serverless"
    assert capture_backend[-1]["placement_spillover"] is None


def test_preflight_cli_placement_overrides_defaults(capture_backend, monkeypatch) -> None:
    monkeypatch.delenv(TAUBENCH_ENV_IMAGE_ENV, raising=False)
    build_preflight_backend(
        _args(placement_mode="serverless", placement_spillover="strict")
    )
    assert capture_backend[-1]["placement_mode"] == "serverless"
    assert capture_backend[-1]["placement_spillover"] == "strict"


def test_preflight_serverless_mode_drops_cks_spillover_default(
    capture_backend, monkeypatch
) -> None:
    """Regression: CWSANDBOX_PLACEMENT_MODE=serverless plus the implicit
    cks_then_serverless default made the SDK reject every create."""
    monkeypatch.delenv(TAUBENCH_ENV_IMAGE_ENV, raising=False)
    monkeypatch.delenv("CWSANDBOX_PLACEMENT_SPILLOVER", raising=False)
    monkeypatch.setenv("CWSANDBOX_PLACEMENT_MODE", "serverless")
    build_preflight_backend(_args())
    assert capture_backend[-1]["placement_mode"] == "serverless"
    assert capture_backend[-1]["placement_spillover"] is None
