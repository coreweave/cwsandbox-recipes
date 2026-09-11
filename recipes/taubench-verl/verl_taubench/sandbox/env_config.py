"""Production scale controls for the sandboxed τ-bench environment plane.

Environment variables override YAML config at tool construction time. Values are
parsed and validated as positive integers so pool sizing and HTTP transport limits
never arrive as untyped strings from OmegaConf interpolation.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

TAUBENCH_POOL_SIZE_ENV = "TAUBENCH_POOL_SIZE"
TAUBENCH_MAX_CONNECTIONS_ENV = "TAUBENCH_MAX_CONNECTIONS"
TAUBENCH_ENV_IMAGE_ENV = "TAUBENCH_ENV_IMAGE"
CWSANDBOX_API_KEY_ENV = "CWSANDBOX_API_KEY"
CWSANDBOX_RUNNER_IDS_ENV = "CWSANDBOX_RUNNER_IDS"
CWSANDBOX_PLACEMENT_MODE_ENV = "CWSANDBOX_PLACEMENT_MODE"
CWSANDBOX_PLACEMENT_SPILLOVER_ENV = "CWSANDBOX_PLACEMENT_SPILLOVER"
CWSANDBOX_SERVERLESS_AUTH_ENV = "CWSANDBOX_SERVERLESS_AUTH"

DEFAULT_MAX_CONNECTIONS_FLOOR = 512
# Matches the pool size in config/tool_config/taubench_sandbox_tool_config.yaml,
# which is where sizing is documented and normally set.
DEFAULT_POOL_SIZE = 64
# Serverless is the recipe default everywhere (tool YAML included): endpoint
# creates on CKS need Gateway API routes and fail hard without them.
DEFAULT_PLACEMENT_MODE = "serverless"
DEFAULT_CPU_PLACEMENT_SPILLOVER = "cks_then_serverless"


def resolve_serverless_auth(
    config_auth: str | None = None, *, environment: Mapping[str, str] = os.environ
) -> str:
    """Select credentials independently of serverless placement."""
    mode = (environment.get(CWSANDBOX_SERVERLESS_AUTH_ENV) or config_auth or "wandb").strip().lower()
    if mode not in {"wandb", "coreweave"}:
        raise ValueError(f"{CWSANDBOX_SERVERLESS_AUTH_ENV} must be 'wandb' or 'coreweave'")
    return mode


def _parse_positive_int(name: str, value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer, got {parsed}")
    return parsed


def resolve_pool_size(config_size: int | None = None, *, default: int = DEFAULT_POOL_SIZE) -> int:
    """Return the warm-pool size, preferring ``TAUBENCH_POOL_SIZE`` when set."""
    env = os.environ.get(TAUBENCH_POOL_SIZE_ENV)
    if env is not None and env != "":
        return _parse_positive_int(TAUBENCH_POOL_SIZE_ENV, env)
    if config_size is not None:
        return int(config_size)
    return default


def resolve_max_connections(
    pool_size: int,
    config_max: int | None = None,
) -> int:
    """Return the HTTP transport connection ceiling.

    Always clamps to at least ``max(pool_size, 512)`` so the warm pool cannot
    exhaust the shared ``httpx`` client. ``TAUBENCH_MAX_CONNECTIONS`` overrides
    the configured preference when set, but cannot lower that safety floor.
    """
    env = os.environ.get(TAUBENCH_MAX_CONNECTIONS_ENV)
    if env is not None and env != "":
        requested = _parse_positive_int(TAUBENCH_MAX_CONNECTIONS_ENV, env)
        return max(requested, pool_size, DEFAULT_MAX_CONNECTIONS_FLOOR)
    if config_max is not None:
        return max(int(config_max), pool_size, DEFAULT_MAX_CONNECTIONS_FLOOR)
    return max(pool_size, DEFAULT_MAX_CONNECTIONS_FLOOR)


def resolve_backend_image(
    config_image: str = "python:3.11",
) -> tuple[str, bool]:
    """Return ``(container_image, install_tau_bench)``.

    When ``TAUBENCH_ENV_IMAGE`` is set the image is assumed to ship tau-bench, so
    the per-sandbox ``pip install`` step is skipped.
    """
    env = os.environ.get(TAUBENCH_ENV_IMAGE_ENV)
    if env is not None and env != "":
        return env, False
    return config_image, True


def resolve_runner_ids(config_ids: list[str] | None = None) -> list[str] | None:
    """Return CKS runner pins, preferring ``CWSANDBOX_RUNNER_IDS`` when set."""
    env = os.environ.get(CWSANDBOX_RUNNER_IDS_ENV)
    if env is not None and env.strip() != "":
        ids = [part.strip() for part in env.split(",") if part.strip()]
        return ids or None
    if config_ids:
        ids = [str(part).strip() for part in config_ids if str(part).strip()]
        return ids or None
    return None


def resolve_placement_mode(config_mode: str | None = None) -> str:
    """Return ``cks`` or ``serverless``. Defaults to CKS after enabling a runner."""
    env = os.environ.get(CWSANDBOX_PLACEMENT_MODE_ENV)
    mode = (env if env is not None and env != "" else config_mode) or DEFAULT_PLACEMENT_MODE
    mode = str(mode).strip().lower()
    if mode not in {"cks", "serverless"}:
        raise ValueError(f"{CWSANDBOX_PLACEMENT_MODE_ENV} must be 'cks' or 'serverless', got {mode!r}")
    return mode


def resolve_placement_spillover(
    config_spillover: str | None = None,
    placement_mode: str | None = None,
) -> str | None:
    """Return CPU-pool spillover, or ``None`` when spillover does not apply.

    The ``cks_then_serverless`` default only makes sense when placement starts
    on CKS. The SDK rejects that combination under ``placement_mode=serverless``
    ("requires placement_mode=cks"), so serverless placement resolves to no
    spillover unless a compatible value is set explicitly.
    """
    env = os.environ.get(CWSANDBOX_PLACEMENT_SPILLOVER_ENV)
    spillover = env if env is not None and env != "" else config_spillover
    if spillover is None or str(spillover).strip() == "":
        if placement_mode == "serverless":
            return None
        spillover = DEFAULT_CPU_PLACEMENT_SPILLOVER
    spillover = str(spillover).strip().lower()
    if spillover not in {"strict", "cks_then_serverless", "serverless_then_cks"}:
        raise ValueError(
            f"{CWSANDBOX_PLACEMENT_SPILLOVER_ENV} must be 'strict', "
            f"'cks_then_serverless', or 'serverless_then_cks', got {spillover!r}"
        )
    if placement_mode == "serverless" and spillover == "cks_then_serverless":
        # Stale default (e.g. copied from .env.example); incompatible with
        # serverless placement, so drop it rather than fail every create.
        return None
    return spillover
