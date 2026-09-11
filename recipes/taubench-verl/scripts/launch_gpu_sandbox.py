#!/usr/bin/env python3
"""Launch veRL GRPO training on a CoreWeave GPU sandbox.

Fires ``scripts/train_grpo.sh`` as the sandbox main process via
:mod:`verl_taubench.sandbox.trainer`. Secrets are injected by name through the
cwsandbox secret store; this CLI never reads or prints secret values.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from collections.abc import Mapping, MutableMapping
from dataclasses import replace
from pathlib import Path

from verl_taubench.sandbox.env_config import CWSANDBOX_SERVERLESS_AUTH_ENV, resolve_serverless_auth
from verl_taubench.sandbox.trainer import (
    DEFAULT_CONTAINER_IMAGE,
    DEFAULT_CPU,
    DEFAULT_GPU_COUNT,
    DEFAULT_GPU_TYPE,
    DEFAULT_MAX_LIFETIME_SECONDS,
    DEFAULT_MEMORY,
    GpuTrainerLauncher,
    TrainerSandboxConfig,
    TrainingJobFailed,
    config_from_env,
    parse_secret_names,
    parse_secret_store,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch veRL GRPO on a GPU CoreWeave sandbox.")
    parser.add_argument(
        "--serverless-auth", choices=("wandb", "coreweave"), default=None,
        help="CPU serverless sandbox credentials (default: CWSANDBOX_SERVERLESS_AUTH or wandb).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Recipe root to mount into the sandbox (default: $PROJECT_ROOT or cwd).",
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=None,
        help=f"Number of GPUs to request (default: $N_GPUS or {DEFAULT_GPU_COUNT}).",
    )
    parser.add_argument(
        "--gpu-type",
        default=None,
        help=f"GPU type to request (default: $GPU_TYPE or {DEFAULT_GPU_TYPE}).",
    )
    parser.add_argument("--cpu", default=None, help=f"CPU request/limit (default: {DEFAULT_CPU}).")
    parser.add_argument(
        "--memory",
        default=None,
        help=f"Memory request/limit (default: {DEFAULT_MEMORY}).",
    )
    parser.add_argument(
        "--image",
        default=None,
        help=f"Container image (default: {DEFAULT_CONTAINER_IMAGE}).",
    )
    parser.add_argument(
        "--max-lifetime-seconds",
        type=int,
        default=None,
        help=f"Sandbox lifetime cap (default: {DEFAULT_MAX_LIFETIME_SECONDS}).",
    )
    parser.add_argument(
        "--hydra-override",
        action="append",
        default=None,
        dest="hydra_override",
        metavar="OVERRIDE",
        help=(
            "Append a Hydra override passed to train_grpo.sh after the default "
            "sandbox tool config override."
        ),
    )
    parser.add_argument(
        "--secret-name",
        action="append",
        default=None,
        dest="secret_names",
        metavar="NAME",
        help=(
            "Secret name to inject via the cwsandbox secret store. "
            "When provided, replaces TRAINER_SECRET_NAMES and defaults."
        ),
    )
    parser.add_argument(
        "--secret-store",
        default=None,
        help=(
            "cwsandbox secret store to resolve secret names from (default: "
            "TRAINER_SECRET_STORE or 'wandb'). Pass 'none' to skip the store "
            "and inject the values from the host environment instead — "
            "required on orgs with no configured secret store."
        ),
    )
    return parser


def _resolve_config(args: argparse.Namespace) -> TrainerSandboxConfig:
    base = config_from_env(args.project_root)
    overrides = {}
    if args.gpu_count is not None:
        overrides["gpu_count"] = args.gpu_count
    if args.gpu_type is not None:
        overrides["gpu_type"] = args.gpu_type
    if args.cpu is not None:
        overrides["cpu"] = args.cpu
    if args.memory is not None:
        overrides["memory"] = args.memory
    if args.image is not None:
        overrides["container_image"] = args.image
    if args.max_lifetime_seconds is not None:
        overrides["max_lifetime_seconds"] = args.max_lifetime_seconds
    if args.hydra_override:
        overrides["hydra_overrides"] = (*base.hydra_overrides, *args.hydra_override)
    if args.secret_names:
        overrides["secret_names"] = parse_secret_names(args.secret_names)
    secret_store = getattr(args, "secret_store", None)
    if secret_store is not None:
        overrides["secret_store"] = parse_secret_store(secret_store)
    if not overrides:
        return base
    return replace(base, **overrides)


def _validate_auth_environment(environment: Mapping[str, str] = os.environ) -> None:
    """Require credentials used by the trainer and host-side pool cleanup."""
    if not environment.get("CWSANDBOX_API_KEY"):
        raise ValueError(
            "cwsandbox auth requires CWSANDBOX_API_KEY "
            "(CoreWeave API access token; see "
            "https://docs.coreweave.com/products/sandboxes/get-started)"
        )
    placement_mode = environment.get("CWSANDBOX_PLACEMENT_MODE") or "serverless"
    serverless_auth = resolve_serverless_auth(environment=environment)
    if placement_mode.strip().lower() == "serverless" and serverless_auth == "wandb" and not environment.get("WANDB_API_KEY"):
        raise ValueError(
            "serverless sandbox auth and host-side cleanup require WANDB_API_KEY"
        )


def _ensure_wandb_run_id(environment: MutableMapping[str, str] = os.environ) -> str:
    """Fix the wandb run id up front so weave traces can bind to the run.

    The id is forwarded into the GPU sandbox (FORWARDED_ENV_KEYS): the driver's
    wandb.init adopts it and every AgentLoopWorker passes it to
    weave client.set_wandb_run_context, linking rollout traces to the run.
    """
    run_id = environment.get("WANDB_RUN_ID")
    if not run_id:
        try:
            from wandb.sdk.lib.runid import generate_id

            run_id = generate_id()
        except Exception:
            run_id = uuid.uuid4().hex[:8]
        environment["WANDB_RUN_ID"] = run_id
    entity = environment.get("WANDB_ENTITY", "")
    project = environment.get("PROJECT_NAME") or environment.get("WANDB_PROJECT", "")
    if entity and project:
        print(f"wandb run id {run_id}: https://wandb.ai/{entity}/{project}/runs/{run_id}")
    else:
        print(f"wandb run id {run_id}")
    return run_id


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.serverless_auth is not None:
        os.environ[CWSANDBOX_SERVERLESS_AUTH_ENV] = args.serverless_auth
    try:
        _validate_auth_environment()
    except ValueError as exc:
        print(f"authentication error: {exc}", file=sys.stderr)
        return 2
    _ensure_wandb_run_id()
    config = _resolve_config(args)
    launcher = GpuTrainerLauncher(config)
    try:
        launcher.launch_and_wait()
    except TrainingJobFailed as exc:
        print(str(exc), file=sys.stderr)
        return exc.returncode if exc.returncode else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
