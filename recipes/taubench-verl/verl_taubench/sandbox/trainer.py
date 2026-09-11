"""GPU training sandbox launcher for veRL GRPO jobs.

Unlike :class:`~verl_taubench.sandbox.pool.CwSandboxBackend`, which provisions
long-lived CPU env sandboxes on CKS (with ``cks_then_serverless`` spillover),
this module uses ``cwsandbox`` with ``placement_mode="cks"`` to fire
``scripts/train_grpo.sh`` as the sandbox main process on a CUDA image.
Python auth is ``CWSANDBOX_API_KEY``.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shlex
import time as _time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Tuple

from verl_taubench.events import log_event, short_id
from verl_taubench.sandbox.env_config import (
    CWSANDBOX_API_KEY_ENV,
    CWSANDBOX_PLACEMENT_MODE_ENV,
    CWSANDBOX_PLACEMENT_SPILLOVER_ENV,
    CWSANDBOX_RUNNER_IDS_ENV,
    CWSANDBOX_SERVERLESS_AUTH_ENV,
    resolve_runner_ids,
)
from verl_taubench.sandbox.tags import RECIPE_TAG

logger = logging.getLogger(__name__)

DEFAULT_TRAINER_TAG = "verl-taubench-trainer"
DEFAULT_CPU_ENV_TAG_PREFIX = "verl-taubench-env-"
DEFAULT_CONTAINER_IMAGE = (
    "verlai/verl:vllm024.dev2@"
    "sha256:b867883b0dd011363e69ab2ab344922a28c5bd0409e2a324e3ee70fb27ca7543"
)
DEFAULT_WORKSPACE = "/workspace"
DEFAULT_GPU_COUNT = 8
DEFAULT_GPU_TYPE = "H100"
DEFAULT_CPU = "32"
DEFAULT_MEMORY = "256Gi"
DEFAULT_MAX_LIFETIME_SECONDS = 12 * 3600
DEFAULT_STARTUP_TIMEOUT_SECONDS = 900.0
DEFAULT_POLL_RETRY_BUDGET_SECONDS = 300.0
DEFAULT_POLL_RPC_TIMEOUT_SECONDS = 30.0
DEFAULT_SECRET_STORE = "wandb"
DEFAULT_SECRET_NAMES: Tuple[str, ...] = ("WANDB_API_KEY", "HF_TOKEN")
# The tool config path is a grpo_trainer.yaml default, so nothing needs
# prepending here. CLI --hydra-override values still land in this tuple.
DEFAULT_HYDRA_OVERRIDES: Tuple[str, ...] = ()

FORWARDED_ENV_KEYS: Tuple[str, ...] = (
    "DATA_DIR",
    "MODEL_PATH",
    "EXPERIMENT_NAME",
    "PROJECT_NAME",
    "TAUBENCH_SIMULATOR_URL",
    "TAUBENCH_SIMULATOR_MODEL",
    "TAUBENCH_POOL_SIZE",
    "TAUBENCH_MAX_CONNECTIONS",
    "TAUBENCH_ENV_IMAGE",
    "TAUBENCH_ENV_CPU",
    "TAUBENCH_ENV_MEMORY",
    "TAUBENCH_DOMAIN",
    "TAUBENCH_TASK_SPLIT",
    "TAUBENCH_TEST_END_INDEX",
    "TAUBENCH_PARTIAL_CREDIT",
    "ROLLOUT_TEMPERATURE",
    "TRACE_BACKEND",
    "CHECKPOINT_DIR",
    "WANDB_ENTITY",
    "WANDB_PROJECT",
    # Pre-generated run id: the driver's wandb.init adopts it, and each
    # AgentLoopWorker binds its weave client to it (set_wandb_run_context) so
    # rollout traces land on the training run in the workspace.
    "WANDB_RUN_ID",
    "WANDB_MODE",
    "CW_ENDPOINT",
    "CW_BUCKET",
)

_MOUNT_TOP_LEVEL_FILES = frozenset({"README.md", "pyproject.toml", "uv.lock"})
_MOUNT_DIR_PREFIXES = ("verl_taubench/", "config/", "scripts/")

# Directory names excluded even inside allowlisted trees.
_EXCLUDE_DIR_NAMES = frozenset(
    {
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".git",
        "__marimo__",
        "wandb",
        "checkpoints",
        "outputs",
        "logs",
        "sky_workdir",
        "ray_logs",
        ".ipynb_checkpoints",
    }
)

# File names excluded anywhere in the tree.
_EXCLUDE_FILE_NAMES = frozenset({".env"})


class TrainingJobFailed(Exception):
    """The training sandbox main process exited with a non-zero status."""

    def __init__(self, returncode: int, sandbox_id: Optional[str] = None):
        self.returncode = returncode
        self.sandbox_id = sandbox_id
        message = f"training sandbox exited with returncode {returncode}"
        if sandbox_id:
            message += f" (sandbox_id={sandbox_id})"
        super().__init__(message)


def _new_cpu_env_tag() -> str:
    return f"{DEFAULT_CPU_ENV_TAG_PREFIX}{uuid.uuid4()}"


@dataclass(frozen=True)
class TrainerSandboxConfig:
    """Configuration for a single GPU training sandbox job."""

    project_root: Path
    gpu_count: int = DEFAULT_GPU_COUNT
    gpu_type: str = DEFAULT_GPU_TYPE
    cpu: str = DEFAULT_CPU
    memory: str = DEFAULT_MEMORY
    container_image: str = DEFAULT_CONTAINER_IMAGE
    max_lifetime_seconds: int = DEFAULT_MAX_LIFETIME_SECONDS
    workspace: str = DEFAULT_WORKSPACE
    tags: Tuple[str, ...] = (RECIPE_TAG, DEFAULT_TRAINER_TAG)
    secret_store: str = DEFAULT_SECRET_STORE
    secret_names: Tuple[str, ...] = DEFAULT_SECRET_NAMES
    hydra_overrides: Tuple[str, ...] = DEFAULT_HYDRA_OVERRIDES
    extra_env: dict[str, str] = field(default_factory=dict)


def parse_secret_store(raw: str | None) -> str:
    """Resolve the secret store name: unset keeps the default, ``none``/empty
    disables the store and switches to plain-env injection of the secrets."""
    if raw is None:
        return DEFAULT_SECRET_STORE
    value = raw.strip()
    if value.lower() in ("", "none"):
        return ""
    return value


def parse_secret_names(raw: str | Iterable[str] | None) -> Tuple[str, ...]:
    """Parse comma-separated or iterable secret names: trim, dedupe, preserve order."""
    if raw is None:
        return DEFAULT_SECRET_NAMES
    if isinstance(raw, str):
        parts = raw.split(",")
    else:
        parts = list(raw)
    seen: set[str] = set()
    result: list[str] = []
    for part in parts:
        name = part.strip()
        if not name:
            raise ValueError("secret name must be non-empty")
        if name not in seen:
            seen.add(name)
            result.append(name)
    if not result:
        raise ValueError("at least one secret name required")
    return tuple(result)


def _is_excluded_within_allowlist(rel_posix: str) -> bool:
    parts = rel_posix.split("/")
    if any(part in _EXCLUDE_DIR_NAMES for part in parts):
        return True
    if parts[-1] in _EXCLUDE_FILE_NAMES:
        return True
    return parts[-1].endswith((".pyc", ".pyo", ".wandb"))


def _is_allowed_mount(rel_posix: str) -> bool:
    if rel_posix in _MOUNT_TOP_LEVEL_FILES:
        return True
    return any(rel_posix.startswith(prefix) for prefix in _MOUNT_DIR_PREFIXES)


def _iter_project_files(project_root: Path) -> Iterable[Tuple[Path, str]]:
    """Yield ``(absolute_path, workspace_relative_posix)`` for mountable files."""
    root = project_root.resolve()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        yield path, rel


def _validated_mount_source(abs_path: Path, rel_posix: str, project_root: Path) -> Path:
    if abs_path.is_symlink():
        raise ValueError(f"symlink mount file is not allowed: {rel_posix}")
    resolved = abs_path.resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(
            f"resolved mount path escapes project root: {rel_posix} -> {resolved}"
        ) from exc
    return resolved


def _read_text_mount_content(abs_path: Path, rel_posix: str) -> str:
    try:
        return abs_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"allowlisted mount file is not valid UTF-8 text: {rel_posix}"
        ) from exc


def _file_mount_entry(abs_path: Path, mount_path: str, rel_posix: str) -> dict[str, str]:
    return {"mount_path": mount_path, "file_content": _read_text_mount_content(abs_path, rel_posix)}


def collect_mounted_files(
    project_root: Path,
    *,
    workspace: str = DEFAULT_WORKSPACE,
) -> List[dict[str, str]]:
    """Build text-only ``mounted_files`` entries for ``Sandbox.run``.

    Mounts only the recipe allowlist (``README.md``, ``pyproject.toml``,
    ``uv.lock``, ``verl_taubench/``, ``config/``, ``scripts/``) while excluding
    virtual environments, caches, secrets, generated ``data/``, and other
    artifacts.
    """
    mounts: List[dict[str, str]] = []
    root = project_root.resolve()
    workspace_root = workspace.rstrip("/")
    for abs_path, rel_posix in _iter_project_files(root):
        if not _is_allowed_mount(rel_posix):
            continue
        if _is_excluded_within_allowlist(rel_posix):
            continue
        source = _validated_mount_source(abs_path, rel_posix, root)
        mounts.append(_file_mount_entry(source, f"{workspace_root}/{rel_posix}", rel_posix))
    return mounts


def _build_secrets(config: TrainerSandboxConfig, Secret: type) -> List[object]:
    return [Secret(store=config.secret_store, name=name) for name in config.secret_names]


def _plain_env_secrets(config: TrainerSandboxConfig) -> dict[str, str]:
    """Read secret values from the host environment for plain-env injection.

    Used when ``secret_store`` is empty (``TRAINER_SECRET_STORE=none``): orgs
    without a configured cwsandbox secret store (the platform only ships a
    W&B-provider store) cannot use server-resolved secrets, so the values ride
    the create request as ``environment_variables`` instead. Fails loudly on a
    missing value — a trainer that starts without WANDB_API_KEY/HF_TOKEN dies
    minutes in, after the GPUs are already allocated.
    """
    values: dict[str, str] = {}
    missing: list[str] = []
    for name in config.secret_names:
        value = os.environ.get(name)
        if value:
            values[name] = value
        else:
            missing.append(name)
    if missing:
        raise ValueError(
            "TRAINER_SECRET_STORE=none injects secrets from the host environment, "
            f"but these are unset: {', '.join(missing)}. Export them or trim "
            "TRAINER_SECRET_NAMES."
        )
    return values


def _build_environment(config: TrainerSandboxConfig, env_tag: str) -> dict[str, str]:
    workspace = config.workspace.rstrip("/")
    extra = dict(config.extra_env)
    existing_pythonpath = extra.pop("PYTHONPATH", None)
    extra.pop("TAUBENCH_ENV_TAG", None)
    pythonpath_parts = [workspace]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)

    env = {
        "N_GPUS": str(config.gpu_count),
        "PROJECT_ROOT": workspace,
        "PYTHONPATH": os.pathsep.join(pythonpath_parts),
        "TAUBENCH_ENV_TAG": env_tag,
    }
    env.update(extra)
    auth = os.environ.get(CWSANDBOX_API_KEY_ENV)
    if auth:
        env[CWSANDBOX_API_KEY_ENV] = auth
    # Forward placement and credential selection for nested CPU sandboxes.
    for name in (
        CWSANDBOX_RUNNER_IDS_ENV,
        CWSANDBOX_PLACEMENT_MODE_ENV,
        CWSANDBOX_PLACEMENT_SPILLOVER_ENV,
        CWSANDBOX_SERVERLESS_AUTH_ENV,
    ):
        value = os.environ.get(name)
        if value:
            env[name] = value
    return env


def _stop_best_effort(sandbox: Any) -> None:
    with contextlib.suppress(BaseException):
        sandbox.stop().result()


class GpuTrainerLauncher:
    """Launch ``train_grpo.sh`` on a GPU CoreWeave sandbox and wait for completion."""

    def __init__(self, config: TrainerSandboxConfig):
        self.config = config

    def _sdk(self):
        try:
            from cwsandbox import NetworkOptions, ResourceOptions, Sandbox, Secret
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ImportError(
                "GpuTrainerLauncher requires cwsandbox. Install the sandbox extra: "
                'uv pip install -e ".[sandbox]"'
            ) from exc
        return Sandbox, NetworkOptions, ResourceOptions, Secret

    def _train_command(self) -> Tuple[str, ...]:
        workspace = self.config.workspace.rstrip("/")
        script = f"{workspace}/scripts/train_grpo.sh"
        override_args = " ".join(shlex.quote(o) for o in self.config.hydra_overrides)
        if override_args:
            shell = (
                f"cd {shlex.quote(workspace)} && exec bash {shlex.quote(script)} {override_args}"
            )
        else:
            shell = f"cd {shlex.quote(workspace)} && exec bash {shlex.quote(script)}"
        return ("bash", "-lc", shell)

    def launch_and_wait(self, *, log_sink: Optional[Callable[[str], None]] = None) -> int:
        """Provision the sandbox, stream main-process logs, and return the exit code.

        Raises :class:`TrainingJobFailed` when the main process exits non-zero.
        Propagates SDK errors instead of reporting success. On lifecycle failures
        after ``Sandbox.run``, stops the sandbox best-effort without masking the
        original error.
        """
        sink = log_sink or (lambda line: print(line, end=""))
        Sandbox, NetworkOptions, ResourceOptions, Secret = self._sdk()
        config = self.config
        project_root = config.project_root.resolve()
        if not project_root.is_dir():
            raise FileNotFoundError(f"project root does not exist: {project_root}")
        runner_ids = resolve_runner_ids()
        if not runner_ids:
            raise ValueError(
                "Set CWSANDBOX_RUNNER_IDS to the intended CKS runner name; "
                "refusing automatic GPU runner selection"
            )

        mounted_files = collect_mounted_files(
            project_root,
            workspace=config.workspace,
        )
        log_event(f"mounting {len(mounted_files)} project files into {config.workspace}")
        resources = ResourceOptions(
            requests={"cpu": config.cpu, "memory": config.memory},
            limits={"cpu": config.cpu, "memory": config.memory},
            gpu={"count": int(config.gpu_count), "type": config.gpu_type},
        )
        network = NetworkOptions()
        env_tag = _new_cpu_env_tag()
        environment_variables = _build_environment(config, env_tag)
        command = self._train_command()
        run_kwargs: dict[str, Any] = {
            "container_image": config.container_image,
            "resources": resources,
            "network": network,
            "mounted_files": mounted_files,
            "environment_variables": environment_variables,
            "max_lifetime_seconds": config.max_lifetime_seconds,
            # A status RPC timeout is not a failed training process. Let the
            # SDK retry transient polls on this same sandbox before cleanup.
            # These controls are separate from the overall startup deadline.
            "poll_retry_budget_seconds": DEFAULT_POLL_RETRY_BUDGET_SECONDS,
            "poll_rpc_timeout_seconds": DEFAULT_POLL_RPC_TIMEOUT_SECONDS,
            "tags": list(config.tags),
            # CPU-pool placement settings must not redirect the GPU trainer.
            "placement_mode": "cks",
            "placement_spillover": "strict",
            "runner_ids": runner_ids,
        }
        if config.secret_store:
            run_kwargs["secrets"] = _build_secrets(config, Secret)
        else:
            # No secret store on this org: inject values as plain env vars.
            environment_variables.update(_plain_env_secrets(config))
        log_event(
            f"creating GPU sandbox: {config.gpu_count}x {config.gpu_type}, "
            f"cpu={config.cpu}, mem={config.memory}, "
            f"placement={run_kwargs['placement_mode']}, "
            f"runners={','.join(runner_ids)}, "
            f"image={str(config.container_image).split('@')[0]}"
        )
        launch_started = _time.monotonic()
        sandbox = Sandbox.run(*command, **run_kwargs)
        log_event(
            f"GPU sandbox {sandbox.sandbox_id} created; waiting for running state "
            f"(startup timeout={DEFAULT_STARTUP_TIMEOUT_SECONDS:g}s, "
            f"poll retry budget={DEFAULT_POLL_RETRY_BUDGET_SECONDS:g}s)"
        )

        try:
            sandbox.wait(timeout=DEFAULT_STARTUP_TIMEOUT_SECONDS)
            log_event(
                f"GPU sandbox {short_id(sandbox.sandbox_id)} running on runner "
                f"{getattr(sandbox, 'runner_id', None) or 'serverless'} "
                f"({_time.monotonic() - launch_started:.1f}s); streaming trainer logs"
            )
            # The gRPC log stream can end mid-run (long silent phases, transient
            # drops) while the job is still training. A stream ending is NOT
            # evidence of completion: reattach until the sandbox itself leaves
            # the running state, bounded by its lifetime cap. Treating stream
            # EOF as completion once made wait_until_complete time out and the
            # error path kill a healthy run.
            deadline = _time.monotonic() + float(config.max_lifetime_seconds or 12 * 3600) + 600.0
            since = None
            missed_tail = False
            while True:
                try:
                    for line in sandbox.stream_logs(follow=True, since_time=since):
                        since = datetime.now(timezone.utc)
                        sink(line)
                except Exception:
                    # Fast jobs may reach a terminal state before the SDK
                    # attaches its follow stream; handled like a stream end,
                    # but the lines it would have carried must be re-fetched.
                    missed_tail = True
                if not _sandbox_alive(sandbox):
                    break
                if _time.monotonic() > deadline:
                    raise RuntimeError(
                        "sandbox exceeded its lifetime cap without reaching a terminal state"
                    )
                log_event(
                    f"log stream ended while sandbox {short_id(sandbox.sandbox_id)} "
                    "is still running; reattaching"
                )
                _time.sleep(5.0)
            sandbox.wait_until_complete(timeout=600.0).result()
            if missed_tail:
                # Fetch the lines the broken stream never delivered.
                try:
                    for line in sandbox.stream_logs(follow=False, since_time=since):
                        sink(line)
                except Exception as exc:
                    logger.warning("could not retrieve historical trainer logs: %s", exc)

            returncode = sandbox.returncode
            if returncode is None:
                # cwsandbox documents returncode as optional even after a
                # COMPLETED sandbox (for example when an older gateway did not
                # record it). wait_until_complete() above already raises for a
                # failed or terminated sandbox, so lack of this optional field
                # must not turn a successful run into a launcher failure.
                logger.warning(
                    "training sandbox completed without a recorded returncode; "
                    "treating the completed terminal state as success"
                )
                returncode = 0
            if returncode != 0:
                sandbox_id = getattr(sandbox, "sandbox_id", None)
                raise TrainingJobFailed(returncode, sandbox_id=sandbox_id)
            return returncode
        except KeyboardInterrupt:
            log_event("interrupted; stopping the training sandbox and its environment pool")
            _stop_best_effort(sandbox)
            raise
        except BaseException:
            _stop_best_effort(sandbox)
            raise
        finally:
            # The CPU pool is created from INSIDE the trainer sandbox, so
            # stopping that sandbox (crash, Ctrl+C, SDK error) orphans every
            # pool sandbox: the in-sandbox exit trap never runs. Reap from the
            # host by the run's unique tag, always.
            _reap_env_pool(env_tag)


def _reap_env_pool(env_tag: str) -> None:
    """Stop every CPU env sandbox carrying this run's tag. Never raises."""
    try:
        from verl_taubench.sandbox.cleanup import cleanup_tag

        stopped = cleanup_tag(env_tag, settle_seconds=10.0)
        if stopped:
            log_event(f"reaped {stopped} environment sandbox(es) tagged {env_tag}")
    except BaseException as exc:  # never mask the original failure
        logger.warning(
            "could not reap environment sandboxes tagged %s: %s; "
            "run: python -m verl_taubench.sandbox.cleanup --tag %s",
            env_tag,
            exc,
            env_tag,
        )


def _sandbox_alive(sandbox: Any) -> bool:
    """True while the sandbox is in a pre-terminal state.

    A transient status failure counts as alive: keep streaming rather than
    letting the error path stop a healthy run; the lifetime deadline bounds us.
    """
    try:
        status = sandbox.get_status()
        status = status.result() if hasattr(status, "result") else status
        return any(
            state in str(status).lower() for state in ("running", "creating", "pending", "starting")
        )
    except Exception:
        # Status probe failed (transient, or an SDK without get_status): a
        # terminal sandbox has a returncode, a running one does not.
        try:
            return sandbox.returncode is None
        except Exception:
            return True


def config_from_env(project_root: Optional[Path] = None) -> TrainerSandboxConfig:
    """Build :class:`TrainerSandboxConfig` from CLI-friendly environment variables."""
    root = project_root or Path(os.environ.get("PROJECT_ROOT", ".")).resolve()
    gpu_count = int(os.environ.get("N_GPUS", str(DEFAULT_GPU_COUNT)))
    extra_env: dict[str, str] = {}
    for key in FORWARDED_ENV_KEYS:
        value = os.environ.get(key)
        if value is not None:
            extra_env[key] = value

    trainer_secret_names = os.environ.get("TRAINER_SECRET_NAMES")
    secret_names = (
        parse_secret_names(trainer_secret_names)
        if trainer_secret_names is not None
        else DEFAULT_SECRET_NAMES
    )
    secret_store = parse_secret_store(os.environ.get("TRAINER_SECRET_STORE"))

    return TrainerSandboxConfig(
        project_root=root,
        gpu_count=gpu_count,
        gpu_type=os.environ.get("GPU_TYPE", DEFAULT_GPU_TYPE),
        cpu=os.environ.get("TRAINER_CPU", DEFAULT_CPU),
        memory=os.environ.get("TRAINER_MEMORY", DEFAULT_MEMORY),
        container_image=os.environ.get("TRAINER_IMAGE", DEFAULT_CONTAINER_IMAGE),
        max_lifetime_seconds=int(
            os.environ.get("MAX_LIFETIME_SECONDS", str(DEFAULT_MAX_LIFETIME_SECONDS))
        ),
        secret_store=secret_store,
        secret_names=secret_names,
        extra_env=extra_env,
    )
