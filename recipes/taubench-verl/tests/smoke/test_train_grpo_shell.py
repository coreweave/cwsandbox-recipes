"""Shell-behavior tests for scripts/train_grpo.sh.

Executes the real launch script inside a temporary project fixture with fake
``ray`` and ``python`` executables on PATH. No real Ray cluster, network, or GPU
workloads are required. Run-ID generation uses the installed W&B SDK.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

FAKE_PYTHON = """\
#!/usr/bin/env python3
import os
import sys
from pathlib import Path

log_path = os.environ["SHELL_TEST_LOG"]
with open(log_path, "a", encoding="utf-8") as handle:
    handle.write("PYTHON " + " ".join(repr(arg) for arg in sys.argv) + "\\n")

args = sys.argv[1:]
if len(args) >= 3 and args[0] == "-m" and args[1] == "pip":
    sys.exit(0)

if args and args[0] == "-c":
    os.execv(os.environ["SHELL_TEST_PYTHON"], [os.environ["SHELL_TEST_PYTHON"], *args])

if len(args) >= 2 and args[0] == "-m" and args[1] == "verl_taubench.sandbox.cleanup":
    sys.exit(int(os.environ.get("CLEANUP_EXIT_CODE", "0")))

if any("preprocess_taubench" in arg for arg in args):
    save_dir = "./data/processed"
    split = "train"
    index = 0
    while index < len(args):
        if args[index] == "--local-save-dir" and index + 1 < len(args):
            save_dir = args[index + 1]
        if args[index] == "--task-split" and index + 1 < len(args):
            split = args[index + 1]
        index += 1
    target = Path(save_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{split}.parquet").write_bytes(b"PAR1")
    sys.exit(0)

if len(args) >= 2 and args[0] == "-m" and args[1] == "verl.trainer.main_ppo":
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(
            "TRAINER_ENV "
            + os.environ.get("WANDB_RUN_ID", "")
            + " "
            + os.environ.get("VERL_USE_EXTERNAL_MODULES", "")
            + "\\n"
        )
        handle.write("WANDB_MODE " + os.environ.get("WANDB_MODE", "") + "\\n")
    if os.environ.get("CREATE_CHECKPOINT") == "1":
        root = Path("checkpoints") / os.environ.get("PROJECT_NAME", "verl-taubench-coreweave") / os.environ.get(
            "EXPERIMENT_NAME", "qwen2.5-7b_grpo"
        )
        (root / "global_step_1").mkdir(parents=True, exist_ok=True)
        (root / "global_step_1" / "data.pt").write_bytes(b"checkpoint")
    sys.exit(int(os.environ.get("TRAINER_EXIT_CODE", "0")))

if any("upload_checkpoints.py" in arg for arg in args):
    sys.exit(0)

sys.stderr.write("unexpected python invocation: " + " ".join(args) + "\\n")
sys.exit(99)
"""

FAKE_RAY = """\
#!/usr/bin/env python3
import os
import sys

log_path = os.environ["SHELL_TEST_LOG"]
with open(log_path, "a", encoding="utf-8") as handle:
    handle.write("RAY " + " ".join(sys.argv[1:]) + "\\n")
sys.exit(0)"""

# Cleanup must be session-scoped: a pkill on the recipe's Ray temp dir, never
# a global `ray stop` (which would kill SkyPilot's runtime Ray on pods).
FAKE_PKILL = """\
#!/usr/bin/env python3
import os
import sys

log_path = os.environ["SHELL_TEST_LOG"]
with open(log_path, "a", encoding="utf-8") as handle:
    handle.write("PKILL " + " ".join(sys.argv[1:]) + "\\n")
sys.exit(0)
"""


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def shell_fixture(tmp_path: Path):
    project = tmp_path / "recipe"
    scripts_src = REPO_ROOT / "scripts"
    shutil.copytree(scripts_src, project / "scripts")
    shutil.copy2(REPO_ROOT / "pyproject.toml", project / "pyproject.toml")
    trainer_dir = project / "verl_taubench" / "trainer"
    trainer_dir.mkdir(parents=True)
    shutil.copy2(
        REPO_ROOT / "verl_taubench" / "trainer" / "grpo_trainer.yaml",
        trainer_dir / "grpo_trainer.yaml",
    )
    (project / "verl_taubench" / "__init__.py").write_text('"""fixture package"""\n', encoding="utf-8")
    (project / "data" / "processed").mkdir(parents=True)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(bin_dir / "python", FAKE_PYTHON)
    _write_executable(bin_dir / "ray", FAKE_RAY)
    _write_executable(bin_dir / "pkill", FAKE_PKILL)

    log_file = tmp_path / "command.log"
    yield {
        "project": project,
        "bin_dir": bin_dir,
        "log_file": log_file,
    }


def _run_train_grpo(
    shell_fixture,
    *,
    extra_env: dict[str, str] | None = None,
    extra_args: list[str] | None = None,
    unset_env: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PATH"] = f"{shell_fixture['bin_dir']}{os.pathsep}{env.get('PATH', '')}"
    env["SHELL_TEST_LOG"] = str(shell_fixture["log_file"])
    env["SHELL_TEST_PYTHON"] = sys.executable
    env.setdefault("WANDB_API_KEY", "test-wandb-key")
    env.setdefault("HF_TOKEN", "test-hf-token")
    if extra_env:
        env.update(extra_env)
    if unset_env:
        for key in unset_env:
            env.pop(key, None)

    command = ["bash", str(shell_fixture["project"] / "scripts" / "train_grpo.sh")]
    if extra_args:
        command.extend(extra_args)

    return subprocess.run(
        command,
        cwd=shell_fixture["project"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _log_lines(shell_fixture) -> list[str]:
    log_file: Path = shell_fixture["log_file"]
    if not log_file.exists():
        return []
    return log_file.read_text(encoding="utf-8").splitlines()


@pytest.mark.parametrize("mode,expected", [("online", "shared"), ("shared", "shared"), ("offline", "offline"), ("disabled", "disabled")])
def test_sdk_reporting_uses_shared_mode_only_for_online_training(shell_fixture, mode, expected):
    result = _run_train_grpo(shell_fixture, extra_env={"WANDB_MODE": mode})
    assert result.returncode == 0, result.stderr
    assert f"WANDB_MODE {expected}" in _log_lines(shell_fixture)


def _ray_lines(shell_fixture) -> list[str]:
    return [line for line in _log_lines(shell_fixture) if line.startswith("RAY ")]


def _python_lines(shell_fixture) -> list[str]:
    return [line for line in _log_lines(shell_fixture) if line.startswith("PYTHON ")]


def _cleanup_lines(shell_fixture) -> list[str]:
    return [
        line
        for line in _python_lines(shell_fixture)
        if "verl_taubench.sandbox.cleanup" in line
    ]


def test_starts_single_local_ray_head_on_port_6379(shell_fixture) -> None:
    """Production break: SkyPilot worker join starts Ray with --address instead of --head."""
    result = _run_train_grpo(shell_fixture)
    assert result.returncode == 0, result.stderr

    ray_lines = _ray_lines(shell_fixture)
    start_lines = [line for line in ray_lines if line.startswith("RAY start")]
    assert len(start_lines) == 1
    assert "--head" in start_lines[0]
    assert "--port=6379" in start_lines[0]
    assert "--dashboard-port=8265" in start_lines[0]
    assert "--address" not in start_lines[0]
    assert all("sleep infinity" not in line for line in ray_lines)


def test_missing_data_runs_preprocess_then_trainer(shell_fixture) -> None:
    """Production break: mounted sandbox has no data/; trainer would fail without preprocess."""
    result = _run_train_grpo(shell_fixture)
    assert result.returncode == 0, result.stderr

    python_lines = _python_lines(shell_fixture)
    preprocess_lines = [line for line in python_lines if "preprocess_taubench" in line]
    trainer_lines = [line for line in python_lines if "verl.trainer.main_ppo" in line]

    assert len(preprocess_lines) == 2
    assert any("'--task-split' 'train'" in line for line in preprocess_lines)
    assert any("'--task-split' 'test'" in line for line in preprocess_lines)
    assert len(trainer_lines) == 1

    data_dir = shell_fixture["project"] / "data" / "processed"
    assert (data_dir / "train.parquet").is_file()
    assert (data_dir / "test.parquet").is_file()


def test_existing_data_skips_preprocess(shell_fixture) -> None:
    """Production break: regenerating parquet on every launch is slow and non-deterministic."""
    data_dir = shell_fixture["project"] / "data" / "processed"
    (data_dir / "train.parquet").write_bytes(b"EXISTING_TRAIN")
    (data_dir / "test.parquet").write_bytes(b"EXISTING_TEST")

    result = _run_train_grpo(shell_fixture)
    assert result.returncode == 0, result.stderr

    python_lines = _python_lines(shell_fixture)
    assert not any("preprocess_taubench" in line for line in python_lines)
    assert (data_dir / "train.parquet").read_bytes() == b"EXISTING_TRAIN"
    assert (data_dir / "test.parquet").read_bytes() == b"EXISTING_TEST"


def test_skip_project_install_skips_pip(shell_fixture) -> None:
    """Production break: pre-baked images should not reinstall the mounted project."""
    result = _run_train_grpo(shell_fixture, extra_env={"SKIP_PROJECT_INSTALL": "1"})
    assert result.returncode == 0, result.stderr

    python_lines = _python_lines(shell_fixture)
    assert not any("pip" in line and "install" in line for line in python_lines)


def test_default_project_install_runs_pip(shell_fixture) -> None:
    result = _run_train_grpo(shell_fixture)
    assert result.returncode == 0, result.stderr

    python_lines = _python_lines(shell_fixture)
    pip_lines = [line for line in python_lines if "'-m'" in line and "'pip'" in line and "install" in line]
    # One cryptography overwrite (distro package the image cannot uninstall)
    # plus the editable project install with the sandbox extra.
    assert len(pip_lines) == 2
    assert "cryptography" in pip_lines[0] and "--ignore-installed" in pip_lines[0]
    editable = pip_lines[1]
    assert "'-e'" in editable
    assert "sandbox" in editable
    assert "vllm" not in editable


def test_default_project_install_uses_sandbox_extra_not_bare_editable(shell_fixture) -> None:
    """GPU sandbox needs cwsandbox + httpx for the CPU rollout pool."""
    result = _run_train_grpo(shell_fixture)
    assert result.returncode == 0, result.stderr

    pip_line = next(
        line
        for line in _python_lines(shell_fixture)
        if "'-m'" in line and "'pip'" in line and "install" in line and "'-e'" in line
    )
    assert "'.[sandbox]'" in pip_line or "'-e' '.[sandbox]'" in pip_line


@pytest.mark.parametrize("n_gpus", ["1", "8"])
def test_n_gpus_flows_into_hydra_overrides(shell_fixture, n_gpus: str) -> None:
    """Production break: hard-coded GPU counts ignore sandbox N_GPUS."""
    result = _run_train_grpo(shell_fixture, extra_env={"N_GPUS": n_gpus})
    assert result.returncode == 0, result.stderr

    trainer_line = next(line for line in _python_lines(shell_fixture) if "verl.trainer.main_ppo" in line)
    assert f"actor_rollout_ref.rollout.data_parallel_size={n_gpus}" in trainer_line
    assert f"trainer.n_gpus_per_node={n_gpus}" in trainer_line


def test_argv_hydra_overrides_are_preserved(shell_fixture) -> None:
    result = _run_train_grpo(
        shell_fixture,
        extra_args=["trainer.total_epochs=1", "actor_rollout_ref.rollout.n=2"],
    )
    assert result.returncode == 0, result.stderr

    trainer_line = next(line for line in _python_lines(shell_fixture) if "verl.trainer.main_ppo" in line)
    assert "trainer.total_epochs=1" in trainer_line
    assert "actor_rollout_ref.rollout.n=2" in trainer_line


def test_trainer_loads_checkpoint_hook_with_stable_wandb_run_id(shell_fixture) -> None:
    result = _run_train_grpo(shell_fixture, unset_env=["WANDB_RUN_ID"])
    assert result.returncode == 0, result.stderr

    env_line = next(
        line for line in _log_lines(shell_fixture) if line.startswith("TRAINER_ENV ")
    )
    assert re.fullmatch(
        r"TRAINER_ENV [a-z0-9]{8} verl_taubench\.trainer\.checkpointing", env_line
    )

    result = _run_train_grpo(shell_fixture, unset_env=["WANDB_RUN_ID"])
    assert result.returncode == 0, result.stderr
    next_env_line = [line for line in _log_lines(shell_fixture) if line.startswith("TRAINER_ENV ")][-1]
    assert next_env_line != env_line


def test_final_checkpoint_command_is_recovery_for_same_run(shell_fixture) -> None:
    result = _run_train_grpo(
        shell_fixture,
        extra_env={
            "CW_ACCESS_KEY": "access",
            "CW_SECRET_KEY": "secret",
            "WANDB_RUN_ID": "existing-run-id",
            "CREATE_CHECKPOINT": "1",
        },
    )
    assert result.returncode == 0, result.stderr

    upload_line = next(
        line for line in _python_lines(shell_fixture) if "upload_checkpoints.py" in line
    )
    assert "'--project' 'verl-taubench-coreweave'" in upload_line
    assert "'--experiment' 'qwen2.5-7b_grpo'" in upload_line
    assert "'--wandb-run-id' 'existing-run-id'" in upload_line


def _pkill_lines(fixture) -> list:
    log = fixture["log_file"].read_text(encoding="utf-8").splitlines()
    return [line for line in log if line.startswith("PKILL ")]


def test_ray_cleanup_is_session_scoped_after_success(shell_fixture) -> None:
    result = _run_train_grpo(shell_fixture)
    assert result.returncode == 0, result.stderr

    # Only the session temp dir is targeted; never a global `ray stop`.
    assert any("ray-taubench" in line for line in _pkill_lines(shell_fixture))
    assert not any("stop" in line for line in _ray_lines(shell_fixture))


def test_ray_cleanup_is_session_scoped_after_trainer_failure(shell_fixture) -> None:
    result = _run_train_grpo(shell_fixture, extra_env={"TRAINER_EXIT_CODE": "17"})
    assert result.returncode == 17, result.stderr

    assert any("ray-taubench" in line for line in _pkill_lines(shell_fixture))
    assert not any("stop" in line for line in _ray_lines(shell_fixture))


@pytest.mark.parametrize("trainer_exit_code", ["0", "17"])
def test_cpu_sandbox_cleanup_runs_with_exact_tag_on_success_and_failure(
    shell_fixture, trainer_exit_code: str
) -> None:
    result = _run_train_grpo(
        shell_fixture,
        extra_env={
            "TAUBENCH_ENV_TAG": "verl-taubench-env-run-123",
            "TRAINER_EXIT_CODE": trainer_exit_code,
        },
    )
    assert result.returncode == int(trainer_exit_code), result.stderr

    cleanup_lines = _cleanup_lines(shell_fixture)
    assert len(cleanup_lines) == 1
    assert "'--tag' 'verl-taubench-env-run-123'" in cleanup_lines[0]


@pytest.mark.parametrize("trainer_exit_code", ["0", "17"])
def test_cleanup_failure_does_not_mask_trainer_exit_status(
    shell_fixture, trainer_exit_code: str
) -> None:
    result = _run_train_grpo(
        shell_fixture,
        extra_env={
            "TAUBENCH_ENV_TAG": "verl-taubench-env-run-123",
            "TRAINER_EXIT_CODE": trainer_exit_code,
            "CLEANUP_EXIT_CODE": "9",
        },
    )
    assert result.returncode == int(trainer_exit_code), result.stderr
    assert len(_cleanup_lines(shell_fixture)) == 1


def test_missing_cpu_env_tag_skips_cleanup_instead_of_broad_reap(shell_fixture) -> None:
    result = _run_train_grpo(
        shell_fixture,
        unset_env=["TAUBENCH_ENV_TAG"],
    )
    assert result.returncode == 0, result.stderr
    assert _cleanup_lines(shell_fixture) == []


@pytest.mark.parametrize("missing_var", ["WANDB_API_KEY", "HF_TOKEN"])
def test_missing_secrets_fail_before_install_or_ray(shell_fixture, missing_var: str) -> None:
    """Production break: missing secrets should abort before pip install or Ray startup."""
    result = _run_train_grpo(shell_fixture, unset_env=[missing_var])
    assert result.returncode != 0

    assert _python_lines(shell_fixture) == []
    assert _ray_lines(shell_fixture) == []


def test_preprocess_uses_taubench_domain_and_test_end_index(shell_fixture) -> None:
    result = _run_train_grpo(
        shell_fixture,
        extra_env={
            "TAUBENCH_DOMAIN": "airline",
            "TAUBENCH_TEST_END_INDEX": "25",
        },
    )
    assert result.returncode == 0, result.stderr

    python_lines = _python_lines(shell_fixture)
    train_line = next(
        line for line in python_lines if "preprocess_taubench" in line and "'--task-split' 'train'" in line
    )
    test_line = next(
        line for line in python_lines if "preprocess_taubench" in line and "'--task-split' 'test'" in line
    )

    assert "'--domain'" in train_line and "'airline'" in train_line
    assert "'--domain'" in test_line and "'airline'" in test_line
    assert "'--end-index'" in test_line and "'25'" in test_line
