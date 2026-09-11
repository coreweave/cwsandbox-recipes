"""Behavioral structure checks for the single-node SkyPilot fallback task."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SANDBOX_TASK_PATH = REPO_ROOT / "skypilot" / "verl-taubench-sandbox.yaml"
SIMULATOR_TASK_PATH = REPO_ROOT / "skypilot" / "taubench-simulator.yaml"

# Every credential the tasks accept. SkyPilot only masks values it knows are
# secrets, so any of these declared under `envs:` would be echoed into task logs.
_SECRET_KEYS = frozenset(
    {
        "CWSANDBOX_API_KEY",
        "WANDB_API_KEY",
        "HF_TOKEN",
        "CW_ACCESS_KEY",
        "CW_SECRET_KEY",
        "OPENAI_API_KEY",
    }
)


def _load_task(path: Path = SANDBOX_TASK_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        task = yaml.safe_load(handle)
    assert isinstance(task, dict)
    return task


def test_sandbox_task_ships_workdir_and_declares_secrets() -> None:
    task = _load_task()
    assert task["num_nodes"] == 1
    assert task["workdir"] == "."
    assert _SECRET_KEYS <= set(task["secrets"])


def test_credentials_are_secrets_with_null_placeholders() -> None:
    """A credential under `envs:` would leak into task logs, and a non-null
    placeholder would silently become the value when the caller forgets --secret."""
    for path in (SANDBOX_TASK_PATH, SIMULATOR_TASK_PATH):
        task = _load_task(path)
        assert not (_SECRET_KEYS & set(task.get("envs") or {})), path.name
        for name, value in (task.get("secrets") or {}).items():
            assert name in _SECRET_KEYS, f"{path.name}: {name} is not a credential"
            # null = required at launch; "" = optional with an empty default.
            # Anything else would be a committed credential.
            assert value in (None, ""), f"{path.name}: {name} must be a placeholder"


_FAKE_SKY_PYTHON = """\
#!/usr/bin/env python3
import os
import sys

args = sys.argv[1:]
if args and args[0] == "-c":
    code = args[1]
    if "token_urlsafe" in code:
        print("generated-secret-value")
    elif "uuid" in code:
        print(os.environ["FAKE_TAG_SUFFIX"])
    else:
        sys.exit(91)
    sys.exit(0)

if args[:2] == ["-m", "verl_taubench.sandbox.preflight"]:
    with open(os.environ["SKY_RUN_LOG"], "a", encoding="utf-8") as handle:
        handle.write(
            "preflight "
            f"tag={os.environ.get('TAUBENCH_ENV_TAG', '')} "
            f"token_set={bool(os.environ.get('TAUBENCH_ENV_TOKEN'))}\\n"
        )
    sys.exit(0)

sys.exit(92)
"""

_FAKE_SKY_BASH = """\
#!/usr/bin/env python3
import os
import sys

with open(os.environ["SKY_RUN_LOG"], "a", encoding="utf-8") as handle:
    handle.write(
        "trainer "
        f"tag={os.environ.get('TAUBENCH_ENV_TAG', '')} "
        f"token_set={bool(os.environ.get('TAUBENCH_ENV_TOKEN'))} "
        f"argv={sys.argv[1:]}\\n"
    )
"""


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _run_sandbox_task(
    tmp_path: Path,
    suffix: str,
    *,
    cwsandbox_api_key: str | None = "sandbox-api-secret",
    placement_mode: str = "serverless",
    serverless_auth: str = "wandb",
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    task = _load_task(SANDBOX_TASK_PATH)
    workdir = tmp_path / suffix
    (workdir / "data" / "processed").mkdir(parents=True)
    (workdir / "data" / "processed" / "train.parquet").write_bytes(b"PAR1")
    (workdir / "data" / "processed" / "test.parquet").write_bytes(b"PAR1")
    home = workdir / "home"
    (home / ".local" / "bin").mkdir(parents=True)
    fake_bin = workdir / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "python", _FAKE_SKY_PYTHON)
    _write_executable(fake_bin / "bash", _FAKE_SKY_BASH)

    log_path = workdir / "run.log"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env.get('PATH', '')}",
            "HOME": str(home),
            "WANDB_API_KEY": "wandb-secret",
            "HF_TOKEN": "hf-secret",
            "CWSANDBOX_PLACEMENT_MODE": placement_mode,
            "CWSANDBOX_SERVERLESS_AUTH": serverless_auth,
            "TAUBENCH_SIMULATOR_URL": "http://simulator.example",
            "TAUBENCH_DOMAIN": "retail",
            "TAUBENCH_TASK_SPLIT": "train",
            "FAKE_TAG_SUFFIX": suffix,
            "SKY_RUN_LOG": str(log_path),
        }
    )
    if cwsandbox_api_key is None:
        env.pop("CWSANDBOX_API_KEY", None)
    else:
        env["CWSANDBOX_API_KEY"] = cwsandbox_api_key
    env.pop("TAUBENCH_ENV_TAG", None)

    result = subprocess.run(
        ["/bin/bash", "-eu", "-c", task["run"]],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    lines = log_path.read_text(encoding="utf-8").splitlines() if log_path.exists() else []
    return result, lines


def test_sandbox_fallback_generates_unique_job_tag_before_preflight_and_training(
    tmp_path: Path,
) -> None:
    first, first_lines = _run_sandbox_task(tmp_path, "run-one")
    second, second_lines = _run_sandbox_task(tmp_path, "run-two")

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert first_lines == [
        "preflight tag=verl-taubench-env-run-one token_set=True",
        (
            "trainer tag=verl-taubench-env-run-one token_set=True "
            "argv=['scripts/train_grpo.sh', 'trainer.total_epochs=3']"
        ),
    ]
    assert second_lines[0] == "preflight tag=verl-taubench-env-run-two token_set=True"
    assert "tag=verl-taubench-env-run-two" in second_lines[1]
    for secret in ("generated-secret-value", "wandb-secret", "hf-secret", "sandbox-api-secret"):
        assert secret not in first.stderr
    assert "generated-secret-value" not in "\n".join(first_lines)


def test_serverless_sandbox_task_does_not_require_coreweave_api_key(tmp_path: Path) -> None:
    result, lines = _run_sandbox_task(
        tmp_path,
        "serverless-wandb-auth",
        cwsandbox_api_key=None,
    )

    assert result.returncode == 0, result.stderr
    assert len(lines) == 2


def test_serverless_coreweave_auth_requires_coreweave_api_key(tmp_path: Path) -> None:
    result, lines = _run_sandbox_task(
        tmp_path, "serverless-coreweave-auth", cwsandbox_api_key=None,
        serverless_auth="coreweave",
    )
    assert result.returncode != 0
    assert "CWSANDBOX_API_KEY" in result.stderr
    assert lines == []


def test_cks_sandbox_task_requires_coreweave_api_key(tmp_path: Path) -> None:
    result, lines = _run_sandbox_task(
        tmp_path,
        "cks-coreweave-auth",
        cwsandbox_api_key=None,
        placement_mode="cks",
    )

    assert result.returncode != 0
    assert "CWSANDBOX_API_KEY" in result.stderr
    assert lines == []
