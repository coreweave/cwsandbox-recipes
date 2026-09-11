"""Tests for the GPU trainer sandbox launcher against a fake cwsandbox SDK.

No network, cloud sandboxes, or credentials. The fake mirrors the 1.x
signatures exercised by :mod:`verl_taubench.sandbox.trainer`.
"""

from __future__ import annotations

import asyncio
import shlex
import sys
import types
from argparse import Namespace
from pathlib import Path
from typing import Any, Iterator, List, Optional

import pytest

from scripts.launch_gpu_sandbox import (
    _ensure_wandb_run_id,
    _resolve_config,
    _validate_auth_environment,
    main,
)
from verl_taubench.sandbox.tags import RECIPE_TAG
from verl_taubench.sandbox.trainer import (
    DEFAULT_CONTAINER_IMAGE,
    DEFAULT_GPU_COUNT,
    DEFAULT_GPU_TYPE,
    DEFAULT_SECRET_NAMES,
    DEFAULT_SECRET_STORE,
    DEFAULT_TRAINER_TAG,
    FORWARDED_ENV_KEYS,
    GpuTrainerLauncher,
    TrainerSandboxConfig,
    TrainingJobFailed,
    _build_environment,
    _new_cpu_env_tag,
    collect_mounted_files,
    config_from_env,
    parse_secret_names,
    parse_secret_store,
)


def test_launcher_generates_id_with_installed_wandb_sdk(monkeypatch):
    def unexpected_uuid():
        pytest.fail("Installed W&B SDK must generate the ID, not the UUID fallback")

    monkeypatch.setattr("scripts.launch_gpu_sandbox.uuid", types.SimpleNamespace(uuid4=unexpected_uuid))
    environment = {}
    run_id = _ensure_wandb_run_id(environment)
    assert len(run_id) == 8
    assert environment["WANDB_RUN_ID"] == run_id


def test_launcher_preserves_existing_run_id():
    environment = {"WANDB_RUN_ID": "existing-run-id"}
    assert _ensure_wandb_run_id(environment) == "existing-run-id"


class FakeStreamReader:
    def __init__(self, lines: List[str]):
        self._lines = list(lines)

    def __iter__(self) -> Iterator[str]:
        yield from self._lines


class FakeRef:
    def __init__(self, value: Any = None, raises: Optional[Exception] = None):
        self._value = value
        self._raises = raises

    def result(self, timeout: Optional[float] = None) -> Any:
        if self._raises is not None:
            raise self._raises
        return self._value


class FakeCancellation(BaseException):
    """Realistic cancellation signal that is not an Exception."""


class FakeSandbox:
    instances: List["FakeSandbox"] = []
    fail_run = False
    fail_wait = False
    fail_stream_logs = False
    fail_historical_logs = False
    fail_wait_until_complete_result = False
    force_returncode_none = False
    wait_base_exception: Optional[BaseException] = None
    stop_base_exception: Optional[BaseException] = None

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self.sandbox_id = f"trainer-fake-{len(FakeSandbox.instances) + 1}"
        self.calls: List[str] = []
        self._returncode = 0
        self.log_lines = ["epoch 1 loss=0.5\n", "done\n"]
        self.log_follows: List[bool] = []
        self.stopped = False
        FakeSandbox.instances.append(self)

    @classmethod
    def run(cls, *args: str, **kwargs: Any) -> "FakeSandbox":
        if kwargs.get("profile_ids") is not None or kwargs.get("profile_names") is not None:
            raise TypeError("profile_ids/profile_names were removed in cwsandbox 1.x")
        if cls.fail_run:
            raise RuntimeError("sandbox API unavailable")
        sandbox = cls(**kwargs)
        sandbox.command = list(args)
        sandbox.calls.append("run")
        if cls.force_returncode_none:
            sandbox._returncode = None
        return sandbox

    @property
    def returncode(self) -> Optional[int]:
        return self._returncode

    def wait(self, timeout: Optional[float] = None) -> "FakeSandbox":
        self.calls.append("wait")
        if FakeSandbox.wait_base_exception is not None:
            raise FakeSandbox.wait_base_exception
        if FakeSandbox.fail_wait:
            raise TimeoutError("sandbox did not reach RUNNING")
        return self

    def stream_logs(self, *, follow: bool = False, **kw: Any) -> FakeStreamReader:
        self.calls.append("stream_logs")
        self.log_follows.append(follow)
        if follow and FakeSandbox.fail_stream_logs:
            raise RuntimeError("sandbox is already terminal")
        if not follow and FakeSandbox.fail_historical_logs:
            raise RuntimeError("historical logs unavailable")
        return FakeStreamReader(self.log_lines)

    # Number of times get_status reports "running" before turning terminal;
    # exercises the launcher's stream-reattach loop.
    alive_status_polls = 0

    def get_status(self) -> str:
        self.calls.append("get_status")
        if FakeSandbox.alive_status_polls > 0:
            FakeSandbox.alive_status_polls -= 1
            return "running"
        return "completed"

    def wait_until_complete(self, **kw: Any) -> FakeRef:
        self.calls.append("wait_until_complete")
        exc = None
        if FakeSandbox.fail_wait_until_complete_result:
            exc = RuntimeError("wait_until_complete result failed")
        return FakeRef(self, raises=exc)

    def stop(self, **kw: Any) -> FakeRef:
        self.calls.append("stop")
        self.stopped = True
        if FakeSandbox.stop_base_exception is not None:
            raise FakeSandbox.stop_base_exception
        return FakeRef(None)


class FakeNetworkOptions:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.deny_egress = kwargs.get("deny_egress")
        self.deny_ingress = kwargs.get("deny_ingress")
        self.egress = kwargs.get("egress")


class FakeResourceOptions:
    def __init__(self, *, requests=None, limits=None, gpu=None):
        self.requests = requests
        self.limits = limits
        self.gpu = gpu


class FakeSecret:
    def __init__(self, *, store: str, name: str, env_var: Optional[str] = None, field: str = ""):
        self.store = store
        self.name = name
        self.env_var = env_var or name
        self.field = field


@pytest.fixture
def fake_sdk(monkeypatch):
    monkeypatch.setenv("CWSANDBOX_RUNNER_IDS", "my-cluster")
    FakeSandbox.instances = []
    for flag in (
        "fail_run",
        "fail_wait",
        "fail_stream_logs",
        "fail_historical_logs",
        "fail_wait_until_complete_result",
        "alive_status_polls",
        "force_returncode_none",
    ):
        setattr(FakeSandbox, flag, False)
    FakeSandbox.wait_base_exception = None
    FakeSandbox.stop_base_exception = None

    module = types.ModuleType("cwsandbox")
    module.Sandbox = FakeSandbox
    module.NetworkOptions = FakeNetworkOptions
    module.ResourceOptions = FakeResourceOptions
    module.Secret = FakeSecret
    monkeypatch.setitem(sys.modules, "cwsandbox", module)
    yield FakeSandbox
    FakeSandbox.instances = []


@pytest.fixture
def sample_project(tmp_path: Path) -> Path:
    root = tmp_path / "recipe"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    (root / "uv.lock").write_text("# lock\n")
    (root / "README.md").write_text("# recipe readme\n")
    (root / ".env").write_text("WANDB_API_KEY=super-secret-not-mounted\n")
    (root / ".venv" / "lib").mkdir(parents=True)
    (root / ".venv" / "lib" / "junk.py").write_text("x = 1\n")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "mod.cpython-311.pyc").write_bytes(b"\x00\x01")
    (root / "data" / "processed").mkdir(parents=True)
    (root / "data" / "processed" / "train.parquet").write_bytes(b"PAR1")
    (root / "verl_taubench" / "trainer").mkdir(parents=True)
    (root / "verl_taubench" / "__init__.py").write_text('"""pkg"""\n')
    (root / "verl_taubench" / "trainer" / "grpo_trainer.yaml").write_text("trainer: {}\n")
    (root / "config" / "tool_config").mkdir(parents=True)
    (root / "config" / "tool_config" / "taubench.yaml").write_text("tools: []\n")
    (root / "scripts").mkdir()
    (root / "scripts" / "train_grpo.sh").write_text("#!/usr/bin/env bash\necho train\n")
    return root


def _config(project_root: Path, **kw) -> TrainerSandboxConfig:
    defaults = dict(
        project_root=project_root,
        gpu_count=8,
        gpu_type="H100",
        cpu="32",
        memory="256Gi",
        container_image="verlai/verl:vllm011.latest",
        max_lifetime_seconds=7200,
    )
    defaults.update(kw)
    return TrainerSandboxConfig(**defaults)


def test_collect_mounted_files_uses_allowlist_and_excludes_secrets(sample_project: Path) -> None:
    mounts = collect_mounted_files(sample_project, workspace="/workspace")
    paths = {m["mount_path"] for m in mounts}
    assert "/workspace/pyproject.toml" in paths
    assert "/workspace/uv.lock" in paths
    assert "/workspace/README.md" in paths
    assert "/workspace/verl_taubench/__init__.py" in paths
    assert "/workspace/config/tool_config/taubench.yaml" in paths
    assert "/workspace/scripts/train_grpo.sh" in paths
    assert all(set(m) == {"mount_path", "file_content"} for m in mounts)
    assert not any(".env" in p for p in paths)
    assert not any(".venv" in p for p in paths)
    assert not any("__pycache__" in p for p in paths)
    assert not any("/data/" in p for p in paths)


def test_collect_mounted_files_always_excludes_data_even_when_present(sample_project: Path) -> None:
    mounts = collect_mounted_files(sample_project, workspace="/workspace")
    assert not any("/data/" in m["mount_path"] for m in mounts)
    assert all(isinstance(m["file_content"], str) for m in mounts)


def test_collect_mounted_files_rejects_non_utf8_allowlisted_file(sample_project: Path) -> None:
    (sample_project / "config" / "bad.bin").write_bytes(b"\xff\xfe\xfd")
    with pytest.raises(ValueError, match="not valid UTF-8 text: config/bad.bin"):
        collect_mounted_files(sample_project, workspace="/workspace")


@pytest.mark.parametrize("target_kind", ["internal", "external"])
def test_symlink_mounts_fail_before_sandbox_run(
    fake_sdk, sample_project: Path, tmp_path: Path, target_kind: str
) -> None:
    if target_kind == "internal":
        target = sample_project / "scripts" / "train_grpo.sh"
    else:
        target = tmp_path / "outside.py"
        target.write_text("print('outside')\n", encoding="utf-8")
    (sample_project / "scripts" / f"{target_kind}.py").symlink_to(target)

    launcher = GpuTrainerLauncher(_config(sample_project))
    with pytest.raises(ValueError, match="symlink.*not allowed"):
        launcher.launch_and_wait(log_sink=lambda _line: None)

    assert fake_sdk.instances == []


def test_build_environment_sets_pythonpath_and_preserves_extra(sample_project: Path) -> None:
    config = _config(
        sample_project,
        gpu_count=3,
        extra_env={"PYTHONPATH": "/extra/lib", "MODEL_PATH": "Qwen/Qwen2.5-1.5B-Instruct"},
    )
    env = _build_environment(config, "verl-taubench-env-test")
    assert env["N_GPUS"] == "3"
    assert env["PYTHONPATH"] == "/workspace:/extra/lib"
    assert env["MODEL_PATH"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert env["PROJECT_ROOT"] == "/workspace"


def test_each_launch_gets_unique_generated_cpu_env_tag(
    monkeypatch, sample_project: Path
) -> None:
    monkeypatch.setenv("TAUBENCH_ENV_TAG", "host-must-not-win")

    first_tag, second_tag = _new_cpu_env_tag(), _new_cpu_env_tag()
    first = _build_environment(config_from_env(sample_project), first_tag)
    second = _build_environment(config_from_env(sample_project), second_tag)

    assert first["TAUBENCH_ENV_TAG"].startswith("verl-taubench-env-")
    assert second["TAUBENCH_ENV_TAG"].startswith("verl-taubench-env-")
    assert first["TAUBENCH_ENV_TAG"] != second["TAUBENCH_ENV_TAG"]
    assert first["TAUBENCH_ENV_TAG"] != "host-must-not-win"


def test_cpu_resource_overrides_are_forwarded_to_gpu_trainer(monkeypatch, sample_project):
    monkeypatch.setenv("TAUBENCH_ENV_CPU", "6")
    monkeypatch.setenv("TAUBENCH_ENV_MEMORY", "12Gi")
    env = _build_environment(config_from_env(sample_project), "test-env-tag")
    assert env["TAUBENCH_ENV_CPU"] == "6"
    assert env["TAUBENCH_ENV_MEMORY"] == "12Gi"


def test_config_from_env_defaults_and_overrides(monkeypatch, sample_project: Path) -> None:
    monkeypatch.delenv("N_GPUS", raising=False)
    monkeypatch.delenv("GPU_COUNT", raising=False)
    monkeypatch.delenv("TRAINER_MEMORY", raising=False)
    monkeypatch.setenv("GPU_TYPE", "A100")
    monkeypatch.setenv("TRAINER_CPU", "16")
    monkeypatch.setenv("MODEL_PATH", "meta-llama/Llama-3")

    config = config_from_env(sample_project)
    assert config.project_root == sample_project.resolve()
    assert config.gpu_count == DEFAULT_GPU_COUNT
    assert config.gpu_type == "A100"
    assert config.cpu == "16"
    assert config.memory == "256Gi"
    assert config.container_image == DEFAULT_CONTAINER_IMAGE
    assert config.extra_env["MODEL_PATH"] == "meta-llama/Llama-3"


def test_cli_auth_validator_accepts_serverless_keys() -> None:
    _validate_auth_environment(
        {
            "CWSANDBOX_API_KEY": "direct-secret",
            "WANDB_API_KEY": "wandb-secret",
        }
    )


def test_cli_serverless_coreweave_auth_does_not_require_wandb_for_cleanup():
    _validate_auth_environment({
        "CWSANDBOX_API_KEY": "cw-test-token",
        "CWSANDBOX_SERVERLESS_AUTH": "coreweave",
    })


def test_cli_serverless_auth_override_reaches_trainer(monkeypatch, sample_project):
    monkeypatch.setenv("CWSANDBOX_API_KEY", "cw-test-token")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("CWSANDBOX_SERVERLESS_AUTH", "wandb")
    captured = []
    monkeypatch.setattr(GpuTrainerLauncher, "launch_and_wait", lambda self: captured.append(
        _build_environment(self.config, "test-env-tag")
    ))
    assert main(["--project-root", str(sample_project), "--serverless-auth", "coreweave"]) == 0
    assert captured[0]["CWSANDBOX_SERVERLESS_AUTH"] == "coreweave"
    assert captured[0]["CWSANDBOX_API_KEY"] == "cw-test-token"


def test_cli_rejects_invalid_serverless_auth():
    with pytest.raises(ValueError, match="CWSANDBOX_SERVERLESS_AUTH"):
        _validate_auth_environment({"CWSANDBOX_API_KEY": "cw-test-token", "CWSANDBOX_SERVERLESS_AUTH": "typo"})


def test_cli_auth_validator_accepts_cks_coreweave_key() -> None:
    _validate_auth_environment(
        {
            "CWSANDBOX_API_KEY": "direct-secret",
            "CWSANDBOX_PLACEMENT_MODE": "cks",
        }
    )


@pytest.mark.parametrize("placement_mode", [None, ""])
def test_cli_auth_validator_rejects_serverless_without_wandb_key(
    placement_mode: str | None,
) -> None:
    environment = {"CWSANDBOX_API_KEY": "direct-secret"}
    if placement_mode is not None:
        environment["CWSANDBOX_PLACEMENT_MODE"] = placement_mode
    with pytest.raises(ValueError, match="WANDB_API_KEY"):
        _validate_auth_environment(environment)


@pytest.mark.parametrize(
    "environment",
    [
        {},
        {"WANDB_API_KEY": "wandb-secret"},
        {"WANDB_API_KEY": "wandb-secret", "WANDB_ENTITY": "team-name"},
        {"WANDB_ENTITY": "team-name"},
    ],
)
def test_cli_auth_validator_rejects_missing_or_wandb_only_modes(
    environment: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match="CWSANDBOX_API_KEY"):
        _validate_auth_environment(environment)


def test_cli_auth_failure_does_not_print_secret_values(
    monkeypatch, capsys, sample_project: Path
) -> None:
    monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "do-not-print-this-secret")
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.setattr(
        GpuTrainerLauncher,
        "launch_and_wait",
        lambda self: pytest.fail("launcher must not run without valid auth"),
    )

    assert main(["--project-root", str(sample_project)]) == 2
    captured = capsys.readouterr()
    assert "CWSANDBOX_API_KEY" in captured.err
    assert "do-not-print-this-secret" not in captured.err


def test_cli_resolve_config_overrides_env_defaults(monkeypatch, sample_project: Path) -> None:
    monkeypatch.setenv("N_GPUS", "8")
    args = Namespace(
        project_root=sample_project,
        gpu_count=1,
        gpu_type=None,
        cpu=None,
        memory=None,
        image="custom/image:tag",
        max_lifetime_seconds=None,
        hydra_override=None,
        secret_names=None,
    )
    config = _resolve_config(args)
    assert config.gpu_count == 1
    assert config.gpu_type == DEFAULT_GPU_TYPE
    assert config.container_image == "custom/image:tag"


def test_launch_passes_gpu_network_and_resources(fake_sdk, sample_project: Path) -> None:
    launcher = GpuTrainerLauncher(_config(sample_project, gpu_count=8, gpu_type="H100"))
    launcher.launch_and_wait(log_sink=lambda _line: None)

    sandbox = fake_sdk.instances[0]
    resources = sandbox.kwargs["resources"]
    assert resources.gpu == {"count": 8, "type": "H100"}
    assert resources.requests == {"cpu": "32", "memory": "256Gi"}
    assert resources.limits == {"cpu": "32", "memory": "256Gi"}

    network = sandbox.kwargs["network"]
    assert network.kwargs == {}
    assert sandbox.kwargs["placement_mode"] == "cks"
    assert sandbox.kwargs["placement_spillover"] == "strict"
    assert "auth" not in sandbox.kwargs


@pytest.mark.parametrize("runner_ids", [None, "", "  ", ", ,"])
def test_gpu_launch_refuses_automatic_runner_selection(
    fake_sdk, monkeypatch, sample_project: Path, runner_ids
) -> None:
    if runner_ids is None:
        monkeypatch.delenv("CWSANDBOX_RUNNER_IDS", raising=False)
    else:
        monkeypatch.setenv("CWSANDBOX_RUNNER_IDS", runner_ids)
    with pytest.raises(ValueError, match="CWSANDBOX_RUNNER_IDS"):
        GpuTrainerLauncher(_config(sample_project)).launch_and_wait()
    assert fake_sdk.instances == []


def test_gpu_runner_pin_cannot_be_overridden_by_cpu_pool_placement(
    fake_sdk, monkeypatch, sample_project: Path
) -> None:
    monkeypatch.setenv("CWSANDBOX_PLACEMENT_MODE", "serverless")
    monkeypatch.setenv("CWSANDBOX_PLACEMENT_SPILLOVER", "cks_then_serverless")
    GpuTrainerLauncher(_config(sample_project)).launch_and_wait(log_sink=lambda _: None)
    kwargs = fake_sdk.instances[0].kwargs
    assert kwargs["runner_ids"] == ["my-cluster"]
    assert kwargs["placement_mode"] == "cks"
    assert kwargs["placement_spillover"] == "strict"
    assert kwargs["environment_variables"]["CWSANDBOX_PLACEMENT_MODE"] == "serverless"


def test_default_gpu_request_fits_recipe_runner_policy(
    request, monkeypatch, sample_project: Path
) -> None:
    import yaml

    parse_quantity = pytest.importorskip("cwsandbox._quantity").parse_quantity
    fake_sdk = request.getfixturevalue("fake_sdk")

    for name in ("TRAINER_MEMORY", "TRAINER_CPU", "N_GPUS"):
        monkeypatch.delenv(name, raising=False)
    policy_path = Path(__file__).resolve().parents[2] / "infra" / "runner-policy.yaml"
    policy = yaml.safe_load(policy_path.read_text())["constraints"]["resources"]
    GpuTrainerLauncher(config_from_env(sample_project)).launch_and_wait(log_sink=lambda _: None)
    resources = fake_sdk.instances[0].kwargs["resources"]
    assert parse_quantity(resources.requests["memory"]) <= parse_quantity(policy["max_memory"])
    assert parse_quantity(resources.requests["cpu"]) <= parse_quantity(policy["max_cpu"])
    assert resources.gpu["count"] <= int(policy["max_gpu_count"])


@pytest.mark.parametrize("recovers", [True, False], ids=["transient-outage", "persistent-outage"])
def test_gpu_startup_polling_tolerates_outage_without_recreating_sandbox(
    request, monkeypatch, sample_project: Path, recovers: bool
) -> None:
    # Keep the installed SDK's retry loop real; replace network polling and
    # its clock. The former 30s budget must fail a 75s gateway outage.
    sdk = pytest.importorskip("cwsandbox._sandbox")
    defaults = sdk.SandboxDefaults()
    fake_sdk = request.getfixturevalue("fake_sdk")
    elapsed = 0.0
    wait_timeouts = []
    reaped = []

    async def sleep(delay):
        nonlocal elapsed
        elapsed += delay

    async def poll(*, rpc_timeout_override=None):
        nonlocal elapsed
        elapsed += rpc_timeout_override
        if not recovers or elapsed < 75.0:
            raise sdk.SandboxRequestTimeoutError("Poll sandbox status timed out: Deadline Exceeded")
        return "running"

    def wait(sandbox, timeout=None):
        wait_timeouts.append(timeout)
        # These are the same defaults the real Sandbox constructor resolves
        # when the launcher omits its polling kwargs.
        harness = types.SimpleNamespace(
            _sandbox_id=sandbox.sandbox_id,
            _poll_retry_budget_seconds=sandbox.kwargs.get(
                "poll_retry_budget_seconds", defaults.poll_retry_budget_seconds
            ),
            _poll_rpc_timeout_seconds=sandbox.kwargs.get(
                "poll_rpc_timeout_seconds", defaults.poll_rpc_timeout_seconds
            ),
            _poll_until_stable=poll,
        )
        asyncio.run(sdk.Sandbox._poll_with_retry(harness))
        return sandbox

    monkeypatch.setattr(sdk, "time", types.SimpleNamespace(monotonic=lambda: elapsed))
    monkeypatch.setattr(sdk, "asyncio", types.SimpleNamespace(sleep=sleep))
    monkeypatch.setattr(fake_sdk, "wait", wait)
    monkeypatch.setattr("verl_taubench.sandbox.trainer._reap_env_pool", reaped.append)
    launcher = GpuTrainerLauncher(_config(sample_project))

    if recovers:
        assert launcher.launch_and_wait(log_sink=lambda _line: None) == 0
        assert 75.0 <= elapsed < 120.0
    else:
        with pytest.raises(sdk.SandboxRequestTimeoutError, match="Poll sandbox status timed out"):
            launcher.launch_and_wait(log_sink=lambda _line: None)
        assert 300.0 <= elapsed <= 331.0

    assert len(fake_sdk.instances) == 1
    assert fake_sdk.instances[0].stopped is (not recovers)
    assert wait_timeouts == [900.0]
    assert len(reaped) == 1


def test_launch_uses_supported_v1_runner_selector_without_removed_profile_selector(
    fake_sdk, monkeypatch, sample_project: Path
) -> None:
    monkeypatch.setenv("CWSANDBOX_RUNNER_IDS", "gpu-runner")
    monkeypatch.setenv("CWSANDBOX_PROFILE_NAMES", "removed-v1-selector")

    GpuTrainerLauncher(config_from_env(sample_project)).launch_and_wait(
        log_sink=lambda _line: None
    )

    sandbox = fake_sdk.instances[0]
    assert sandbox.kwargs["runner_ids"] == ["gpu-runner"]
    assert "profile_ids" not in sandbox.kwargs
    assert "profile_names" not in sandbox.kwargs


def test_launch_command_cds_to_workspace_and_execs_train_grpo(fake_sdk, sample_project: Path) -> None:
    launcher = GpuTrainerLauncher(_config(sample_project))
    launcher.launch_and_wait(log_sink=lambda _line: None)

    command = fake_sdk.instances[0].command
    assert command[:2] == ["bash", "-lc"]
    shell = command[2]
    assert "cd /workspace" in shell
    assert "exec bash /workspace/scripts/train_grpo.sh" in shell
    assert "skypilot" not in shell.lower()
    assert "sky " not in shell.lower()


def test_launch_passes_environment_variables(fake_sdk, sample_project: Path) -> None:
    launcher = GpuTrainerLauncher(
        _config(sample_project, gpu_count=4, extra_env={"MODEL_PATH": "Qwen/Qwen2.5-1.5B-Instruct"})
    )
    launcher.launch_and_wait(log_sink=lambda _line: None)

    env = fake_sdk.instances[0].kwargs["environment_variables"]
    assert env["N_GPUS"] == "4"
    assert env["PYTHONPATH"] == "/workspace"
    assert env["MODEL_PATH"] == "Qwen/Qwen2.5-1.5B-Instruct"


def test_reusing_launcher_still_generates_a_fresh_cpu_env_tag_per_run(
    fake_sdk, sample_project: Path
) -> None:
    launcher = GpuTrainerLauncher(_config(sample_project))

    launcher.launch_and_wait(log_sink=lambda _line: None)
    launcher.launch_and_wait(log_sink=lambda _line: None)

    first_env = fake_sdk.instances[0].kwargs["environment_variables"]
    second_env = fake_sdk.instances[1].kwargs["environment_variables"]
    assert first_env["TAUBENCH_ENV_TAG"] != second_env["TAUBENCH_ENV_TAG"]


def test_launch_tags_trainer_sandbox_separately(fake_sdk, sample_project: Path) -> None:
    launcher = GpuTrainerLauncher(_config(sample_project))
    launcher.launch_and_wait(log_sink=lambda _line: None)

    tags = fake_sdk.instances[0].kwargs["tags"]
    assert DEFAULT_TRAINER_TAG in tags
    assert "verl-taubench-env" not in tags


def test_launch_streams_logs_and_waits_for_completion(fake_sdk, sample_project: Path) -> None:
    lines: list[str] = []
    launcher = GpuTrainerLauncher(_config(sample_project))
    rc = launcher.launch_and_wait(log_sink=lines.append)

    sandbox = fake_sdk.instances[0]
    assert sandbox.calls == ["run", "wait", "stream_logs", "get_status", "wait_until_complete"]
    assert lines == ["epoch 1 loss=0.5\n", "done\n"]
    assert rc == 0


def test_fast_terminal_job_falls_back_to_historical_logs(
    fake_sdk, sample_project: Path
) -> None:
    fake_sdk.fail_stream_logs = True
    lines: list[str] = []

    rc = GpuTrainerLauncher(_config(sample_project)).launch_and_wait(log_sink=lines.append)

    sandbox = fake_sdk.instances[0]
    assert sandbox.log_follows == [True, False]
    assert sandbox.calls == [
        "run",
        "wait",
        "stream_logs",
        "get_status",
        "wait_until_complete",
        "stream_logs",
    ]
    assert lines == sandbox.log_lines
    assert rc == 0
    assert sandbox.stopped is False


def test_fast_terminal_nonzero_returncode_is_not_masked_by_log_attach_error(
    fake_sdk, sample_project: Path
) -> None:
    fake_sdk.fail_stream_logs = True
    original_run = FakeSandbox.run

    @classmethod
    def run_terminal_failure(cls, *args, **kwargs):
        sandbox = original_run(*args, **kwargs)
        sandbox._returncode = 23
        return sandbox

    FakeSandbox.run = run_terminal_failure
    try:
        with pytest.raises(TrainingJobFailed) as exc:
            GpuTrainerLauncher(_config(sample_project)).launch_and_wait(
                log_sink=lambda _line: None
            )
        assert exc.value.returncode == 23
        assert fake_sdk.instances[0].log_follows == [True, False]
    finally:
        FakeSandbox.run = original_run


def test_launch_raises_on_nonzero_returncode(fake_sdk, sample_project: Path) -> None:
    launcher = GpuTrainerLauncher(_config(sample_project))
    original_run = FakeSandbox.run

    @classmethod
    def run_with_failure(cls, *args, **kwargs):
        sb = original_run(*args, **kwargs)
        sb._returncode = 17
        return sb

    FakeSandbox.run = run_with_failure
    try:
        with pytest.raises(TrainingJobFailed) as exc:
            launcher.launch_and_wait(log_sink=lambda _line: None)
        assert exc.value.returncode == 17
    finally:
        FakeSandbox.run = original_run


def test_launch_propagates_sdk_failures(fake_sdk, sample_project: Path) -> None:
    fake_sdk.fail_run = True
    launcher = GpuTrainerLauncher(_config(sample_project))
    with pytest.raises(RuntimeError, match="sandbox API unavailable"):
        launcher.launch_and_wait(log_sink=lambda _line: None)


def test_launch_missing_project_root_raises_before_run(fake_sdk, tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    launcher = GpuTrainerLauncher(_config(missing))
    with pytest.raises(FileNotFoundError, match="project root does not exist"):
        launcher.launch_and_wait(log_sink=lambda _line: None)
    assert fake_sdk.instances == []


def test_completed_sandbox_without_optional_returncode_is_success(
    fake_sdk, sample_project: Path
) -> None:
    fake_sdk.force_returncode_none = True
    launcher = GpuTrainerLauncher(_config(sample_project))

    assert launcher.launch_and_wait(log_sink=lambda _line: None) == 0

    sandbox = fake_sdk.instances[0]
    assert sandbox.stopped is False
    assert "stop" not in sandbox.calls


@pytest.mark.parametrize(
    "flag,match",
    [
        ("fail_wait", "did not reach RUNNING"),
        ("fail_wait_until_complete_result", "wait_until_complete result failed"),
    ],
)
def test_lifecycle_failures_stop_sandbox_and_reraise(
    fake_sdk, sample_project: Path, flag: str, match: str
) -> None:
    setattr(fake_sdk, flag, True)
    launcher = GpuTrainerLauncher(_config(sample_project))
    with pytest.raises(Exception, match=match):
        launcher.launch_and_wait(log_sink=lambda _line: None)

    sandbox = fake_sdk.instances[0]
    assert sandbox.stopped is True
    assert "stop" in sandbox.calls


def test_base_exception_after_run_stops_sandbox_without_masking_original(
    fake_sdk, sample_project: Path
) -> None:
    original = FakeCancellation("cancelled")
    fake_sdk.wait_base_exception = original
    fake_sdk.stop_base_exception = FakeCancellation("cleanup also cancelled")

    with pytest.raises(FakeCancellation) as exc:
        GpuTrainerLauncher(_config(sample_project)).launch_and_wait(
            log_sink=lambda _line: None
        )

    assert exc.value is original
    assert fake_sdk.instances[0].stopped is True


def test_launch_passes_secrets_without_mounting_values(fake_sdk, sample_project: Path) -> None:
    launcher = GpuTrainerLauncher(_config(sample_project))
    launcher.launch_and_wait(log_sink=lambda _line: None)

    secrets = fake_sdk.instances[0].kwargs["secrets"]
    secret_names = {s.name for s in secrets}
    assert "WANDB_API_KEY" in secret_names
    assert "HF_TOKEN" in secret_names

    mounted_blob = "".join(m["file_content"] for m in fake_sdk.instances[0].kwargs["mounted_files"])
    assert "super-secret-not-mounted" not in mounted_blob


def test_parse_secret_store_none_disables_store() -> None:
    """Orgs without a configured cwsandbox secret store (the platform ships
    only a W&B-provider store) get 'secret store not found' on every create;
    TRAINER_SECRET_STORE=none switches to plain-env injection instead."""
    assert parse_secret_store(None) == DEFAULT_SECRET_STORE
    assert parse_secret_store("none") == ""
    assert parse_secret_store("None") == ""
    assert parse_secret_store("  ") == ""
    assert parse_secret_store("vault") == "vault"


def test_config_from_env_secret_store_override(monkeypatch, sample_project: Path) -> None:
    monkeypatch.setenv("TRAINER_SECRET_STORE", "none")
    assert config_from_env(sample_project).secret_store == ""
    monkeypatch.setenv("TRAINER_SECRET_STORE", "vault")
    assert config_from_env(sample_project).secret_store == "vault"
    monkeypatch.delenv("TRAINER_SECRET_STORE")
    assert config_from_env(sample_project).secret_store == DEFAULT_SECRET_STORE


def test_launch_without_secret_store_injects_plain_env(
    fake_sdk, sample_project: Path, monkeypatch
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "wb-plain-secret")
    monkeypatch.setenv("HF_TOKEN", "hf-plain-secret")
    launcher = GpuTrainerLauncher(_config(sample_project, secret_store=""))
    launcher.launch_and_wait(log_sink=lambda _line: None)

    sandbox = fake_sdk.instances[0]
    assert "secrets" not in sandbox.kwargs
    env = sandbox.kwargs["environment_variables"]
    assert env["WANDB_API_KEY"] == "wb-plain-secret"
    assert env["HF_TOKEN"] == "hf-plain-secret"


def test_launch_without_secret_store_fails_before_create_on_missing_value(
    fake_sdk, sample_project: Path, monkeypatch
) -> None:
    """A trainer started without its keys dies minutes in, after the GPUs are
    allocated — the launcher must refuse to create the sandbox at all."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf-plain-secret")
    launcher = GpuTrainerLauncher(_config(sample_project, secret_store=""))
    with pytest.raises(ValueError, match="WANDB_API_KEY"):
        launcher.launch_and_wait(log_sink=lambda _line: None)
    assert fake_sdk.instances == []


def test_launch_sets_max_lifetime_and_image(fake_sdk, sample_project: Path) -> None:
    launcher = GpuTrainerLauncher(
        _config(
            sample_project,
            max_lifetime_seconds=12345,
            container_image="verlai/verl:vllm011.latest",
        )
    )
    launcher.launch_and_wait(log_sink=lambda _line: None)

    sandbox = fake_sdk.instances[0]
    assert sandbox.kwargs["max_lifetime_seconds"] == 12345
    assert sandbox.kwargs["container_image"] == "verlai/verl:vllm011.latest"


def test_default_image_at_sdk_boundary_is_verified_vllm_018_digest(
    fake_sdk, monkeypatch, sample_project: Path
) -> None:
    monkeypatch.delenv("TRAINER_IMAGE", raising=False)
    monkeypatch.delenv("CONTAINER_IMAGE", raising=False)
    launcher = GpuTrainerLauncher(config_from_env(sample_project))
    launcher.launch_and_wait(log_sink=lambda _line: None)

    assert fake_sdk.instances[0].kwargs["container_image"] == (
        "verlai/verl:vllm024.dev2@"
        "sha256:b867883b0dd011363e69ab2ab344922a28c5bd0409e2a324e3ee70fb27ca7543"
    )


_ALLOWLISTED_TAUBENCH_KEYS = (
    "TAUBENCH_SIMULATOR_URL",
    "TAUBENCH_SIMULATOR_MODEL",
    "TAUBENCH_POOL_SIZE",
    "TAUBENCH_MAX_CONNECTIONS",
    "TAUBENCH_ENV_IMAGE",
    "TAUBENCH_DOMAIN",
    "TAUBENCH_TASK_SPLIT",
    "TAUBENCH_TEST_END_INDEX",
)

_ALLOWLISTED_CW_KEYS = ("CW_ENDPOINT", "CW_BUCKET")
_ALLOWLISTED_WANDB_IDENTITY_KEYS = (
    "WANDB_ENTITY",
    "WANDB_PROJECT",
)


def test_config_from_env_forwards_allowlisted_host_vars(monkeypatch, sample_project: Path) -> None:
    for key in FORWARDED_ENV_KEYS:
        monkeypatch.setenv(key, f"value-{key}")
    monkeypatch.setenv("TAUBENCH_ENV_TOKEN", "host-bearer-secret")
    monkeypatch.setenv("RANDOM_SECRET", "must-not-forward")

    config = config_from_env(sample_project)
    for key in FORWARDED_ENV_KEYS:
        assert config.extra_env[key] == f"value-{key}"
    assert "TAUBENCH_ENV_TOKEN" not in config.extra_env
    assert "RANDOM_SECRET" not in config.extra_env


def test_config_from_env_forwards_wandb_identity_not_secret_values(
    monkeypatch, sample_project: Path
) -> None:
    monkeypatch.setenv("WANDB_ENTITY", "metrics-team")
    monkeypatch.setenv("WANDB_PROJECT", "training-project")
    monkeypatch.setenv("WANDB_API_KEY", "must-stay-in-secret-store")

    env = _build_environment(config_from_env(sample_project), "verl-taubench-env-test")

    assert env["WANDB_ENTITY"] == "metrics-team"
    assert env["WANDB_PROJECT"] == "training-project"
    assert "WANDB_API_KEY" not in env


def test_build_environment_forwards_cwsandbox_api_key(
    monkeypatch, sample_project: Path
) -> None:
    monkeypatch.setenv("CWSANDBOX_API_KEY", "host-sandbox-token")
    env = _build_environment(config_from_env(sample_project), "verl-taubench-env-test")
    assert env["CWSANDBOX_API_KEY"] == "host-sandbox-token"


def test_build_environment_forwards_placement_settings(
    monkeypatch, sample_project: Path
) -> None:
    """Nested CPU Sandbox.run inside the GPU sandbox must resolve the same
    placement mode and runner selection as the host."""
    monkeypatch.setenv("CWSANDBOX_RUNNER_IDS", "runner-1")
    monkeypatch.setenv("CWSANDBOX_PLACEMENT_MODE", "cks")
    monkeypatch.setenv("CWSANDBOX_PLACEMENT_SPILLOVER", "cks_then_serverless")
    env = _build_environment(config_from_env(sample_project), "verl-taubench-env-test")
    assert env["CWSANDBOX_RUNNER_IDS"] == "runner-1"
    assert env["CWSANDBOX_PLACEMENT_MODE"] == "cks"
    assert env["CWSANDBOX_PLACEMENT_SPILLOVER"] == "cks_then_serverless"


def test_build_environment_omits_unset_placement_settings(
    monkeypatch, sample_project: Path
) -> None:
    for name in (
        "CWSANDBOX_RUNNER_IDS",
        "CWSANDBOX_PLACEMENT_MODE",
        "CWSANDBOX_PLACEMENT_SPILLOVER",
    ):
        monkeypatch.delenv(name, raising=False)
    env = _build_environment(config_from_env(sample_project), "verl-taubench-env-test")
    assert "CWSANDBOX_RUNNER_IDS" not in env


def test_config_from_env_omits_unset_allowlisted_vars(monkeypatch, sample_project: Path) -> None:
    for key in FORWARDED_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    config = config_from_env(sample_project)
    for key in (
        _ALLOWLISTED_TAUBENCH_KEYS
        + _ALLOWLISTED_CW_KEYS
        + _ALLOWLISTED_WANDB_IDENTITY_KEYS
        + (
        "DATA_DIR",
        "MODEL_PATH",
        "EXPERIMENT_NAME",
        "PROJECT_NAME",
        )
    ):
        assert key not in config.extra_env


def test_train_command_runs_the_script_with_no_default_overrides(
    fake_sdk, sample_project: Path
) -> None:
    """The tool config path is a grpo_trainer.yaml default, so the launcher
    passes nothing unless the caller supplied --hydra-override."""
    launcher = GpuTrainerLauncher(_config(sample_project))
    launcher.launch_and_wait(log_sink=lambda _line: None)

    shell = fake_sdk.instances[0].command[2]
    assert shell.endswith("exec bash /workspace/scripts/train_grpo.sh")


def test_cli_hydra_override_appends_to_defaults(monkeypatch, sample_project: Path) -> None:
    monkeypatch.delenv("N_GPUS", raising=False)
    args = Namespace(
        project_root=sample_project,
        gpu_count=None,
        gpu_type=None,
        cpu=None,
        memory=None,
        image=None,
        max_lifetime_seconds=None,
        hydra_override=["data.train_batch_size=4", "actor_rollout_ref.rollout.n=2"],
        secret_names=[],
    )
    config = _resolve_config(args)
    assert config.hydra_overrides == (
        "data.train_batch_size=4",
        "actor_rollout_ref.rollout.n=2",
    )


def test_train_command_quotes_hostile_hydra_overrides(fake_sdk, sample_project: Path) -> None:
    hostile = "data.train_batch_size=4; echo pwned"
    launcher = GpuTrainerLauncher(
        _config(
            sample_project,
            hydra_overrides=(hostile,),
        )
    )
    launcher.launch_and_wait(log_sink=lambda _line: None)

    shell = fake_sdk.instances[0].command[2]
    quoted_hostile = shlex.quote(hostile)
    assert quoted_hostile in shell
    # Semicolon must stay inside quotes — not a separate shell command token.
    after_script = shell.split("train_grpo.sh", 1)[1]
    assert after_script.strip().endswith(quoted_hostile)


def test_parse_secret_names_trim_dedupe_preserves_order() -> None:
    names = parse_secret_names([" WANDB_API_KEY ", "OPENAI_API_KEY", "WANDB_API_KEY", " HF_TOKEN "])
    assert names == ("WANDB_API_KEY", "OPENAI_API_KEY", "HF_TOKEN")


def test_parse_secret_names_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        parse_secret_names(["WANDB_API_KEY", "  "])
    with pytest.raises(ValueError, match="at least one"):
        parse_secret_names([])


def test_config_from_env_secret_names_from_env(monkeypatch, sample_project: Path) -> None:
    monkeypatch.setenv("TRAINER_SECRET_NAMES", " OPENAI_API_KEY , WANDB_API_KEY , OPENAI_API_KEY ")
    config = config_from_env(sample_project)
    assert config.secret_names == ("OPENAI_API_KEY", "WANDB_API_KEY")


def test_config_from_env_default_secret_names(monkeypatch, sample_project: Path) -> None:
    monkeypatch.delenv("TRAINER_SECRET_NAMES", raising=False)
    config = config_from_env(sample_project)
    assert config.secret_names == DEFAULT_SECRET_NAMES


def test_cli_secret_names_replace_env_defaults(monkeypatch, sample_project: Path) -> None:
    monkeypatch.setenv("TRAINER_SECRET_NAMES", "OPENAI_API_KEY")
    args = Namespace(
        project_root=sample_project,
        gpu_count=None,
        gpu_type=None,
        cpu=None,
        memory=None,
        image=None,
        max_lifetime_seconds=None,
        hydra_override=None,
        secret_names=["HF_TOKEN", "CW_ACCESS_KEY", "HF_TOKEN"],
    )
    config = _resolve_config(args)
    assert config.secret_names == ("HF_TOKEN", "CW_ACCESS_KEY")


def test_launch_passes_cli_secret_names(fake_sdk, sample_project: Path) -> None:
    launcher = GpuTrainerLauncher(
        _config(sample_project, secret_names=("OPENAI_API_KEY", "CW_ACCESS_KEY"))
    )
    launcher.launch_and_wait(log_sink=lambda _line: None)

    secret_names = {s.name for s in fake_sdk.instances[0].kwargs["secrets"]}
    assert secret_names == {"OPENAI_API_KEY", "CW_ACCESS_KEY"}


def test_launch_forwards_taubench_env_at_sdk_boundary(
    fake_sdk, monkeypatch, sample_project: Path
) -> None:
    """M1: config_from_env values must reach sandbox.kwargs environment_variables."""
    monkeypatch.setenv("TAUBENCH_SIMULATOR_URL", "http://sim.example/v1")
    monkeypatch.setenv("TAUBENCH_POOL_SIZE", "32")
    monkeypatch.setenv("TAUBENCH_ENV_IMAGE", "registry.example/taubench:1.0")
    monkeypatch.setenv("CW_BUCKET", "my-checkpoints")
    monkeypatch.setenv("TAUBENCH_ENV_TOKEN", "host-bearer-secret")
    monkeypatch.setenv("RANDOM_SECRET", "must-not-forward")

    config = config_from_env(sample_project)
    launcher = GpuTrainerLauncher(config)
    launcher.launch_and_wait(log_sink=lambda _line: None)

    env = fake_sdk.instances[0].kwargs["environment_variables"]
    assert env["TAUBENCH_SIMULATOR_URL"] == "http://sim.example/v1"
    assert env["TAUBENCH_POOL_SIZE"] == "32"
    assert env["TAUBENCH_ENV_IMAGE"] == "registry.example/taubench:1.0"
    assert env["CW_BUCKET"] == "my-checkpoints"
    assert "TAUBENCH_ENV_TOKEN" not in env
    assert "RANDOM_SECRET" not in env


def test_stream_end_while_running_reattaches_instead_of_dying(fake_sdk, monkeypatch, sample_project):
    """A dropped log stream is not completion: the launcher must reattach.

    Regression: stream EOF mid-run used to fall through to wait_until_complete,
    whose timeout then killed a healthy training sandbox.
    """
    monkeypatch.setattr("verl_taubench.sandbox.trainer._time.sleep", lambda s: None)
    FakeSandbox.alive_status_polls = 2
    config = _config(sample_project)
    launcher = GpuTrainerLauncher(config)
    lines: list = []

    assert launcher.launch_and_wait(log_sink=lines.append) == 0

    sandbox = FakeSandbox.instances[-1]
    follows = [c for c in sandbox.calls if c == "stream_logs"]
    # 1 initial + 2 reattaches while status stayed "running".
    assert len(follows) == 3
    assert sandbox.calls.count("get_status") >= 3
    assert not sandbox.stopped, "a reattaching launcher must not stop the sandbox"


def test_ctrl_c_stops_sandbox_and_reaps_the_env_pool(fake_sdk, sample_project, monkeypatch):
    """Ctrl+C must never orphan the CPU pool.

    The pool is created from inside the trainer sandbox, so stopping that
    sandbox leaves its sandboxes running unless the host reaps them by tag.
    """
    reaped: list = []
    monkeypatch.setattr(
        "verl_taubench.sandbox.trainer._reap_env_pool", lambda tag: reaped.append(tag)
    )
    FakeSandbox.wait_base_exception = KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        GpuTrainerLauncher(_config(sample_project)).launch_and_wait(log_sink=lambda _l: None)

    sandbox = FakeSandbox.instances[-1]
    assert sandbox.stopped is True, "the trainer sandbox must be stopped on interrupt"
    assert len(reaped) == 1, "the env pool must be reaped exactly once"
    assert reaped[0].startswith("verl-taubench-env-")
    assert sandbox.kwargs["environment_variables"]["TAUBENCH_ENV_TAG"] == reaped[0]


def test_env_pool_is_reaped_even_when_training_succeeds(fake_sdk, sample_project, monkeypatch):
    reaped: list = []
    monkeypatch.setattr(
        "verl_taubench.sandbox.trainer._reap_env_pool", lambda tag: reaped.append(tag)
    )

    GpuTrainerLauncher(_config(sample_project)).launch_and_wait(log_sink=lambda _l: None)

    assert len(reaped) == 1


def test_trainer_sandbox_carries_the_stable_recipe_tag(fake_sdk, sample_project) -> None:
    """Every sandbox is findable by tag after a crash loses the run id."""
    GpuTrainerLauncher(_config(sample_project)).launch_and_wait(log_sink=lambda _l: None)
    assert RECIPE_TAG in FakeSandbox.instances[-1].kwargs["tags"]
