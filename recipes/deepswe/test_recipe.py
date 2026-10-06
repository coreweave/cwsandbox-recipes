import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from pier.models.task.config import EnvironmentConfig
from pier.models.trial.paths import TrialPaths

from cwsandbox_environment import CWSandboxEnvironment


def environment(tmp_path, image="example.invalid/task:v1.1"):
    return CWSandboxEnvironment(
        environment_dir=tmp_path,
        environment_name="task",
        session_id="trial",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=EnvironmentConfig(
            docker_image=image, cpus=2, memory_mb=8192, allow_internet=False
        ),
    )


def test_agent_image_does_not_upload_hidden_tests(tmp_path, monkeypatch):
    env = environment(tmp_path)
    sandbox = Mock(sandbox_id="test-id")
    run = Mock(return_value=sandbox)
    monkeypatch.setattr("cwsandbox_environment.Sandbox.run", run)
    env.exec = AsyncMock()
    env.upload_file = AsyncMock()
    asyncio.run(env.start())
    assert run.call_args.kwargs["network"].deny_egress
    assert run.call_args.kwargs["placement_mode"] == "serverless"
    assert run.call_args.kwargs["auth"] == "coreweave_api_key"
    assert run.call_args.kwargs["environment_variables"] == {}
    env.upload_file.assert_not_called()


def verifier_dockerfile(tmp_path, extra=""):
    lines = ["FROM example.invalid/task:v1.1"]
    lines += [
        f"COPY {name} /tests/{name}"
        for name in ["test.sh", "test.patch", "grader.py", "config.json"]
    ]
    lines += ["RUN chmod +x /tests/test.sh"]
    (tmp_path / "Dockerfile").write_text("\n".join(lines) + extra)


def test_verifier_copies_only_declared_files(tmp_path, monkeypatch):
    verifier_dockerfile(tmp_path)
    env = environment(tmp_path, image=None)
    monkeypatch.setattr("cwsandbox_environment.Sandbox.run", Mock(return_value=Mock()))
    env.exec = AsyncMock()
    env.upload_file = AsyncMock()
    asyncio.run(env.start())
    assert env.task_env_config.docker_image == "example.invalid/task:v1.1"
    assert [c.args[1] for c in env.upload_file.call_args_list] == [
        "/tests/test.sh",
        "/tests/test.patch",
        "/tests/grader.py",
        "/tests/config.json",
    ]


def test_verifier_never_silently_skips_build_instructions(tmp_path):
    verifier_dockerfile(tmp_path, "\nRUN install-more-dependencies\n")
    with pytest.raises(ValueError, match="Unsupported verifier Dockerfile"):
        environment(tmp_path, image=None)


def test_exec_keeps_shell_metacharacters_in_environment_values(tmp_path):
    env = environment(tmp_path)
    env.sandbox = Mock()
    env.sandbox.exec.return_value.result.return_value = SimpleNamespace(
        stdout="ok", stderr="", returncode=0
    )
    asyncio.run(env.exec("printf ok", env={"VALUE": "$(touch /bad); `id` '"}))
    argv = env.sandbox.exec.call_args.args[0]
    assert argv == ["env", "VALUE=$(touch /bad); `id` '", "bash", "-c", "printf ok"]


def test_stop_can_be_called_twice(tmp_path):
    env = environment(tmp_path)
    sandbox = Mock()
    env.sandbox = sandbox
    asyncio.run(env.stop())
    asyncio.run(env.stop())
    sandbox.stop.assert_called_once_with(missing_ok=True)


def test_remote_shell_never_exposes_host_environment(tmp_path, monkeypatch):
    from host_agent import RemoteShell

    monkeypatch.setenv("WANDB_API_KEY", "not-a-real-key")
    shell = RemoteShell(None, None)
    assert "WANDB_API_KEY" not in shell.get_template_vars()
    assert "not-a-real-key" not in str(shell.serialize())


def test_shell_timeout_is_returned_to_the_agent():
    from cwsandbox.exceptions import SandboxTimeoutError

    from host_agent import RemoteShell

    async def scenario():
        remote = SimpleNamespace(
            exec=AsyncMock(side_effect=SandboxTimeoutError("timeout"))
        )
        shell = RemoteShell(remote, asyncio.get_running_loop())
        return await asyncio.to_thread(shell.execute, {"command": "sleep 999"})

    output = asyncio.run(scenario())
    assert output["returncode"] == -1
    assert "timed out" in output["exception_info"]


def test_infrastructure_failure_is_not_a_model_observation():
    from cwsandbox.exceptions import SandboxNotRunningError

    from host_agent import RemoteShell

    async def scenario():
        remote = SimpleNamespace(
            exec=AsyncMock(side_effect=SandboxNotRunningError("gone"))
        )
        shell = RemoteShell(remote, asyncio.get_running_loop())
        return await asyncio.to_thread(shell.execute, {"command": "true"})

    with pytest.raises(SandboxNotRunningError):
        asyncio.run(scenario())


def test_archive_download_rejects_path_traversal(tmp_path):
    import io
    import tarfile

    env = environment(tmp_path)
    env.exec = AsyncMock(return_value=SimpleNamespace(return_code=0))

    async def malicious_archive(source, destination):
        with tarfile.open(destination, "w:gz") as tar:
            item = tarfile.TarInfo("../escaped")
            item.size = 4
            tar.addfile(item, io.BytesIO(b"oops"))

    env.download_file = malicious_archive
    with pytest.raises(tarfile.OutsideDestinationError):
        asyncio.run(env.download_dir("/logs/agent", tmp_path / "output"))
    assert not (tmp_path / "escaped").exists()


@pytest.mark.parametrize(
    "errors,expected_executions",
    [
        (["RateLimitError"] * 3, 3),
        (["RateLimitError", None], 2),
        (["AuthenticationError"], 1),
        ([None], 1),
    ],
)
def test_pier_retries_only_eligible_errors(
    tmp_path, monkeypatch, errors, expected_executions
):
    from pier.models.trial.config import TaskConfig, TrialConfig
    from pier.trial.hooks import TrialEvent
    from pier.trial.queue import TrialQueue

    from run import build_config, parse_args, preserve_failed_attempt

    monkeypatch.setenv("CWSANDBOX_API_KEY", "not-a-real-key")
    config = build_config(parse_args(["--tasks", str(tmp_path), "--agent", "nop"]))
    config.retry.min_wait_sec = 0
    executions = []

    class FakeTrial:
        def __init__(self, trial_config):
            self.config = trial_config
            self.trial_dir = trial_config.trials_dir / trial_config.trial_name
            self.hooks = []

        @classmethod
        async def create(cls, trial_config):
            return cls(trial_config)

        def add_hook(self, event, hook):
            if event == TrialEvent.END:
                self.hooks.append(hook)

        async def run(self):
            executions.append(len(executions) + 1)
            self.trial_dir.mkdir()
            (self.trial_dir / "diagnostic.txt").write_text(str(len(executions)))
            error = errors[len(executions) - 1]
            result = SimpleNamespace(
                id=str(len(executions)),
                exception_info=SimpleNamespace(exception_type=error) if error else None,
            )
            for hook in self.hooks:
                await hook(SimpleNamespace(config=self.config, result=result))
            return result

    monkeypatch.setattr("pier.trial.trial.Trial", FakeTrial)
    queue = TrialQueue(n_concurrent=1, retry_config=config.retry)
    queue.on_trial_ended(preserve_failed_attempt)
    trial = TrialConfig(
        task=TaskConfig(path=tmp_path), trial_name="test-trial", trials_dir=tmp_path
    )
    asyncio.run(queue.submit(trial))
    assert len(executions) == expected_executions
    archived = list((tmp_path / "failed-attempts").rglob("diagnostic.txt"))
    assert {p.read_text() for p in archived} == {
        str(n) for n in executions if errors[n - 1]
    }


def test_pier_backoff_is_bounded(tmp_path, monkeypatch):
    from pier.trial.queue import TrialQueue

    from run import build_config, parse_args

    monkeypatch.setenv("CWSANDBOX_API_KEY", "not-a-real-key")
    config = build_config(
        parse_args(
            [
                "--tasks",
                str(tmp_path),
                "--agent",
                "nop",
                "--concurrency",
                "2",
            ]
        )
    )
    assert config.n_concurrent_trials == 2
    queue = TrialQueue(n_concurrent=2, retry_config=config.retry)
    assert [queue._calculate_backoff_delay(n) for n in range(4)] == [30, 60, 60, 60]


def test_parallel_agent_steps_leave_threads_for_remote_shell(monkeypatch):
    import threading

    from pier.models.job.config import JobConfig

    import run

    async def scenario():
        loop = asyncio.get_running_loop()
        barrier = threading.Barrier(40, timeout=5)

        def agent_step():
            barrier.wait()
            return asyncio.run_coroutine_threadsafe(
                asyncio.to_thread(lambda: "remote command completed"), loop
            ).result(timeout=5)

        class FakeJob:
            def on_trial_ended(self, hook):
                pass

            async def run(self):
                return await asyncio.gather(
                    *[asyncio.to_thread(agent_step) for _ in range(40)]
                )

        monkeypatch.setattr(run.Job, "create", AsyncMock(return_value=FakeJob()))
        return await run.execute(JobConfig(n_concurrent_trials=40))

    assert asyncio.run(scenario()) == ["remote command completed"] * 40
