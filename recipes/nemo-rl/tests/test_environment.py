from unittest.mock import Mock, patch

import pytest

from nemo_rl_sandbox.environment import SandboxRewardClient


def test_score_is_argv_data_and_cleanup_runs_on_failure():
    sandbox = Mock(sandbox_id="test-reward-sandbox")
    sandbox.exec.return_value.result.return_value = Mock(
        returncode=1, stderr="grader failed"
    )
    with patch("nemo_rl_sandbox.environment.Sandbox.run", return_value=sandbox):
        with pytest.raises(RuntimeError, match="grader failed"):
            with SandboxRewardClient() as environment:
                environment.score(["$(touch /tmp/injected)"], ["4"])
    command = sandbox.exec.call_args.args[0]
    assert command[:2] == ["python", "/tmp/grader.py"]
    assert "$(touch /tmp/injected)" in command[2]
    sandbox.stop.assert_called_once_with(missing_ok=True)


def test_cpu_startup_failure_stops_sandbox():
    sandbox = Mock(sandbox_id="test-reward-sandbox")
    sandbox.write_file.return_value.result.side_effect = RuntimeError("startup failed")
    with patch("nemo_rl_sandbox.environment.Sandbox.run", return_value=sandbox):
        with pytest.raises(RuntimeError, match="startup failed"):
            SandboxRewardClient()
    sandbox.stop.assert_called_once_with(missing_ok=True)
