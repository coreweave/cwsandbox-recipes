"""Check sandbox cleanup with SDK failures and real process signals."""

import json
from pathlib import Path
import signal
import subprocess
import sys
import textwrap
from unittest.mock import Mock

import pytest

from nemo_rl_sandbox import lifecycle


@pytest.fixture
def sandbox(monkeypatch):
    sandbox = Mock(sandbox_id="test-owned-sandbox")
    monkeypatch.setattr(lifecycle.Sandbox, "run", Mock(return_value=sandbox))
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    yield sandbox
    assert {sig: signal.getsignal(sig) for sig in previous} == previous


@pytest.mark.parametrize("body_fails", [False, True])
def test_normal_and_error_exit_stop_owned_sandbox(tmp_path, sandbox, body_fails):
    original = RuntimeError("training failed")
    caught = None
    try:
        with lifecycle.managed_sandbox(tmp_path, "sleep", "infinity", tags=["test"]):
            assert (
                tmp_path / "sandbox-id.txt"
            ).read_text().strip() == sandbox.sandbox_id
            if body_fails:
                raise original
    except RuntimeError as error:
        caught = error
    assert caught is (original if body_fails else None)
    sandbox.stop.assert_called_once_with(missing_ok=True)
    sandbox.stop.return_value.result.assert_called_once_with(timeout=60)
    lifecycle.Sandbox.run.assert_called_once_with("sleep", "infinity", tags=["test"])
    assert json.loads((tmp_path / "cleanup.json").read_text()) == {
        "sandbox_id": sandbox.sandbox_id,
        "stopped": True,
    }


@pytest.mark.parametrize("body_fails", [False, True])
def test_cleanup_failure_records_uncertainty_and_preserves_original(
    tmp_path,
    sandbox,
    body_fails,
    capsys,
):
    original = RuntimeError("training failed")
    cleanup_failure = TimeoutError("stop deadline exceeded")
    sandbox.stop.return_value.result.side_effect = cleanup_failure
    with pytest.raises(Exception) as caught:
        with lifecycle.managed_sandbox(tmp_path, "sleep", "infinity"):
            if body_fails:
                raise original
    assert caught.value is (original if body_fails else cleanup_failure)
    assert json.loads((tmp_path / "cleanup.json").read_text())["stopped"] is False
    assert f"scripts/stop_sandbox.py {sandbox.sandbox_id}" in capsys.readouterr().err


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize("phase", ["create", "body"])
def test_real_signals_stop_sandbox_and_defer_creation_interrupt(
    tmp_path, signum, phase
):
    # Isolate signals in a child so a regression cannot terminate the test runner.
    code = textwrap.dedent("""
        import json
        import os
        from pathlib import Path
        import signal
        import sys
        from unittest.mock import patch
        from nemo_rl_sandbox.lifecycle import managed_sandbox

        output, signum, phase = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        events = []

        class Stopped:
            def result(self, timeout):
                assert timeout == 60
                # Repeated interrupts must not interrupt the shutdown request.
                os.kill(os.getpid(), signal.SIGINT)
                os.kill(os.getpid(), signal.SIGTERM)
                events.append("stop-completed")

        class Sandbox:
            sandbox_id = "test-interrupted-sandbox"

            def stop(self, missing_ok):
                assert missing_ok
                assert (output / "sandbox-id.txt").read_text().strip() == self.sandbox_id
                events.append("stop-started")
                return Stopped()

        def create(*command, **options):
            assert command == ("sleep", "infinity")
            events.append("create-started")
            if phase == "create":
                os.kill(os.getpid(), signum)
                # Creation must finish before the deferred interrupt is raised.
                assert not (output / "sandbox-id.txt").exists()
                events.append("create-finished-after-signal")
            return Sandbox()

        with patch("nemo_rl_sandbox.lifecycle.Sandbox.run", side_effect=create):
            try:
                with managed_sandbox(output, "sleep", "infinity"):
                    events.append("body-entered")
                    assert phase == "body"
                    os.kill(os.getpid(), signum)
                    raise AssertionError("signal failed to interrupt body")
            except SystemExit as error:
                assert error.code == 128 + signum
                events.append("interrupt-propagated")
            else:
                raise AssertionError("interrupt did not propagate")

        assert {sig: signal.getsignal(sig) for sig in previous} == previous
        (output / "events.json").write_text(json.dumps(events))
    """)
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), str(int(signum)), phase],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    middle = "create-finished-after-signal" if phase == "create" else "body-entered"
    assert json.loads((tmp_path / "events.json").read_text()) == [
        "create-started",
        middle,
        "stop-started",
        "stop-completed",
        "interrupt-propagated",
    ]
    assert json.loads((tmp_path / "cleanup.json").read_text()) == {
        "sandbox_id": "test-interrupted-sandbox",
        "stopped": True,
    }


def test_status_write_failure_preserves_training_error(
    tmp_path, sandbox, monkeypatch, capsys
):
    write_text = Path.write_text

    def fail_status(path, *args, **kwargs):
        if path.name == "cleanup.json":
            raise OSError("disk full")
        return write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_status)
    original = RuntimeError("training failed")
    with pytest.raises(RuntimeError) as caught:
        with lifecycle.managed_sandbox(tmp_path, "sleep", "infinity"):
            raise original
    assert caught.value is original
    sandbox.stop.return_value.result.assert_called_once_with(timeout=60)
    assert "Could not save cleanup status: disk full" in capsys.readouterr().err
