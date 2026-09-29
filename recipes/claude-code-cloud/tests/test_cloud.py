import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

SPEC = importlib.util.spec_from_file_location("cloud", Path(__file__).parents[1] / "cloud.py")
cloud = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cloud)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    config = {
        "environment": "ccpool_test",
        "tag": "claude-cloud:test",
        "max_workers": 2,
        "worker_hours": 8,
        "idle_minutes": 15,
    }
    token = tmp_path / "order.jwt"
    token.write_text("test.one-use.token")
    for key, value in {
        "CLAUDE_RUNNER_ORDER_ID": "order-1",
        "CLAUDE_RUNNER_POOL_ID": "ccpool_test",
        "CLAUDE_RUNNER_WORK_ORDER_FILE": str(token),
        "CLAUDE_RUNNER_SESSION_ID": "session_test",
        "CWSANDBOX_API_KEY": "control-key",
        cloud.SECRET: "environment-key",
    }.items():
        monkeypatch.setenv(key, value)
    listing = Mock(return_value=[])
    submit = Mock(return_value=SimpleNamespace(sandbox_id="worker-1"))
    monkeypatch.setattr(cloud, "active", listing)
    monkeypatch.setattr(cloud, "run_box", submit)
    return config, tmp_path / "state", listing, submit


def test_order_replay_never_submits_twice(setup):
    config, directory, _, submit = setup
    assert cloud.spawn(config, directory) == 0
    assert cloud.spawn(config, directory) == 0
    submit.assert_called_once()
    record = json.loads(next((directory / "orders").glob("*.json")).read_text())
    assert record == {"sandbox_id": "worker-1", "session_id": "session_test"}
    assert "token" not in json.dumps(record)


def test_worker_receives_only_scoped_credential(setup):
    config, directory, _, submit = setup
    assert cloud.spawn(config, directory) == 0
    env = submit.call_args.args[2]
    assert env[cloud.SECRET] == "test.one-use.token"
    assert "CWSANDBOX_API_KEY" not in env
    assert "environment-key" not in str(submit.call_args)


def test_uncertain_create_fails_closed_even_after_replay(setup):
    config, directory, _, submit = setup
    submit.side_effect = TimeoutError("secret request body")
    assert cloud.spawn(config, directory) == 2
    assert cloud.spawn(config, directory) == 2
    submit.assert_called_once()


def test_capacity_failure_can_retry(setup):
    config, directory, listing, submit = setup
    listing.return_value = [object(), object()]
    assert cloud.spawn(config, directory) == 1
    submit.assert_not_called()
    listing.return_value = []
    assert cloud.spawn(config, directory) == 0


def test_failed_list_does_not_claim_order(setup):
    config, directory, listing, submit = setup
    listing.side_effect = TimeoutError()
    assert cloud.spawn(config, directory) == 1
    submit.assert_not_called()
    assert not list((directory / "orders").glob("*.json"))


@pytest.mark.parametrize(
    "variable,value",
    [
        ("CLAUDE_RUNNER_POOL_ID", "ccpool_wrong"),
        ("CLAUDE_RUNNER_ORDER_ID", ""),
        ("CLAUDE_RUNNER_WORK_ORDER_FILE", "/missing"),
    ],
)
def test_invalid_orders_never_create(setup, monkeypatch, variable, value):
    config, directory, _, submit = setup
    monkeypatch.setenv(variable, value)
    assert cloud.spawn(config, directory) == 2
    submit.assert_not_called()


def test_ambiguous_deployment_adopts_exactly_one_host(tmp_path, monkeypatch):
    path = tmp_path / "deployment.json"
    cloud.save(path, {"config": {"tag": "test"}})
    monkeypatch.setattr(cloud, "active", lambda tag: [SimpleNamespace(sandbox_id="host-1")])
    assert cloud.deployment(path)["sandbox_id"] == "host-1"
    assert json.loads(path.read_text())["sandbox_id"] == "host-1"


def test_stop_waits_for_host_before_enumerating_workers(tmp_path, monkeypatch):
    events = []
    host, worker = Mock(), Mock()
    host.stop.side_effect = lambda **kw: events.append("stop host") or Mock()
    host.wait_until_complete.side_effect = lambda **kw: events.append("wait host") or Mock()
    worker.stop.side_effect = lambda **kw: events.append("stop worker") or Mock()
    worker.wait_until_complete.side_effect = lambda **kw: events.append("wait worker") or Mock()
    monkeypatch.setattr(cloud, "get_box", lambda _: host)

    def listing(tag):
        events.append("list " + tag)
        return [worker] if tag.endswith(":worker") else []

    monkeypatch.setattr(cloud, "active", listing)
    path = tmp_path / "deployment.json"
    path.touch()
    cloud.stop(SimpleNamespace(state=path), {"sandbox_id": "host", "config": {"tag": "test"}})
    assert events == [
        "stop host",
        "wait host",
        "list test:worker",
        "stop worker",
        "wait worker",
        "list test",
    ]
    assert not path.exists()


def test_worker_selection_rejects_other_deployments(monkeypatch):
    monkeypatch.setattr(cloud, "active", lambda _: [SimpleNamespace(sandbox_id="ours")])
    with pytest.raises(cloud.RecipeError):
        cloud.select_worker({"config": {"tag": "test"}}, "theirs")


def test_entrypoint_does_not_retry_failed_workload(tmp_path):
    import subprocess
    import sys

    marker = tmp_path / "attempts"
    command = (
        "from pathlib import Path; p=Path("
        + repr(str(marker))
        + "); p.write_text(p.read_text()+'x' if p.exists() else 'x'); raise SystemExit(7)"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(Path(cloud.__file__).with_name("entrypoint.py")),
            sys.executable,
            "-c",
            command,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert marker.read_text() == "x"
    assert "code 7" in result.stdout


def test_parallel_deploy_rejected_before_api_call(tmp_path, monkeypatch):
    import fcntl
    import sys

    state = tmp_path / "deployment.json"
    monkeypatch.setenv("CWSANDBOX_API_KEY", "test-control-key")
    monkeypatch.setattr(
        sys, "argv", ["cloud.py", "--state", str(state), "deploy", "--environment", "ccpool_test"]
    )
    submit = Mock()
    monkeypatch.setattr(cloud, "run_box", submit)
    with state.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(cloud.RecipeError, match="Another deployment operation"):
            cloud.main()
    submit.assert_not_called()
    assert not state.exists()


@pytest.mark.parametrize("missing", [False, True])
def test_read_reports_file_errors_without_sdk_details(tmp_path, monkeypatch, capsys, missing):
    import sys

    state = tmp_path / "deployment.json"
    cloud.save(state, {"sandbox_id": "host", "config": {"tag": "test"}})
    monkeypatch.setenv("CWSANDBOX_API_KEY", "test-control-key")
    monkeypatch.setattr(
        sys,
        "argv",
        ["cloud.py", "--state", str(state), "read", "--worker", "ours", "--path", "/proof"],
    )
    box = Mock()
    monkeypatch.setattr(cloud, "get_box", lambda _: box)
    monkeypatch.setattr(cloud, "active", lambda _: [SimpleNamespace(sandbox_id="ours")])
    if missing:
        box.read_file.return_value.result.side_effect = cloud.SandboxFileError(
            "sensitive SDK request details", filepath="/proof"
        )
        with pytest.raises(cloud.RecipeError, match="Check that the path exists") as error:
            cloud.main()
        assert "sensitive" not in str(error.value)
        assert error.value.__suppress_context__
        assert capsys.readouterr().out == ""
    else:
        box.read_file.return_value.result.return_value = b"CWS_CLOUD_OK\n"
        cloud.main()
        assert capsys.readouterr().out == "CWS_CLOUD_OK\n"
    box.read_file.assert_called_once_with("/proof")
