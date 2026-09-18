import json
import os
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import run


def record(**fields):
    result = SimpleNamespace(**fields)
    result.model_dump = lambda **_: fields
    return result


def test_upload_allowlist_excludes_secrets_and_verifier(tmp_path):
    for name in run.INPUTS:
        target = tmp_path / name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes((run.ROOT / name).read_bytes())
    (tmp_path / ".env").write_text("SECRET=private")
    (tmp_path / "verify.py").write_text("private verifier")
    assert set(run.inputs(tmp_path)) == set(run.INPUTS)
    (tmp_path / "TASK.md").unlink()
    (tmp_path / "TASK.md").symlink_to(tmp_path / ".env")
    with pytest.raises(ValueError, match="symlinked"):
        run.inputs(tmp_path)


def test_exact_remote_endpoint_preserved():
    url = "wss://codex-cloud-environments.chatgpt.com/unique/path?ticket=opaque"
    env = SimpleNamespace(
        type="self_hosted", workspace_directory=run.WORKSPACE, id="env_123", remote_url=url
    )
    assert run.environment(SimpleNamespace(environment=env)).remote_url == url
    sandbox = MagicMock()
    sandbox.exec.return_value.result.return_value.returncode = 0
    run.start_executor(sandbox, env, time.monotonic() + 10)
    command = sandbox.exec.call_args.args[0][-1]
    assert url in command
    env.remote_url = "https://api.openai.com@attacker.invalid/"
    with pytest.raises(ValueError, match="endpoint"):
        run.environment(SimpleNamespace(environment=env))


@pytest.mark.parametrize("stage", ["Sandbox bootstrap", "Executor launch"])
def test_setup_failure_identifies_stage_and_preserves_diagnostics(stage):
    sandbox = MagicMock()
    failed = SimpleNamespace(returncode=1, stdout="install output", stderr="package unavailable")
    sandbox.exec.return_value.result.side_effect = (
        [failed] if stage == "Sandbox bootstrap" else [SimpleNamespace(returncode=0), failed]
    )
    env = SimpleNamespace(remote_url="wss://api.openai.com/test", id="env_test")
    with pytest.raises(RuntimeError) as error:
        run.start_executor(sandbox, env, time.monotonic() + 10)
    message = str(error.value)
    assert f"{stage} failed (exit code 1)" in message
    assert "install output" in message and "package unavailable" in message
    assert "inspect executor.log" not in message


def test_setup_diagnostic_redacts_before_truncating(monkeypatch):
    for name in run.KEYS:
        monkeypatch.setenv(name, f"test-secret-{name}")
    secrets = " ".join(os.environ[name] for name in run.KEYS)
    sandbox = MagicMock()
    sandbox.exec.return_value.result.return_value = SimpleNamespace(
        returncode=17,
        stdout="old output" + "x" * 3000 + secrets + "y" * 1995,
        stderr=secrets + "\ninstallation failed",
    )
    with pytest.raises(RuntimeError) as error:
        run.execute(sandbox, ["test"], time.monotonic() + 5)
    message = str(error.value)
    assert "exit code 17" in message and "installation failed" in message
    assert "old output" not in message
    assert "[REDACTED]" in message
    assert all(os.environ[name] not in message for name in run.KEYS)
    stdout = message.split("stdout:\n", 1)[1].split("\nstderr:", 1)[0]
    assert len(stdout) == 2000
    assert stdout == "CTED]" + "y" * 1995


def test_setup_failure_with_empty_output():
    sandbox = MagicMock()
    sandbox.exec.return_value.result.return_value = SimpleNamespace(
        returncode=2, stdout="", stderr=""
    )
    with pytest.raises(RuntimeError, match="stdout:\n\\(empty\\)\nstderr:\n\\(empty\\)"):
        run.execute(sandbox, ["test"], time.monotonic() + 5)


def test_cleanup_compute_after_api_delete_failure(tmp_path):
    client = MagicMock()
    client.beta.agents.sessions.delete.side_effect = RuntimeError("API unavailable")
    sandbox = MagicMock(sandbox_id="sb_1")
    journal = {"session_id": "sess_1", "sandbox_id": "sb_1", "sandbox_tag": "recipe-test"}
    with patch.object(run.Sandbox, "list") as listing:
        listing.return_value.result.return_value = [sandbox]
        errors = run.cleanup(client, journal, tmp_path)
    sandbox.stop.assert_called_once_with(missing_ok=True)
    assert errors == ["API session: RuntimeError"]
    assert json.loads((tmp_path / "run.json").read_text())["sandbox_stopped"] is True


def test_cleanup_api_after_compute_failure(tmp_path):
    client = MagicMock()
    journal = {"session_id": "sess_1", "sandbox_tag": "recipe-test"}
    with patch.object(run.Sandbox, "list", side_effect=RuntimeError("unavailable")):
        errors = run.cleanup(client, journal, tmp_path)
    client.beta.agents.sessions.delete.assert_called_once_with("sess_1")
    assert errors == ["CWS sandbox: RuntimeError"]
    assert journal["session_deleted"] is True


def test_collects_child_histories_and_all_pages(tmp_path):
    client = MagicMock()
    sessions = client.beta.agents.sessions
    sessions.turns.list.return_value = iter([record(id="root", subagent_id=None)])
    sessions.items.list.return_value = iter([record(id="root-item")])
    sessions.subagents.list.return_value = iter([record(id="child")])
    sessions.subagents.turns.list.return_value = iter(
        [record(id=f"child-turn-{i}", subagent_id="child") for i in range(120)]
    )
    sessions.subagents.items.list.return_value = iter(
        [record(id="child-command", turn_id="child-turn-119", type="command_execution")]
    )
    run.collect_evidence(client, "session", tmp_path, time.monotonic() + 5)
    assert len(json.loads((tmp_path / "turns.json").read_text())) == 121
    assert json.loads((tmp_path / "items.json").read_text())[-1]["id"] == "child-command"
    sessions.subagents.items.list.assert_called_once_with(
        "child",
        session_id="session",
        order="asc",
        limit=100,
    )


def test_turn_ignores_previous_root_and_subagent_completion(tmp_path):
    client = MagicMock()
    sessions = client.beta.agents.sessions
    sessions.turns.list.return_value = [record(id="old-root")]
    events = [
        record(type="agent.session.turn.completed", turn={"id": "old-root"}),
        record(type="agent.session.turn.completed", turn={"id": "child", "subagent_id": "child"}),
        record(type="agent.session.turn.completed", turn={"id": "new-root", "subagent_id": None}),
    ]
    sessions.events.stream.return_value.__enter__.return_value = iter(events)
    turn_id = run.run_turn(
        client, "sess_1", "prompt", tmp_path, time.monotonic() + 5, time.monotonic()
    )
    assert turn_id == "new-root"
    saved = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert len(saved) == 3
    assert all("received_at" in item and "elapsed_seconds" in item for item in saved)


def test_silent_stream_deadline():
    def silent():
        time.sleep(0.2)
        yield record(type="irrelevant")

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        next(run.events_until(silent(), started + 0.02))
    assert time.monotonic() - started < 0.15


def test_existing_output_rejected_before_provisioning(tmp_path):
    client = MagicMock()
    with patch.object(run, "inputs", return_value={}):
        with pytest.raises(FileExistsError):
            run.run(SimpleNamespace(output_dir=tmp_path, model="test"), client)
    client.beta.agents.sessions.create.assert_not_called()


def test_logs_redact_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_EXECUTOR_API_KEY", "test-secret-not-real")
    run.dump(tmp_path / "run.json", {"message": "test-secret-not-real"})
    assert "test-secret-not-real" not in (tmp_path / "run.json").read_text()


def test_completed_cleanup_is_idempotent(tmp_path):
    client = MagicMock()
    journal = {
        "session_id": "sess_1",
        "sandbox_tag": "recipe-test",
        "session_deleted": True,
        "sandbox_stopped": True,
    }
    with patch.object(run.Sandbox, "list") as listing:
        assert run.cleanup(client, journal, tmp_path) == []
    listing.assert_not_called()
    client.beta.agents.sessions.delete.assert_not_called()


def test_failed_sandbox_creation_still_deletes_api_session(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_EXECUTOR_API_KEY", "test-placeholder")
    client = MagicMock()
    session = record(
        id="sess_test",
        environment=SimpleNamespace(
            type="self_hosted",
            workspace_directory=run.WORKSPACE,
            id="env_test",
            remote_url="wss://codex-cloud-environments.chatgpt.com/exact-path",
        ),
    )
    client.beta.agents.sessions.create.return_value = session
    journal_path = tmp_path / "new"
    with (
        patch.object(run, "inputs", return_value={}),
        patch.object(run.Sandbox, "run", side_effect=RuntimeError("creation failed")),
        patch.object(run.Sandbox, "list") as listing,
    ):
        listing.return_value.result.return_value = []
        with pytest.raises(RuntimeError, match="creation failed"):
            run.run(SimpleNamespace(output_dir=journal_path, model="test"), client)
    client.beta.agents.sessions.delete.assert_called_once_with("sess_test")
    creation = client.beta.agents.sessions.create.call_args.kwargs
    assert creation["agent"]["multi_agent"] == {
        "enabled": True,
        "max_concurrent_subagents": 3,
    }
    assert json.loads((journal_path / "run.json").read_text())["session_deleted"]


def test_cleanup_attempts_both_resources_when_journal_write_fails(tmp_path):
    client = MagicMock()
    sandbox = MagicMock(sandbox_id="sb_1")
    journal = {"session_id": "sess_1", "sandbox_id": "sb_1", "sandbox_tag": "recipe-test"}
    with (
        patch.object(run.Sandbox, "list") as listing,
        patch.object(run, "dump", side_effect=OSError("disk full")),
    ):
        listing.return_value.result.return_value = [sandbox]
        errors = run.cleanup(client, journal, tmp_path)
    client.beta.agents.sessions.delete.assert_called_once_with("sess_1")
    sandbox.stop.assert_called_once_with(missing_ok=True)
    assert any("Cleanup journal" in item for item in errors)


def test_env_example_values_are_empty():
    from dotenv import dotenv_values

    assert dotenv_values(run.ROOT / ".env.example") == {name: "" for name in run.KEYS}


def test_malformed_fixtures_rejected():
    payloads = run.inputs()
    payloads["fixtures/requests.csv"] = b"wrong,column\n1,2\n"
    with pytest.raises(ValueError, match="columns"):
        run.validate_inputs(payloads)
    payloads = run.inputs()
    payloads["fixtures/deployments.json"] = b"not json"
    with pytest.raises(ValueError):
        run.validate_inputs(payloads)


@pytest.mark.parametrize("auth_mode", ["coreweave", "wandb"])
@pytest.mark.parametrize("sandbox_mode", [None, "serverless", "cks"])
def test_full_mocked_run_orchestration(tmp_path, monkeypatch, auth_mode, sandbox_mode):
    monkeypatch.setenv("OPENAI_EXECUTOR_API_KEY", "test-executor-placeholder")
    monkeypatch.setenv("OPENAI_API_KEY", "test-application-placeholder")
    client = MagicMock()
    session = record(
        id="sess_test",
        environment=SimpleNamespace(
            type="self_hosted",
            workspace_directory=run.WORKSPACE,
            id="env_test",
            remote_url="wss://codex-cloud-environments.chatgpt.com/exact-path",
        ),
    )
    client.beta.agents.sessions.create.return_value = session
    client.beta.agents.environments.retrieve.return_value.status = "connected"
    sandbox = MagicMock(sandbox_id="sandbox_test")
    sandbox.exec.return_value.result.return_value.returncode = 0
    sandbox.read_file.return_value.result.return_value = b"synthetic result"
    output = tmp_path / "new"
    order = []
    with (
        patch.object(run.Sandbox, "run", return_value=sandbox) as create_sandbox,
        patch.object(run.Sandbox, "list") as listing,
        patch.object(run, "run_turn", side_effect=lambda *a: order.append(a[2]) or "turn"),
        patch.object(run, "collect_evidence"),
    ):
        listing.return_value.result.return_value = [sandbox]
        run.run(
            SimpleNamespace(
                output_dir=output, model="test", sandbox_auth=auth_mode, sandbox_mode=sandbox_mode
            ),
            client,
        )
    assert len(order) == 2
    assert order[0] == (run.ROOT / "TASK.md").read_text()
    assert order[1] == run.FOLLOWUP
    assert create_sandbox.call_args.kwargs["environment_variables"] == {
        "CODEX_API_KEY": "test-executor-placeholder",
        "HOME": "/workspace/home",
    }
    assert create_sandbox.call_args.kwargs["max_lifetime_seconds"] == 1200
    assert create_sandbox.call_args.kwargs["auth"] == run.AUTH_MODES[auth_mode]
    assert create_sandbox.call_args.kwargs["placement_mode"] == sandbox_mode
    assert create_sandbox.call_args.kwargs["placement_spillover"] == "strict"
    assert listing.call_args.kwargs["auth"] == run.AUTH_MODES[auth_mode]
    assert {call.args[0] for call in sandbox.write_file.call_args_list} == {
        f"{run.WORKSPACE}/{name}" for name in run.INPUTS
    }
    assert {path.name for path in (output / "output").iterdir()} == set(run.ARTIFACTS)
    client.beta.agents.sessions.delete.assert_called_once_with("sess_test")
    sandbox.stop.assert_called_once_with(missing_ok=True)
    journal = json.loads((output / "run.json").read_text())
    assert journal["status"] == "completed" and journal["cleanup_errors"] == []
    assert journal["sandbox_auth"] == auth_mode
    assert journal["sandbox_mode"] == sandbox_mode


def test_wandb_cleanup_uses_saved_mode_for_recovery(tmp_path):
    client = MagicMock()
    journal = {
        "session_id": "sess_1",
        "sandbox_id": "sb_1",
        "sandbox_tag": "recipe-test",
        "sandbox_auth": "wandb",
    }
    sandbox = MagicMock()
    with (
        patch.object(run.Sandbox, "list") as listing,
        patch.object(run.Sandbox, "from_id") as lookup,
    ):
        listing.return_value.result.return_value = []
        lookup.return_value.result.return_value = sandbox
        assert run.cleanup(client, journal, tmp_path) == []
    listing.assert_called_once_with(tags=["recipe-test"], auth=run.AuthStrategy.WANDB)
    lookup.assert_called_once_with("sb_1", auth=run.AuthStrategy.WANDB)
    sandbox.stop.assert_called_once_with(missing_ok=True)


def test_wandb_preflight_does_not_require_coreweave_key(tmp_path, monkeypatch):
    for name in run.KEYS:
        monkeypatch.delenv(name, raising=False)
    for name in ("WANDB_API_KEY", "OPENAI_API_KEY", "OPENAI_EXECUTOR_API_KEY"):
        monkeypatch.setenv(name, "test-placeholder")
    with patch.object(run, "OpenAI"), patch.object(run, "run") as execute:
        assert run.main(["run", "--sandbox-auth", "wandb", "--output-dir", str(tmp_path)]) == 0
    assert execute.call_args.args[0].sandbox_auth == "wandb"


def test_wrong_auth_key_fails_before_provisioning(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("CWSANDBOX_API_KEY", "test-placeholder")
    with patch.object(run, "OpenAI") as client, pytest.raises(SystemExit):
        run.main(["run", "--sandbox-auth", "wandb", "--output-dir", str(tmp_path)])
    client.assert_not_called()


@pytest.mark.parametrize("command", ["run", "cleanup"])
@pytest.mark.parametrize("auth_mode", ["coreweave", "wandb"])
def test_explicit_env_file_overrides_exported_credentials(
    tmp_path, monkeypatch, command, auth_mode
):
    for name in run.KEYS:
        monkeypatch.setenv(name, f"exported-{name}")
    env_file = tmp_path / ".env"
    expected = {name: f"file-{name}" for name in run.KEYS}
    env_file.write_text("\n".join(f"{name}={value}" for name, value in expected.items()))
    (tmp_path / "run.json").write_text(json.dumps({"sandbox_auth": auth_mode}))
    argv = [command, "--env-file", str(env_file), "--output-dir", str(tmp_path)]
    if command == "run":
        argv.extend(["--sandbox-auth", auth_mode])
    with (
        patch.object(run, "OpenAI") as client,
        patch.object(run, "run"),
        patch.object(run, "cleanup", return_value=[]),
    ):
        # Check credentials at client creation, before any cloud operation.
        def check_credentials(**_):
            assert {name: os.environ[name] for name in run.KEYS} == expected
            return MagicMock()

        client.side_effect = check_credentials
        assert run.main(argv) == 0
        client.assert_called_once()


def test_env_file_preserves_exports_for_omitted_variables(tmp_path, monkeypatch):
    for name in run.KEYS:
        monkeypatch.setenv(name, f"exported-{name}")
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=file-application-key\n")
    with patch.object(run, "OpenAI"), patch.object(run, "run"):
        assert run.main(["run", "--env-file", str(env_file), "--output-dir", str(tmp_path)]) == 0
    assert os.environ["OPENAI_API_KEY"] == "file-application-key"
    for name in set(run.KEYS) - {"OPENAI_API_KEY"}:
        assert os.environ[name] == f"exported-{name}"


def test_blank_env_file_key_does_not_fall_back_to_export(tmp_path, monkeypatch, capsys):
    for name in run.KEYS:
        monkeypatch.setenv(name, f"exported-{name}")
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=\n")
    with patch.object(run, "OpenAI") as client, pytest.raises(SystemExit) as error:
        run.main(["run", "--env-file", str(env_file), "--output-dir", str(tmp_path)])
    assert error.value.code == 2
    assert "Missing environment variables: OPENAI_API_KEY" in capsys.readouterr().err
    client.assert_not_called()
