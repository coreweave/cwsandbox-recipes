import hashlib
import io
import json
import sys
import tarfile
from types import SimpleNamespace

import pytest

import evaluate
import run
import transport


def archive(files):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return stream.getvalue()


@pytest.mark.parametrize("name", ["/etc/file", "output/../../file", "elsewhere/file"])
def test_export_rejects_unsafe_paths(tmp_path, name):
    with pytest.raises(ValueError, match="Unsafe"):
        run.extract_results(archive({name: b"data"}), tmp_path, ("secret-token",))
    assert not list(tmp_path.iterdir())


def test_export_checks_all_files_before_writing(tmp_path):
    blob = archive({"output/ok.json": b"{}", "output/key.json": b"secret-token"})
    with pytest.raises(ValueError, match="Credential"):
        run.extract_results(blob, tmp_path, ("secret-token",))
    assert not list(tmp_path.iterdir())


def test_export_rejects_links(tmp_path):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        info = tarfile.TarInfo("output/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/tmp/escape"
        tar.addfile(info)
    with pytest.raises(ValueError, match="non-regular"):
        run.extract_results(buffer.getvalue(), tmp_path, ("secret-token",))


def test_verify_distinguishes_wrong_answer_from_failed_execution(tmp_path):
    task = SimpleNamespace(id="task-1")
    raw = b'{"response": {}}'
    (tmp_path / (hashlib.sha256(task.id.encode()).hexdigest() + ".json")).write_bytes(raw)
    row = {
        "task_id": task.id,
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "status": "ok",
        "valid": False,
        "correct": False,
    }
    result = evaluate.verify_results([task], [row], tmp_path)
    assert result["complete"] and result["failed"] == 0
    assert result["correct"] == 0 and result["valid"] == 0
    assert not evaluate.verify_results([task], [], tmp_path)["complete"]
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate.verify_results([task], [row, row], tmp_path)
    row["status"] = "failed"
    assert evaluate.verify_results([task], [row], tmp_path)["failed"] == 1
    row["raw_sha256"] = "incorrect"
    with pytest.raises(ValueError, match="hash mismatch"):
        evaluate.verify_results([task], [row], tmp_path)


@pytest.mark.parametrize("status", [200, 429])
def test_native_transport_preserves_prompt_schema_output_and_errors(monkeypatch, status):
    body = {
        "model": evaluate.MODEL,
        "temperature": 0,
        "max_tokens": 4096,
        "messages": [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Task with no answer key"},
        ],
        "response_format": {"json_schema": {"schema": {"type": "object"}}},
    }
    raw_text = '{"probabilities":{"yes":0.75,"no":0.25}}'
    response = {
        "model": evaluate.MODEL,
        "content": [{"type": "text", "text": raw_text}],
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }
    if status != 200:
        response = {"error": {"type": "rate_limit_error"}}

    def post(url, native, headers, timeout):
        assert url == "https://api.anthropic.com/v1/messages"
        assert native["system"] == "System prompt"
        assert native["messages"] == body["messages"][1:]
        assert native["output_config"]["format"]["schema"] == {"type": "object"}
        assert headers["x-api-key"] == "test-token"
        assert timeout == 30
        return status, response, 1.5

    monkeypatch.setitem(sys.modules, "jevbench.adapters.base", SimpleNamespace(http_post_json=post))
    code, result, latency = transport.native_transport(
        "https://api.anthropic.com/v1/chat/completions",
        body,
        {"Authorization": "Bearer test-token"},
        30,
    )
    assert code == status and latency == 1.5
    if status == 200:
        assert result["choices"][0]["message"]["content"] == raw_text
        assert result["usage"] == {"prompt_tokens": 10, "completion_tokens": 20}
        assert result["native_response"] is response
        assert "test-token" not in json.dumps(result)
    else:
        assert result is response


class Operation:
    def __init__(self, value=None):
        self.value = value

    def result(self, **kwargs):
        return self.value


@pytest.mark.parametrize(
    "failure", [None, "evaluation", "artifact", "start", "stop", "slow_setup", "expired_setup"]
)
def test_lifecycle_retrieves_evidence_and_stops_on_failure(monkeypatch, tmp_path, failure):
    monkeypatch.setenv("CWSANDBOX_API_KEY", "sandbox-test-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "model-test-secret")
    monkeypatch.setattr(run, "source_archive", lambda: b"public source")
    elapsed = 300 if failure == "slow_setup" else 901 if failure == "expired_setup" else 0
    clock = iter([1000, 1000 + elapsed])
    monkeypatch.setattr(run, "monotonic", lambda: next(clock))

    class FakeSandbox:
        sandbox_id = None
        stopped = False

        def __init__(self, **kwargs):
            self.writes = {}
            assert kwargs["auth"] == run.AuthStrategy.COREWEAVE_API_KEY
            assert kwargs["environment_variables"] == {
                "ANTHROPIC_API_KEY": "model-test-secret",
                "PYTHONUNBUFFERED": "1",
            }
            assert kwargs["max_lifetime_seconds"] == 1200

        def start(self):
            self.sandbox_id = "test-sandbox"
            if failure == "start":
                raise RuntimeError("start failed model-test-secret")
            return Operation()

        def wait(self, **kwargs):
            return self

        def exec(self, command, **kwargs):
            if command[0] == "python":
                assert failure != "expired_setup", "Must reserve time for artifact retrieval"
                assert kwargs["timeout_seconds"] == (600 if failure == "slow_setup" else 900)
            code = 3 if failure == "evaluation" and command[0] == "python" else 0
            process = Operation(
                SimpleNamespace(stdout_bytes=b"model-test-secret", stderr_bytes=b"")
            )
            process.returncode = code
            return process

        def write_file(self, path, contents):
            self.writes[path] = contents
            return Operation()

        def read_file(self, path):
            if failure == "artifact":
                raise RuntimeError("artifact failed")
            return Operation(archive({"output/verification.json": b"{}"}))

        def stop(self, **kwargs):
            FakeSandbox.stopped = True
            if failure == "stop":
                raise RuntimeError("stop failed")
            return Operation()

    monkeypatch.setattr(run, "Sandbox", FakeSandbox)
    output = tmp_path / "results"
    code = run.run(SimpleNamespace(output=output, limit=2, budget_usd=2.0))
    assert FakeSandbox.stopped
    assert code == (0 if failure in (None, "slow_setup") else 1)
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["sandbox_id"] == "test-sandbox"
    if failure == "expired_setup":
        assert "Setup exhausted" in manifest["error"]
        assert manifest["artifacts_retrieved"]
    if failure == "stop":
        assert "cleanup_error" in manifest
    else:
        assert manifest["sandbox_stopped"]
    for file in output.rglob("*"):
        if file.is_file():
            assert b"model-test-secret" not in file.read_bytes()
            assert b"sandbox-test-secret" not in file.read_bytes()


@pytest.mark.parametrize(
    "argv", [["--limit", "0"], ["--limit", "73"], ["--budget-usd", "nan"], ["--budget-usd", "-1"]]
)
def test_invalid_config_never_provisions(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["run.py", *argv])
    with pytest.raises(SystemExit) as error:
        run.main()
    assert error.value.code == 2


def test_missing_credentials_never_fetches_or_provisions(monkeypatch, tmp_path):
    monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Set CWSANDBOX_API_KEY"):
        run.run(SimpleNamespace(output=tmp_path / "unused"))
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("filled_key", [None, "CWSANDBOX_API_KEY", "ANTHROPIC_API_KEY"])
def test_unfilled_template_never_fetches_or_provisions(monkeypatch, tmp_path, filled_key):
    template = (run.ROOT / ".env.example").read_text()
    for key in ("CWSANDBOX_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    if filled_key:
        template = template.replace(f"{filled_key}=", f"{filled_key}=test-token")
    (tmp_path / ".env").write_text(template)
    monkeypatch.setattr(run, "ROOT", tmp_path)
    output = tmp_path / "unused"
    monkeypatch.setattr(sys, "argv", ["run.py", "--output", str(output)])

    def unexpected(*args, **kwargs):
        pytest.fail("Incomplete credentials must fail before fetching or provisioning")

    monkeypatch.setattr(run, "source_archive", unexpected)
    monkeypatch.setattr(run, "Sandbox", unexpected)
    assert run.main() == 1
    assert not output.exists()
