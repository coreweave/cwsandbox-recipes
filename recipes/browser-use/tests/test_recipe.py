import json
from types import SimpleNamespace

import pytest

import run
from verify import verify

ANSWER = {"title": "Example Domain", "information_link": "https://iana.org/help/example-domains"}
OBSERVED = {
    "title": "Example Domain",
    "url": "https://example.com/",
    "links": [ANSWER["information_link"]],
}


def test_verify_requires_browser_evidence():
    assert verify(ANSWER, OBSERVED)
    assert not verify({**ANSWER, "title": "example.com"}, OBSERVED)
    assert not verify(ANSWER, {**OBSERVED, "links": []})
    assert not verify(ANSWER, {**OBSERVED, "url": "https://attacker.example/"})
    link = "https://iana.org.attacker.example/"
    assert not verify({**ANSWER, "information_link": link}, {**OBSERVED, "links": [link]})


class Ref:
    def __init__(self, value=None):
        self.value = value

    def result(self, **kwargs):
        return self.value


class FakeSandbox:
    sandbox_id = "test-sandbox"

    def __init__(self, **kwargs):
        self.stopped = False

    def start(self):
        return Ref()

    def wait(self, **kwargs):
        pass

    def write_file(self, *args):
        return Ref()

    def exec(self, *args, **kwargs):
        return Ref(SimpleNamespace(returncode=1, stdout="test-key", stderr="setup failed"))

    def stop(self):
        self.stopped = True
        return Ref()


@pytest.fixture
def environment(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setenv("WANDB_ENTITY", "test-team")


def test_setup_failure_stops_sandbox_and_redacts_logs(tmp_path, monkeypatch, environment):
    sandbox = FakeSandbox()
    monkeypatch.setattr(run, "Sandbox", lambda **kwargs: sandbox)
    result = run.run_worker(0, tmp_path)
    assert sandbox.stopped
    assert not result["passed"]
    assert result["stopped"]
    assert (tmp_path / "worker-0/setup.stdout.log").read_text() == "[REDACTED]"
    assert (
        json.loads((tmp_path / "worker-0/sandbox.json").read_text())["sandbox_id"] == "test-sandbox"
    )


def test_stop_failure_is_reported(tmp_path, monkeypatch, environment):
    sandbox = FakeSandbox()

    def fail():
        raise TimeoutError()

    sandbox.stop = fail
    monkeypatch.setattr(run, "Sandbox", lambda **kwargs: sandbox)
    result = run.run_worker(0, tmp_path)
    assert not result["stopped"]
    assert result["cleanup_error"] == "TimeoutError"


def test_cleanup_rejects_different_team(tmp_path, monkeypatch, environment):
    run.save(tmp_path / "run.json", {"entity": "other-team"})
    with pytest.raises(ValueError, match="match the team"):
        run.cleanup(tmp_path)


def test_refuses_to_overwrite_run(tmp_path, monkeypatch, environment):
    monkeypatch.setattr("sys.argv", ["run.py", "--output", str(tmp_path)])
    with pytest.raises(FileExistsError):
        run.main()


def test_rejects_placeholder_credentials(monkeypatch, environment):
    monkeypatch.setenv("WANDB_API_KEY", "[WANDB-API-KEY]")
    with pytest.raises(ValueError, match="WANDB_API_KEY"):
        run.check_environment()
