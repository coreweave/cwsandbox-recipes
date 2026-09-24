import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import run


def fake_sandbox(monkeypatch, result=None):
    sandbox = Mock(sandbox_id="test-sandbox")
    sandbox.exec.return_value.result.return_value = result or SimpleNamespace(
        returncode=0, stdout="ok", stderr=""
    )
    monkeypatch.setattr(run.Sandbox, "run", Mock(return_value=sandbox))
    monkeypatch.setenv("CWSANDBOX_API_KEY", "test-placeholder")
    return sandbox


@pytest.mark.parametrize("returncode,expected", [(0, 0), (7, 1)])
def test_test_exit_and_cleanup(monkeypatch, tmp_path, returncode, expected):
    sandbox = fake_sandbox(
        monkeypatch, SimpleNamespace(returncode=returncode, stdout="output", stderr="detail")
    )
    assert run.run(run.parse_args(["--output", str(tmp_path)])) == expected
    report = json.loads((tmp_path / "result.json").read_text())
    assert report["returncode"] == returncode
    assert report["stopped"]
    sandbox.stop.assert_called_once_with(missing_ok=True)
    kwargs = run.Sandbox.run.call_args.kwargs
    assert kwargs["auth"] == run.AuthStrategy.COREWEAVE_API_KEY
    assert "environment_variables" not in kwargs


@pytest.mark.parametrize("stage", ["wait", "prepare", "test", "cleanup"])
def test_errors_save_results_and_attempt_cleanup(monkeypatch, tmp_path, stage):
    sandbox = fake_sandbox(monkeypatch)
    if stage == "wait":
        sandbox.wait.side_effect = TimeoutError("secret must not be printed")
    elif stage == "prepare":
        sandbox.write_file.return_value.result.side_effect = RuntimeError("secret")
    elif stage == "test":
        monkeypatch.setattr(run, "prepare", Mock(return_value=""))
        sandbox.exec.return_value.result.side_effect = TimeoutError("secret")
    else:
        sandbox.stop.return_value.result.side_effect = RuntimeError("secret")
    assert run.run(run.parse_args(["--output", str(tmp_path)])) == 1
    sandbox.stop.assert_called_once()
    report = json.loads((tmp_path / "result.json").read_text())
    assert report["status"] == "error"
    assert "secret" not in (tmp_path / "summary.md").read_text()


def test_review_outage_preserves_failed_tests(monkeypatch, tmp_path):
    fake_sandbox(monkeypatch, SimpleNamespace(returncode=7, stdout="", stderr="failed"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-placeholder")
    monkeypatch.setattr(run, "prepare", Mock(return_value="diff"))
    monkeypatch.setattr(run, "review", Mock(side_effect=TimeoutError("secret")))
    args = run.parse_args(
        [
            "--repo",
            "owner/repo",
            "--head",
            "a" * 40,
            "--base",
            "b" * 40,
            "--review",
            "--output",
            str(tmp_path),
        ]
    )
    assert run.run(args) == 1
    report = json.loads((tmp_path / "result.json").read_text())
    assert report["status"] == "failed"
    assert report["returncode"] == 7
    assert report["review_error"] == "TimeoutError"


@pytest.mark.parametrize(
    "argv",
    [
        ["--repo", "owner/repo"],
        ["--head", "a" * 40],
        ["--repo", "owner/repo;id", "--head", "a" * 40],
        ["--repo", "owner/repo", "--head", "main"],
        ["--base", "b" * 40],
        ["--review"],
        ["--timeout", "0"],
        ["--timeout", "301"],
    ],
)
def test_invalid_inputs_rejected_before_provisioning(argv):
    with pytest.raises(SystemExit):
        run.parse_args(argv)


def test_git_checkout_verification_and_merge_base(monkeypatch):
    sandbox = fake_sandbox(monkeypatch)
    args = run.parse_args(["--repo", "owner/repo", "--head", "a" * 40, "--base", "b" * 40])

    def result(command, **kwargs):
        value = args.head if command[:2] == ["git", "rev-parse"] else "diff"
        return Mock(result=Mock(return_value=SimpleNamespace(stdout=value)))

    sandbox.exec.side_effect = result
    assert run.prepare(sandbox, args) == "diff"
    commands = [call.args[0] for call in sandbox.exec.call_args_list]
    assert ["git", "fetch", "--no-tags", "origin", args.head, args.base] in commands
    assert ["git", "diff", "--no-ext-diff", f"{args.base}...{args.head}", "--"] in commands
    sandbox.exec.side_effect = None
    sandbox.exec.return_value.result.return_value.stdout = "wrong-sha"
    with pytest.raises(RuntimeError, match="does not match"):
        run.prepare(sandbox, args)


def test_report_escapes_untrusted_output(monkeypatch, tmp_path, capsys):
    malicious = "::error::fake failure\n</pre><script>alert(1)</script>"
    fake_sandbox(monkeypatch, SimpleNamespace(returncode=0, stdout=malicious, stderr=""))
    summary = tmp_path / "github-summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert run.run(run.parse_args(["--output", str(tmp_path)])) == 0
    assert "::error::" not in capsys.readouterr().out
    assert "<script>" not in summary.read_text()
    assert "&lt;script&gt;" in summary.read_text()


def test_failure_annotation_does_not_include_sandbox_output(monkeypatch, tmp_path, capsys):
    malicious = "::error::injected annotation\n</pre><script>bad()</script>"
    fake_sandbox(monkeypatch, SimpleNamespace(returncode=1, stdout=malicious, stderr=malicious))
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert run.run(run.parse_args(["--output", str(tmp_path)])) == 1
    console = capsys.readouterr().out
    assert console.count("::error::") == 1
    assert (
        "::error::Sandbox CI failed. See the job summary and sandbox-ci-results artifact."
        in console
    )
    assert "injected annotation" not in console
    assert "<script>" not in console


def test_truncation_preserves_failure_at_end(monkeypatch, tmp_path):
    output = "install noise\n" * 4000 + "AssertionError: expected 5, got -1\n"
    fake_sandbox(monkeypatch, SimpleNamespace(returncode=1, stdout=output, stderr=output))
    assert run.run(run.parse_args(["--output", str(tmp_path)])) == 1
    report = json.loads((tmp_path / "result.json").read_text())
    assert report["output_truncated"] is True
    for field in ("stdout", "stderr"):
        assert len(report[field]) == run.OUTPUT_LIMIT
        assert report[field].endswith("AssertionError: expected 5, got -1\n")


def test_environment_template_preserves_exported_values():
    import os
    import subprocess

    names = ("CWSANDBOX_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "SANDBOX_REVIEW_MODEL")
    env = dict(os.environ, **{name: "existing-value" for name in names})
    script = 'set -a; source "$1"; set +a; ' + " && ".join(
        f'test "${{{name}}}" = existing-value' for name in names
    )
    subprocess.run(
        ["bash", "-c", script, "bash", str(run.ROOT / ".env.example")], env=env, check=True
    )


def test_documented_ci_copy_keeps_pytest_collection_clean(tmp_path):
    import re
    import subprocess
    import sys

    readme = (run.ROOT / "README.md").read_text()
    setup = readme.split("## Add to GitHub Actions", 1)[1]
    commands = re.search(r"```bash\n(.*?)\n\s*```", setup, re.DOTALL).group(1)
    commands = commands.replace("[RECIPES-CHECKOUT]", str(run.ROOT.parent.parent))
    subprocess.run(["bash", "-eu", "-c", commands], cwd=tmp_path, check=True)
    (tmp_path / "test_application.py").write_text(
        "def test_application():\n    assert 2 + 3 == 5\n"
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 test collected" in result.stdout
    assert "test_application.py::test_application" in result.stdout
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    ignored = subprocess.run(
        ["git", "check-ignore", "ci/sandbox/.env", "ci/sandbox/outputs/result.json"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert ignored.stdout.splitlines() == ["ci/sandbox/.env", "ci/sandbox/outputs/result.json"]


@pytest.mark.parametrize("override", [None, "claude-sonnet-4-6"])
def test_review_model_setting_is_scoped_and_returns_text(monkeypatch, override):
    import io

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-placeholder")
    monkeypatch.setenv("ANTHROPIC_MODEL", "unrelated-tool-model")
    monkeypatch.delenv("SANDBOX_REVIEW_MODEL", raising=False)
    if override:
        monkeypatch.setenv("SANDBOX_REVIEW_MODEL", override)
    response = io.BytesIO(json.dumps({"content": [{"type": "text", "text": "Advisory"}]}).encode())
    request = Mock(return_value=response)
    monkeypatch.setattr(run.urllib.request, "urlopen", request)
    assert run.review("diff", {"returncode": 0, "stdout": "", "stderr": ""}) == "Advisory"
    payload = json.loads(request.call_args.args[0].data)
    assert payload["model"] == (override or "claude-sonnet-5")
    assert payload["thinking"] == {"type": "disabled"}
