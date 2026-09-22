"""Offline checks: no CoreWeave credentials or cluster access required."""

import io
import json
import stat
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pytest
import yaml

import sandbox as recipe
import sandbox_service

ROOT = Path(__file__).resolve().parents[1]
CKS_ADDRESS = "tc-placeholder-cks-address"
SANDBOX_ADDRESS = "tc-placeholder-sandbox-address"


def completed(value):
    return SimpleNamespace(result=lambda: value)


@pytest.fixture
def fake_sandbox(monkeypatch):
    class FakeSandbox:
        sandbox_id = "example-sandbox"
        stopped = False
        fail_commands = False

        def __init__(self):
            self.requests = []
            self.commands = []
            self.files = {}

        def run(self, *args, **kwargs):
            self.requests.append((args, kwargs))
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stopped = True

        def wait(self, timeout):
            pass

        def exec(self, command, **kwargs):
            self.commands.append(command)
            if command[:2] == ["sh", "-ec"]:
                subprocess.run(["sh", "-n"], input=command[2], text=True, check=True)
            stdout = "hello from CKS\n" if command[0] == "curl" else ""
            return completed(
                SimpleNamespace(
                    returncode=1 if self.fail_commands else 0,
                    stdout=CKS_ADDRESS if self.fail_commands else stdout,
                    stderr=SANDBOX_ADDRESS,
                )
            )

        def write_file(self, path, contents):
            self.files[path] = contents
            return completed(None)

        def read_file(self, path):
            return completed(json.dumps({"listenAddr": SANDBOX_ADDRESS}).encode())

    instance = FakeSandbox()
    monkeypatch.setattr(recipe, "Sandbox", instance)
    return instance


@pytest.mark.parametrize("mode", ["fetch", "serve", "roundtrip"])
def test_modes_transfer_secret_and_stop(mode, fake_sandbox, tmp_path, monkeypatch, capsys):
    source = tmp_path / "cks.addr"
    source.write_text(CKS_ADDRESS)
    output = tmp_path / "private" / "sandbox.addr"
    args = ["sandbox.py", mode, "--image", "example/demo:0.7.0", "--address-out", str(output)]
    if mode != "serve":
        args += ["--cks-address", str(source)]
    monkeypatch.setattr(sys, "argv", args)

    def finish(prompt):
        assert output.read_text().strip() == SANDBOX_ADDRESS
        assert stat.S_IMODE(output.stat().st_mode) == 0o600
        assert stat.S_IMODE(output.parent.stat().st_mode) == 0o700
        return ""

    monkeypatch.setattr("builtins.input", finish)
    recipe.main()
    assert fake_sandbox.stopped
    assert not output.exists()
    options = fake_sandbox.requests[0][1]
    assert options["placement_mode"] == "serverless"
    assert options["placement_spillover"] == "strict"
    assert options["max_lifetime_seconds"] == 1800
    assert not options.get("services")
    if mode != "serve":
        assert fake_sandbox.files["/tmp/tailcat/cks.addr"] == CKS_ADDRESS.encode()
    captured = capsys.readouterr()
    for address in (CKS_ADDRESS, SANDBOX_ADDRESS):
        assert address not in captured.out + captured.err
        assert address not in json.dumps(fake_sandbox.commands)


def test_remote_failure_stops_sandbox_without_printing_diagnostics(
    fake_sandbox, monkeypatch, capsys
):
    monkeypatch.setattr(sys, "argv", ["sandbox.py", "serve", "--image", "example/demo:0.7.0"])
    fake_sandbox.fail_commands = True
    with pytest.raises(RuntimeError, match="exit 1") as error:
        recipe.main()
    assert fake_sandbox.stopped
    captured = capsys.readouterr()
    for address in (CKS_ADDRESS, SANDBOX_ADDRESS):
        assert address not in str(error.value) + captured.out + captured.err


def test_interrupt_revokes_local_address(fake_sandbox, tmp_path, monkeypatch):
    output = tmp_path / "sandbox.addr"
    monkeypatch.setattr(
        sys,
        "argv",
        ["sandbox.py", "serve", "--image", "example/demo:0.7.0", "--address-out", str(output)],
    )

    def interrupt(prompt):
        assert output.exists()
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupt)
    recipe.main()
    assert fake_sandbox.stopped
    assert not output.exists()


def test_missing_address_does_not_provision(fake_sandbox, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["sandbox.py", "fetch", "--image", "example/demo:0.7.0"])
    with pytest.raises(SystemExit):
        recipe.main()
    assert not fake_sandbox.requests


def test_private_write_refuses_overwrite_and_symlink(tmp_path):
    address = tmp_path / "address"
    recipe.write_private(address, "original")
    symlink = tmp_path / "link"
    symlink.symlink_to(address)
    for path in (address, symlink):
        with pytest.raises(FileExistsError):
            recipe.write_private(path, "replacement")
    assert address.read_text() == "original\n"


@pytest.fixture
def service_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), sandbox_service.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_http_greeting_and_roundtrip(service_url, monkeypatch):
    with urlopen(service_url, timeout=5) as response:
        assert response.status == 200
        assert response.read() == b"hello from sandbox\n"

    def upstream(url, timeout):
        assert url == "http://127.0.0.1:18080/"
        return io.BytesIO(b"hello from CKS\n")

    monkeypatch.setattr(sandbox_service, "urlopen", upstream)
    with urlopen(service_url + "/roundtrip", timeout=5) as response:
        assert response.read() == b"hello from sandbox\nupstream: hello from CKS\n"


def test_http_upstream_failure_and_unknown_path(service_url, monkeypatch):
    def unavailable(*args, **kwargs):
        raise URLError("connection refused")

    monkeypatch.setattr(sandbox_service, "urlopen", unavailable)
    for path, status in [("/roundtrip", 502), ("/missing", 404)]:
        with pytest.raises(HTTPError) as error:
            urlopen(service_url + path, timeout=5)
        assert error.value.code == status
        error.value.close()


def test_manifests_parse_with_shell_variables_preserved():
    documents = []
    for path in sorted((ROOT / "k8s").glob("*.yaml")):
        rendered = path.read_text().replace("${TAILCAT_IMAGE}", "example/demo:0.7.0")
        documents.extend(yaml.safe_load_all(rendered))
    service = next(doc for doc in documents if doc["kind"] == "Service")
    assert service["spec"]["type"] == "ClusterIP"
    for doc in documents:
        if doc["kind"] not in {"Deployment", "Job"}:
            continue
        pod = doc["spec"]["template"]["spec"]
        assert pod["securityContext"]["runAsNonRoot"]
        assert not pod.get("hostNetwork")
        for container in pod["containers"]:
            assert container["securityContext"]["capabilities"]["drop"] == ["ALL"]
            if container["command"] == ["sh", "-ec"]:
                subprocess.run(["sh", "-n"], input=container["args"][0], text=True, check=True)
    job = next(doc for doc in documents if doc["kind"] == "Job")
    assert job["spec"]["activeDeadlineSeconds"] == 180
    caller = job["spec"]["template"]["spec"]["containers"][0]
    assert "${REQUEST_PATH}" in caller["args"][0]
