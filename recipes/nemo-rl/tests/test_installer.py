"""Exercise bounded downloads, archive integrity, and safe source extraction."""

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
from pathlib import Path
import subprocess
import tarfile
import threading

import pytest


spec = importlib.util.spec_from_file_location(
    "install_nemo_source",
    Path(__file__).resolve().parents[1] / "scripts" / "install_nemo_source.py",
)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def test_stalled_connection_is_killed_and_retried(tmp_path, capsys):
    release = threading.Event()
    connected = threading.Event()

    class StalledHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            connected.set()
            release.wait(10)

    server = ThreadingHTTPServer(("127.0.0.1", 0), StalledHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(b"incomplete earlier attempt")
    try:
        with pytest.raises(RuntimeError, match="exhausted") as error:
            installer.download_with_retry(
                f"http://127.0.0.1:{server.server_port}",
                archive,
                attempts=2,
                attempt_seconds=0.3,
            )
        assert isinstance(error.value.__cause__, subprocess.TimeoutExpired)
        assert connected.is_set()
        assert not archive.exists()
        assert "attempt 2/2" in capsys.readouterr().out
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join()


def test_failed_attempt_can_recover(monkeypatch, tmp_path):
    archive = tmp_path / "source.tar.gz"
    calls = []

    def run(command, *, check, timeout):
        calls.append(command)
        if len(calls) == 1:
            archive.write_bytes(b"partial")
            raise subprocess.CalledProcessError(1, command)
        assert not archive.exists()
        archive.write_bytes(b"complete")

    monkeypatch.setattr(installer.subprocess, "run", run)
    installer.download_with_retry("https://example.invalid/source", archive)
    assert len(calls) == 2
    assert archive.read_bytes() == b"complete"


def test_checksum_failure_extracts_nothing(tmp_path):
    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(b"corrupted archive")
    target = tmp_path / "extracted"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        installer.extract_verified(archive, target)
    assert not target.exists()


@pytest.mark.parametrize("unsafe", ["traversal", "symlink", None])
def test_extracts_only_safe_members(monkeypatch, tmp_path, unsafe):
    archive = tmp_path / "source.tar.gz"
    root = f"RL-{installer.COMMIT}"
    with tarfile.open(archive, "w:gz") as output:
        member = tarfile.TarInfo(f"{root}/nemo_rl/example.py")
        member.size = len(b"source")
        output.addfile(member, io.BytesIO(b"source"))
        if unsafe:
            member = tarfile.TarInfo(
                f"{root}/../../escape" if unsafe == "traversal" else f"{root}/link"
            )
            if unsafe == "symlink":
                member.type = tarfile.SYMTYPE
                member.linkname = "../../escape"
            output.addfile(member)
    monkeypatch.setattr(
        installer, "SHA256", hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    target = tmp_path / "extracted"
    if unsafe:
        with pytest.raises(ValueError, match="Unsafe archive member"):
            installer.extract_verified(archive, target)
        assert not target.exists()
    else:
        installer.extract_verified(archive, target)
        assert (target / root / "nemo_rl/example.py").read_bytes() == b"source"
