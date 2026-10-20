"""Tailcat tunnels between SUNK and the Gym sandbox. Standard library only (Python 3.10+).

Tailcat is a userspace WireGuard tunnel. Both ends connect out to a relay, so neither side
needs an open inbound port. ``serve`` exposes local ports and prints a secret address;
``forward`` uses that address to make the ports local on the other side.

The address is a credential: anyone who has it can connect. This module writes it only to
files with mode 0600 and sends Tailcat's output to a private log file.
"""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
import time
import urllib.request
from pathlib import Path

from async_grpo_tool_use.procs import log, spawn

VERSION = "0.7.0"
URL = f"https://github.com/tailscale/tailcat/releases/download/v{VERSION}/tailcat_{VERSION}_linux_amd64.tar.gz"
SHA256 = "23c0b1887a5ec422f0d18a9c52b4f5357815febdaae738a1eb54036d10bd9ee6"


def write_private(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(text)


def install(directory: Path) -> Path:
    """Download the pinned Tailcat binary into ``directory`` and check its SHA-256."""
    binary = directory / "tailcat"
    if binary.is_file():
        return binary
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "tailcat.tar.gz"
    urllib.request.urlretrieve(URL, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"tailcat download has SHA-256 {digest}, expected {SHA256}")
    with tarfile.open(archive) as tar:
        member = next(m for m in tar.getmembers() if Path(m.name).name == "tailcat")
        member.name = "tailcat"
        tar.extract(member, directory)
    archive.unlink()
    binary.chmod(0o755)
    return binary


def serve(binary: Path, ports: list[int], address_file: Path, log_dir: Path, timeout: float = 120.0):
    """Expose local ``ports``; write the new secret address to ``address_file``."""
    log_dir.mkdir(parents=True, exist_ok=True)
    info = log_dir / f"serve-{address_file.stem}.json"
    info.unlink(missing_ok=True)
    out = os.open(info, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    err = os.open(log_dir / f"serve-{address_file.stem}.log", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    proc = spawn(
        [str(binary), "--json", "serve", "--key=new", "--full-address", ",".join(map(str, ports))],
        stdout=out,
        stderr=err,
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"tailcat serve exited with {proc.returncode}")
        if info.stat().st_size:
            write_private(address_file, json.loads(info.read_text())["listenAddr"])
            info.unlink()
            log(f"tunnel: serving ports {ports}")
            return proc
        time.sleep(1)
    raise SystemExit("tailcat serve did not report an address")


def forward(binary: Path, address: str, mappings: list[str], log_dir: Path):
    """Make the remote ports in ``mappings`` (``local:remote``) local on 127.0.0.1."""
    err = os.open(log_dir / "forward.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    proc = spawn(
        [str(binary), "--key=new", "forward", "--bind=127.0.0.1", address, *mappings],
        stdout=err,
        stderr=err,
    )
    log(f"tunnel: forwarding {mappings}")
    return proc
