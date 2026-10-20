"""Process helpers for the container entrypoints. Standard library only.

Each entrypoint is the container's PID 1. Linux ignores SIGTERM for PID 1 unless it has a
handler, so a sandbox stop would wait 5 minutes (the kill grace period). ``supervise()``
adds a handler that stops the child processes and exits.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path

_children: list[subprocess.Popen] = []


def log(message: str) -> None:
    print(f"[grpo] {message}", flush=True)


def require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} must be set (the launcher sets it)")
    return value


def stop_children(timeout: float = 20.0) -> None:
    for proc in _children:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.monotonic() + timeout
    for proc in _children:
        try:
            proc.wait(max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            proc.kill()


def supervise() -> None:
    """Stop the children and exit on SIGTERM or SIGINT."""

    def handler(signum: int, _frame: object) -> None:
        log(f"received signal {signum}; stopping")
        stop_children()
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def spawn(
    cmd: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    prefix: str = "",
    cwd: str | None = None,
    stdout: int | None = None,
    stderr: int | None = None,
):
    """Start a child; with ``prefix``, tag each of its output lines. ``stdout``/``stderr``
    take file descriptors, for output that must stay private."""
    proc = subprocess.Popen(
        list(cmd),
        env={**os.environ, **(env or {})},
        cwd=cwd,
        stdout=subprocess.PIPE if prefix else stdout,
        stderr=subprocess.STDOUT if prefix else stderr,
        text=bool(prefix),
        bufsize=1 if prefix else -1,
    )
    if prefix:

        def pump() -> None:
            for line in proc.stdout:
                sys.stdout.write(f"{prefix}{line}")
                sys.stdout.flush()

        threading.Thread(target=pump, daemon=True).start()
    _children.append(proc)
    return proc


def wait_any(procs: Sequence[subprocess.Popen], poll: float = 1.0) -> int:
    """Wait until one child exits, stop the rest, and return the first exit code."""
    while True:
        for proc in procs:
            code = proc.poll()
            if code is not None:
                stop_children()
                return code
        time.sleep(poll)


def run(cmd: Sequence[str], **kwargs) -> None:
    subprocess.run(list(cmd), check=True, **kwargs)


def wait_for_file(path: Path, poll: float = 2.0) -> Path:
    while not path.is_file():
        time.sleep(poll)
    return path


def read_env_file(path: Path) -> dict[str, str]:
    pairs = (line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
    return {key.strip(): value.strip() for key, value in pairs}


def http_status(url: str, headers: Mapping[str, str] | None = None, timeout: float = 10.0) -> int:
    """HTTP status of a GET, or 0 when nothing answers."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=dict(headers or {})), timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except OSError:
        return 0


def wait_http(name: str, url: str, *, ok, headers: Mapping[str, str] | None = None, timeout: float = 3600.0) -> None:
    deadline = time.monotonic() + timeout
    while not ok(http_status(url, headers)):
        if time.monotonic() > deadline:
            raise SystemExit(f"{name} not ready at {url}")
        time.sleep(10)
    log(f"{name} ready")


def gpu_count() -> int:
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
    return sum(1 for line in out.splitlines() if line.startswith("GPU "))
