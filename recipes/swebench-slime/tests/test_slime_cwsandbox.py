"""Offline tests for slime_cwsandbox.CWSandbox against a fake cwsandbox SDK.

The fake mirrors the real SDK's call shapes, including wrapping transport
errors from file operations in SandboxFileError, so these pin the adapter's
contract without credentials or network access.
"""

from __future__ import annotations

import asyncio
import enum
import sys
import types
from types import SimpleNamespace

import pytest

import slime_cwsandbox
from slime_cwsandbox import CWSandbox


class Unavailable(Exception):
    pass


class RequestTimeout(Exception):
    pass


class Exhausted(Exception):
    pass


class FileError(Exception):
    pass


class PlacementMode(enum.Enum):
    UNSPECIFIED = "unspecified"
    SERVERLESS = "serverless"
    CKS = "cks"


class FakeSDKSandbox:
    instances: list[FakeSDKSandbox] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.sandbox_id = None
        self.start_calls = 0
        self.start_errors: list[BaseException] = []
        self.wait_error: BaseException | None = None
        self.stop_hangs = False
        self.stopped = False
        self.execs: list[list[str]] = []
        self.exec_results: list[object] = []
        self.files: dict[str, bytes] = {}
        self.stream_errors: list[BaseException] = []
        FakeSDKSandbox.instances.append(self)

    async def start(self):
        self.start_calls += 1
        if self.start_errors:
            raise self.start_errors.pop(0)
        self.sandbox_id = "sb-1"

    def __await__(self):
        async def _running():
            if self.wait_error:
                raise self.wait_error
            return self

        return _running().__await__()

    async def stop(self, *, graceful_shutdown_seconds=10.0, missing_ok=False):
        if self.stop_hangs:
            await asyncio.sleep(3600)
        self.stopped = True

    def exec(self, argv, *, timeout_seconds=None):
        self.execs.append(list(argv))
        result = self.exec_results.pop(0) if self.exec_results else (0, "", "")

        async def _run():
            if isinstance(result, BaseException):
                raise result
            code, out, err = result
            return SimpleNamespace(returncode=code, stdout=out, stderr=err)

        return _run()

    async def write_file(self, path, data, *, timeout_seconds=None):
        self.files[path] = data

    async def write_file_streaming(self, path, source, *, timeout_seconds=None):
        if self.stream_errors:
            cause = self.stream_errors.pop(0)
            raise FileError(f"stream to {path} failed") from cause
        self.files[path] = source if isinstance(source, bytes) else b"".join(source)

    async def read_file(self, path, *, timeout_seconds=None):
        if path not in self.files:
            raise FileError(path)
        return self.files[path]


@pytest.fixture(autouse=True)
def fake_sdk(monkeypatch):
    FakeSDKSandbox.instances = []
    mod = types.ModuleType("cwsandbox")
    mod.Sandbox = FakeSDKSandbox
    mod.PlacementMode = PlacementMode
    mod.SandboxUnavailableError = Unavailable
    mod.SandboxRequestTimeoutError = RequestTimeout
    mod.SandboxResourceExhaustedError = Exhausted
    monkeypatch.setitem(sys.modules, "cwsandbox", mod)
    for name in list(slime_cwsandbox.os.environ):
        if name.startswith("SLIME_AGENT_"):
            monkeypatch.delenv(name)
    return mod


def make(**kwargs) -> CWSandbox:
    sb = CWSandbox("python:3.12", rpc_retries=kwargs.pop("rpc_retries", 3), **kwargs)
    sb.rpc_backoff_base_sec = 0.0
    return sb


def booted(**kwargs) -> tuple[CWSandbox, FakeSDKSandbox]:
    sb = make(**kwargs)
    asyncio.run(sb.__aenter__())
    return sb, FakeSDKSandbox.instances[-1]


def test_create_passes_settings(monkeypatch):
    monkeypatch.setenv("SLIME_AGENT_CWSANDBOX_PLACEMENT_MODE", "Serverless")
    monkeypatch.setenv("SLIME_AGENT_CWSANDBOX_TAGS", "swebench-slime, run-7")
    sb, raw = booted(timeout=900, cpu="4", memory="8Gi", request_timeout=1200)
    assert sb.sandbox_id == "sb-1"
    assert raw.kwargs == {
        "container_image": "python:3.12",
        "max_lifetime_seconds": 900,
        "resources": {"cpu": "4", "memory": "8Gi"},
        "tags": ["swebench-slime", "run-7"],
        "request_timeout_seconds": 1200.0,
        "placement_mode": "serverless",
    }
    asyncio.run(sb.__aexit__(None, None, None))
    assert raw.stopped


def test_bad_placement_mode_fails_at_construction(monkeypatch):
    monkeypatch.setenv("SLIME_AGENT_CWSANDBOX_PLACEMENT_MODE", "serverles")
    with pytest.raises(ValueError):
        CWSandbox("img")


def test_create_retries_reuse_one_handle():
    sb = make()

    def first_start_fails(**kwargs):
        raw = FakeSDKSandbox(**kwargs)
        raw.start_errors = [Unavailable("lost response")]
        return raw

    sys.modules["cwsandbox"].Sandbox = first_start_fails
    asyncio.run(sb.__aenter__())
    assert len(FakeSDKSandbox.instances) == 1, "a retry must not build a second handle (second sandbox)"
    assert FakeSDKSandbox.instances[0].start_calls == 2


def test_create_request_timeout_is_not_retried():
    sb = make()

    def start_times_out(**kwargs):
        raw = FakeSDKSandbox(**kwargs)
        raw.start_errors = [RequestTimeout("deadline")]
        return raw

    sys.modules["cwsandbox"].Sandbox = start_times_out
    with pytest.raises(RequestTimeout):
        asyncio.run(sb.__aenter__())
    assert FakeSDKSandbox.instances[0].start_calls == 1


def test_failed_wait_stops_the_sandbox():
    sb = make()

    def never_runs(**kwargs):
        raw = FakeSDKSandbox(**kwargs)
        raw.wait_error = RuntimeError("sandbox failed")
        return raw

    sys.modules["cwsandbox"].Sandbox = never_runs
    with pytest.raises(RuntimeError, match="sandbox failed"):
        asyncio.run(sb.__aenter__())
    assert FakeSDKSandbox.instances[0].stopped


def test_stuck_stop_is_bounded():
    sb, raw = booted()
    raw.stop_hangs = True
    sb.stop_timeout_sec = 0.05
    asyncio.run(asyncio.wait_for(sb.__aexit__(None, None, None), timeout=5))


def test_exec_wraps_user_and_env():
    sb, raw = booted()
    asyncio.run(sb.exec("echo hi"))
    asyncio.run(sb.exec("claude -p x", user="agent", env={"ANTHROPIC_BASE_URL": "http://h:1"}))
    assert raw.execs[0] == ["bash", "-l", "-c", "echo hi"]
    assert raw.execs[1] == [
        "runuser", "-u", "agent", "--",
        "env", "ANTHROPIC_BASE_URL=http://h:1",
        "bash", "-l", "-c", "claude -p x",
    ]  # fmt: skip


def test_exec_check_and_idempotent_retry():
    sb, raw = booted()
    raw.exec_results = [(3, "", "boom")]
    with pytest.raises(RuntimeError, match="exit=3"):
        asyncio.run(sb.exec("false", check=True))

    raw.execs.clear()
    raw.exec_results = [Unavailable("severed"), (0, "ok", "")]
    assert asyncio.run(sb.exec("cat f")) == (0, "ok", "")
    assert len(raw.execs) == 2

    raw.execs.clear()
    raw.exec_results = [Unavailable("severed"), (0, "ok", "")]
    with pytest.raises(Unavailable):
        asyncio.run(sb.exec("setsid run &", idempotent=False))
    assert len(raw.execs) == 1, "a non-idempotent command must not be replayed"


def test_streamed_upload_retries_wrapped_transient_error(tmp_path):
    sb, raw = booted()
    host = tmp_path / "node22.tar"
    host.write_bytes(b"x" * (CWSandbox.stream_chunk_bytes + 5))
    raw.stream_errors = [Unavailable("blip")]
    asyncio.run(sb.write_file("/tmp/node22.tar", host))
    assert raw.files["/tmp/node22.tar"] == host.read_bytes()


def test_large_bytes_stream_and_small_bytes_do_not(monkeypatch):
    sb, raw = booted()
    sb.stream_threshold_bytes = 8
    calls = []
    real_stream = raw.write_file_streaming

    async def spy(path, source, **kw):
        calls.append(path)
        await real_stream(path, source, **kw)

    raw.write_file_streaming = spy
    asyncio.run(sb.write_file("/tmp/small", b"1234"))
    asyncio.run(sb.write_file("/tmp/large", b"123456789"))
    assert calls == ["/tmp/large"]
    assert raw.files["/tmp/small"] == b"1234"


def test_non_root_write_creates_parent_as_user_then_chowns():
    sb, raw = booted()
    asyncio.run(sb.write_file("/workspace/setup/before.sh", "set -e\n", user="agent"))
    assert raw.execs == [
        ["runuser", "-u", "agent", "--", "bash", "-l", "-c",
         "test -d /workspace/setup || mkdir -p /workspace/setup"],
        ["bash", "-l", "-c", "chown agent: /workspace/setup/before.sh"],
    ]  # fmt: skip
    assert raw.files["/workspace/setup/before.sh"] == b"set -e\n"


def test_root_write_to_tmp_needs_no_exec():
    sb, raw = booted()
    asyncio.run(sb.write_file("/tmp/a.txt", "hello"))
    assert raw.execs == []


def test_read_file_decodes_and_swallows_missing():
    sb, raw = booted()
    raw.files["/tmp/out"] = "héllo".encode()
    assert asyncio.run(sb.read_file("/tmp/out")) == "héllo"
    assert asyncio.run(sb.read_file("/missing")) == ""
