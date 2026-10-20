import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from async_grpo_tool_use.sampler import SamplerClient, make_receiver


@pytest.fixture
def vllm():
    """A vLLM stand-in that records the dev-mode routes the receiver calls."""
    calls = []

    class FakeVllm(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return None

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            calls.append((self.path, json.loads(body) if body else None))
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeVllm)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", calls
    server.shutdown()


class FakeS3:
    """The bucket, as a dict of key -> bytes."""

    def __init__(self):
        self.objects = {}

    def get_paginator(self, name):
        def paginate(Bucket, Prefix):
            return [{"Contents": [{"Key": key} for key in self.objects if key.startswith(Prefix)]}]

        return SimpleNamespace(paginate=paginate)

    def download_file(self, bucket, key, path, Config=None):
        Path(path).write_bytes(self.objects[key])


@pytest.fixture
def receiver(tmp_path, vllm):
    url, _ = vllm
    s3 = FakeS3()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_receiver(tmp_path / "weights", url, s3, "bucket"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield SamplerClient(f"http://127.0.0.1:{server.server_port}"), tmp_path / "weights", s3
    server.shutdown()


def _export(path, payload):
    path.mkdir(parents=True)
    (path / "model.safetensors").write_bytes(payload)
    (path / "config.json").write_text("{}")
    return path


def _upload(s3, prefix, payload):
    s3.objects[f"{prefix}/model.safetensors"] = payload
    s3.objects[f"{prefix}/config.json"] = b"{}"


def test_load_from_the_bucket_swaps_weights_while_paused(vllm, receiver):
    _, calls = vllm
    client, weights, s3 = receiver
    _upload(s3, "runs/r/weights/v000001", b"x" * 1000)

    result = client.load(s3="runs/r/weights/v000001", version=1)

    loaded = weights / "v000001"
    assert (loaded / "model.safetensors").read_bytes() == b"x" * 1000 and result["download_s"] >= 0
    assert [path for path, _ in calls] == ["/pause?mode=keep", "/collective_rpc", "/reset_prefix_cache", "/resume"]
    assert calls[1][1] == {"method": "reload_weights", "kwargs": {"weights_path": str(loaded)}}


def test_each_load_deletes_older_downloads(receiver):
    client, weights, s3 = receiver
    for version in (1, 2):
        _upload(s3, f"w/v{version:06d}", b"w")
        client.load(s3=f"w/v{version:06d}", version=version)
    assert sorted(p.name for p in weights.iterdir()) == ["v000002"]


def test_load_in_place_from_a_shared_directory(tmp_path, vllm, receiver):
    _, calls = vllm
    client, _, _ = receiver
    export = _export(tmp_path / "shared", b"w")
    client.load(path=export)
    assert calls[1][1]["kwargs"]["weights_path"] == str(export)
    assert export.exists()  # the trainer owns it


def test_loading_a_missing_export_fails_without_pausing(vllm, receiver):
    _, calls = vllm
    client, _, _ = receiver
    with pytest.raises(Exception, match="400"):
        client.load(s3="w/v000007", version=7)
    assert calls == []
