"""Self-hosted sampler (``SAMPLER=vllm``): vLLM on your own GPU, with weight swaps from the trainer.

Runs in a GPU sandbox, or as a second Slurm job step next to the trainer:

1. Download the model and start ``vllm serve`` in dev mode, which adds the routes RL needs:
   ``/pause``, ``/resume`` and ``/collective_rpc``.
2. Start the weight receiver on 127.0.0.1. The trainer asks it to load each new export: it
   pauses the requests in flight (``mode=keep``), reloads the weights from disk, resets the
   prefix cache and resumes the requests, so a rollout continues on the new weights.
3. Expose vLLM and the receiver with ``tailcat serve`` and write the tunnel address to
   ``<run dir>/sampler.addr``. Gym forwards vLLM's port and the trainer forwards both.

With ``SAMPLER_REPLICAS=N`` (GPU sandboxes) N samplers run, one per sandbox. Replica ``i`` is at
local ports ``VLLM_PORT + i`` and ``RECEIVER_PORT + i`` on the trainer and in Gym
(:func:`forwards`); Gym spreads the rollouts over them and the trainer loads each export into
all of them.

Receiver routes:

* ``POST /load?path=P``: load the export at ``P``, a directory the trainer shares with this
  process (Slurm: same node). The trainer deletes it.
* ``POST /load?s3=<prefix>&version=N``: download the export from ``s3://$HOTLOAD_BUCKET/<prefix>``
  into ``SAMPLER_WEIGHTS_DIR``, load it and delete older ones (GPU sandboxes). The tunnel carries
  only this request: Tailcat is fine for rollouts but too slow for gigabytes of weights.

:class:`SamplerClient` is the trainer's side.

Usage: /opt/nemo_rl_venv/bin/python -m async_grpo_tool_use.sampler <run dir>
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from async_grpo_tool_use import caios

VLLM_PORT = 21100
RECEIVER_PORT = 21200


def forwards(replica: int, receiver: bool) -> list[str]:
    """Tailcat ``local:remote`` mappings that make replica ``replica`` local at its own ports."""
    mappings = [f"{VLLM_PORT + replica}:{VLLM_PORT}"]
    if receiver:
        mappings.append(f"{RECEIVER_PORT + replica}:{RECEIVER_PORT}")
    return mappings


SERVED_MODEL = "policy"
# The vLLM that NeMo-RL's own vLLM workers use, prebuilt in the NeMo-RL image.
VLLM = "/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/vllm"


def call(url: str, data: bytes | None = None, timeout: float = 600, method: str = "POST") -> bytes:
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def swap_weights(vllm_url: str, path: Path) -> float:
    """Pause vLLM's requests, reload the weights at ``path``, resume. Return the seconds paused."""
    started = time.perf_counter()
    call(f"{vllm_url}/pause?mode=keep")
    try:
        body = {"method": "reload_weights", "kwargs": {"weights_path": str(path)}}
        call(f"{vllm_url}/collective_rpc", json.dumps(body).encode())
        # Requests admitted from now on must not reuse KV computed with the old weights.
        call(f"{vllm_url}/reset_prefix_cache")
    finally:
        call(f"{vllm_url}/resume")
    return time.perf_counter() - started


def make_receiver(weights_dir: Path, vllm_url: str, s3: Any = None, bucket: str = "") -> type[BaseHTTPRequestHandler]:
    lock = threading.Lock()  # one swap at a time

    class Receiver(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            return None

        def reply(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:
            url = urllib.parse.urlsplit(self.path)
            query = {key: values[0] for key, values in urllib.parse.parse_qs(url.query).items()}
            from_bucket = {"s3", "version"} <= query.keys() and s3 is not None
            if url.path != "/load" or not (from_bucket or "path" in query):
                return self.reply(404, {"error": "POST /load?path=P or /load?s3=<prefix>&version=N"})
            with lock:
                downloaded_s = 0.0
                if from_bucket:
                    path = weights_dir / f"v{int(query['version']):06d}"
                    started = time.perf_counter()
                    caios.download_dir(s3, bucket, query["s3"], path)
                    downloaded_s = time.perf_counter() - started
                else:
                    path = Path(query["path"])
                if not path.is_dir() or not any(path.glob("*.safetensors")):
                    return self.reply(400, {"error": f"no export at {path}"})
                paused = swap_weights(vllm_url, path)
                if from_bucket:  # each export is full, so older ones are not needed
                    for old in weights_dir.glob("v*"):
                        if old.name < path.name:
                            shutil.rmtree(old, ignore_errors=True)
            self.reply(200, {"paused_s": round(paused, 2), "download_s": round(downloaded_s, 2)})

    return Receiver


class SamplerClient:
    """The trainer's side: send an export and load it, through the tunnel on 127.0.0.1."""

    def __init__(self, receiver_url: str) -> None:
        self.url = receiver_url.rstrip("/")

    def load(self, *, path: Path | None = None, s3: str | None = None, version: int | None = None) -> dict:
        """Load an export in place (``path``) or from the bucket (``s3`` prefix and ``version``).
        Return the seconds the sampler paused and spent downloading."""
        query = urllib.parse.urlencode({"path": str(path)} if path is not None else {"s3": s3, "version": version})
        return json.loads(call(f"{self.url}/load?{query}", b"", timeout=1800))


def main(run_dir: Path) -> int:
    from async_grpo_tool_use import model_prep, tunnel
    from async_grpo_tool_use.inference import DeploymentSettings
    from async_grpo_tool_use.procs import gpu_count, log, require, spawn, supervise, wait_any, wait_for_file, wait_http
    from async_grpo_tool_use.run_trainer import install_cwtrain

    supervise()
    weights_dir = Path(os.environ.get("SAMPLER_WEIGHTS_DIR", "/dev/shm/sampler-weights"))
    model_dir = Path(os.environ.get("SAMPLER_MODEL_DIR", "/work/models/base"))
    if os.environ.get("SAMPLER_WAIT_FOR_MODEL") == "1":
        wait_for_file(model_dir / ".prepared")  # next to the trainer on Slurm: it downloads the model
    else:
        model_prep.prepare(require("MODEL_REPO"), require("MODEL_REVISION"), model_dir)
    # Same limits as the Dedicated Inference deployment, which Gym's settings assume.
    settings = DeploymentSettings()
    gpus = gpu_count()  # one engine over all of them; collective_rpc reaches every worker
    log(f"sampler: starting vLLM on {model_dir}, {gpus} GPU(s)")
    vllm = spawn(
        [VLLM, "serve", str(model_dir), "--served-model-name", SERVED_MODEL, "--host", "127.0.0.1",
         "--port", str(VLLM_PORT), "--max-model-len", str(settings.max_model_len),
         "--max-num-seqs", str(settings.concurrency), "--generation-config", "vllm", "--tensor-parallel-size", str(gpus),
         "--gpu-memory-utilization", os.environ.get("SAMPLER_GPU_MEMORY", "0.85")],
        env={**os.environ, "VLLM_SERVER_DEV_MODE": "1"},
        prefix="[vllm] ",
    )  # fmt: skip
    vllm_url = f"http://127.0.0.1:{VLLM_PORT}"
    wait_http("vLLM", f"{vllm_url}/health", ok=lambda status: status == 200, timeout=1800)
    bucket = os.environ.get("HOTLOAD_BUCKET", "")
    if bucket:  # GPU sandboxes: exports arrive through object storage
        install_cwtrain()
    s3 = caios.client() if bucket else None
    receiver = ThreadingHTTPServer(("127.0.0.1", RECEIVER_PORT), make_receiver(weights_dir, vllm_url, s3, bucket))
    threading.Thread(target=receiver.serve_forever, daemon=True).start()
    tailcat = tunnel.install(Path(os.environ.get("TAILCAT_DIR", "/opt/tailcat")))
    tunnel.serve(tailcat, [VLLM_PORT, RECEIVER_PORT], run_dir / "sampler.addr", run_dir)
    log(f"sampler: serving {SERVED_MODEL} and the weight receiver")
    return wait_any([vllm])


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
