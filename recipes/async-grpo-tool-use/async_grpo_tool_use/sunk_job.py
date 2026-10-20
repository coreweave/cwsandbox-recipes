"""Slurm batch script for SUNK: the NeMo-RL trainer on one 8x H100 node.

The launcher submits this with ``sbatch``. It runs on the compute node (host Python 3.10):

1. Import the NeMo-RL image into the shared cache (first run only).
2. Forward the Gym sandbox's agent ports 21002-21004 to 127.0.0.1 here, over a Tailcat tunnel.
3. Run ``run_trainer`` as a job step in a pyxis container with all 8 GPUs.

Rollouts come from a Dedicated Inference deployment, which Gym calls over HTTPS.

Usage (from the launcher): sbatch ... --wrap "python3 -m async_grpo_tool_use.sunk_job <run dir>"
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from async_grpo_tool_use import tunnel
from async_grpo_tool_use.procs import log, read_env_file, run, spawn, supervise, wait_any, wait_for_file

TRAINER_IMAGE = "nvcr.io#nvidia/nemo-rl@sha256:75c4821c7f24d6a045d22ce51756bd7ea9ec208020c90233723b9d7a9727d681"
IMAGE_NAME = "nemo-rl-v0.7.0"
TRAINER_PYTHON = "/opt/nemo_rl_venv/bin/python"
GYM_PORTS = ("21002", "21003", "21004")  # train agent, eval agent, eval server


def image_path(cache: Path) -> Path:
    return cache / "images" / f"{IMAGE_NAME}.sqsh"


def import_image(cache: Path) -> Path:
    """Import the image into the cache if missing. The import writes a temporary file and
    renames it when done, so a stopped job never leaves a broken image. The first run takes
    about 50 minutes (the image is about 40 GB); later runs skip this."""
    final = image_path(cache)
    if final.is_file():
        return final
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_suffix(".sqsh.tmp")
    tmp.unlink(missing_ok=True)
    log(f"sunk: importing {TRAINER_IMAGE} (first run only)")
    run(["srun", "--ntasks=1", "--exact", "--gres=gpu:0", "--cpus-per-task=16", "--mem=64G", "--job-name=import",
         "enroot", "import", "--output", str(tmp), f"docker://{TRAINER_IMAGE}"])  # fmt: skip
    tmp.rename(final)
    log(f"sunk: cached {final}")
    return final


def step(image: Path, mounts: dict[Path, str], command: list[str]) -> list[str]:
    """An ``srun`` step with all GPUs of the job's node in a pyxis container."""
    mount = ",".join(f"{host}:{inside}" for host, inside in mounts.items())
    return [
        "srun", "--ntasks=1", "--exact", "--gres=gpu:8", "--cpus-per-task=112", "--job-name=trainer",
        f"--container-image={image}", f"--container-mounts={mount}", "--container-writable", "--no-container-mount-home",
        "--container-workdir=/workspace", *command,
    ]  # fmt: skip


def refit_dir() -> str:
    return f"/dev/shm/dapo-refit-{os.environ.get('SLURM_JOB_ID', os.getpid())}"


def report_gpus(every_s: int = 2) -> None:
    """Print one ``@@metric`` line with every GPU's utilization every ``every_s`` seconds, for the
    live dashboard (``DASHBOARD_METRICS=1``). The job log carries it to the launcher."""
    if not shutil.which("nvidia-smi"):
        return
    query = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
             "--format=csv,noheader,nounits", f"--loop={every_s}"]  # fmt: skip
    smi = subprocess.Popen(query, stdout=subprocess.PIPE, text=True)
    count = int(os.environ.get("SLURM_GPUS_ON_NODE") or 8)

    def pump() -> None:
        rows = []
        for line in smi.stdout:
            parts = [part.strip() for part in line.split(",")]
            rows.append([float(part) if part.replace(".", "", 1).isdigit() else 0.0 for part in parts])
            if len(rows) == count:
                row = {"ts": time.time(), "k": "gpu", "util": [r[0] for r in rows], "mem_mib": [r[1] for r in rows],
                       "power_w": [r[2] for r in rows]}  # fmt: skip
                print("@@metric " + json.dumps(row), flush=True)
                rows = []

    threading.Thread(target=pump, daemon=True).start()


def main(run_dir: Path, overrides: list[str]) -> int:
    supervise()
    cache = run_dir.parent.parent / "cache"
    os.environ.update(read_env_file(run_dir / "secrets.env"))
    os.environ["PYTHONPATH"] = "/workspace"
    # Weights are exported to node memory: every policy rank writes its shard there, and the
    # upload reads it back, without a round trip through shared storage. The dir is this job's.
    os.environ.setdefault("REFIT_EXPORT_DIR", refit_dir())
    (cache / "work").mkdir(parents=True, exist_ok=True)

    log(f"sunk: node {os.uname().nodename}, run dir {run_dir}")
    image = import_image(cache)
    tailcat = tunnel.install(cache / "bin")
    gym_address = wait_for_file(run_dir / "gym.addr").read_text().strip()
    tunnel.forward(tailcat, gym_address, list(GYM_PORTS), run_dir)
    if os.environ.get("DASHBOARD_METRICS") == "1":
        report_gpus()

    mounts = {run_dir / "code": "/workspace", cache / "work": "/work", run_dir: "/job"}
    # Mounted at the same path, so this script can clean up whatever the container left there.
    refit = Path(os.environ["REFIT_EXPORT_DIR"])
    refit.mkdir(parents=True, exist_ok=True)
    mounts[refit] = str(refit)
    trainer = spawn(step(image, mounts, [TRAINER_PYTHON, "-m", "async_grpo_tool_use.run_trainer", *overrides]))
    try:
        return wait_any([trainer])
    finally:
        # A failed run can leave up to three 14 GiB exports in node memory.
        shutil.rmtree(os.environ["REFIT_EXPORT_DIR"], ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), sys.argv[2:]))
