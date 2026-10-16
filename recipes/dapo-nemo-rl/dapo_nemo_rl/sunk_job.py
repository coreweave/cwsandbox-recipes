"""Slurm batch script for SUNK: the NeMo-RL trainer on one 8x H100 node.

The launcher submits this with ``sbatch``. It runs on the compute node (host Python 3.10):

1. Import the NeMo-RL image into the shared cache (first run only).
2. Forward the Gym sandbox's agent ports 21002-21004 to 127.0.0.1 here, over a Tailcat tunnel.
3. Run ``run_trainer`` as a job step in a pyxis container with all 8 GPUs.

Rollouts come from a Dedicated Inference deployment, which Gym calls over HTTPS.

Usage (from the launcher): sbatch ... --wrap "python3 -m dapo_nemo_rl.sunk_job <run dir>"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dapo_nemo_rl import tunnel
from dapo_nemo_rl.procs import log, read_env_file, run, spawn, supervise, wait_any, wait_for_file

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


def main(run_dir: Path, overrides: list[str]) -> int:
    supervise()
    cache = run_dir.parent.parent / "cache"
    os.environ.update(read_env_file(run_dir / "secrets.env"))
    os.environ["PYTHONPATH"] = "/workspace"
    (cache / "work").mkdir(parents=True, exist_ok=True)

    log(f"sunk: node {os.uname().nodename}, run dir {run_dir}")
    image = import_image(cache)
    tailcat = tunnel.install(cache / "bin")
    gym_address = wait_for_file(run_dir / "gym.addr").read_text().strip()
    tunnel.forward(tailcat, gym_address, list(GYM_PORTS), run_dir)

    mounts = {run_dir / "code": "/workspace", cache / "work": "/work", run_dir: "/job"}
    trainer = spawn(step(image, mounts, [TRAINER_PYTHON, "-m", "dapo_nemo_rl.run_trainer", *overrides]))
    return wait_any([trainer])


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), sys.argv[2:]))
