"""Entrypoint of the trainer, in a Slurm job step or a GPU sandbox: patch NeMo-RL, stage the
model, calibrate, run GRPO.

1. Patch NeMo-RL and download the model.
2. With Dedicated Inference (``SAMPLER=di``): upload the model to object storage (once, cached)
   and copy it to this run's prefix. The launcher then creates the inference deployment from
   that copy. The self-hosted sampler (``SAMPLER=vllm``) downloads the model itself.
3. Prepare the data and run the verifier calibration while the sampler starts.
4. Wait for the sampler, then run GRPO. Rollouts come from Gym and the sampler.

Arguments are extra Hydra overrides for NeMo-RL. ``/job`` is the run directory.

Usage: /opt/nemo_rl_venv/bin/python -m async_grpo_tool_use.run_trainer [hydra overrides...]
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
from pathlib import Path

from async_grpo_tool_use import calibrate, dapo_data, model_prep
from async_grpo_tool_use.procs import (
    gpu_count,
    log,
    read_env_file,
    require,
    run,
    spawn,
    supervise,
    wait_any,
    wait_for_file,
    wait_http,
)
from async_grpo_tool_use.run_gym import EVAL_AGENT_PORT, EVAL_SERVER_PORT, TRAIN_AGENT_PORT
from async_grpo_tool_use.sunk_job import report_gpus

WORKSPACE = Path("/workspace")
NEMO_RL_DIR = Path(os.environ.get("NEMO_RL_DIR", "/opt/nemo-rl"))
CONFIG_DIR = WORKSPACE / "config"
CONFIG = CONFIG_DIR / "grpo_async_qwen.yaml"
PATCHES = sorted((WORKSPACE / "patches").glob("*.patch"))
MODEL_DIR = Path("/work/models/base")
DATA_DIR = Path("/work/data")
CALIBRATION_REPORT = Path("/work/logs/calibration.json")
JOB_DIR = Path("/job")
CWTRAIN = "cwtrain==0.1.0"


def apply_patches() -> None:
    """Apply each patch once; refuse a NeMo-RL checkout it was not written for."""
    for patch in PATCHES:
        git = ["git", "-C", str(NEMO_RL_DIR), "apply"]
        if subprocess.run([*git, "--reverse", "--check", str(patch)], capture_output=True).returncode == 0:
            log(f"already applied: {patch.name}")
            continue
        check = subprocess.run([*git, "--check", str(patch)], capture_output=True, text=True)
        if check.returncode != 0:
            raise SystemExit(f"{patch.name} does not apply; this recipe is tested with NeMo-RL v0.7.0\n{check.stderr}")
        subprocess.run([*git, str(patch)], check=True)
        log(f"applied {patch.name}")


def expose_recipe_to_worker_venvs() -> None:
    """Ray imports the export worker in NeMo-RL's prebuilt per-worker venvs before any runtime
    env applies, so put the recipe on every venv's path with a ``.pth`` file."""
    venvs = os.environ.setdefault("NEMO_RL_VENV_DIR", "/opt/ray_venvs")
    for python in [sys.executable, *sorted(glob.glob(f"{venvs}/*/bin/python"))]:
        purelib = subprocess.run(
            [python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
        Path(purelib, "async_grpo_tool_use.pth").write_text(f"{WORKSPACE}\n")


def install_cwtrain() -> None:
    """The weight synchronizer hot-loads with cwtrain, which the image lacks."""
    run(["uv", "pip", "install", "--quiet", "--python", sys.executable, CWTRAIN])


def stage_model() -> None:
    """Upload the model once to a cached prefix, then copy it to this run's prefix.

    Each run's deployment gets its own copy: two deployments that load one model path can
    collide in the inference service's weight registry."""
    from async_grpo_tool_use import caios

    s3 = caios.client()
    bucket, cache, target = require("HOTLOAD_BUCKET"), require("MODEL_CACHE_PREFIX"), require("MODEL_RUN_PREFIX")
    if not caios.is_complete(s3, bucket, cache):
        log(f"trainer: uploading the model to s3://{bucket}/{cache}")
        caios.upload_dir(s3, MODEL_DIR, bucket, cache)
    caios.copy_prefix(s3, bucket, cache, target)
    (JOB_DIR / "model.staged").write_text(f"s3://{bucket}/{target}\n")
    log(f"trainer: model staged at s3://{bucket}/{target}")


def run_calibration(train_data: Path, eval_data: Path) -> None:
    """Grade known-right and known-wrong answers with the sandboxed eval verifier."""
    answers = sorted({row["expected_answer"] for row in dapo_data.read(eval_data)})
    answers += [row["expected_answer"] for row in dapo_data.read(train_data)[:50]]
    log(f"calibration: grading {len(answers) * 4} known cases in CoreWeave sandboxes")
    report = calibrate.run(f"http://127.0.0.1:{EVAL_SERVER_PORT}", answers, CALIBRATION_REPORT)
    for kind, row in report.items():
        log(f"calibration: {kind:<15} {row['agree']:.0%} graded as expected ({row['cases']} cases)")
    log(f"calibration: details in {CALIBRATION_REPORT}")


def grpo_command(extra: list[str], environ: dict[str, str]) -> list[str]:
    overrides = [f"grpo.max_num_steps={environ.get('MAX_NUM_STEPS', '20')}"]
    if environ.get("WANDB_API_KEY"):
        overrides.append("logger.wandb_enabled=true")
    return [sys.executable, "examples/nemo_gym/run_grpo_nemo_gym.py", "--config", str(CONFIG), *overrides, *extra]


def main(argv: list[str]) -> int:
    supervise()
    require("COREWEAVE_API_TOKEN")
    os.environ["PYTHONUNBUFFERED"] = "1"

    if os.environ.get("DASHBOARD_METRICS") == "1":
        report_gpus()
    dedicated_inference = os.environ.get("SAMPLER", "di") == "di"
    log(f"trainer: patching NeMo-RL in {NEMO_RL_DIR}")
    if os.environ.get("HOTLOAD_BUCKET"):  # weights go through object storage, with cwtrain's client
        install_cwtrain()
    apply_patches()
    expose_recipe_to_worker_venvs()
    model_prep.prepare(require("MODEL_REPO"), require("MODEL_REVISION"), MODEL_DIR)
    if dedicated_inference:
        stage_model()
    max_samples = int(os.environ["DAPO_MAX_SAMPLES"]) if os.environ.get("DAPO_MAX_SAMPLES") else None
    train_data, eval_data = dapo_data.prepare(DATA_DIR, max_samples)

    # The Tailcat tunnel makes the Gym sandbox's agents local on this node.
    agents = {
        "GYM_TRAIN_AGENT_URL": f"http://127.0.0.1:{TRAIN_AGENT_PORT}",
        "GYM_EVAL_AGENT_URL": f"http://127.0.0.1:{EVAL_AGENT_PORT}",
    }
    os.environ.update(agents)
    # Gym agents have no health route; FastAPI answers 404 for "/". The tunnel itself answers
    # with a 5xx while nothing listens behind it.
    for name, url in agents.items():
        wait_http(name, url + "/", ok=lambda s: 200 <= s < 500)
    run_calibration(train_data, eval_data)

    log("trainer: waiting for the sampler")
    os.environ.update(read_env_file(wait_for_file(JOB_DIR / "inference.env")))
    log(f"trainer: starting async GRPO on {gpu_count()} GPUs, rollouts from {require('INFERENCE_BASE_URL')}")
    if os.environ.get("WANDB_API_KEY"):
        # wandb.init reads the run id from here, so the dashboard can link the run before it starts.
        os.environ["WANDB_RUN_ID"] = require("RUN_TAG")
    return wait_any([spawn(grpo_command(argv, dict(os.environ)), cwd=str(NEMO_RL_DIR))])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
