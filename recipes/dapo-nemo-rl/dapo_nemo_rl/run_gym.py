"""Entrypoint of the Gym CPU sandbox.

Installs NeMo-Gym and starts two agents. The trainer on SUNK calls them through a Tailcat
tunnel. Gym samples from the Dedicated Inference deployment over HTTPS:

* training agent: ``math_with_judge``, grades in the Gym process
* evaluation agent: ``math_sandbox_eval`` (from ``gym_env/``), grades in CoreWeave sandboxes

The tunnel makes Gym's ports local on the SUNK node, so they start at 21001 to stay clear of
NeMo-RL and Ray ports there.

Usage: python -m dapo_nemo_rl.run_gym
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from dapo_nemo_rl import tunnel
from dapo_nemo_rl.procs import log, read_env_file, require, run, spawn, supervise, wait_any, wait_for_file

GYM_REPO = "https://github.com/NVIDIA-NeMo/Gym.git"
GYM_DIR = Path("/opt/gym")
RUN_DIR = Path("/workspace/run")
EVAL_ENV = Path("/workspace/gym_env/math_sandbox_eval")
HEAD_PORT = 21001
TRAIN_AGENT_PORT = 21002
EVAL_AGENT_PORT = 21003
EVAL_SERVER_PORT = 21004  # the trainer calls its /calibrate route directly
TOKEN_IDS_MODEL = Path("/workspace/gym_env/token_ids_model.py")
TRANSFORMERS = "transformers==5.14.1"
CONFIGS = (
    "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml",
    "resources_servers/math_with_judge/configs/math_with_judge.yaml",
    "resources_servers/math_sandbox_eval/configs/math_sandbox_eval.yaml",
)


def overrides_yaml(rollout_url: str, served_model: str, tokenizer: str, revision: str) -> str:
    """Gym config overlay: policy endpoint, localhost binds and fixed ports."""
    return f"""\
policy_base_url: {rollout_url}
policy_api_key: ${{oc.env:COREWEAVE_API_TOKEN}}
policy_model_name: {served_model}
port_range_low: 21010
port_range_high: 25000
skip_venv_if_present: true  # use the eval server venv that install() built
head_server:
  host: 127.0.0.1
  port: {HEAD_PORT}
policy_model:
  responses_api_models:
    vllm_model:
      host: 127.0.0.1
      uses_reasoning_parser: false
      # Sends token IDs to /v1/completions; see gym_env/token_ids_model.py.
      entrypoint: token_ids_app.py
      tokenizer: {tokenizer}
      tokenizer_revision: {revision}
math_with_judge:
  resources_servers:
    math_with_judge:
      host: 127.0.0.1
math_with_judge_simple_agent:
  responses_api_agents:
    simple_agent:
      host: 127.0.0.1
      port: {TRAIN_AGENT_PORT}
math_sandbox_eval:
  resources_servers:
    math_sandbox_eval:
      host: 127.0.0.1
      port: {EVAL_SERVER_PORT}
math_eval_agent:
  responses_api_agents:
    simple_agent:
      host: 127.0.0.1
      port: {EVAL_AGENT_PORT}
"""


def install(ref: str) -> Path:
    """Gym's per-server venvs install Gym from its source tree, so it runs from a checkout."""
    log(f"gym: installing NeMo-Gym {ref}")
    run([sys.executable, "-m", "pip", "install", "--quiet", f"uv=={os.environ.get('UV_VERSION', '0.12.19')}"])
    run(["git", "clone", "--quiet", "--depth", "1", "--branch", ref, GYM_REPO, str(GYM_DIR)])
    run(["uv", "venv", "--quiet", "--python", sys.executable, ".venv"], cwd=GYM_DIR)
    run(["uv", "pip", "install", "--quiet", "--python", ".venv/bin/python", "-e", "."], cwd=GYM_DIR)
    # Gym finds servers inside its own tree.
    server = GYM_DIR / "resources_servers" / EVAL_ENV.name
    shutil.copytree(EVAL_ENV, server)
    # cwsandbox needs cryptography, which Gym's pyproject excludes from server venvs.
    build_server_venv(server, ["-r", str(server / "requirements.txt"), "cryptography>=42"])
    model = GYM_DIR / "responses_api_models" / "vllm_model"
    shutil.copy(TOKEN_IDS_MODEL, model / "token_ids_app.py")
    build_server_venv(model, ["-e", f"{GYM_DIR}[dev]", TRANSFORMERS])  # the model server tokenizes
    return GYM_DIR / ".venv" / "bin" / "gym"


def build_server_venv(server: Path, requirements: list[str]) -> None:
    """Build a Gym server's venv ourselves; ``skip_venv_if_present`` makes Gym use it.

    Gym's pyproject excludes some packages from every server venv it builds. uv applies that
    rule only when it runs inside the Gym tree, so install from /tmp. Pin ray and openai to the
    main venv's versions, as Gym does."""
    main_python = GYM_DIR / ".venv" / "bin" / "python"
    pins = subprocess.run(
        [str(main_python), "-c", "import openai, ray; print(f'ray[default]=={ray.__version__} openai=={openai.__version__}')"],
        capture_output=True, text=True, check=True,
    ).stdout.split()  # fmt: skip
    venv = server / ".venv"
    run(["uv", "venv", "--quiet", "--seed", "--python", sys.executable, str(venv)], cwd="/tmp")
    run(["uv", "pip", "install", "--quiet", "--python", str(venv / "bin" / "python"), *requirements, *pins], cwd="/tmp")


def main() -> int:
    supervise()
    require("COREWEAVE_API_TOKEN")  # for the inference gateway
    for name in ("CWSANDBOX_API_KEY", "CWSANDBOX_RUNNER_ID", "DAPO_RUN_TAG"):
        require(name)  # used by the eval server's cwsandbox provider
    gym = install(os.environ.get("GYM_REF", "v0.6.0"))
    tailcat = tunnel.install(Path("/opt/tailcat"))
    ports = [TRAIN_AGENT_PORT, EVAL_AGENT_PORT, EVAL_SERVER_PORT]
    tunnel.serve(tailcat, ports, RUN_DIR / "gym.addr", RUN_DIR)  # the launcher passes this to SUNK
    log("gym: waiting for the inference deployment")
    inference = read_env_file(wait_for_file(RUN_DIR / "inference.env"))
    rollout_url = inference["INFERENCE_BASE_URL"]
    overlay = RUN_DIR / "gym_overrides.yaml"
    overlay.write_text(
        overrides_yaml(rollout_url, inference["INFERENCE_MODEL"], require("MODEL_REPO"), require("MODEL_REVISION"))
    )
    log(f"gym: starting servers against {rollout_url}")
    args = [arg for path in (*CONFIGS, str(overlay)) for arg in ("--config", path)]
    return wait_any([spawn([str(gym), "env", "start", *args], cwd=str(GYM_DIR))])


if __name__ == "__main__":
    raise SystemExit(main())
