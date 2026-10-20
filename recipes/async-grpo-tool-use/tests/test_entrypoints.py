"""Container entrypoint helpers, run locally with ordinary child processes."""

import sys
import time

from async_grpo_tool_use import procs, run_gym, run_trainer


def test_wait_any_returns_first_exit_and_stops_the_rest():
    quick = procs.spawn([sys.executable, "-c", "import sys; sys.exit(3)"])
    slow = procs.spawn([sys.executable, "-c", "import time; time.sleep(60)"])
    started = time.monotonic()
    assert procs.wait_any([quick, slow], poll=0.1) == 3
    assert slow.poll() is not None and time.monotonic() - started < 30


def test_read_env_file(tmp_path):
    path = tmp_path / "peers.env"
    path.write_text("INFERENCE_BASE_URL=https://gw.example/v1\n\n")
    assert procs.read_env_file(path) == {"INFERENCE_BASE_URL": "https://gw.example/v1"}


def test_http_status_is_zero_when_nothing_listens():
    assert procs.http_status("http://127.0.0.1:9/", timeout=1) == 0


def test_gym_overlay_binds_localhost_outside_nemo_rl_ports():
    text = run_gym.overrides_yaml("https://gw.example/v1", "async-grpo-1", "Qwen/Qwen2.5-Math-7B", "abc")
    assert "policy_base_url: https://gw.example/v1" in text and "policy_model_name: async-grpo-1" in text
    assert "policy_api_key: ${oc.env:COREWEAVE_API_TOKEN}" in text  # the token stays out of the file
    assert "entrypoint: token_ids_app.py" in text and "tokenizer_revision: abc" in text
    ports = (run_gym.HEAD_PORT, run_gym.TRAIN_AGENT_PORT, run_gym.EVAL_AGENT_PORT, run_gym.EVAL_SERVER_PORT)
    assert all(f"port: {port}" in text for port in ports) and "0.0.0.0" not in text
    assert min(ports) > 20000 and len(set(ports)) == 4


def test_grpo_command_overrides():
    cmd = run_trainer.grpo_command(["grpo.seed=1"], {"MAX_NUM_STEPS": "2", "WANDB_API_KEY": "k"})
    assert cmd[1] == "examples/nemo_gym/run_grpo_nemo_gym.py"
    assert cmd[-3:] == ["grpo.max_num_steps=2", "logger.wandb_enabled=true", "grpo.seed=1"]


def test_trainer_step_uses_the_whole_node(tmp_path):
    from async_grpo_tool_use import sunk_job

    image = sunk_job.image_path(tmp_path)
    assert image == tmp_path / "images" / "nemo-rl-v0.7.0.sqsh"
    step = sunk_job.step(image, {tmp_path: "/workspace"}, ["python3", "-m", "m"])
    assert step[:3] == ["srun", "--ntasks=1", "--exact"] and "--gres=gpu:8" in step
    assert f"--container-image={image}" in step
    assert f"--container-mounts={tmp_path}:/workspace" in step and step[-3:] == ["python3", "-m", "m"]


def test_cached_image_is_not_imported_again(tmp_path):
    from async_grpo_tool_use import sunk_job

    sunk_job.image_path(tmp_path).parent.mkdir(parents=True)
    sunk_job.image_path(tmp_path).write_text("")
    assert sunk_job.import_image(tmp_path) == sunk_job.image_path(tmp_path)  # no srun


def test_tunnel_address_file_is_private(tmp_path):
    from async_grpo_tool_use import tunnel

    tunnel.write_private(tmp_path / "x.addr", "tcSECRET")
    assert (tmp_path / "x.addr").stat().st_mode & 0o777 == 0o600
