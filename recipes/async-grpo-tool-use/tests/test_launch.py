"""Launcher wiring with fakes for the cwsandbox SDK, SUNK, cwic and object storage. Nothing is created."""

import argparse
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from async_grpo_tool_use import launch

ROOT = Path(__file__).resolve().parents[1]


def done(value=None):
    return SimpleNamespace(result=lambda: value)


class FakeSandbox:
    def __init__(self, kwargs, sandbox_id):
        self.kwargs, self.sandbox_id = kwargs, sandbox_id
        self.files = {"/workspace/run/gym.addr": b"tcGYM\n"}
        self.stopped = False
        self.status = "running"
        self.restarts = 0

    def wait(self, timeout=None):
        return self

    def read_file(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        return done(self.files[path])

    def write_file(self, path, contents):
        self.files[path] = contents
        return done()

    def get_status(self):
        return self.status

    @property
    def container_statuses(self):
        return (SimpleNamespace(restart_count=self.restarts),)

    def stream_logs(self, follow):
        return iter(["(policy_model)  + uvicorn==0.54.0\n", "gym ready\n"])

    def stop(self, **kwargs):
        self.stopped = True
        return done()


class FakeSDK:
    def __init__(self):
        self.created = []
        sdk = self

        class Sandbox:
            @staticmethod
            def run(*command, **kwargs):
                sandbox = FakeSandbox(kwargs, f"sb-{len(sdk.created)}")
                sandbox.command = command
                sdk.created.append(sandbox)
                return sandbox

            @staticmethod
            def list(tags):
                return done([s for s in sdk.created if set(tags) <= set(s.kwargs["tags"])])

        self.Sandbox = Sandbox
        self.ResourceOptions = lambda **kw: kw


class FakeSunk:
    def __init__(self, states=("RUNNING", "COMPLETED")):
        self.files, self.private, self.calls = {}, set(), []
        self.states = list(states)
        self.job_command = None
        self.host = "sunk"

    def sync(self, sources, dest):
        self.calls.append(("sync", [s.name for s in sources], dest))

    def write_private(self, path, data):
        self.files[path] = data
        self.private.add(path)
        if path.endswith("gym.addr"):  # the job starts and the trainer stages the model
            self.files[path.replace("gym.addr", "model.staged")] = b"s3://bucket/base\n"

    def read(self, path):
        data = self.files.get(path)
        return data.decode() if data else None

    def submit(self, sbatch_args, command):
        self.job_command = (sbatch_args, command)
        return "123"

    def state(self, job_id):
        return self.states.pop(0) if len(self.states) > 1 else self.states[0]

    def exit_code(self, job_id):
        return 7

    def cancel(self, job_id):
        self.calls.append(("cancel", job_id))

    def run(self, command, **kwargs):
        self.calls.append(("run", command))
        return ""

    def tail(self, path):
        return SimpleNamespace(stdout=iter(["step 1\n"]), terminate=lambda: None)


class FakeInference:
    def __init__(self):
        self.calls, self.spec = [], None

    def bucket_zone(self, bucket):
        return "US-EAST-02A"

    def create_access_key(self, name, seconds):
        self.calls.append("key")
        return {"accessKeyID": "AK", "secretKey": "SK", "expiry": "later"}

    def create_gateway(self, name, zone):
        self.calls.append(("gateway", zone))
        return "gw-1", "https://gw.example"

    def create_deployment(self, spec):
        self.spec = spec
        return "dep-1"

    def wait_deployment(self, deployment_id):
        self.calls.append(("ready", deployment_id))

    def delete_deployment(self, deployment_id):
        self.calls.append(("delete deployment", deployment_id))

    def delete_gateway(self, gateway_id):
        self.calls.append(("delete gateway", gateway_id))

    def delete_access_key(self, key_id):
        self.calls.append(("revoke", key_id))


class FakeS3:
    def __init__(self):
        self.deleted = []

    def get_paginator(self, name):
        return SimpleNamespace(paginate=lambda Bucket, Prefix: [{"Contents": [{"Key": Prefix + "base/config.json"}]}])

    def delete_objects(self, Bucket, Delete):
        self.deleted += [o["Key"] for o in Delete["Objects"]]

    def list_multipart_uploads(self, Bucket, Prefix):
        return {}


def _launcher(sunk=None):
    environ = {"CWSANDBOX_RUNNER_ID": "my-runner", "CWSANDBOX_API_KEY": "cw-token", "SUNK_DATA_DIR": "/mnt/data/me",
               "HF_TOKEN": "hf"}  # fmt: skip
    args = argparse.Namespace(max_num_steps=2, overrides=["grpo.seed=1"])
    config = launch.config_from_env(args, environ)
    config = launch.LaunchConfig(**{**config.__dict__, "sunk": sunk or FakeSunk()})
    sdk, s3 = FakeSDK(), FakeS3()
    launcher = launch.Launcher(config, sdk, inference=FakeInference(), s3_factory=lambda key, zone: s3,
                               sink=lambda line: None)  # fmt: skip
    launcher.fake_s3 = s3
    return launcher, sdk


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(launch.time, "sleep", lambda s: None)
    monkeypatch.setattr(launch, "check_write", lambda s3, bucket, prefix: None)


def test_missing_write_access_stops_before_anything_is_created(monkeypatch):
    def denied(s3, bucket, prefix):
        raise PermissionError(f"PutObject denied on s3://{bucket}/{prefix}")

    monkeypatch.setattr(launch, "check_write", denied)
    monkeypatch.setattr(launch.time, "monotonic", iter(range(0, 10_000, 30)).__next__)
    launcher, sdk = _launcher()
    with pytest.raises(RuntimeError, match="write access"):
        launcher.run()
    calls = launcher.inference.calls
    assert not sdk.created and not any(c[0] == "gateway" for c in calls if isinstance(c, tuple))
    assert ("revoke", "AK") in calls  # the temporary key is still revoked


def test_run_wires_gym_sunk_job_and_deployment():
    launcher, sdk = _launcher()
    sunk, inference, tag = launcher.sunk, launcher.inference, launcher.run_tag
    assert launcher.run() == 0
    [gym] = sdk.created
    assert gym.command == ("python", "-m", "async_grpo_tool_use.run_gym")
    assert gym.kwargs["runner_ids"] == ["my-runner"] and tag in gym.kwargs["tags"]
    assert "gpu" not in gym.kwargs["resources"]
    run_dir = f"/mnt/data/me/async-grpo-tool-use/runs/{tag}"
    assert sunk.files[f"{run_dir}/gym.addr"] == b"tcGYM"  # Gym's address reached SUNK
    # The deployment loads the run's model copy and hot-loads from the run's prefix.
    prefix = f"cookbook/async-grpo-tool-use/runs/{tag}"
    spec = inference.spec
    assert spec["model"] == {"name": tag, "bucket": "gradient-hotloading", "path": f"{prefix}/base"}
    assert spec["hotLoad"]["pathPrefix"] == f"{prefix}/hotload" and spec["runtime"]["engine"] == "dynamo-vllm"
    assert spec["gatewayIds"] == ["gw-1"] and ("ready", "dep-1") in inference.calls
    # Both sides got the endpoint, privately on SUNK.
    endpoint = gym.files["/workspace/run/inference.env"].decode()
    assert "INFERENCE_BASE_URL=https://gw.example/v1" in endpoint and "HOTLOAD_DEPLOYMENT_ID=dep-1" in endpoint
    assert {f"{run_dir}/secrets.env", f"{run_dir}/gym.addr", f"{run_dir}/inference.env"} <= sunk.private
    secrets = sunk.files[f"{run_dir}/secrets.env"].decode()
    assert "HF_TOKEN=hf" in secrets and "MAX_NUM_STEPS=2" in secrets and "AWS_SECRET_ACCESS_KEY=SK" in secrets
    assert "AWS_REGION=US-EAST-02A" in secrets and f"MODEL_RUN_PREFIX={prefix}/base" in secrets
    sbatch_args, command = sunk.job_command
    assert "--gres=gpu:8" in sbatch_args and "--partition=h100" in sbatch_args
    assert command[-3:] == ["async_grpo_tool_use.sunk_job", run_dir, "grpo.seed=1"]
    # Cleanup: job, sandboxes, deployment before gateway, the run's objects, the key.
    assert ("cancel", "123") in sunk.calls and gym.stopped
    calls = inference.calls
    assert calls.index(("delete deployment", "dep-1")) < calls.index(("delete gateway", "gw-1"))
    assert launcher.fake_s3.deleted == [f"{prefix}/base/config.json"] and ("revoke", "AK") in calls


def test_cleanup_runs_every_step_after_a_failure():
    launcher, sdk = _launcher(FakeSunk(states=("FAILED",)))  # the job dies before staging the model
    assert launcher.run() == 1
    assert launcher.deployment_id is None and ("delete gateway", "gw-1") in launcher.inference.calls
    assert ("revoke", "AK") in launcher.inference.calls and sdk.created[0].stopped


def test_failed_job_returns_its_exit_code():
    launcher, _ = _launcher(FakeSunk(states=("RUNNING", "FAILED")))
    assert launcher.run() == 7


def test_gym_stop_ends_the_run():
    launcher, sdk = _launcher(FakeSunk(states=("RUNNING",)))
    launcher.start()
    launcher.tail = None
    sdk.created[0].status = "failed"
    with pytest.raises(launch.PeerStopped):
        launcher.follow_job()


def test_gym_restart_ends_the_run():
    launcher, sdk = _launcher(FakeSunk(states=("RUNNING",)))
    launcher.start()
    launcher.tail = None
    sdk.created[0].restarts = 1  # the container restarted: tunnel and endpoint file are gone
    with pytest.raises(launch.PeerStopped):
        launcher.follow_job()


def test_gym_log_drops_install_lines():
    lines = []
    launcher, sdk = _launcher()
    launcher.sink = lines.append
    launcher.gym = FakeSandbox({}, "sb")
    launcher.stream_gym_log()
    deadline = time.monotonic() + 5
    while not lines and time.monotonic() < deadline:
        pass
    assert lines == ["[gym] gym ready\n"]


def test_mounts_only_recipe_sources():
    mounts = {m["mount_path"] for m in launch.collect_mounted_files(ROOT)}
    assert "/workspace/async_grpo_tool_use/run_gym.py" in mounts
    assert "/workspace/gym_env/math_sandbox_eval/app.py" in mounts
    assert not any(p.endswith(".env") or "/tests/" in p or "__pycache__" in p for p in mounts)


def test_required_settings():
    args = argparse.Namespace(max_num_steps=None, overrides=[])
    with pytest.raises(SystemExit):
        launch.config_from_env(args, {"CWSANDBOX_RUNNER_ID": "r", "CWSANDBOX_API_KEY": "k"})


def test_create_retries_only_capacity_errors():
    class SandboxResourceExhaustedError(Exception):
        pass

    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise SandboxResourceExhaustedError("runner capacity exhausted")
        return "sandbox"

    assert launch.create_with_retry(flaky, "test") == "sandbox" and len(calls) == 3
    with pytest.raises(ValueError):
        launch.create_with_retry(lambda: (_ for _ in ()).throw(ValueError("bad spec")), "test")


@pytest.mark.skipif(not os.environ.get("NEMO_RL_SRC"), reason="set NEMO_RL_SRC to a NeMo-RL v0.7.0 checkout")
def test_patches_apply_to_nemo_rl_v070(tmp_path):
    src = Path(os.environ["NEMO_RL_SRC"])
    work = tmp_path / "rl"
    subprocess.run(["git", "clone", "-q", str(src), str(work)], check=True)
    for patch in sorted((ROOT / "patches").glob("*.patch")):
        subprocess.run(["git", "-C", str(work), "apply", str(patch)], check=True)
    for rel in ("nemo_rl/algorithms/grpo.py", "nemo_rl/environments/nemo_gym.py", "nemo_rl/algorithms/utils.py"):
        compile((work / rel).read_text(), rel, "exec")


def test_stop_tagged_retries_the_sandboxes_whose_stop_failed(monkeypatch):
    monkeypatch.setattr(launch.time, "sleep", lambda s: None)

    class StopError(Exception):
        pass

    calls = {}

    def sandbox(name, failures):
        def stop(missing_ok):
            calls[name] = calls.get(name, 0) + 1

            def result():
                if calls[name] <= failures:
                    raise StopError("dns lookup failed")

            return SimpleNamespace(result=result)

        return SimpleNamespace(stop=stop)

    sandboxes = [sandbox("a", 0), sandbox("b", 1), sandbox("c", 0)]
    sdk = SimpleNamespace(
        Sandbox=SimpleNamespace(list=lambda tags: done(sandboxes)),
        exceptions=SimpleNamespace(CWSandboxError=StopError),
    )
    assert launch.stop_tagged(sdk, "run") == 3
    assert calls == {"a": 1, "b": 2, "c": 1}


def test_wandb_settings_reach_the_trainer_and_gym(monkeypatch):
    monkeypatch.setattr(launch, "wandb_entity", lambda key: "my-team")
    environ = {"CWSANDBOX_RUNNER_ID": "r", "CWSANDBOX_API_KEY": "k", "SUNK_DATA_DIR": "/mnt/data/me",
               "WANDB_API_KEY": "wk"}  # fmt: skip
    config = launch.config_from_env(argparse.Namespace(max_num_steps=3, overrides=["grpo.val_period=2"]), environ)
    for env in (config.env, config.gym_env):
        assert env["WANDB_ENTITY"] == "my-team" and env["WANDB_PROJECT"] == "async-grpo-tool-use"
    assert config.gym_env["MAX_NUM_STEPS"] == "3" and config.gym_env["VAL_PERIOD"] == "2"
    assert config.gym_env["EVAL_ROWS"] == "120"


def test_no_wandb_key_means_no_wandb_settings():
    environ = {"CWSANDBOX_RUNNER_ID": "r", "CWSANDBOX_API_KEY": "k", "SUNK_DATA_DIR": "/mnt/data/me"}
    config = launch.config_from_env(argparse.Namespace(max_num_steps=None, overrides=[]), environ)
    assert "WANDB_API_KEY" not in config.env and "WANDB_API_KEY" not in config.gym_env
    assert config.gym_env["VAL_PERIOD"] == "10" and config.env["MAX_NUM_STEPS"] == "20"
