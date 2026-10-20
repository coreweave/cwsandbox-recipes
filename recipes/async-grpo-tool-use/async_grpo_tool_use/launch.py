"""Start a training run: NeMo-RL trainer, a sampler, and NeMo-Gym in a CoreWeave CPU sandbox.

Where the trainer runs (``TRAINER``):

* ``sunk`` (default): a Slurm job on one 8-GPU node of SUNK, or of any Slurm cluster with pyxis
  that you reach over ssh.
* ``sandbox``: a CoreWeave GPU sandbox with one GPU, for small models.

Where rollouts are sampled (``SAMPLER``):

* ``di`` (default, ``TRAINER=sunk`` only): a CoreWeave Dedicated Inference hot-load deployment.
* ``vllm``: the self-hosted vLLM of ``async_grpo_tool_use.sampler``, on part of the trainer's
  node (``TRAINER=sunk``) or in a second one-GPU sandbox (``TRAINER=sandbox``).

1. With ``SAMPLER=di``: create a temporary object storage key (``cwic``) and an inference
   gateway (``cwtrain``). With ``TRAINER=sandbox``: create the storage key only; the weights go
   from the trainer sandbox to the sampler sandbox through the bucket.
2. Start the Gym sandbox. It serves its agent ports through a Tailcat tunnel.
3. Start the trainer: copy the recipe to SUNK and submit a Slurm job (``sunk_job``), or start the
   trainer sandbox (``trainer_sandbox``) and, for its sampler, the sampler sandbox.
4. With ``SAMPLER=di``: once the trainer has staged the base model in the bucket, create a
   hot-load deployment of it behind the gateway and wait until it serves. With ``SAMPLER=vllm``:
   wait for the sampler's tunnel address. Pass the sampler to the trainer and to Gym.
5. Stream the trainer's log until it ends. Then delete everything the run created.

Usage: python -m async_grpo_tool_use.launch [--max-num-steps N] [-- hydra overrides...]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shlex
import sys
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from async_grpo_tool_use.dapo_data import AIME_PROBLEMS, EVAL_REPEATS
from async_grpo_tool_use.inference import DeploymentSettings, Inference, deployment_spec
from async_grpo_tool_use.sunk import Sunk
from async_grpo_tool_use.trainer_sandbox import EXIT_LINE

RECIPE_TAG = "async-grpo"
WORKSPACE = "/workspace"
GYM_IMAGE = "python:3.13.16"
# The image sunk_job imports on Slurm, as a registry reference for sandboxes.
TRAINER_IMAGE = "nvcr.io/nvidia/nemo-rl@sha256:75c4821c7f24d6a045d22ce51756bd7ea9ec208020c90233723b9d7a9727d681"
MODEL_REPO = "Qwen/Qwen2.5-Math-7B"
MODEL_REVISION = "b101308fe89651ea5ce025f25317fea6fc07e96e"
# The default with TRAINER=sandbox: it trains on one GPU, and vLLM serves it on another.
SMALL_MODEL_REPO = "Qwen/Qwen2.5-Math-1.5B"
SMALL_MODEL_REVISION = "4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
GPU_SANDBOX_TYPE = "rtxp6000-8x"  # serverless GPU sandboxes: RTX PRO 6000 Blackwell, 96 GB
MOUNT_DIRS = ("async_grpo_tool_use/", "config/", "patches/", "gym_env/")
WANDB_PROJECT = "async-grpo-tool-use"
# Copied from the host environment to the SUNK job when set.
FORWARDED_ENV = ("MAX_NUM_STEPS", "DAPO_MAX_SAMPLES", "HF_TOKEN", "GYM_REMOTE_CONCURRENCY", "DASHBOARD_METRICS")
# Copied from the host environment to the Gym sandbox when set: the sandbox pool, the dashboard
# metrics and the share of training rollouts traced in Agent Lens.
GYM_FORWARDED_ENV = (
    "SANDBOX_POOL_MIN", "SANDBOX_POOL_MAX", "SANDBOX_POOL_IDLE_S", "SANDBOX_PLACEMENT", "DASHBOARD_METRICS",
    "AGENT_LENS_SAMPLE",
)  # fmt: skip
ACTIVE_STATES = ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED")


class PeerStopped(RuntimeError):
    """The Gym sandbox or the Slurm job stopped early."""


def log(message: str) -> None:
    print(f"[grpo] {message}", flush=True)


def stop_tagged(sdk: Any, tag: str, attempts: int = 3) -> int:
    """Stop every sandbox with ``tag``. The stops run concurrently: one at a time, a run with
    hundreds of rollout sandboxes took over 30 minutes to clean up. Stops that fail (a DNS
    hiccup at this volume) are retried."""
    sandboxes = sdk.Sandbox.list(tags=[tag]).result()
    pending = list(sandboxes)
    for attempt in range(attempts):
        ops = [(sandbox, sandbox.stop(missing_ok=True)) for sandbox in pending]
        pending, error = [], None
        for sandbox, op in ops:
            try:
                op.result()
            except sdk.exceptions.CWSandboxError as exc:
                pending.append(sandbox)
                error = exc
        if not pending:
            return len(sandboxes)
        time.sleep(5)
    raise RuntimeError(f"{len(pending)} sandbox(es) did not stop: {error}")


@dataclass(frozen=True)
class LaunchConfig:
    runner_id: str  # CoreWeave sandbox runner for the Gym sandbox; empty: serverless
    api_token: str  # CoreWeave API token: sandboxes, inference requests and hot-loads
    sunk: Sunk | None  # None with TRAINER=sandbox
    sunk_dir: str  # e.g. /mnt/data/<you>/async-grpo-tool-use; holds runs/ and cache/
    project_root: Path
    trainer: str = "sunk"  # sunk or sandbox
    sampler: str = "di"  # di or vllm
    model_repo: str = MODEL_REPO
    model_revision: str = MODEL_REVISION
    gpu_type: str = GPU_SANDBOX_TYPE
    trainer_cpu: str = "16"
    trainer_memory: str = "40Gi"
    sampler_cpu: str = "8"
    sampler_replicas: int = 1  # sampler sandboxes, one GPU each (TRAINER=sandbox)
    sampler_memory: str = "32Gi"
    bucket: str = "gradient-hotloading"
    bucket_prefix: str = "cookbook/async-grpo-tool-use"  # the inference service must be able to read it
    deployment: DeploymentSettings = field(default_factory=DeploymentSettings)
    partition: str = "h100"
    time_limit: str = "12:00:00"
    gym_cpu: str = "8"
    gym_memory: str = "32Gi"
    max_lifetime_seconds: int = 12 * 3600
    hydra_overrides: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    gym_env: dict[str, str] = field(default_factory=dict)


def collect_mounted_files(project_root: Path) -> list[dict[str, str]]:
    """The recipe source files the Gym sandbox needs, mounted under /workspace."""
    root = project_root.resolve()
    mounts = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_file() and rel.startswith(MOUNT_DIRS) and "__pycache__" not in rel:
            mounts.append({"mount_path": f"{WORKSPACE}/{rel}", "file_content": path.read_text("utf-8")})
    return mounts


def retry(action: Callable[[], Any], what: str, timeout: float = 1800.0, every: float = 10.0) -> Any:
    """Run ``action`` until it returns a value that is not None. ``PeerStopped`` raises at once."""
    deadline = time.monotonic() + timeout
    error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            value = action()
            if value is not None:
                return value
        except PeerStopped:
            raise
        except Exception as exc:  # noqa: BLE001 - e.g. the container is not up yet
            error = exc
        time.sleep(every)
    raise RuntimeError(f"timed out: {what} ({error})")


def create_with_retry(create: Callable[[], Any], what: str, timeout: float = 900.0) -> Any:
    """Retry a create rejected for lack of capacity ("unavailable", "resource exhausted").
    Other errors raise at once."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return create()
        except Exception as exc:
            capacity = any(word in type(exc).__name__ for word in ("Unavailable", "ResourceExhausted"))
            if not capacity or time.monotonic() > deadline:
                raise
            log(f"{what}: no capacity yet; retrying")
        time.sleep(20)


def sandbox_alive(sandbox: Any) -> bool:
    """False once the sandbox stopped or its container restarted (a restart loses the
    tunnel and the endpoint file, so the run cannot recover)."""
    try:
        status = str(sandbox.get_status()).lower()
        restarts = sum(c.restart_count or 0 for c in sandbox.container_statuses or ())
    except Exception:  # noqa: BLE001 - a failed status call does not mean the sandbox stopped
        return True
    return restarts == 0 and any(state in status for state in ("running", "creating", "pending", "starting"))


def check_write(s3: Any, bucket: str, prefix: str) -> None:
    """Fail early if the storage key cannot list, read and write under ``prefix``.

    cwtrain's check starts and aborts a multipart upload, so it leaves nothing behind."""
    from cwtrain.hotload.snapshot import check_snapshot_access

    check_snapshot_access(s3, bucket, prefix)


def s3_client(key: dict[str, str], region: str) -> Any:
    from cwtrain.infra.storage import s3_client

    return s3_client(region_name=region, aws_access_key_id=key["accessKeyID"], aws_secret_access_key=key["secretKey"])


class Launcher:
    """Owns one run. ``sdk`` is the ``cwsandbox`` module (fakes in tests)."""

    def __init__(
        self,
        config: LaunchConfig,
        sdk: Any,
        *,
        inference: Inference | None = None,
        s3_factory: Callable[[dict[str, str], str], Any] = s3_client,
        sink: Callable[[str], None] | None = None,
    ):
        self.config = config
        self.sdk = sdk
        self.sunk = config.sunk
        if inference is None:
            from cwtrain import InferenceClient

            inference = Inference(InferenceClient(token=config.api_token))
        self.inference = inference
        self.s3_factory = s3_factory
        self.sink = sink or (lambda line: print(line, end="", flush=True))
        self.run_tag = f"{RECIPE_TAG}-{uuid.uuid4().hex[:8]}"
        self.run_dir = f"{config.sunk_dir}/runs/{self.run_tag}"
        self.run_prefix = f"{config.bucket_prefix}/runs/{self.run_tag}"
        model = f"{config.model_repo.split('/')[-1]}-{config.model_revision[:8]}"
        self.model_cache_prefix = f"{config.bucket_prefix}/models/{model}"
        self.gym: Any = None
        self.trainer: Any = None  # the trainer sandbox (TRAINER=sandbox)
        self.trainer_exit: int | None = None  # from the trainer sandbox's last log line
        self.samplers: list[Any] = []  # the sampler sandboxes (TRAINER=sandbox), SAMPLER_REPLICAS of them
        self.sampler_gpus = 0  # GPUs of the self-hosted sampler, once it serves
        self.job_id: str | None = None
        self.tail: Any = None
        self.access_key: dict[str, str] | None = None
        self.bucket_zone = ""
        self.gateway_id: str | None = None
        self.endpoint = ""
        self.deployment_id: str | None = None

    # ------------------------------------------------------------------ resources

    def create_storage_key(self) -> None:
        self.bucket_zone = self.inference.bucket_zone(self.config.bucket)
        self.access_key = self.inference.create_access_key(self.run_tag, self.config.max_lifetime_seconds + 3600)
        uri = f"s3://{self.config.bucket}/{self.run_prefix}"
        s3 = self.s3_factory(self.access_key, self.bucket_zone)
        # A new key can take a few seconds to work; a missing permission keeps failing.
        retry(lambda: check_write(s3, self.config.bucket, self.run_prefix) or True, f"write access to {uri}",
              timeout=60, every=10)  # fmt: skip
        log(f"storage: temporary access key can write {uri}")

    def create_gateway(self) -> None:
        self.gateway_id, self.endpoint = self.inference.create_gateway(self.run_tag, self.config.deployment.zone)
        log(f"inference: gateway {self.gateway_id} ready at {self.endpoint}")

    def create_deployment(self) -> None:
        settings = self.config.deployment
        spec = deployment_spec(
            self.run_tag, self.gateway_id, self.config.bucket, f"{self.run_prefix}/base",
            f"{self.run_prefix}/hotload", settings,
        )  # fmt: skip
        self.deployment_id = self.inference.create_deployment(spec)
        log(f"inference: deployment {self.deployment_id} requested ({settings.replicas} x {settings.instance_type})")
        self.inference.wait_deployment(self.deployment_id)
        log("inference: deployment ready")

    def inference_env(self) -> bytes:
        values = {
            "INFERENCE_BASE_URL": f"{self.endpoint.rstrip('/')}/v1",
            "INFERENCE_MODEL": self.run_tag,
            "HOTLOAD_DEPLOYMENT_ID": self.deployment_id,
        }
        return "".join(f"{key}={value}\n" for key, value in values.items()).encode()

    # ------------------------------------------------------------------ gym sandbox

    def start_gym(self) -> Any:
        sdk = self.sdk
        limits = {"cpu": self.config.gym_cpu, "memory": self.config.gym_memory}
        env = {
            "PYTHONPATH": WORKSPACE,
            "RUN_TAG": self.run_tag,
            "MODEL_REPO": self.config.model_repo,  # Gym's model server tokenizes with the model's tokenizer
            "MODEL_REVISION": self.config.model_revision,
            "COREWEAVE_API_TOKEN": self.config.api_token,  # for the inference gateway
            # Gym's episode server creates the rollout sandboxes.
            "CWSANDBOX_API_KEY": self.config.api_token,
            "CWSANDBOX_RUNNER_ID": self.config.runner_id,
            **self.config.gym_env,
        }
        self.gym = create_with_retry(
            lambda: sdk.Sandbox.run(
                "python",
                "-m",
                "async_grpo_tool_use.run_gym",
                container_image=GYM_IMAGE,
                resources=sdk.ResourceOptions(requests=limits, limits=limits),
                mounted_files=collect_mounted_files(self.config.project_root),
                environment_variables=env,
                max_lifetime_seconds=self.config.max_lifetime_seconds,
                tags=[RECIPE_TAG, self.run_tag, f"{RECIPE_TAG}-gym"],
                **gym_placement(self.config.runner_id),
            ),
            "gym",
        )
        log(f"gym: sandbox {self.gym.sandbox_id} requested (CPU)")
        return self.gym

    def gym_alive(self) -> bool:
        return sandbox_alive(self.gym)

    def stream_gym_log(self) -> None:
        self.stream_sandbox_log(self.gym, "[gym] ")

    def stream_sandbox_log(self, sandbox: Any, prefix: str) -> None:
        """Print a sandbox's output with ``prefix``, without package install lines."""
        noise = re.compile(
            r"^(\(\w+\) )?\s*[+-] \S+==|^(\(\w+\) )?(Resolved|Prepared|Installed|Uninstalled|Audited|Checked) "
        )

        def pump() -> None:
            try:
                for line in sandbox.stream_logs(follow=True):
                    if not noise.search(line):
                        self.sink(f"{prefix}{line}" if line.endswith("\n") else f"{prefix}{line}\n")
                    if sandbox is self.trainer and EXIT_LINE in line:
                        self.trainer_exit = int(line.rsplit(EXIT_LINE, 1)[1].strip())
            except Exception:  # noqa: BLE001 - the stream ends when the sandbox stops
                pass

        threading.Thread(target=pump, daemon=True).start()

    # ------------------------------------------------------------------ GPU sandboxes

    def start_gpu_sandbox(self, role: str, module: str, args: list[str], cpu: str, memory: str) -> Any:
        """A serverless sandbox with one GPU on the NeMo-RL image, running ``module``."""
        limits = {"cpu": cpu, "memory": memory}
        env = {**self.trainer_env(), "PYTHONPATH": WORKSPACE, "PYTHONUNBUFFERED": "1"}
        sandbox = create_with_retry(
            lambda: self.sdk.Sandbox.run(
                "/opt/nemo_rl_venv/bin/python",
                "-m",
                module,
                *args,
                container_image=TRAINER_IMAGE,
                resources={"requests": limits, "limits": limits, "gpu": {"count": 1, "type": self.config.gpu_type}},
                mounted_files=collect_mounted_files(self.config.project_root),
                environment_variables=env,
                max_lifetime_seconds=self.config.max_lifetime_seconds,
                tags=[RECIPE_TAG, self.run_tag, f"{RECIPE_TAG}-{role}"],
                placement_mode="serverless",
            ),  # fmt: skip
            f"{role} GPU sandbox",
            timeout=3600,
        )
        log(f"{role}: sandbox {sandbox.sandbox_id} requested (1 GPU)")
        return sandbox

    def sampler_env(self, addresses: list[str]) -> bytes:
        """What the trainer and Gym need to reach the self-hosted samplers through the tunnel:
        replica ``i`` is local at ports ``VLLM_PORT + i`` and ``RECEIVER_PORT + i``."""
        from async_grpo_tool_use.sampler import RECEIVER_PORT, SERVED_MODEL, VLLM_PORT

        replicas = range(len(addresses))
        values = {
            "INFERENCE_BASE_URL": f"http://127.0.0.1:{VLLM_PORT}/v1",
            "INFERENCE_BASE_URLS": ",".join(f"http://127.0.0.1:{VLLM_PORT + i}/v1" for i in replicas),
            "INFERENCE_MODEL": SERVED_MODEL,
            "SAMPLER_ADDRS": ",".join(addresses),
            "SAMPLER_URLS": ",".join(f"http://127.0.0.1:{RECEIVER_PORT + i}" for i in replicas),
            # On Slurm the sampler is a step on the trainer's node and loads each export in place.
            # GPU sandboxes send them through object storage.
            "SAMPLER_TRANSFER": "shared" if self.config.trainer == "sunk" else "bucket",
        }
        return "".join(f"{key}={value}\n" for key, value in values.items()).encode()

    # ------------------------------------------------------------------ sunk job

    def trainer_env(self) -> dict[str, str]:
        values = {
            "RUN_TAG": self.run_tag,
            "MODEL_REPO": self.config.model_repo,
            "MODEL_REVISION": self.config.model_revision,
            "COREWEAVE_API_TOKEN": self.config.api_token,
            "SAMPLER": self.config.sampler,
            **self.config.env,
        }
        if "WANDB_API_KEY" in values:  # the dashboard's own W&B run joins this group
            values["WANDB_RUN_GROUP"] = self.run_tag
        if self.access_key:  # the trainer uploads the weights; the sandbox sampler downloads them
            values.update({
                "AWS_ACCESS_KEY_ID": self.access_key["accessKeyID"],
                "AWS_SECRET_ACCESS_KEY": self.access_key["secretKey"],
                "AWS_REGION": self.bucket_zone,
                "HOTLOAD_BUCKET": self.config.bucket,
            })  # fmt: skip
        if self.config.sampler == "di":  # the trainer stages the model for the deployment
            values.update(MODEL_CACHE_PREFIX=self.model_cache_prefix, MODEL_RUN_PREFIX=f"{self.run_prefix}/base")
        elif self.config.trainer == "sandbox":
            values.update(SAMPLER_WEIGHTS_PREFIX=f"{self.run_prefix}/weights")
        return values

    def secrets_env(self) -> bytes:
        return "".join(f"{key}={value}\n" for key, value in self.trainer_env().items()).encode()

    def submit_job(self, gym_address: str) -> str:
        sunk, run_dir = self.sunk, self.run_dir
        root = self.config.project_root
        sunk.sync([root / d.rstrip("/") for d in MOUNT_DIRS], f"{run_dir}/code")
        sunk.write_private(f"{run_dir}/secrets.env", self.secrets_env())
        sunk.write_private(f"{run_dir}/gym.addr", gym_address.encode())
        self.job_id = sunk.submit(
            [
                f"--partition={self.config.partition}", "--nodes=1", "--gres=gpu:8", "--cpus-per-task=112", "--mem=0",
                f"--time={self.config.time_limit}", f"--job-name={self.run_tag}", f"--output={run_dir}/job.log",
                f"--chdir={run_dir}",
            ],
            ["env", f"PYTHONPATH={run_dir}/code", "python3", "-m", "async_grpo_tool_use.sunk_job", run_dir,
             *self.config.hydra_overrides],
        )  # fmt: skip
        log(f"sunk: job {self.job_id} submitted ({run_dir})")
        return self.job_id

    def stream_log(self) -> None:
        self.tail = self.sunk.tail(f"{self.run_dir}/job.log")

        def pump() -> None:
            for line in self.tail.stdout:
                self.sink(line)

        threading.Thread(target=pump, daemon=True).start()

    def check_peers(self) -> str:
        """Raise ``PeerStopped`` if the trainer, a sampler sandbox or the Gym sandbox ended.
        Return the trainer's state."""
        if self.trainer is not None:
            state = "RUNNING" if sandbox_alive(self.trainer) else "STOPPED"
            if state == "STOPPED":
                raise PeerStopped("the trainer sandbox stopped")
        else:
            state = self.sunk.state(self.job_id)
            if state not in ACTIVE_STATES:
                raise PeerStopped(f"sunk: job {self.job_id} {state}")
        for replica, sampler in enumerate(self.samplers):
            if not sandbox_alive(sampler):
                raise PeerStopped(f"sampler-{replica} stopped or restarted; see the [sampler-{replica}] lines")
        if not self.gym_alive():
            raise PeerStopped("the Gym sandbox stopped or restarted; see the [gym] lines")
        return state

    def wait_for_sunk_file(self, name: str, what: str) -> str:
        def read() -> str | None:
            self.check_peers()
            return (self.sunk.read(f"{self.run_dir}/{name}") or "").strip() or None

        return retry(read, what, timeout=4 * 3600, every=20)

    def follow_job(self) -> int:
        """Wait until the trainer ends. Return its exit code."""
        deadline = time.monotonic() + self.config.max_lifetime_seconds + 600
        while True:
            time.sleep(30)
            if self.trainer is not None:
                if not sandbox_alive(self.trainer):
                    # The exit code can be recorded a few seconds after the terminal status.
                    code = None
                    for _ in range(6):
                        time.sleep(5)
                        code = self.trainer.returncode
                        if code is not None:
                            break
                    if code is None:
                        code = self.trainer_exit
                    status = self.trainer.get_status()
                    log(f"trainer: sandbox {str(getattr(status, 'value', status)).lower()}, exit code {code}")
                    return 1 if code is None else code
                self.check_peers()
                continue
            try:
                self.check_peers()
            except PeerStopped:
                state = self.sunk.state(self.job_id)
                if state in ACTIVE_STATES:
                    raise  # the Gym sandbox stopped while the job runs
                time.sleep(5)  # let the last lines arrive
                log(f"sunk: job {self.job_id} {state}")
                return 0 if state == "COMPLETED" else self.sunk.exit_code(self.job_id) or 1
            if time.monotonic() > deadline:
                raise RuntimeError("the job did not finish within its lifetime")

    # ------------------------------------------------------------------ run

    def start(self) -> None:
        log(f"run tag {self.run_tag}")
        if self.config.sampler == "di" or self.config.trainer == "sandbox":
            self.create_storage_key()
        if self.config.sampler == "di":
            self.create_gateway()
        gym = self.start_gym()
        gym.wait(timeout=1800)
        self.stream_gym_log()

        def read_gym_address() -> str | None:
            if not self.gym_alive():
                raise PeerStopped("the Gym sandbox stopped or restarted; see the [gym] lines")
            return gym.read_file(f"{WORKSPACE}/run/gym.addr").result().decode().strip() or None

        gym_address = retry(read_gym_address, "the Gym tunnel address")
        if self.config.trainer == "sandbox":
            self.start_sandbox_trainer(gym_address)
            addresses = [self.read_sandbox_file(sampler, "sampler.addr", f"sampler-{replica}'s tunnel address")
                         for replica, sampler in enumerate(self.samplers)]  # fmt: skip
            env = self.sampler_env(addresses)
            self.trainer.write_file(f"{WORKSPACE}/run/inference.env", env).result()
        else:
            self.submit_job(gym_address)
            self.stream_log()
            if self.config.sampler == "di":
                self.wait_for_sunk_file(
                    "model.staged", "the trainer to stage the model (the job may be waiting for a node)"
                )
                self.create_deployment()
                env = self.inference_env()
            else:
                env = self.sampler_env([self.wait_for_sunk_file("sampler.addr", "the sampler step's tunnel address")])
            self.sunk.write_private(f"{self.run_dir}/inference.env", env)
        gym.write_file(f"{WORKSPACE}/run/inference.env", env).result()
        if self.config.sampler == "vllm":
            self.sampler_gpus = int(self.config.env["SAMPLER_GPUS"]) * self.config.sampler_replicas
        log("sampler: passed to the trainer and to Gym")

    def start_sandbox_trainer(self, gym_address: str) -> None:
        """Start the sampler and the trainer sandboxes together: they all pull the NeMo-RL image."""
        self.samplers = [
            self.start_gpu_sandbox(f"sampler-{replica}", "async_grpo_tool_use.sampler", [f"{WORKSPACE}/run"],
                                   self.config.sampler_cpu, self.config.sampler_memory)
            for replica in range(self.config.sampler_replicas)
        ]  # fmt: skip
        self.trainer = self.start_gpu_sandbox("trainer", "async_grpo_tool_use.trainer_sandbox",
                                              list(self.config.hydra_overrides), self.config.trainer_cpu,
                                              self.config.trainer_memory)  # fmt: skip
        self.trainer.wait(timeout=3600)
        self.trainer.write_file(f"{WORKSPACE}/run/gym.addr", gym_address.encode()).result()
        self.stream_sandbox_log(self.trainer, "")
        for replica, sampler in enumerate(self.samplers):
            sampler.wait(timeout=3600)
            self.stream_sandbox_log(sampler, f"[sampler-{replica}] ")

    def read_sandbox_file(self, sandbox: Any, name: str, what: str) -> str:
        def read() -> str | None:
            self.check_peers()
            return sandbox.read_file(f"{WORKSPACE}/run/{name}").result().decode().strip() or None

        return retry(read, what, timeout=3600, every=15)

    def stop_all(self) -> None:
        """Delete everything the run created. Each step runs even if an earlier one fails."""

        def step(what: str, action: Callable[[], Any], manual: str) -> None:
            try:
                action()
            except KeyboardInterrupt:
                # A second Ctrl+C must not leave billed resources behind.
                log(f"interrupted while trying to {what}; continuing cleanup (if it stuck, run: {manual})")
            except Exception as exc:  # noqa: BLE001
                log(f"could not {what} ({exc}); run: {manual}")

        if self.tail is not None:
            self.tail.terminate()
        if self.job_id:
            run_dir = shlex.quote(self.run_dir)

            def cancel() -> None:
                self.sunk.cancel(self.job_id)
                # Tokens, keys and endpoints; the logs stay.
                self.sunk.run(f"rm -f {run_dir}/secrets.env {run_dir}/inference.env {run_dir}/*.addr", check=False)
                log(f"sunk: job {self.job_id} cancelled")

            step("cancel the Slurm job", cancel, f"ssh {self.sunk.host} scancel {self.job_id}")

        def stop_sandboxes() -> None:  # the Gym sandbox and the rollout sandboxes it started
            log(f"stopped {stop_tagged(self.sdk, self.run_tag)} sandbox(es)")

        step("stop sandboxes", stop_sandboxes, f"python -m async_grpo_tool_use.cleanup --tag {self.run_tag}")
        if self.deployment_id:  # before the gateway, which cannot be deleted while in use

            def delete_deployment() -> None:
                self.inference.delete_deployment(self.deployment_id)
                log(f"inference: deployment {self.deployment_id} deleted")

            step("delete the deployment", delete_deployment, f"cwic inference deployment delete {self.deployment_id}")
        if self.gateway_id:

            def delete_gateway() -> None:
                self.inference.delete_gateway(self.gateway_id)
                log(f"inference: gateway {self.gateway_id} deleted")

            step("delete the gateway", delete_gateway, f"cwic inference gateway delete {self.gateway_id}")
        if self.access_key:
            from async_grpo_tool_use import caios

            def delete_objects() -> None:
                s3 = self.s3_factory(self.access_key, self.bucket_zone)
                count = caios.delete_prefix(s3, self.config.bucket, self.run_prefix)
                log(f"storage: deleted {count} object(s) under s3://{self.config.bucket}/{self.run_prefix}")

            step("delete the run's objects", delete_objects, f"delete s3://{self.config.bucket}/{self.run_prefix}/")

            def revoke_key() -> None:
                self.inference.delete_access_key(self.access_key["accessKeyID"])
                log("storage: access key revoked")

            step("revoke the access key", revoke_key, "cwic cwobject access-key list")
        log("cleanup: done")

    def run(self) -> int:
        try:
            self.start()
            code = self.follow_job()
            log(f"trainer exited with {code}")
            return code
        except PeerStopped as exc:
            log(str(exc))
            return 1
        except KeyboardInterrupt:
            log("interrupted")
            return 130
        finally:
            self.stop_all()


def wandb_entity(api_key: str) -> str:
    """The default entity of a W&B API key."""
    auth = base64.b64encode(f"api:{api_key}".encode()).decode()
    request = urllib.request.Request(
        "https://api.wandb.ai/graphql", data=json.dumps({"query": "query { viewer { entity } }"}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Basic {auth}"},
    )  # fmt: skip
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)["data"]["viewer"]["entity"]


def wandb_env(environ: dict[str, str]) -> dict[str, str]:
    """W&B settings for the trainer (metrics) and the Gym sandbox (Agent Lens, Weave). Empty
    without ``WANDB_API_KEY``."""
    key = environ.get("WANDB_API_KEY")
    if not key:
        return {}
    return {
        "WANDB_API_KEY": key,
        "WANDB_ENTITY": environ.get("WANDB_ENTITY") or wandb_entity(key),
        "WANDB_PROJECT": environ.get("WANDB_PROJECT") or WANDB_PROJECT,
    }


def gym_placement(runner_id: str) -> dict[str, Any]:
    """The Gym sandbox runs on your runner if you name one, else on serverless capacity."""
    if not runner_id:
        return {"placement_mode": "serverless"}
    return {"placement_mode": "cks", "placement_spillover": "strict", "runner_ids": [runner_id]}


def val_period(overrides: tuple[str, ...], config: Path) -> int:
    """``grpo.val_period`` from the Hydra overrides, else from the config file."""
    for override in overrides:
        if override.startswith("grpo.val_period="):
            return int(override.split("=", 1)[1])
    return int(re.search(r"^\s+val_period: (\d+)", config.read_text(), re.MULTILINE).group(1))


def config_from_env(args: argparse.Namespace, environ: dict[str, str]) -> LaunchConfig:
    def need(name: str) -> str:
        if not environ.get(name):
            raise SystemExit(f"set {name}")
        return environ[name]

    root = Path(__file__).resolve().parents[1]
    wandb = wandb_env(environ)
    env = {key: environ[key] for key in FORWARDED_ENV if environ.get(key)}
    if args.max_num_steps is not None:
        env["MAX_NUM_STEPS"] = str(args.max_num_steps)
    env.setdefault("MAX_NUM_STEPS", "20")
    gym_env = {key: environ[key] for key in GYM_FORWARDED_ENV if environ.get(key)}
    # The Gym sandbox labels each validation round with its step and logs it once all its rows are graded.
    gym_env.update(
        MAX_NUM_STEPS=env["MAX_NUM_STEPS"],
        VAL_PERIOD=str(val_period(tuple(args.overrides), root / "config" / "grpo_async_qwen.yaml")),
        EVAL_ROWS=str(AIME_PROBLEMS * EVAL_REPEATS),
    )
    trainer, sampler = environ.get("TRAINER", "sunk"), environ.get("SAMPLER", "di")
    if (trainer, sampler) not in (("sunk", "di"), ("sunk", "vllm"), ("sandbox", "vllm")):
        raise SystemExit("set TRAINER=sunk with SAMPLER=di or vllm, or TRAINER=sandbox with SAMPLER=vllm")
    on_slurm = trainer == "sunk"
    replicas = int(environ.get("SAMPLER_REPLICAS", "1"))
    if replicas < 1 or (on_slurm and replicas != 1):
        raise SystemExit("SAMPLER_REPLICAS is the number of sampler GPU sandboxes; it needs TRAINER=sandbox")
    if sampler == "vllm":
        # Throughput metrics only; the sampler step takes SAMPLER_GPUS, a sandbox has one GPU.
        env["SAMPLER_GPUS"] = environ.get("SAMPLER_GPUS", "4") if on_slurm else "1"
        if on_slurm:
            env["TRAINER_GPUS"] = str(8 - int(env["SAMPLER_GPUS"]))
    if not on_slurm:
        env.update(TRAINER_GPUS="1")
    runner_id = environ.get("CWSANDBOX_RUNNER_ID", "")
    if not runner_id and environ.get("SANDBOX_PLACEMENT", "serverless") != "serverless":
        raise SystemExit("SANDBOX_PLACEMENT=cks or cks_then_serverless needs CWSANDBOX_RUNNER_ID")
    small = not on_slurm
    defaults = DeploymentSettings()
    return LaunchConfig(
        runner_id=runner_id,
        api_token=need("CWSANDBOX_API_KEY"),
        sunk=Sunk(host=environ.get("SUNK_SSH_HOST", "sunk"), key=environ.get("SUNK_SSH_KEY") or None)
        if on_slurm
        else None,
        sunk_dir=need("SUNK_DATA_DIR").rstrip("/") + "/async-grpo-tool-use" if on_slurm else "",
        project_root=root,
        trainer=trainer,
        sampler=sampler,
        model_repo=environ.get("MODEL_REPO") or (SMALL_MODEL_REPO if small else MODEL_REPO),
        model_revision=environ.get("MODEL_REVISION") or (SMALL_MODEL_REVISION if small else MODEL_REVISION),
        gpu_type=environ.get("GPU_SANDBOX_TYPE", GPU_SANDBOX_TYPE),
        trainer_memory=environ.get("TRAINER_SANDBOX_MEMORY", "40Gi"),
        sampler_memory=environ.get("SAMPLER_SANDBOX_MEMORY", "32Gi"),
        sampler_replicas=replicas,
        bucket=need("HOTLOAD_BUCKET") if sampler == "di" or not on_slurm else environ.get("HOTLOAD_BUCKET", ""),
        bucket_prefix=environ.get("HOTLOAD_BUCKET_PREFIX", "cookbook/async-grpo-tool-use").strip("/"),
        deployment=DeploymentSettings(
            zone=environ.get("INFERENCE_ZONE", defaults.zone),
            instance_type=environ.get("INFERENCE_INSTANCE_TYPE", defaults.instance_type),
            replicas=int(environ.get("INFERENCE_REPLICAS", defaults.replicas)),
        ),
        partition=environ.get("SUNK_PARTITION", "h100"),
        hydra_overrides=tuple(args.overrides),
        env={**env, **wandb},
        gym_env={**gym_env, **wandb},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Start an async GRPO run with tool use: a trainer, a sampler and CoreWeave Sandboxes."
    )
    parser.add_argument("--max-num-steps", type=int, default=None)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides for NeMo-RL (after --)")
    args = parser.parse_args(argv)
    if not os.environ.get("CWSANDBOX_API_KEY"):
        print("set CWSANDBOX_API_KEY (a CoreWeave API access token)", file=sys.stderr)
        return 2
    # cwic runs as a subprocess next to cwsandbox's gRPC client; keep gRPC's fork notices quiet.
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    import cwsandbox

    return Launcher(config_from_env(args, dict(os.environ)), cwsandbox).run()


if __name__ == "__main__":
    raise SystemExit(main())
