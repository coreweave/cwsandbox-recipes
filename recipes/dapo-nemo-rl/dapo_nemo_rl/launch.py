"""Start a training run: NeMo-RL on SUNK, rollouts from CoreWeave Dedicated Inference,
NeMo-Gym in a CoreWeave CPU sandbox.

1. Create a temporary object storage key (``cwic``) and an inference gateway (``cwtrain``).
2. Start the Gym sandbox. It serves its agent ports through a Tailcat tunnel.
3. Copy the recipe to SUNK and submit a Slurm job (``sunk_job``). The trainer stages the base
   model in the bucket.
4. Create a hot-load deployment of the staged model behind the gateway and wait until it serves.
   Pass its endpoint to the trainer and to Gym.
5. Stream the Slurm job log until the job ends. Then delete everything the run created.

Usage: python -m dapo_nemo_rl.launch [--max-num-steps N] [-- hydra overrides...]
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from dapo_nemo_rl.inference import DeploymentSettings, Inference, deployment_spec
from dapo_nemo_rl.sunk import Sunk

RECIPE_TAG = "dapo-nemo-rl"
WORKSPACE = "/workspace"
GYM_IMAGE = "python:3.13.16"
MODEL_REPO = "Qwen/Qwen2.5-Math-7B"
MODEL_REVISION = "b101308fe89651ea5ce025f25317fea6fc07e96e"
MOUNT_DIRS = ("dapo_nemo_rl/", "config/", "patches/", "gym_env/")
# Copied from the host environment to the SUNK job when set.
FORWARDED_ENV = ("MAX_NUM_STEPS", "DAPO_MAX_SAMPLES", "WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT", "HF_TOKEN")
ACTIVE_STATES = ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED")


class PeerStopped(RuntimeError):
    """The Gym sandbox or the Slurm job stopped early."""


def log(message: str) -> None:
    print(f"[dapo] {message}", flush=True)


@dataclass(frozen=True)
class LaunchConfig:
    runner_id: str  # CoreWeave sandbox runner for the Gym and grading sandboxes
    api_token: str  # CoreWeave API token: sandboxes, inference requests and hot-loads
    sunk: Sunk
    sunk_dir: str  # e.g. /mnt/data/<you>/dapo-nemo-rl; holds runs/ and cache/
    project_root: Path
    bucket: str = "gradient-hotloading"
    bucket_prefix: str = "cookbook/dapo-nemo-rl"  # the inference service must be able to read it
    deployment: DeploymentSettings = field(default_factory=DeploymentSettings)
    partition: str = "h100"
    time_limit: str = "12:00:00"
    gym_cpu: str = "8"
    gym_memory: str = "32Gi"
    max_lifetime_seconds: int = 12 * 3600
    hydra_overrides: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)


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
            log(f"{what}: no capacity on the runner yet; retrying")
        time.sleep(20)


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
        self.model_cache_prefix = f"{config.bucket_prefix}/models/{MODEL_REPO.split('/')[-1]}-{MODEL_REVISION[:8]}"
        self.gym: Any = None
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
        log(f"storage: temporary access key for s3://{self.config.bucket}/{self.run_prefix}")

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
            "DAPO_RUN_TAG": self.run_tag,
            "MODEL_REPO": MODEL_REPO,  # Gym's model server tokenizes with the model's tokenizer
            "MODEL_REVISION": MODEL_REVISION,
            "COREWEAVE_API_TOKEN": self.config.api_token,  # for the inference gateway
            # Gym's eval server creates grading sandboxes on the same runner.
            "CWSANDBOX_API_KEY": self.config.api_token,
            "CWSANDBOX_RUNNER_ID": self.config.runner_id,
        }
        self.gym = create_with_retry(
            lambda: sdk.Sandbox.run(
                "python",
                "-m",
                "dapo_nemo_rl.run_gym",
                container_image=GYM_IMAGE,
                resources=sdk.ResourceOptions(requests=limits, limits=limits),
                mounted_files=collect_mounted_files(self.config.project_root),
                environment_variables=env,
                max_lifetime_seconds=self.config.max_lifetime_seconds,
                tags=[RECIPE_TAG, self.run_tag, f"{RECIPE_TAG}-gym"],
                placement_mode="cks",
                placement_spillover="strict",
                runner_ids=[self.config.runner_id],
            ),
            "gym",
        )
        log(f"gym: sandbox {self.gym.sandbox_id} requested (CPU)")
        return self.gym

    def gym_alive(self) -> bool:
        """False once the sandbox stopped or its container restarted (a restart loses the
        tunnel and the endpoint file, so the run cannot recover)."""
        try:
            status = str(self.gym.get_status()).lower()
            restarts = sum(c.restart_count or 0 for c in self.gym.container_statuses or ())
        except Exception:  # noqa: BLE001 - a failed status call does not mean the sandbox stopped
            return True
        return restarts == 0 and any(state in status for state in ("running", "creating", "pending", "starting"))

    def stream_gym_log(self) -> None:
        """Print the Gym sandbox's output with a ``[gym]`` prefix, without package install lines."""
        noise = re.compile(
            r"^(\(\w+\) )?\s*[+-] \S+==|^(\(\w+\) )?(Resolved|Prepared|Installed|Uninstalled|Audited|Checked) "
        )

        def pump() -> None:
            try:
                for line in self.gym.stream_logs(follow=True):
                    if not noise.search(line):
                        self.sink(f"[gym] {line}" if line.endswith("\n") else f"[gym] {line}\n")
            except Exception:  # noqa: BLE001 - the stream ends when the sandbox stops
                pass

        threading.Thread(target=pump, daemon=True).start()

    # ------------------------------------------------------------------ sunk job

    def secrets_env(self) -> bytes:
        values = {
            "DAPO_RUN_TAG": self.run_tag,
            "MODEL_REPO": MODEL_REPO,
            "MODEL_REVISION": MODEL_REVISION,
            "COREWEAVE_API_TOKEN": self.config.api_token,
            "AWS_ACCESS_KEY_ID": self.access_key["accessKeyID"],
            "AWS_SECRET_ACCESS_KEY": self.access_key["secretKey"],
            "AWS_REGION": self.bucket_zone,
            "HOTLOAD_BUCKET": self.config.bucket,
            "MODEL_CACHE_PREFIX": self.model_cache_prefix,
            "MODEL_RUN_PREFIX": f"{self.run_prefix}/base",
            **self.config.env,
        }
        return "".join(f"{key}={value}\n" for key, value in values.items()).encode()

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
            ["env", f"PYTHONPATH={run_dir}/code", "python3", "-m", "dapo_nemo_rl.sunk_job", run_dir,
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
        """Raise ``PeerStopped`` if the job or the Gym sandbox ended. Return the job state."""
        state = self.sunk.state(self.job_id)
        if state not in ACTIVE_STATES:
            raise PeerStopped(f"sunk: job {self.job_id} {state}")
        if not self.gym_alive():
            raise PeerStopped("the Gym sandbox stopped or restarted; see the [gym] lines")
        return state

    def wait_for_sunk_file(self, name: str, what: str) -> str:
        def read() -> str | None:
            self.check_peers()
            return (self.sunk.read(f"{self.run_dir}/{name}") or "").strip() or None

        return retry(read, what, timeout=4 * 3600, every=20)

    def follow_job(self) -> int:
        """Wait until the job ends. Return its exit code."""
        deadline = time.monotonic() + self.config.max_lifetime_seconds + 600
        while True:
            time.sleep(30)
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
        self.create_storage_key()
        self.create_gateway()
        gym = self.start_gym()
        gym.wait(timeout=1800)
        self.stream_gym_log()

        def read_gym_address() -> str | None:
            if not self.gym_alive():
                raise PeerStopped("the Gym sandbox stopped or restarted; see the [gym] lines")
            return gym.read_file(f"{WORKSPACE}/run/gym.addr").result().decode().strip() or None

        gym_address = retry(read_gym_address, "the Gym tunnel address")
        self.submit_job(gym_address)
        self.stream_log()
        self.wait_for_sunk_file("model.staged", "the trainer to stage the model (the job may be waiting for a node)")
        self.create_deployment()
        env = self.inference_env()
        self.sunk.write_private(f"{self.run_dir}/inference.env", env)
        gym.write_file(f"{WORKSPACE}/run/inference.env", env).result()

    def stop_all(self) -> None:
        """Delete everything the run created. Each step runs even if an earlier one fails."""

        def step(what: str, action: Callable[[], Any], manual: str) -> None:
            try:
                action()
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

        def stop_sandboxes() -> None:  # the Gym sandbox and the grading sandboxes it started
            sandboxes = self.sdk.Sandbox.list(tags=[self.run_tag]).result()
            for sandbox in sandboxes:
                sandbox.stop(missing_ok=True).result()
            log(f"stopped {len(sandboxes)} sandbox(es)")

        step("stop sandboxes", stop_sandboxes, f"python -m dapo_nemo_rl.cleanup --tag {self.run_tag}")
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
            from dapo_nemo_rl import caios

            def delete_objects() -> None:
                s3 = self.s3_factory(self.access_key, self.bucket_zone)
                count = caios.delete_prefix(s3, self.config.bucket, self.run_prefix)
                log(f"storage: deleted {count} object(s) under s3://{self.config.bucket}/{self.run_prefix}")

            step("delete the run's objects", delete_objects, f"delete s3://{self.config.bucket}/{self.run_prefix}/")

            def revoke_key() -> None:
                self.inference.delete_access_key(self.access_key["accessKeyID"])
                log("storage: access key revoked")

            step("revoke the access key", revoke_key, "cwic cwobject access-key list")

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


def config_from_env(args: argparse.Namespace, environ: dict[str, str]) -> LaunchConfig:
    def need(name: str) -> str:
        if not environ.get(name):
            raise SystemExit(f"set {name}")
        return environ[name]

    env = {key: environ[key] for key in FORWARDED_ENV if environ.get(key)}
    if args.max_num_steps is not None:
        env["MAX_NUM_STEPS"] = str(args.max_num_steps)
    defaults = DeploymentSettings()
    return LaunchConfig(
        runner_id=need("CWSANDBOX_RUNNER_ID"),
        api_token=need("CWSANDBOX_API_KEY"),
        sunk=Sunk(host=environ.get("SUNK_SSH_HOST", "sunk"), key=environ.get("SUNK_SSH_KEY") or None),
        sunk_dir=need("SUNK_DATA_DIR").rstrip("/") + "/dapo-nemo-rl",
        project_root=Path(__file__).resolve().parents[1],
        bucket=environ.get("HOTLOAD_BUCKET", "gradient-hotloading"),
        bucket_prefix=environ.get("HOTLOAD_BUCKET_PREFIX", "cookbook/dapo-nemo-rl").strip("/"),
        deployment=DeploymentSettings(
            zone=environ.get("INFERENCE_ZONE", defaults.zone),
            instance_type=environ.get("INFERENCE_INSTANCE_TYPE", defaults.instance_type),
            replicas=int(environ.get("INFERENCE_REPLICAS", defaults.replicas)),
        ),
        partition=environ.get("SUNK_PARTITION", "h100"),
        hydra_overrides=tuple(args.overrides),
        env=env,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Start a DAPO training run on SUNK, Dedicated Inference and sandboxes."
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
