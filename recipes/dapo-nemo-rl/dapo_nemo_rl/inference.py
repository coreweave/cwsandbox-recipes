"""Dedicated Inference and object storage resources for one run.

The launcher uses this module to create, and at the end delete:

* an inference gateway (the HTTPS endpoint Gym calls) and a hot-load deployment of the base
  model behind it, with ``cwtrain`` (the Dedicated Inference client library),
* a temporary object storage access key for the SUNK trainer, with ``cwic``.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any, Callable

Runner = Callable[[list[str]], str]


def run_cwic(args: list[str]) -> str:
    done = subprocess.run(["cwic", *args], capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"cwic {' '.join(args[:3])} failed: {done.stderr.strip() or done.stdout.strip()}")
    return done.stdout


@dataclass(frozen=True)
class DeploymentSettings:
    zone: str = "US-WEST-04A"
    instance_type: str = "gd-8xh100ib-i128"
    engine_version: str = "0.30.0+crwv.5"  # dynamo-vllm with hot-load support, vLLM 0.30
    replicas: int = 2
    gpus_per_replica: int = 1
    concurrency: int = 256  # sequences per replica (vLLM --max-num-seqs)
    max_model_len: int = 3072


def deployment_spec(
    name: str, gateway_id: str, bucket: str, model_path: str, hotload_prefix: str, s: DeploymentSettings
) -> dict[str, Any]:
    """The ``CreateDeploymentRequest`` body. Hot-load can only be enabled at creation."""
    return {
        "name": name,
        "gatewayIds": [gateway_id],
        "runtime": {
            "engine": "dynamo-vllm",
            "version": s.engine_version,
            "engineConfig": {"max-model-len": str(s.max_model_len), "gpu-memory-utilization": "0.90"},
        },
        "resources": {"instanceType": s.instance_type, "gpuCount": s.gpus_per_replica},
        "model": {"name": name, "bucket": bucket, "path": model_path},
        "autoscaling": {"min": s.replicas, "max": s.replicas, "concurrency": s.concurrency},
        "traffic": {"weight": 100},
        "hotLoad": {"bucket": bucket, "pathPrefix": hotload_prefix, "transitionMode": "TRANSITION_MODE_ASYNC"},
    }


class Inference:
    """``client`` is a ``cwtrain.InferenceClient`` (a fake in tests); ``cwic`` runs the CLI."""

    def __init__(self, client: Any, cwic: Runner = run_cwic) -> None:
        self.client = client
        self.cwic = cwic

    # ------------------------------------------------------- gateway, deployment

    def create_gateway(self, name: str, zone: str) -> tuple[str, str]:
        """Create a gateway with CoreWeave auth. Return (id, https endpoint) once READY."""
        from cwtrain.infra import lifecycle

        spec = {"name": name, "zones": [zone], "coreWeaveAuth": {}, "bodyBasedRouting": {"apiType": "API_TYPE_OPENAI"}}
        gateway = lifecycle.wait_for_gateway(self.client, self.client.create_gateway(spec).id)
        return gateway.id, gateway.endpoints[0]

    def create_deployment(self, spec: dict[str, Any]) -> str:
        return self.client.create_deployment(spec).id

    def wait_deployment(self, deployment_id: str) -> None:
        from cwtrain.infra import lifecycle

        lifecycle.wait_for_deployment(self.client, deployment_id)

    def delete_deployment(self, deployment_id: str) -> None:
        """Delete and wait until it is gone, so its gateway can be deleted next."""
        from cwtrain.infra import lifecycle

        lifecycle.delete_deployment(self.client, deployment_id, timeout=900)

    def delete_gateway(self, gateway_id: str) -> None:
        from cwtrain.infra import lifecycle

        lifecycle.delete_gateway(self.client, gateway_id)

    # ----------------------------------------------------------- object storage

    def json(self, *args: str) -> Any:
        return json.loads(self.cwic([*args, "-o", "json"]))

    def bucket_zone(self, bucket: str) -> str:
        return self.json("cwobject", "bucket", "get", bucket)["location"]

    def create_access_key(self, name: str, seconds: int) -> dict[str, str]:
        """A temporary access key: ``accessKeyID``, ``secretKey``, ``expiry``."""
        return self.json("cwobject", "access-key", "create", "--name", name, "--duration", str(seconds))

    def delete_access_key(self, key_id: str) -> None:
        self.cwic(["cwobject", "access-key", "delete", key_id, "--yes"])
