"""Fail-fast reachability check for the sandbox environment plane.

Provisions exactly one sandbox, drives a full episode-shaped exchange against it
(``/health`` -> ``/reset`` -> ``/step`` -> ``/reward``), then tears it down.

Run this before the trainer. The failure modes it catches -- runner profile
without public ingress, missing bearer token, tau-bench install failure inside
the image, env server bound to localhost -- would otherwise surface as a hang on
the first rollout, after the GPUs are already allocated and billing.

    python -m verl_taubench.sandbox.preflight --domain retail --task-split train
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from typing import Optional

from verl_taubench.sandbox.client import HttpxTransport
from verl_taubench.sandbox.env_config import (
    CWSANDBOX_SERVERLESS_AUTH_ENV,
    resolve_backend_image,
    resolve_placement_mode,
    resolve_placement_spillover,
    resolve_runner_ids,
)
from verl_taubench.sandbox.pool import CwSandboxBackend, SandboxPool


def build_preflight_backend(args: argparse.Namespace):
    """Construct the preflight backend with the same rules as production."""
    container_image, install_tau_bench = resolve_backend_image(args.container_image)
    placement_mode = resolve_placement_mode(getattr(args, "placement_mode", None))
    return CwSandboxBackend(
        domain=args.domain,
        task_split=args.task_split,
        container_image=container_image,
        install_tau_bench=install_tau_bench,
        env_port=args.env_port,
        cpu=args.cpu,
        memory=args.memory,
        token=os.environ.get("TAUBENCH_ENV_TOKEN"),
        max_lifetime_seconds=args.max_lifetime_seconds,
        placement_mode=placement_mode,
        placement_spillover=resolve_placement_spillover(
            getattr(args, "placement_spillover", None),
            placement_mode=placement_mode,
        ),
        runner_ids=resolve_runner_ids(None),
    )


async def _run(args: argparse.Namespace) -> int:
    backend = build_preflight_backend(args)
    transport = HttpxTransport(timeout=args.timeout)
    pool = SandboxPool(backend, transport, size=1, reset_timeout_s=args.timeout)

    started = time.monotonic()
    try:
        print(f"[preflight] provisioning 1 sandbox (domain={args.domain} split={args.task_split})...")
        async with pool.lease(args.task_index, episode_id="preflight") as lease:
            elapsed = time.monotonic() - started
            print(f"[preflight] sandbox {lease.handle.sandbox_id} ready at {lease.handle.base_url} in {elapsed:.1f}s")

            health = await lease.client.health()
            print(f"[preflight] /health   -> {health}")

            instruction = lease.metadata.get("instruction", "")
            print(f"[preflight] /reset    -> task_index={lease.metadata.get('task_index')} "
                  f"instruction={instruction[:80]!r}")

            # A read-only tool call: exercises the DB without mutating it.
            step = await lease.client.step(args.probe_tool, {})
            print(f"[preflight] /step     -> source={step.get('source')} "
                  f"observation={str(step.get('observation'))[:120]!r}")

            # Records a respond action, which reward's task.outputs check needs.
            await lease.client.step("respond", {"content": "preflight probe"})

            reward = await lease.client.reward()
            print(f"[preflight] /reward   -> {reward.get('reward')} (info keys: "
                  f"{sorted((reward.get('info') or {}).keys())})")

        print(f"[preflight] OK in {time.monotonic() - started:.1f}s")
        return 0
    except Exception as exc:
        print(f"[preflight] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        if "service_urls" in str(exc) or "service_address" in str(exc):
            print(
                "[preflight] hint: declare a public HTTPS Service at create time "
                "(services=[Service(port=..., visibility='public', endpoint=Endpoint(kind='https'))]).",
                file=sys.stderr,
            )
        return 1
    finally:
        await pool.aclose()


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default=os.environ.get("TAUBENCH_DOMAIN", "retail"))
    parser.add_argument("--task-split", default=os.environ.get("TAUBENCH_TASK_SPLIT", "train"))
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--container-image", default="python:3.11")
    parser.add_argument("--env-port", type=int, default=8080)
    parser.add_argument("--cpu", default=os.environ.get("TAUBENCH_ENV_CPU") or "4")
    parser.add_argument("--memory", default=os.environ.get("TAUBENCH_ENV_MEMORY") or "8Gi")
    parser.add_argument("--max-lifetime-seconds", type=int, default=3600)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--placement-mode",
        default=None,
        help="cks (default) or serverless. Overrides CWSANDBOX_PLACEMENT_MODE.",
    )
    parser.add_argument(
        "--placement-spillover",
        default=None,
        help="cks_then_serverless (default), strict, or serverless_then_cks.",
    )
    parser.add_argument(
        "--probe-tool",
        default="list_all_product_types",
        help="A read-only τ-bench tool to smoke-test (retail default).",
    )
    parser.add_argument(
        "--cwsandbox-auth",
        action="store_true",
        help="Ignored; preflight selects cwsandbox auth from the placement mode.",
    )
    parser.add_argument("--serverless-auth", choices=("wandb", "coreweave"), default=None)
    args = parser.parse_args(argv)
    if args.serverless_auth is not None:
        os.environ[CWSANDBOX_SERVERLESS_AUTH_ENV] = args.serverless_auth
    return asyncio.run(_run(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
