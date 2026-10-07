"""Run slime's single-GPU colocated GRPO smoke test in a GPU sandbox.

Megatron training and SGLang rollouts share one GPU (slime ``--colocate``).
The sandbox is stopped when the run ends, fails, or is interrupted.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cwsandbox
from cwsandbox import ResourceOptions, Sandbox

from recipe_common import pick_gpu_type

HERE = Path(__file__).resolve().parent
DEFAULT_IMAGE = "slimerl/slime:nightly-dev-20260930a-cu129"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", default=os.environ.get("SLIME_IMAGE", DEFAULT_IMAGE))
    parser.add_argument("--gpu-type", default=os.environ.get("GPU_TYPE") or None)
    parser.add_argument("--cpu", default="8")
    parser.add_argument("--memory", default="32Gi")
    parser.add_argument("--timeout", type=int, default=2700, help="Client-side budget for the run, seconds.")
    args = parser.parse_args()

    gpu_type = pick_gpu_type(cwsandbox.list_runners(healthy_only=True), args.gpu_type)
    t0 = time.time()
    sb = Sandbox.run(
        container_image=args.image,
        resources=ResourceOptions(
            requests={"cpu": args.cpu, "memory": args.memory},
            limits={"cpu": args.cpu, "memory": args.memory},
            gpu={"count": 1, "type": gpu_type},
        ),
        placement_mode="serverless",
        max_lifetime_seconds=3600,
        tags=["swebench-slime", "gpu-trainer-smoke"],
    )
    print(f"sandbox {sb.sandbox_id}: {args.image} on 1x {gpu_type}", flush=True)
    try:
        sb.wait(timeout=2400)
        print(f"running after {time.time() - t0:.0f}s (first pull of the ~20 GiB image is slowest)", flush=True)
        sb.write_file("/root/trainer_smoke.sh", (HERE / "trainer_smoke.sh").read_bytes()).result()
        # Detach so the run doesn't depend on one long-lived exec stream; poll its log instead.
        sb.exec(
            [
                "bash",
                "-c",
                "setsid bash -c 'bash /root/trainer_smoke.sh > /root/run.log 2>&1; echo $? > /root/run.done'"
                " < /dev/null > /dev/null 2>&1 &",
            ],
            check=True,
        ).result()
        seen, deadline, code = 0, time.time() + args.timeout, None
        while time.time() < deadline and code is None:
            time.sleep(10)
            log = sb.exec(["cat", "/root/run.log"], timeout_seconds=30).result().stdout
            for line in log.splitlines()[seen:]:
                print(line, flush=True)
            seen = len(log.splitlines())
            done = sb.exec(["bash", "-c", "cat /root/run.done 2>/dev/null"], timeout_seconds=30).result()
            code = int(done.stdout) if done.stdout.strip() else None
        if code is None:
            raise SystemExit(f"timed out after {args.timeout}s")
        print(f"{'PASS' if code == 0 else 'FAIL'}: exit {code} after {time.time() - t0:.0f}s", flush=True)
        raise SystemExit(code)
    finally:
        sb.stop(graceful_shutdown_seconds=0, missing_ok=True).result()
        print(f"stopped {sb.sandbox_id}", flush=True)


if __name__ == "__main__":
    main()
