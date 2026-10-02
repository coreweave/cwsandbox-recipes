"""Run the compact NeMo RL integration on one GPU and collect its checkpoint."""

import argparse
import os
from pathlib import Path

from cwsandbox import AuthStrategy
from nemo_rl_sandbox.lifecycle import managed_sandbox


IMAGE = "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("results/sandbox"))
    parser.add_argument("--lifetime-seconds", type=int, default=3600)
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("--steps must be at least 2 to exercise updated-policy rollouts")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a new, empty output directory")
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    key = os.environ["CWSANDBOX_API_KEY"]
    with managed_sandbox(
        args.output,
        "sleep",
        "infinity",
        auth=AuthStrategy.COREWEAVE_API_KEY,
        placement_mode="serverless",
        container_image=IMAGE,
        resources={"cpu": "8", "memory": "32Gi", "gpu": {"count": 1}},
        environment_variables={
            "CWSANDBOX_API_KEY": key,
            "PYTHONPATH": "/opt/nemo-rl:/workspace/recipe",
        },
        max_lifetime_seconds=args.lifetime_seconds,
        tags=["nemo-rl-trainer"],
    ) as sandbox:
        print(f"GPU sandbox: {sandbox.sandbox_id}", flush=True)
        files = list((root / "nemo_rl_sandbox").glob("*.py"))
        files += [
            root / "scripts" / name
            for name in (
                "install_smoke.sh",
                "install_nemo_source.py",
                "requirements-smoke.txt",
                "train_smoke.py",
                "verify_smoke.py",
            )
        ]
        for source in files:
            sandbox.write_file(
                "/workspace/recipe/" + str(source.relative_to(root)),
                source.read_bytes(),
            ).result(timeout=600)
        for name, command, timeout in (
            ("install", ["bash", "/workspace/recipe/scripts/install_smoke.sh"], 900),
            (
                "train",
                [
                    "python",
                    "/workspace/recipe/scripts/train_smoke.py",
                    "--steps",
                    str(args.steps),
                ],
                args.lifetime_seconds,
            ),
        ):
            process = sandbox.exec(command, timeout_seconds=timeout)
            # Stream stdout so package installation and training show progress.
            with (args.output / f"{name}.log").open("w") as log:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
            result = process.result(timeout=timeout + 30)
            (args.output / f"{name}.log").write_text(
                result.stdout + "\n" + result.stderr
            )
            if result.returncode:
                raise RuntimeError(
                    f"{name} failed; see {args.output / (name + '.log')}"
                )
        verification = sandbox.exec(
            ["python", "/workspace/recipe/scripts/verify_smoke.py"], timeout_seconds=300
        ).result(timeout=330)
        (args.output / "verify.log").write_text(
            verification.stdout + "\n" + verification.stderr
        )
        if verification.returncode:
            raise RuntimeError("Checkpoint reload failed; see verify.log")
        archive = sandbox.exec(
            ["tar", "-czf", "/tmp/result.tar.gz", "-C", "/workspace/results", "smoke"],
            timeout_seconds=120,
        ).result(timeout=150)
        if archive.returncode:
            raise RuntimeError("Failed to archive the adapter checkpoint")
        for name in ("summary.json", "metrics.json", "reload.json"):
            (args.output / name).write_bytes(
                sandbox.read_file("/workspace/results/smoke/" + name).result(
                    timeout=120
                )
            )
        (args.output / "result.tar.gz").write_bytes(
            sandbox.read_file("/tmp/result.tar.gz").result(timeout=120)
        )
    print(f"Artifacts: {args.output.resolve()}; GPU and CPU reward sandboxes stopped.")


if __name__ == "__main__":
    main()
