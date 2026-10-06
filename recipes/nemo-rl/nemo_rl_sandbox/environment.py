"""CPU sandbox lifecycle and batched reward verification."""

import json
from pathlib import Path

from cwsandbox import AuthStrategy, Sandbox


class SandboxRewardClient:
    def __init__(self, *, lifetime_seconds=3600):
        self.sandbox = Sandbox.run(
            "sleep",
            "infinity",
            auth=AuthStrategy.COREWEAVE_API_KEY,
            placement_mode="serverless",
            container_image="python:3.11",
            resources={"cpu": "1", "memory": "1Gi"},
            max_lifetime_seconds=lifetime_seconds,
            tags=["nemo-rl-reward"],
        )
        self.calls = 0
        try:
            print(
                json.dumps(
                    {
                        "event": "cpu_sandbox_created",
                        "sandbox_id": self.sandbox.sandbox_id,
                    }
                ),
                flush=True,
            )
            self.sandbox.write_file(
                "/tmp/grader.py", Path(__file__).with_name("grader.py").read_bytes()
            ).result(timeout=120)
        except BaseException:
            self.close()
            raise

    def score(self, responses: list[str], expected: list[str]) -> list[float]:
        payload = json.dumps({"responses": responses, "expected": expected})
        # Pass the JSON as an argv element, never interpolate generated text in a shell.
        result = self.sandbox.exec(
            ["python", "/tmp/grader.py", payload], timeout_seconds=30
        ).result(timeout=60)
        if result.returncode:
            raise RuntimeError(
                f"Reward environment exited {result.returncode}: {result.stderr}"
            )
        values = json.loads(result.stdout)
        if len(values) != len(responses):
            raise RuntimeError("Reward environment returned the wrong batch size")
        self.calls += 1
        return values

    def close(self):
        self.sandbox.stop(missing_ok=True).result(timeout=60)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
