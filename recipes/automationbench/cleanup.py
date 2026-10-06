"""Delete only the sandboxes recorded by one stopped recipe job."""

import argparse
import json
from pathlib import Path

from cwsandbox import AuthStrategy, Sandbox, SandboxNotFoundError


def cleanup(output):
    ids = {
        json.loads(line)["sandbox_id"]
        for line in (output / "sandbox-ids.jsonl").read_text().splitlines()
        if line.strip()
    }
    failures = []
    for sandbox_id in sorted(ids):
        try:
            Sandbox.delete(
                sandbox_id, auth=AuthStrategy.COREWEAVE_API_KEY, missing_ok=True
            ).result(timeout=90)
            try:
                sandbox = Sandbox.from_id(
                    sandbox_id, auth=AuthStrategy.COREWEAVE_API_KEY
                ).result(timeout=30)
                state = str(sandbox.status)
            except SandboxNotFoundError:
                state = "not_found"
            print(json.dumps({"sandbox_id": sandbox_id, "state": state}))
            if state not in {"completed", "failed", "not_found"}:
                failures.append(sandbox_id)
        except Exception as exc:  # noqa: BLE001 -- continue cleanup of the other recorded IDs
            print(
                json.dumps({"sandbox_id": sandbox_id, "error_type": type(exc).__name__})
            )
            failures.append(sandbox_id)
    if failures:
        raise SystemExit(
            "Some sandboxes are not confirmed stopped; inspect the reported IDs"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    cleanup(parser.parse_args().output)
