"""Sandbox entry point: upstream measurement, scoring, and evidence verification."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from transport import native_transport

MODEL = "claude-haiku-4-5-20251001"


def verify_results(tasks, records, raw_dir):
    expected = {task.id for task in tasks}
    actual = [row["task_id"] for row in records]
    if len(actual) != len(set(actual)) or not set(actual).issubset(expected):
        raise ValueError("Duplicate or unexpected task results")
    for row in records:
        name = hashlib.sha256(row["task_id"].encode()).hexdigest() + ".json"
        evidence = (raw_dir / name).read_bytes()
        if hashlib.sha256(evidence).hexdigest() != row["raw_sha256"]:
            raise ValueError("Raw response hash mismatch")
    return {
        "planned": len(tasks),
        "attempted": len(records),
        "complete": set(actual) == expected,
        "failed": sum(row["status"] != "ok" for row in records),
        "valid": sum(row["valid"] for row in records),
        "correct": sum(row["correct"] for row in records),
        "raw_hashes_verified": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--budget-usd", type=float, required=True)
    args = parser.parse_args()
    sys.path.insert(0, "/work/jevbench")
    from jevbench.adapters import openai_compat
    from jevbench.cli import main as cli
    from jevbench.tasks import load_jsonl

    output = Path("/work/output")
    output.mkdir(exist_ok=True)
    # Summarize exactly the selected cohort, including when running a smoke sample.
    original = Path("/work/jevbench/datasets/public/original.jsonl")
    lines = [line for line in original.read_text().splitlines() if line.strip()]
    tasks_path = output / "tasks.jsonl"
    tasks_path.write_text("\n".join(lines[: args.limit]) + "\n")
    openai_compat.http_post_json = native_transport
    result = cli(
        [
            "run",
            "--tasks",
            str(tasks_path),
            "--adapter",
            "openai_compat",
            "--endpoint",
            "https://api.anthropic.com/v1",
            "--model",
            MODEL,
            "--key-env",
            "ANTHROPIC_API_KEY",
            "--results",
            str(output / "results.jsonl"),
            "--raw-dir",
            str(output / "raw"),
            "--ledger",
            str(output / "ledger.jsonl"),
            "--cap-usd",
            str(args.budget_usd),
            "--reserve-usd",
            "0.02",
            "--price-in-per-m",
            "1.00",
            "--price-out-per-m",
            "5.00",
            "--manifest",
            str(output / "manifest.json"),
            "--run-label",
            "cwsandbox-public-original",
        ]
    )
    cli(
        [
            "summarize",
            "--tasks",
            str(tasks_path),
            "--results",
            str(output / "results.jsonl"),
            "--ledger",
            str(output / "ledger.jsonl"),
            "--public-export",
            str(output / "summary.json"),
        ]
    )
    records = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    verification = verify_results(load_jsonl(str(tasks_path)), records, output / "raw")
    (output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification), flush=True)
    # Incorrect or invalid model answers are benchmark outcomes, not infrastructure failures.
    return result or (0 if verification["complete"] and not verification["failed"] else 1)


if __name__ == "__main__":
    raise SystemExit(main())
