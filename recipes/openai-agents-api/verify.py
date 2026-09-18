"""Local checks for downloaded artifacts and recorded OpenAI API evidence.

Never upload this file or tests/ to the worker. Downloaded Python is parsed, not
executed: execution evidence must come from the recorded API command items.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import shlex
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).parent / "fixtures"
SPECIALISTS = ("latency", "errors", "capacity")


class VerificationError(ValueError):
    """A required artifact or API evidence check failed."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise VerificationError(f"Cannot read JSON from {path.name}: {exc}") from exc


def expected_metrics(fixtures: Path = FIXTURES) -> dict[str, dict]:
    """Recompute ground truth from the local inputs, without agent-written code."""
    cutoff = _load(fixtures / "deployments.json")["cutoff"]
    result = {}
    for specialist in SPECIALISTS:
        filename = "capacity.csv" if specialist == "capacity" else "requests.csv"
        with (fixtures / filename).open(newline="") as source:
            rows = list(csv.DictReader(source))
        grouped: dict[tuple[str, str], list[dict]] = {}
        for row in rows:
            window = "before" if row["timestamp"] < cutoff else "after"
            grouped.setdefault((window, row["region"]), []).append(row)
        groups = []
        for (window, region), values in sorted(grouped.items()):
            group = {"window": window, "region": region}
            if specialist == "latency":
                latencies = sorted(int(row["latency_ms"]) for row in values)
                group.update(
                    count=len(values),
                    p50_ms=latencies[math.ceil(0.50 * len(values)) - 1],
                    p95_ms=latencies[math.ceil(0.95 * len(values)) - 1],
                )
            elif specialist == "errors":
                errors = sum(int(row["status_code"]) >= 500 for row in values)
                group.update(
                    count=len(values),
                    error_count=errors,
                    error_rate=round(errors / len(values), 6),
                )
            else:
                replicas = [int(row["replicas"]) for row in values]
                group.update(
                    samples=len(values),
                    mean_utilization_pct=round(
                        sum(float(row["utilization_pct"]) for row in values) / len(values), 6
                    ),
                    max_queue_depth=max(int(row["queue_depth"]) for row in values),
                    min_replicas=min(replicas),
                    max_replicas=max(replicas),
                )
            groups.append(group)
        result[specialist] = {
            "specialist": specialist,
            "groups": groups,
            "assessment": "correlation_only",
        }
    return result


def _compare(actual: Any, expected: Any, location: str) -> None:
    if isinstance(expected, dict):
        _check(
            isinstance(actual, dict) and actual.keys() == expected.keys(),
            f"Wrong fields at {location}",
        )
        for key in expected:
            _compare(actual[key], expected[key], f"{location}.{key}")
    elif isinstance(expected, list):
        _check(
            isinstance(actual, list) and len(actual) == len(expected),
            f"Wrong group count at {location}",
        )
        for index, value in enumerate(expected):
            _compare(actual[index], value, f"{location}[{index}]")
    elif isinstance(expected, int | float):
        _check(
            type(actual) in (int, float)
            and math.isfinite(actual)
            and abs(actual - expected) <= 0.000001,
            f"Wrong numeric metric at {location}: expected {expected}, got {actual!r}",
        )
    else:
        _check(actual == expected, f"Wrong value at {location}: expected {expected!r}")


def _text(path: Path) -> str:
    try:
        text = path.read_text()
    except OSError as exc:
        raise VerificationError(f"Missing artifact {path.name}") from exc
    _check(bool(text.strip()), f"Empty artifact {path.name}")
    return text


def _runs_script(command: str, name: str) -> bool:
    """Accept an unambiguous invocation, optionally inside one shell wrapper."""
    try:
        parts = shlex.split(command)
        if (
            len(parts) == 3
            and Path(parts[0]).name in ("bash", "sh", "zsh")
            and parts[1] in ("-c", "-lc")
        ):
            parts = shlex.split(parts[2])
    except ValueError:
        return False
    return parts == ["python3", f"output/{name}.py"]


def _evidence(
    events: list[dict], turns: list[dict], items: list[dict], expected_session_id: str | None
) -> dict:
    created = {}
    for envelope in events:
        event = envelope.get("event", envelope)
        if event.get("type") != "agent.session.subagent.created":
            continue
        child = event.get("subagent", {})
        if child.get("id") and child.get("session_id"):
            previous = created.get(child["id"])
            _check(
                not previous or previous["session_id"] == child["session_id"],
                "Subagent records belong to different API sessions",
            )
            created[child["id"]] = child
    _check(len(created) >= 3, "Need API creation records for three distinct subagent IDs")
    sessions = {child["session_id"] for child in created.values()}
    _check(len(sessions) == 1, "Specialists belong to different API sessions")
    session_id = next(iter(sessions))
    _check(
        expected_session_id is None or session_id == expected_session_id,
        "Delegation evidence belongs to another API session",
    )
    successful_turns = {
        turn["id"]: turn
        for turn in turns
        if turn.get("session_id") == session_id
        and turn.get("status") == "completed"
        and turn.get("id")
    }
    roots = [turn for turn in successful_turns.values() if turn.get("subagent_id") is None]
    _check(
        len(roots) >= 2, "Need two completed root turns in the same session, including follow-up"
    )
    # Task text can be absent or encrypted. Attribute each role through an actual
    # command -> completed turn -> created subagent chain instead of instructions.
    children = {}
    command_ids = {}
    zero_exit_codes = {}
    missing_exit_codes = []
    for name in SPECIALISTS:
        commands = []
        for item in items:
            turn = successful_turns.get(item.get("turn_id"), {})
            code = item.get("exit_code")
            if (
                item.get("type") == "command_execution"
                and item.get("id")
                and turn.get("subagent_id") in created
                and item.get("status") == "completed"
                and (code is None or (type(code) is int and code == 0))
                and _runs_script(item.get("command", ""), name)
            ):
                commands.append(item)
        _check(bool(commands), f"No completed Python execution attributed to {name} subagent")
        owners = {successful_turns[item["turn_id"]]["subagent_id"] for item in commands}
        _check(len(owners) == 1, f"Multiple subagents executed {name}; attribution is ambiguous")
        children[name] = created[next(iter(owners))]
        command_ids[name] = [item["id"] for item in commands]
        zero_exit_codes[name] = any(type(item.get("exit_code")) is int for item in commands)
        missing_exit_codes.extend(item["id"] for item in commands if item.get("exit_code") is None)
    ids = {child["id"] for child in children.values()}
    _check(len(ids) == 3, "Specialists must have three distinct API subagent IDs")

    intervals = []
    for turn in successful_turns.values():
        start, end = turn.get("started_at"), turn.get("completed_at")
        if (
            turn.get("subagent_id") in ids
            and type(start) in (int, float)
            and type(end) in (int, float)
            and math.isfinite(start)
            and math.isfinite(end)
            and end > start
        ):
            intervals.append(turn)
    overlaps = []
    for index, first in enumerate(intervals):
        for second in intervals[index + 1 :]:
            if first["subagent_id"] == second["subagent_id"]:
                continue
            seconds = min(first["completed_at"], second["completed_at"]) - max(
                first["started_at"], second["started_at"]
            )
            if seconds > 0:
                overlaps.append({"turn_ids": [first["id"], second["id"]], "seconds": seconds})
    return {
        "session_id": session_id,
        "delegation_verified": True,
        "specialist_subagent_ids": {name: child["id"] for name, child in children.items()},
        "completed_command_ids": command_ids,
        "command_evidence_basis": "API completed status and command/turn/subagent IDs",
        "exit_code_evidence": {
            "all_specialists_have_zero_exit": all(zero_exit_codes.values()),
            "zero_exit_verified_by_specialist": zero_exit_codes,
            "missing_for_command_ids": missing_exit_codes,
            "limitation": "An omitted exit code is unknown, not an observed zero exit code.",
        },
        "completed_root_turns": len(roots),
        "overlap": {
            "established": bool(overlaps),
            "basis": "API subagent turn started_at/completed_at; not shell process timestamps",
            "pairs": overlaps,
            "limitation": "No pair overlap does not establish that execution was serial.",
        },
    }


def verify_output(
    output_dir: str | Path,
    *,
    events: list[dict] | None = None,
    turns: list[dict] | None = None,
    items: list[dict] | None = None,
    expected_session_id: str | None = None,
    require_overlap: bool = False,
    require_exit_codes: bool = False,
) -> dict:
    """Validate artifact metrics and actual delegation; never run downloaded code."""
    output = Path(output_dir)
    for name, expected in expected_metrics().items():
        _compare(_load(output / f"{name}.json"), expected, f"{name}.json")
        source = _text(output / f"{name}.py")
        try:
            ast.parse(source, filename=f"{name}.py")
        except SyntaxError as exc:
            raise VerificationError(f"Invalid Python in {name}.py: {exc}") from exc
    incident = _text(output / "incident.md").lower()
    followup = _text(output / "followup.md").lower()
    for filename in ("latency.json", "errors.json", "capacity.json", "deployments.json"):
        _check(filename in incident, f"incident.md must cite {filename}")
    for label, report in (("incident.md", incident), ("followup.md", followup)):
        _check(
            "synthetic" in report and "correlation" in report,
            f"{label} must state synthetic-data and correlation limitations",
        )
    _check(
        "deploy-b" in incident and "west" in incident,
        "incident.md must identify the affected deployment and region",
    )
    _check(
        "incident.md" in followup and any(f"{name}.json" in followup for name in SPECIALISTS),
        "followup.md must refer to the existing report and specialist artifacts",
    )
    evidence = _evidence(events or [], turns or [], items or [], expected_session_id)
    _check(
        not require_overlap or evidence["overlap"]["established"],
        "Subagent turn overlap could not be established from API timestamps",
    )
    _check(
        not require_exit_codes or evidence["exit_code_evidence"]["all_specialists_have_zero_exit"],
        "Zero exit codes could not be established for every specialist from API records",
    )
    return {
        "verified": True,
        "metrics_verified": True,
        "python_syntax_verified": True,
        "report_checks": "Required references and limitations; not a semantic review",
        **evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Local run directory")
    parser.add_argument("--require-overlap", action="store_true")
    parser.add_argument("--require-exit-codes", action="store_true")
    args = parser.parse_args()
    try:
        events = [
            json.loads(line)
            for line in (args.output_dir / "events.jsonl").read_text().splitlines()
            if line.strip()
        ]
        run = _load(args.output_dir / "run.json")
        _check(
            isinstance(run, dict) and run.get("status") == "completed",
            "run.json must record a completed run",
        )
        _check(
            isinstance(run.get("session_id"), str) and bool(run["session_id"]),
            "run.json must identify the API session",
        )
        result = verify_output(
            args.output_dir / "output",
            events=events,
            turns=_load(args.output_dir / "turns.json"),
            items=_load(args.output_dir / "items.json"),
            expected_session_id=run.get("session_id"),
            require_overlap=args.require_overlap,
            require_exit_codes=args.require_exit_codes,
        )
        print(json.dumps(result, indent=2))
    except (VerificationError, OSError, ValueError) as exc:
        print(json.dumps({"verified": False, "error": str(exc)}, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
