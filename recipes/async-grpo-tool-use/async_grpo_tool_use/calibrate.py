"""Verifier calibration: check the sandboxed math verifier before training.

For each ground-truth answer, build responses with a known grade and send them to the
Gym eval server's ``/calibrate`` route, which grades them in CoreWeave sandboxes:

* ``boxed_correct``:   ``... \\boxed{<answer>}``          expected reward 1
* ``answer_correct``:  ``... Answer: <answer>`` (DAPO's format)  expected reward 1
* ``boxed_wrong``:     a different answer in ``\\boxed{}``  expected reward 0
* ``answer_wrong``:    a different answer after ``Answer:``  expected reward 0

The report gives the share of cases graded as expected, per kind. A low score for the
correct kinds means the reward misses right answers in that format. Standard library only.
"""

from __future__ import annotations

import json
import urllib.request
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

KINDS = {"boxed_correct": 1.0, "answer_correct": 1.0, "boxed_wrong": 0.0, "answer_wrong": 0.0}


def wrong_answer(answer: str) -> str:
    try:
        return str(int(answer) + 1)
    except ValueError:
        return "0" if answer.strip() != "0" else "1"


def build_cases(answers: Iterable[str]) -> list[dict[str, str]]:
    cases = []
    for answer in answers:
        wrong = wrong_answer(answer)
        text = "We work through the problem step by step.\n"
        responses = {
            "boxed_correct": f"{text}The answer is \\boxed{{{answer}}}.",
            "answer_correct": f"{text}Answer: {answer}",
            "boxed_wrong": f"{text}The answer is \\boxed{{{wrong}}}.",
            "answer_wrong": f"{text}Answer: {wrong}",
        }
        cases.extend({"kind": kind, "expected": answer, "response": response} for kind, response in responses.items())
    return cases


def summarize(cases: list[dict[str, str]], results: list[dict[str, Any]]) -> dict[str, Any]:
    right: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    for case, result in zip(cases, results, strict=True):
        total[case["kind"]] += 1
        right[case["kind"]] += int(result["reward"] == KINDS[case["kind"]])
    return {kind: {"agree": right[kind] / total[kind], "cases": total[kind]} for kind in KINDS if total[kind]}


def run(server_url: str, answers: Iterable[str], out: Path, timeout: float = 1800.0) -> dict[str, Any]:
    cases = build_cases(answers)
    body = json.dumps({"cases": [{"expected": c["expected"], "response": c["response"]} for c in cases]}).encode()
    request = urllib.request.Request(
        server_url.rstrip("/") + "/calibrate", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        results = json.loads(resp.read())["results"]
    report = summarize(cases, results)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"report": report, "cases": cases, "results": results}, indent=1))
    return report
