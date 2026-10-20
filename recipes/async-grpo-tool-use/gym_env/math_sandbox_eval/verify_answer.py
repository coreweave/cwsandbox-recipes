"""Grade math answers with math-verify. Runs inside a CoreWeave sandbox.

Input: a JSON file with a list of ``{"expected": str, "response": str}``.
Output (stdout): a JSON list of ``{"reward": 0.0 or 1.0, "extracted": str or null}``.

The rule matches Gym's ``math_with_judge`` (judge off): the expected answer is wrapped
in ``\\boxed{}`` and compared with any LaTeX or plain expression in the response.

Usage: python3 verify_answer.py cases.json
"""

import json
import logging
import sys

from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

logging.getLogger("math_verify").setLevel(logging.CRITICAL)
VERIFY = math_metric(
    gold_extraction_target=(LatexExtractionConfig(),),
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
)


def strip_delimiters(text: str) -> str:
    text = text.strip()
    for left, right in (("$$", "$$"), ("$", "$"), ("\\(", "\\)"), ("\\[", "\\]")):
        if text.startswith(left) and text.endswith(right) and len(text) > len(left) + len(right):
            return text[len(left) : -len(right)].strip()
    return text


def grade(expected: str, response: str) -> dict:
    try:
        score, extracted = VERIFY(["\\boxed{" + strip_delimiters(expected) + "}"], [response])
        prediction = extracted[1][0] if extracted and extracted[1] else None
        return {"reward": float(score), "extracted": None if prediction is None else str(prediction)}
    except BaseException:  # math-verify raises TimeoutException, a BaseException
        return {"reward": 0.0, "extracted": None}


if __name__ == "__main__":
    with open(sys.argv[1]) as fh:
        cases = json.load(fh)
    print(json.dumps([grade(case["expected"], case["response"]) for case in cases]))
