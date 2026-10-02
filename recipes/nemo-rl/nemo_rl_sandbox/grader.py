"""Deterministic arithmetic verifier; runs in a CPU sandbox."""

import json
import re
import sys


def score_response(response: str, expected: str) -> float:
    """Reward a correct final integer and give a small brevity reward.

    The auxiliary reward gives this tiny smoke run a learning signal even when
    all sampled answers have the same correctness. It is not a benchmark score.
    """
    numbers = re.findall(r"(?<!\w)-?\d+(?:\.\d+)?(?!\w)", response)
    correct = bool(numbers) and numbers[-1] == str(expected)
    return float(correct) + 0.1 / (1 + len(response))


def grade(payload):
    if len(payload["responses"]) != len(payload["expected"]):
        raise ValueError("Response and expected-answer counts must match")
    return [
        score_response(r, e) for r, e in zip(payload["responses"], payload["expected"])
    ]


if __name__ == "__main__":
    print(json.dumps(grade(json.loads(sys.argv[1]))))
