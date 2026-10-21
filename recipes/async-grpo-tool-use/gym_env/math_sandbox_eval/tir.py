"""Pure helpers of the tool-integrated reasoning server (``tir_app.py``), kept free of Gym and
sandbox imports so they can be unit-tested anywhere."""

import json
from typing import Any, Optional

TIMEOUT_RETURN_CODE = 124  # what coreutils `timeout` returns when it kills the command


def truncate(text: str, limit: int) -> str:
    """Keep the head and the tail of a long output; the model rarely needs the middle."""
    if len(text) <= limit:
        return text
    marker = f"\n... ({len(text) - limit} characters cut) ...\n"
    head = limit // 2
    tail = limit - head
    return text[:head] + marker + text[-tail:]


def render_output(stdout: str, stderr: str, return_code: int, timeout_s: float, limit: int) -> str:
    """What the model sees in the ```output block: stdout, plus the error line when the code
    failed. A full traceback costs tokens and says little more than its last line."""
    if return_code == TIMEOUT_RETURN_CODE:
        return truncate(stdout, limit) + f"\nTimeoutError: the code ran longer than {timeout_s:g} s"
    if return_code != 0:
        lines = [line for line in stderr.splitlines() if line.strip()]
        error = lines[-1] if lines else f"the code exited with status {return_code}"
        return truncate(stdout + error, limit)
    return truncate(stdout, limit)


def episode_answer(output: list[dict[str, Any]]) -> Optional[str]:
    """The text of the final assistant message, or None when the episode stopped on a tool call
    (the agent's turn limit): an unfinished episode gets no answer and reward 0."""
    if not output:
        return None
    last = output[-1]
    if last.get("type") != "message" or last.get("role") != "assistant":
        return None
    return "".join(part.get("text", "") for part in last.get("content") or [] if part.get("type") == "output_text")


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content or [])


def episode_example(
    input_items: Any, output: list[dict[str, Any]], expected: str, reward: float, extracted: Optional[str],
    sandbox_id: Optional[str], max_bytes: int = 15_000,
) -> dict[str, Any]:  # fmt: skip
    """One finished episode for the dashboard's examples: the problem, each assistant turn's text,
    the code of each ``execute_python`` call as written to ``/work/cell_<n>.py``, and its output.
    Fields are cut so the row, as JSON, stays under ``max_bytes`` and fits one log line."""
    if isinstance(input_items, str):
        prompt = input_items
    else:
        users = [_text(m.get("content")) for m in input_items or [] if m.get("role") == "user"]
        prompt = users[-1] if users else ""
    turns: list[dict[str, Any]] = []
    for item in output:
        kind = item.get("type")
        if kind == "message" and item.get("role") == "assistant":
            turns.append({"text": _text(item.get("content")).strip()})
        elif kind == "function_call":
            if not turns or "code" in turns[-1]:
                turns.append({"text": ""})
            turn = turns[-1]
            # the reply's text ends with the same code as a ```python block; show it once
            cut = turn["text"].rfind("```python")
            if cut >= 0:
                turn["text"] = turn["text"][:cut].rstrip()
            turn["code"] = json.loads(item["arguments"]).get("code", "")
            turn["cell"] = f"/work/cell_{sum('code' in t for t in turns)}.py"
        elif kind == "function_call_output" and turns and "code" in turns[-1]:
            turns[-1]["output"] = str(item.get("output", ""))
    # JSON escaping can double the size of code full of quotes and newlines: cut harder until it fits
    limits = {"text": 600, "code": 1200, "output": 400}
    for scale in (1, 0.5, 0.25, 0.1):
        row = {
            "k": "example", "prompt": truncate(prompt, int(1000 * scale)),
            "turns": [{key: truncate(value, int(limits[key] * scale)) if key in limits else value
                       for key, value in turn.items()} for turn in turns[:6]],
            "cells": sum("code" in turn for turn in turns), "answer": extracted, "expected": expected,
            "reward": reward, "sandbox": sandbox_id,
        }  # fmt: skip
        if len(json.dumps(row)) < max_bytes:
            break
    return row
