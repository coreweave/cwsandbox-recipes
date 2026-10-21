"""Agent Lens traces and Weave evaluations from the episode server, with fakes for the SDKs."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gym_env" / "math_sandbox_eval"))
import forge  # noqa: E402

# A rollout as Gym's simple agent returns it: one code reply, its output, then the answer.
OUTPUT = [
    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "```python\nprint(17*23)\n```"}],
     "prompt_token_ids": list(range(80)), "generation_token_ids": list(range(20))},
    {"type": "function_call", "name": "execute_python", "arguments": '{"code": "print(17*23)"}', "call_id": "call_a"},
    {"type": "function_call_output", "call_id": "call_a", "output": "391"},
    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "So \\boxed{391}."}],
     "prompt_token_ids": list(range(110)), "generation_token_ids": list(range(12))},
]  # fmt: skip


def test_model_calls_split_on_tool_results():
    calls = forge.model_calls(OUTPUT)
    assert [c.text for c in calls] == ["```python\nprint(17*23)\n```", "So \\boxed{391}."]
    assert calls[0].tool_call_ids == ["call_a"] and calls[1].tool_call_ids == []
    assert (calls[0].input_tokens, calls[0].output_tokens) == (80, 20)
    assert (calls[1].input_tokens, calls[1].output_tokens) == (110, 12)


def test_spans_follow_the_episode_timeline():
    cell = forge.Cell("print(17*23)", "391", started=103.0, ended=104.0)
    first, tool, second = forge.episode_spans(OUTPUT, [cell], started=100.0, ended=106.0, model="m")
    assert (first.started_at, first.ended_at) == (forge.at(100.0), forge.at(103.0))
    assert tool.name == "execute_python" and tool.result == "391" and tool.tool_call_id == "call_a"
    assert (tool.started_at, tool.ended_at) == (forge.at(103.0), forge.at(104.0))
    assert (second.started_at, second.ended_at) == (forge.at(104.0), forge.at(106.0))
    assert second.usage.output_tokens == 12


def test_a_call_the_server_never_ran_gets_no_tool_span():
    spans = forge.episode_spans(OUTPUT, [], started=100.0, ended=106.0, model="m")
    assert [type(s).__name__ for s in spans] == ["LLM", "LLM"]


def test_round_step_is_the_step_validation_ran_after():
    assert [forge.round_step(k, 10, 25) for k in (1, 2, 3)] == [10, 20, 25]
    assert forge.round_step(1, 10, 3) == 3  # a short run validates only at the end


def test_eval_round_logs_when_full_or_idle():
    round_ = forge.EvalRound(size=2, idle_s=60)
    assert round_.add({"n": 1}, now=0) is None
    assert round_.add({"n": 2}, now=1) == [{"n": 1}, {"n": 2}]
    assert round_.add({"n": 3}, now=2) is None
    assert round_.due(now=30) is None and round_.due(now=63) == [{"n": 3}]
    assert round_.due(now=200) is None


class FakeEvaluationLogger:
    created = []

    def __init__(self, **kwargs):
        self.kwargs, self.examples, self.summary = kwargs, [], None
        FakeEvaluationLogger.created.append(self)

    def log_example(self, inputs, output, scores):
        self.examples.append((inputs, output, scores))

    def log_summary(self, summary):
        self.summary = summary


@pytest.fixture
def logged(monkeypatch):
    turns = []
    monkeypatch.setattr(forge.tracing, "init", lambda project, autopatch_integrations: None)
    monkeypatch.setattr(forge.tracing, "log_turn", lambda **kwargs: turns.append(kwargs))
    monkeypatch.setattr(
        forge, "weave", SimpleNamespace(init=lambda project: None, EvaluationLogger=FakeEvaluationLogger)
    )
    FakeEvaluationLogger.created = []
    return turns


def test_forge_is_off_without_a_wandb_key():
    assert forge.Forge.from_env({"WANDB_ENTITY": "team"}) is None


def test_trace_and_evaluation_rows(logged):
    environ = {"WANDB_API_KEY": "k", "WANDB_ENTITY": "team", "RUN_TAG": "run-1", "MODEL_REPO": "Qwen/Qwen2.5-Math-7B",
               "EVAL_ROWS": "2", "VAL_PERIOD": "10", "MAX_NUM_STEPS": "3"}  # fmt: skip
    f = forge.Forge.from_env(environ)
    assert f.project == "team/async-grpo-tool-use" and f.wants("eval")
    cell = forge.Cell("print(17*23)", "391", 103.0, 104.0)
    f.trace(session_id="s1", split="eval", question="What is 17 times 23?", system="Please integrate", output=OUTPUT,
            cells=[cell], started=100.0, ended=106.0, answer="So \\boxed{391}.", expected="391", extracted="391",
            reward=1.0)  # fmt: skip
    [turn] = logged
    assert turn["conversation_id"] == "run-1-s1" and turn["agent_name"] == forge.AGENT_NAME
    assert turn["attributes"]["rl.split"] == "eval" and turn["attributes"]["rl.reward"] == 1.0
    assert len(turn["spans"]) == 3

    for reward in (1.0, 0.0):
        f.add_eval({"question": "q", "expected": "391", "answer": "a", "extracted": None, "cells": 1, "reward": reward})
    [evaluation] = FakeEvaluationLogger.created
    assert evaluation.kwargs["name"] == "aime-2024-step-3"
    assert [scores["correct"] for _, _, scores in evaluation.examples] == [True, False]
    assert evaluation.summary["accuracy"] == 0.5
