"""Smoke test for the τ-bench environment wrapper.

This test uses `user_strategy="human"` so it never calls an LLM API. We monkeypatch
`input()` so the human user simulator returns a canned response and the test stays
fully offline.
"""

import builtins
from unittest import mock

import pytest

from verl_taubench.envs import taubench_env

# We only test retail to avoid dragging in the airline dependency chain, which
# is structurally identical.


def test_create_retail_env_human_only():
    with mock.patch.object(builtins, "input", return_value="OK"):
        env = taubench_env.make_env(
            domain="retail",
            task_split="test",
            user_strategy="human",
            task_index=0,
        )
        reset = taubench_env.reset_env(env, task_index=0)
    assert "observation" in reset
    assert "task" in reset
    assert reset["source"] == "user"


def test_make_env_survives_inclusive_randint():
    """τ-bench's ``Env.__init__`` picks tasks with an INCLUSIVE ``randint``.

    ``random.randint(0, len(tasks))`` can return ``len(tasks)``, which then raises
    ``IndexError`` on ``tasks[i]``. ``make_env`` must never depend on that call, so
    we force the off-by-one value and assert both code paths still work.
    """
    with mock.patch("random.randint", side_effect=lambda a, b: b):
        with mock.patch.object(builtins, "input", return_value="OK"):
            env = taubench_env.make_env(
                domain="retail", task_split="test", user_strategy="human"
            )
            pinned = taubench_env.make_env(
                domain="retail", task_split="test", task_index=5, user_strategy="human"
            )

    # Random selection must land in range and be self-consistent.
    assert 0 <= env.task_index < len(env.tasks)
    assert env.task is env.tasks[env.task_index]
    # An explicit index must be honoured exactly.
    assert pinned.task_index == 5
    assert pinned.task is pinned.tasks[5]


def test_reset_env_preserves_pinned_task():
    """``reset_env(env)`` must not silently re-randomise a pinned task."""
    with mock.patch.object(builtins, "input", return_value="OK"):
        env = taubench_env.make_env(
            domain="retail", task_split="test", task_index=7, user_strategy="human"
        )
        # Force Env.reset()'s internal random pick to a different task.
        with mock.patch("random.randint", return_value=123):
            taubench_env.reset_env(env)

    assert env.task_index == 7, "reset_env(env) re-randomised a pinned task"
    assert env.task is env.tasks[7]


def test_make_env_rejects_out_of_range_task_index():
    """An explicit out-of-range index must fail loudly and deterministically."""
    with mock.patch.object(builtins, "input", return_value="OK"):
        with pytest.raises(IndexError, match="out of range"):
            taubench_env.make_env(
                domain="retail",
                task_split="test",
                task_index=10**9,
                user_strategy="human",
            )


def test_openai_tools_schema_nonempty():
    env = taubench_env.make_env(domain="retail", task_split="test", user_strategy="human")
    tools = taubench_env.env_to_openai_tools(env)
    assert tools, "τ-bench retail env should expose tool schemas"


def test_system_prompt_builds():
    env = taubench_env.make_env(domain="retail", task_split="test", user_strategy="human")
    prompt = taubench_env.build_system_prompt(env)
    assert "Available tools:" in prompt
    assert "<tool_call>" in prompt


