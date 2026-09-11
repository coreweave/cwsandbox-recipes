"""Tests for the in-sandbox τ-bench environment server.

These drive the real `_dispatch` routing and a real `EpisodeState` holding a real
τ-bench env, so the reward semantics under test are the shipped ones. No network
and no sandbox required.
"""

import json
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from typing import Any, Dict

import pytest

from verl_taubench.sandbox.env_server import (
    EpisodeError,
    EpisodeState,
    _dispatch,
    build_server,
)


def _state() -> EpisodeState:
    return EpisodeState(domain="retail", task_split="test")


def test_reset_loads_requested_task_and_returns_metadata() -> None:
    state = _state()
    meta = _dispatch(state, "/reset", {"task_index": 3, "episode_id": "ep-3"})

    assert meta["task_index"] == 3
    assert meta["episode_id"] == "ep-3"
    assert meta["instruction"], "instruction is what seeds the remote simulator"
    assert state.env.task is state.env.tasks[3]
    assert state.env.actions == [], "a fresh episode must start with no actions"


def test_reset_is_a_real_reset_between_episodes() -> None:
    """Consecutive resets must not leak actions or DB mutations across episodes."""
    state = _state()
    _dispatch(state, "/reset", {"task_index": 0})
    _dispatch(state, "/step", {"action_name": "respond", "action_kwargs": {"content": "hello"}})
    assert len(state.env.actions) == 1
    first_hash = state.env.get_data_hash()

    _dispatch(state, "/reset", {"task_index": 0})
    assert state.env.actions == [], "actions leaked across episodes"
    assert state.env.get_data_hash() == first_hash, "database not reloaded on reset"
    assert state.done is False


def test_reset_rejects_out_of_range_and_non_integer_task_index() -> None:
    state = _state()
    with pytest.raises(EpisodeError) as out_of_range:
        _dispatch(state, "/reset", {"task_index": 10**9})
    assert out_of_range.value.status == 400

    with pytest.raises(EpisodeError):
        _dispatch(state, "/reset", {"task_index": "not-an-int"})

    with pytest.raises(EpisodeError):
        _dispatch(state, "/reset", {})


def test_step_before_reset_is_a_conflict() -> None:
    state = _state()
    with pytest.raises(EpisodeError) as exc:
        _dispatch(state, "/step", {"action_name": "respond", "action_kwargs": {"content": "x"}})
    assert exc.value.status == 409


def test_respond_is_recorded_but_not_executed() -> None:
    """The user simulator is remote, so respond must be recorded, never executed.

    Recording is load-bearing: `calculate_reward` scans `env.actions` for respond
    actions to verify `task.outputs`. If we dropped them, reward would silently
    be 0 on every task that has required outputs.
    """
    state = _state()
    _dispatch(state, "/reset", {"task_index": 0})

    result = _dispatch(
        state, "/step", {"action_name": "respond", "action_kwargs": {"content": "the answer is 42"}}
    )

    assert result["recorded_only"] is True
    assert result["observation"] == "", "the sandbox must not produce user text"
    assert result["done"] is False
    recorded = [a for a in state.env.actions if a.name == "respond"]
    assert len(recorded) == 1
    assert recorded[0].kwargs["content"] == "the answer is 42"


def test_tool_call_mutates_db_and_returns_observation() -> None:
    state = _state()
    _dispatch(state, "/reset", {"task_index": 0})
    before = state.env.get_data_hash()

    result = _dispatch(
        state,
        "/step",
        {"action_name": "list_all_product_types", "action_kwargs": {}},
    )

    assert result["recorded_only"] is False
    assert result["observation"], "a real tool must return output"
    assert result["source"] == "list_all_product_types"
    # A read-only tool must not change the database.
    assert state.env.get_data_hash() == before


def test_unknown_tool_does_not_crash_the_server() -> None:
    state = _state()
    _dispatch(state, "/reset", {"task_index": 0})
    result = _dispatch(state, "/step", {"action_name": "no_such_tool", "action_kwargs": {}})
    assert "Unknown action" in result["observation"]


def test_reward_is_computed_and_cached() -> None:
    """`calculate_reward` is destructive, so it must run at most once."""
    state = _state()
    _dispatch(state, "/reset", {"task_index": 0})

    first = _dispatch(state, "/reward", {})
    assert isinstance(first["reward"], float)
    assert state.done is True

    calls = {"n": 0}
    real = state.env.calculate_reward

    def counting():
        calls["n"] += 1
        return real()

    state.env.calculate_reward = counting  # type: ignore[method-assign]
    second = _dispatch(state, "/reward", {})

    assert calls["n"] == 0, "calculate_reward must not be re-run; it is destructive"
    assert second == first


def test_recorded_respond_drives_the_output_check() -> None:
    """Respond recording is what makes `task.outputs` scoring work.

    retail/test task 2 requires the agent to say "10". Saying it must be visible
    to `calculate_reward`, which only sees it via the recorded respond action.
    This is the regression test for routing respond to the simulator *without*
    also recording it in the sandbox.
    """
    outputs_task = 2

    # Never said -> the output check fails.
    silent = _state()
    meta = _dispatch(silent, "/reset", {"task_index": outputs_task})
    assert meta["outputs"] == ["10"], "fixture drifted; pick another task"
    _dispatch(silent, "/step", {"action_name": "respond", "action_kwargs": {"content": "no idea"}})
    silent_reward = _dispatch(silent, "/reward", {})
    assert silent_reward["info"]["outputs"] == {"10": False}

    # Said -> the output check passes for that output.
    spoken = _state()
    _dispatch(spoken, "/reset", {"task_index": outputs_task})
    _dispatch(
        spoken,
        "/step",
        {"action_name": "respond", "action_kwargs": {"content": "You have 10 of them."}},
    )
    spoken_reward = _dispatch(spoken, "/reward", {})
    assert spoken_reward["info"]["outputs"] == {"10": True}


def test_dropping_respond_recording_would_break_reward() -> None:
    """Guards the exact failure mode the split architecture invites.

    If respond actions were forwarded only to the remote simulator and never
    recorded in the sandbox, `env.actions` would hold no respond entries and the
    output check would score 0 even for a correct answer.
    """
    state = _state()
    _dispatch(state, "/reset", {"task_index": 2})
    _dispatch(
        state,
        "/step",
        {"action_name": "respond", "action_kwargs": {"content": "You have 10 of them."}},
    )

    # Simulate the bug: discard the recorded respond actions.
    state.env.actions = [a for a in state.env.actions if a.name != "respond"]
    broken = _dispatch(state, "/reward", {})
    assert broken["info"]["outputs"] == {"10": False}
    assert broken["reward"] == 0.0


def test_step_validates_action_shape() -> None:
    state = _state()
    _dispatch(state, "/reset", {"task_index": 0})

    with pytest.raises(EpisodeError):
        _dispatch(state, "/step", {"action_kwargs": {}})

    with pytest.raises(EpisodeError):
        _dispatch(state, "/step", {"action_name": "respond", "action_kwargs": "not-a-dict"})


@contextmanager
def _running_server(token):
    """Run the real HTTP server on an ephemeral port."""
    server, state = build_server(
        domain="retail", task_split="test", host="127.0.0.1", port=0, token=token
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[0], server.server_address[1]
        yield f"http://{host}:{port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _get(url, token=None, timeout=5):
    """Return (status, body) for a GET, treating HTTP errors as responses."""
    req = urllib.request.Request(url, method="GET")
    if token is not None:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def test_bearer_auth_is_enforced_over_real_http() -> None:
    """`ingress_mode="public"` puts this server on the internet.

    The auth check lives in the request handler, which the `_dispatch`-level
    tests bypass entirely, so it needs a real HTTP exercise.
    """
    token = "s3cret-token-value"
    with _running_server(token) as (base, _):
        ok_status, ok_body = _get(f"{base}/health", token=token)
        assert ok_status == 200 and ok_body["status"] == "ok"

        # No credentials at all.
        assert _get(f"{base}/health")[0] == 401
        # Wrong value, same length -- guards a length-only comparison.
        assert _get(f"{base}/health", token="s3cret-token-valuX")[0] == 401
        # Wrong value, different length.
        assert _get(f"{base}/health", token="nope")[0] == 401


def test_server_without_token_serves_openly() -> None:
    """`--insecure` (token=None) must still work for trusted-network use."""
    with _running_server(None) as (base, _):
        assert _get(f"{base}/health")[0] == 200


def test_main_refuses_to_start_without_a_token() -> None:
    """A public-ingress server with no token would be an open relay."""
    from verl_taubench.sandbox import env_server

    with pytest.raises(SystemExit):
        env_server.main(["--domain", "retail", "--task-split", "test"])


def test_health_and_unknown_path() -> None:
    state = _state()
    health: Dict[str, Any] = _dispatch(state, "/health", {})
    assert health["status"] == "ok"
    assert health["episode_loaded"] is False

    with pytest.raises(EpisodeError) as exc:
        _dispatch(state, "/nope", {})
    assert exc.value.status == 404


def test_action_progress_grades_tool_name_then_arguments() -> None:
    from tau_bench.types import Action  # type: ignore[import-untyped]

    gt = [Action(name="exchange_delivered_order_items", kwargs={"order_id": "#W1", "item_ids": ["1"]})]

    def progress(agent):
        return EpisodeState._action_progress(agent, gt)

    # Never called the tool.
    assert progress([Action(name="get_order_details", kwargs={"order_id": "#W1"})]) == 0.0
    # Right tool, both arguments wrong: name credit only.
    assert progress([Action(name="exchange_delivered_order_items", kwargs={"order_id": "#W9", "item_ids": ["9"]})]) == 0.5
    # Right tool, half the arguments right.
    assert progress([Action(name="exchange_delivered_order_items", kwargs={"order_id": "#W1", "item_ids": ["9"]})]) == 0.75
    # Exact match.
    assert progress([Action(name="exchange_delivered_order_items", kwargs={"order_id": "#W1", "item_ids": ["1"]})]) == 1.0


def test_action_progress_cannot_be_inflated_by_repeating_a_call() -> None:
    from tau_bench.types import Action  # type: ignore[import-untyped]

    gt = [
        Action(name="modify_user_address", kwargs={"user_id": "u1"}),
        Action(name="modify_user_address", kwargs={"user_id": "u2"}),
    ]
    # One correct call cannot satisfy both ground-truth actions: it scores 1.0
    # for its match and the other falls back to name-only credit.
    agent = [Action(name="modify_user_address", kwargs={"user_id": "u1"})]
    assert EpisodeState._action_progress(agent, gt) == pytest.approx(0.5)

    spam = [Action(name="modify_user_address", kwargs={"user_id": "u1"})] * 8
    assert EpisodeState._action_progress(spam, gt) == pytest.approx(0.75)


def test_output_progress_counts_required_strings_said_to_the_user() -> None:
    from tau_bench.types import Action  # type: ignore[import-untyped]

    agent = [
        Action(name="respond", kwargs={"content": "Your total is 1,234 dollars."}),
        Action(name="get_order_details", kwargs={"order_id": "#W1"}),
    ]
    # Comma normalization matches tau-bench's own check.
    assert EpisodeState._output_progress(agent, ["1234"]) == 1.0
    assert EpisodeState._output_progress(agent, ["1234", "refunded"]) == 0.5
    assert EpisodeState._output_progress(agent, []) is None


def test_partial_credit_grades_a_failed_episode_between_zero_and_the_weight() -> None:
    state = EpisodeState(domain="retail", task_split="train", partial_credit=0.3)
    task_index = None
    for candidate in range(60):
        state.reset(candidate)
        goal = [a for a in state.env.task.actions if a.name != "respond"]
        if len(goal) >= 2:
            task_index = candidate
            break
    assert task_index is not None, "no multi-action retail task found in range"

    # Execute one ground-truth action exactly; the rest are missed.
    first = goal[0]
    state.step(first.name, dict(first.kwargs))
    payload = state.reward()

    assert payload["info"]["success"] == 0.0
    assert 0.0 < payload["reward"] < 0.3, "graded credit must be strictly between zero and the weight"
    assert 0.0 < payload["info"]["action_progress"] < 1.0
    assert payload["source"] == "calculate_reward+partial_credit"


def test_strict_reward_unchanged_when_partial_credit_disabled() -> None:
    state = EpisodeState(domain="retail", task_split="test")
    state.reset(2)
    payload = state.reward()
    assert payload["reward"] in (0.0, 1.0)
    assert payload["source"] == "calculate_reward"
    assert "action_match_fraction" not in (payload["info"] or {})
