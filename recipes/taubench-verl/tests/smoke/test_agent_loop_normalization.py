"""Smoke test: dataset-field normalization for ``TauBenchAgentLoop``.

veRL hands non-tensor batch columns to the loop as numpy object arrays, so the
loop normalizes them before any truthiness test or ``dict()`` conversion.
"""

import numpy as np

from verl_taubench.agent.taubench_loop import _normalize_kwargs_value


def test_normalize_kwargs_value_unwraps_numpy() -> None:
    """veRL delivers non-tensor batch columns as numpy object arrays (R3-1)."""
    # The exact shape that used to raise "truth value of an array with more than
    # one element is ambiguous" on every rollout of the first training step.
    multi = np.array(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        dtype=object,
    )
    out = _normalize_kwargs_value(multi)
    assert out == [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    assert bool(out) is True, "`if raw_prompt:` must not raise"

    # A single-message prompt must stay a list, not collapse to a bare dict --
    # otherwise `[dict(m) for m in raw_prompt]` iterates over dict *keys*.
    single = np.array([{"role": "system", "content": "s"}], dtype=object)
    assert _normalize_kwargs_value(single) == [{"role": "system", "content": "s"}]
    assert [dict(m) for m in _normalize_kwargs_value(single)] == [
        {"role": "system", "content": "s"}
    ]

    # 0-d object arrays wrap a single object (e.g. the extra_info dict).
    zero_d = np.empty((), dtype=object)
    zero_d[()] = {"domain": "retail", "task_index": np.int64(3)}
    assert _normalize_kwargs_value(zero_d) == {"domain": "retail", "task_index": 3}

    # numpy scalars unwrap to real Python scalars.
    assert _normalize_kwargs_value(np.int64(7)) == 7
    assert isinstance(_normalize_kwargs_value(np.int64(7)), int)
