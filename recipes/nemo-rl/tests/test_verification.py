"""Exercise the actual numeric guards without importing GPU training packages."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


def run_guard(script, message, **values):
    source = Path(__file__).resolve().parents[1] / "scripts" / script
    tree = ast.parse(source.read_text())
    # Extract the production guard, including its raise, rather than duplicate
    # its condition or import torch, NeMo RL, and Transformers on the launcher.
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(statement, ast.Raise)
            and isinstance(statement.exc, ast.Call)
            and statement.exc.args
            and isinstance(statement.exc.args[0], ast.Constant)
            and statement.exc.args[0].value == message
            for statement in node.body
        )
    ]
    assert len(matches) == 1

    def isfinite(value):
        if isinstance(value, list):
            return SimpleNamespace(all=lambda: all(math.isfinite(x) for x in value))
        return math.isfinite(value)

    code = compile(ast.Module(body=matches, type_ignores=[]), str(source), "exec")
    exec(code, {"math": math, "torch": SimpleNamespace(isfinite=isfinite), **values})


@pytest.mark.parametrize("field", ["loss", "grad_norm"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_training_rejects_nonfinite_loss_and_gradient(field, invalid):
    values = {"loss": 0.0, "grad_norm": 1.0, field: invalid}
    with pytest.raises(RuntimeError, match="finite, nonzero policy gradient"):
        run_guard(
            "train_smoke.py",
            "No finite, nonzero policy gradient; increase rollout diversity",
            **values,
        )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["baseline", "adapted"])
def test_reload_rejects_nonfinite_logits(field, invalid):
    values = {"baseline": [0.0, 1.0], "adapted": [0.0, 2.0], field: [0.0, invalid]}
    with pytest.raises(RuntimeError, match="non-finite logits"):
        run_guard(
            "verify_smoke.py", "Checkpoint reload produced non-finite logits", **values
        )


@pytest.mark.parametrize(
    "delta", [0.0, -1.0, float("nan"), float("inf"), -float("inf")]
)
def test_reload_requires_finite_positive_logit_delta(delta):
    with pytest.raises(RuntimeError, match="did not change model logits"):
        run_guard(
            "verify_smoke.py",
            "Reloaded checkpoint did not change model logits",
            delta=delta,
        )


@pytest.mark.parametrize(
    "tensors",
    [
        {},
        {"weight": [float("nan")]},
        {"weight": [float("inf")]},
        {"weight": [-float("inf")]},
    ],
)
def test_reload_rejects_empty_or_nonfinite_adapter(tensors):
    with pytest.raises(RuntimeError, match="empty or non-finite"):
        run_guard(
            "verify_smoke.py",
            "Checkpoint adapter tensors are empty or non-finite",
            tensors=tensors,
        )


def test_finite_update_and_reload_pass():
    run_guard(
        "train_smoke.py",
        "No finite, nonzero policy gradient; increase rollout diversity",
        loss=0.0,
        grad_norm=0.5,
    )
    run_guard(
        "verify_smoke.py",
        "Checkpoint reload produced non-finite logits",
        baseline=[0.0],
        adapted=[0.5],
    )
    run_guard(
        "verify_smoke.py", "Reloaded checkpoint did not change model logits", delta=0.5
    )
    run_guard(
        "verify_smoke.py",
        "Checkpoint adapter tensors are empty or non-finite",
        tensors={"weight": [0.0, 0.5]},
    )
