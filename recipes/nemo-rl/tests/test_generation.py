"""Check the real sampling call without loading GPU training dependencies."""

import ast
from pathlib import Path
from types import SimpleNamespace


def test_sampling_overrides_qwen_defaults_to_match_loss_and_mask():
    source = Path(__file__).resolve().parents[1] / "scripts/train_smoke.py"
    tree = ast.parse(source.read_text())
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "model"
        and node.func.attr == "generate"
    )
    # These are the pinned Qwen generation defaults. Inheriting them applies
    # a different behavior distribution and can terminate before our mask's EOS.
    defaults = {"repetition_penalty": 1.1, "eos_token_id": [151645, 151643]}
    model = SimpleNamespace(generate=lambda **kwargs: defaults | kwargs)
    tokenizer = SimpleNamespace(eos_token_id=151645, pad_token_id=151645)
    options = eval(
        compile(ast.Expression(call), str(source), "eval"),
        {"model": model, "inputs": {}, "tokenizer": tokenizer},
    )
    assert options["repetition_penalty"] == 1.0
    assert options["temperature"] == options["top_p"] == 1.0
    assert options["top_k"] == 0
    assert options["eos_token_id"] == tokenizer.eos_token_id
    assert options["pad_token_id"] == tokenizer.pad_token_id
