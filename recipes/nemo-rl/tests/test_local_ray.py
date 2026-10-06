"""Verify SUNK's Ray isolation boundary without loading GPU dependencies."""

import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("allocated_cpus, expected_cpus", [(None, 16), ("4", 4)])
def test_ray_stays_local_and_omits_sandbox_key(
    tmp_path, monkeypatch, allocated_cpus, expected_cpus
):
    source = Path(__file__).resolve().parents[1] / "sunk/native_grpo.py"
    tree = ast.parse(source.read_text())
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    # Execute the real startup code, stopping before model loading or compute.
    module = ast.Module(body=[main], type_ignores=[])
    environment = {
        "CWSANDBOX_API_KEY": "test-only-placeholder",
        "RAY_ADDRESS": "ray://unrelated.example.invalid:10001",
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        "PYTHONPATH": "/recipe:/opt/nemo-rl",
    }
    if allocated_cpus is not None:
        environment["SLURM_CPUS_PER_TASK"] = allocated_cpus
    config = {
        "data": {"train_data_path": str(tmp_path / "arithmetic.jsonl")},
        "policy": {"tokenizer": {}},
    }

    class StartupComplete(Exception):
        pass

    ray = SimpleNamespace(init=Mock())
    namespace = {
        "__file__": str(source),
        "argparse": argparse,
        "json": json,
        "os": SimpleNamespace(environ=environment),
        "Path": Path,
        "OmegaConf": SimpleNamespace(
            register_new_resolver=Mock(),
            to_container=lambda value, **_: value,
        ),
        "load_config": lambda _: config,
        "ray": ray,
        "get_tokenizer": Mock(side_effect=StartupComplete),
    }
    monkeypatch.setattr("sys.argv", [str(source)])
    exec(compile(module, str(source), "exec"), namespace)
    with pytest.raises(StartupComplete):
        namespace["main"]()

    ray.init.assert_called_once()
    options = ray.init.call_args.kwargs
    assert options["address"] == "local"
    assert options["include_dashboard"] is False
    assert options["num_cpus"] == expected_cpus
    worker_environment = options["runtime_env"]["env_vars"]
    assert "CWSANDBOX_API_KEY" not in worker_environment
    assert "RAY_ADDRESS" not in worker_environment
    assert "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES" not in worker_environment
    assert worker_environment["PYTHONPATH"] == environment["PYTHONPATH"]
