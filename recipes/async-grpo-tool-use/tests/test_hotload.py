"""Snapshot files, the cwtrain wrapper and the Gym token-ID adapter, with fakes for the API."""

import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from async_grpo_tool_use import hotload
from async_grpo_tool_use import inference as inference_module
from async_grpo_tool_use.inference import DeploymentSettings, Inference, deployment_spec

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gym_env"))
import token_ids_model  # noqa: E402


def _shard(path: Path, tensors: dict[str, str]) -> None:
    header = json.dumps(
        {name: {"dtype": dtype, "shape": [1], "data_offsets": [0, 2]} for name, dtype in tensors.items()}
    )
    path.write_bytes(struct.pack("<Q", len(header)) + header.encode() + b"\0\0")


def test_checkpoint_dtypes_and_model_files(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    _shard(base / "model-00001-of-00001.safetensors", {"a": "BF16", "b": "F32"})
    (base / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00001.safetensors", "b": "model-00001-of-00001.safetensors"}})
    )
    (base / "config.json").write_text("{}")
    (base / "tokenizer.json").write_text("{}")
    assert hotload.checkpoint_dtypes(base) == {"a": "BF16", "b": "F32"}
    export = tmp_path / "export"
    export.mkdir()
    hotload.add_model_files(base, export)
    assert sorted(p.name for p in export.iterdir()) == ["config.json", "tokenizer.json"]


def test_checkpoint_without_an_index_is_one_file(tmp_path):
    _shard(tmp_path / "model.safetensors", {"a": "BF16", "b": "F32"})
    assert hotload.checkpoint_dtypes(tmp_path) == {"a": "BF16", "b": "F32"}
    assert hotload.checkpoint_sizes(tmp_path) == {"a": 2, "b": 2}


def test_deployment_spec_enables_hot_load():
    spec = deployment_spec("run", "gw", "bucket", "p/base", "p/hotload", DeploymentSettings(replicas=3))
    assert spec["hotLoad"] == {"bucket": "bucket", "pathPrefix": "p/hotload", "transitionMode": "TRANSITION_MODE_ASYNC"}
    assert spec["autoscaling"]["min"] == spec["autoscaling"]["max"] == 3
    assert spec["runtime"]["engineConfig"]["max-model-len"] == "3072"


def test_inference_creates_and_deletes_with_cwtrain(monkeypatch):
    from cwtrain.infra import lifecycle

    calls = []
    client = SimpleNamespace(
        create_gateway=lambda spec: calls.append(("gateway", spec)) or SimpleNamespace(id="gw"),
        create_deployment=lambda spec: SimpleNamespace(id="dep"),
    )
    monkeypatch.setattr(
        lifecycle, "wait_for_gateway", lambda c, gid: SimpleNamespace(id=gid, endpoints=("https://gw",))
    )
    monkeypatch.setattr(lifecycle, "wait_for_deployment", lambda c, did: calls.append(("ready", did)))
    monkeypatch.setattr(lifecycle, "delete_deployment", lambda c, did, timeout: calls.append(("delete", did)))
    monkeypatch.setattr(lifecycle, "delete_gateway", lambda c, gid: calls.append(("delete", gid)))
    inference = Inference(client, cwic=lambda args: json.dumps({"location": "US-EAST-02A"}))
    assert inference.create_gateway("run", "US-WEST-04A") == "gw"
    assert inference.wait_gateway("gw") == "https://gw"
    assert calls[0][1]["zones"] == ["US-WEST-04A"] and "coreWeaveAuth" in calls[0][1]
    assert inference.create_deployment({}) == "dep"
    inference.wait_deployment("dep")
    inference.delete_deployment("dep")
    inference.delete_gateway("gw")
    assert calls[1:] == [("ready", "dep"), ("delete", "dep"), ("delete", "gw")]
    assert inference.bucket_zone("b") == "US-EAST-02A"


def test_cwic_runs_with_the_run_token_not_its_own_login(monkeypatch):
    seen = {}

    def fake_run(command, capture_output, text, env):
        seen.update(command=command, token=env["COREWEAVE_API_TOKEN"])
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(inference_module.subprocess, "run", fake_run)
    inference_module.cwic_runner("cw-token")(["cwobject", "bucket", "get", "b"])
    assert seen == {"command": ["cwic", "cwobject", "bucket", "get", "b"], "token": "cw-token"}


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict):
        assert tokenize and add_generation_prompt and messages == [{"role": "user", "content": "1+1?"}]
        return [10, 11, 12]


def test_token_ids_adapter_round_trip():
    body = token_ids_model.completion_request(
        {"model": "m", "messages": [{"role": "user", "content": [{"type": "input_text", "text": "1+1?"}]}],
         "max_tokens": 8, "temperature": 1.0, "logprobs": True, "top_logprobs": 0},
        FakeTokenizer(),
    )  # fmt: skip
    assert body == {"model": "m", "temperature": 1.0, "prompt": [10, 11, 12], "max_tokens": 8, "logprobs": 0,
                    "return_tokens_as_token_ids": True}  # fmt: skip
    response = {"id": "c", "created": 1, "model": "m", "usage": {},
                "choices": [{"text": "2", "finish_reason": "stop",
                             "logprobs": {"tokens": ["token_id:17", "token_id:2"], "token_logprobs": [-0.1, -0.2]}}]}  # fmt: skip
    message = token_ids_model.chat_response(response, body["prompt"])["choices"][0]["message"]
    assert message["prompt_token_ids"] == [10, 11, 12] and message["generation_token_ids"] == [17, 2]
    assert message["generation_log_probs"] == [-0.1, -0.2]
    response["choices"][0]["logprobs"]["tokens"] = ["2", "x"]
    with pytest.raises(ValueError):
        token_ids_model.chat_response(response, body["prompt"])
