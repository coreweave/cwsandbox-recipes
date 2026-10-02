"""Check the exact-token serving contract without network or GPU access."""

import io
import json
from pathlib import Path
import runpy
import urllib.request

import pytest


sample = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "serverless" / "checkpoints.py")
)["sample"]
URI = "wandb-artifact:///example/project/policy:v2"
BASE = "example/base-model"


def response():
    return {
        "id": "test-request",
        "model": BASE,
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        "choices": [
            {
                "text": "43",
                "finish_reason": "stop",
                "logprobs": {
                    "tokens": ["token_id:43", "token_id:2"],
                    "token_logprobs": [-0.1, -0.2],
                },
            }
        ],
    }


def test_completions_keeps_requested_policy_and_exact_tokens(monkeypatch):
    requests = []

    def serve(request, timeout):
        requests.append(request)
        return io.StringIO(json.dumps(response()))

    monkeypatch.setenv("WANDB_API_KEY", "test-only-placeholder")
    monkeypatch.setattr(urllib.request, "urlopen", serve)
    rollout = sample(
        URI,
        [],
        entity="example",
        project="project",
        prompt_token_ids=[10, 20, 30],
        base_model=BASE,
    )
    request = json.loads(requests[0].data)
    assert requests[0].full_url.endswith("/v1/completions")
    assert request["model"] == URI
    assert request["prompt"] == [10, 20, 30]
    assert request["add_special_tokens"] is False
    assert "messages" not in request
    assert rollout["policy_version"] == URI
    assert rollout["served_model"] == BASE
    assert rollout["output_token_ids"] == [43, 2]
    assert rollout["output_logprobs"] == [-0.1, -0.2]


@pytest.mark.parametrize("defect", ["missing_eos", "nonfinite", "wrong_model"])
def test_completions_rejects_unusable_training_evidence(monkeypatch, defect):
    body = response()
    if defect == "missing_eos":
        body["usage"]["completion_tokens"] = 3
    elif defect == "nonfinite":
        body["choices"][0]["logprobs"]["token_logprobs"][0] = float("nan")
    else:
        body["model"] = "unrelated/model"
    monkeypatch.setenv("WANDB_API_KEY", "test-only-placeholder")
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: io.StringIO(json.dumps(body)),
    )
    with pytest.raises(ValueError):
        sample(
            URI,
            [],
            entity="example",
            project="project",
            prompt_token_ids=[10, 20, 30],
            base_model=BASE,
        )


def test_mutable_alias_is_rejected_before_network_access():
    with pytest.raises(ValueError, match="Pin an uploaded adapter version"):
        sample(URI.replace(":v2", ":latest"), [], entity="example", project="project")
