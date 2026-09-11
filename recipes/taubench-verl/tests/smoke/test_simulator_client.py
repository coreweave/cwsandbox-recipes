"""User simulator over the openai SDK: endpoint, auth, and error behavior."""

import json

import httpx
import pytest

from verl_taubench.sandbox.simulator import (
    OpenAICompatibleSimulator,
    resolve_simulator_api_key,
)


def _mock_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _completion(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "cmpl-1",
            "object": "chat.completion",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        },
    )


def test_base_url_gains_v1_suffix_once() -> None:
    bare = OpenAICompatibleSimulator("http://sim.example:8000", model="m", api_key="k")
    suffixed = OpenAICompatibleSimulator("http://sim.example:8000/v1/", model="m", api_key="k")
    assert bare.base_url == "http://sim.example:8000/v1"
    assert suffixed.base_url == "http://sim.example:8000/v1"


def test_api_key_resolution_matches_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "wandb-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("TAUBENCH_SIMULATOR_API_KEY", "selfhosted-key")
    assert resolve_simulator_api_key("https://api.inference.wandb.ai/v1") == "wandb-key"
    assert resolve_simulator_api_key("https://api.openai.com/v1") == "openai-key"
    assert resolve_simulator_api_key("http://sim.internal:8000/v1") == "selfhosted-key"
    assert resolve_simulator_api_key("http://x/v1", explicit="direct") == "direct"


@pytest.mark.asyncio
async def test_start_sends_openai_chat_request_with_bearer() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return _completion("Hi, I need help with an exchange.")

    simulator = OpenAICompatibleSimulator(
        "http://sim.example:8000",
        model="test-model",
        api_key="secret-key",
        http_client=_mock_client(handler),
    )
    opening = await simulator.start("ep-1", "You want to exchange a keyboard.")

    assert opening == "Hi, I need help with an exchange."
    assert seen["url"] == "http://sim.example:8000/v1/chat/completions"
    assert seen["auth"] == "Bearer secret-key"
    assert seen["body"]["model"] == "test-model"
    roles = [m["role"] for m in seen["body"]["messages"]]
    assert roles == ["system", "user"]


@pytest.mark.asyncio
async def test_respond_appends_history_and_http_errors_propagate() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return _completion("opening line")
        return httpx.Response(400, json={"error": {"message": "invalid model ID"}})

    simulator = OpenAICompatibleSimulator(
        "http://sim.example:8000",
        model="test-model",
        api_key="k",
        http_client=_mock_client(handler),
    )
    await simulator.start("ep-1", "instruction")
    with pytest.raises(RuntimeError, match="user simulator 400"):
        await simulator.respond("ep-1", "agent message")


@pytest.mark.asyncio
async def test_respond_requires_started_episode() -> None:
    simulator = OpenAICompatibleSimulator(
        "http://sim.example:8000",
        model="m",
        api_key="k",
        http_client=_mock_client(lambda request: _completion("x")),
    )
    with pytest.raises(KeyError):
        await simulator.respond("never-started", "hello")
