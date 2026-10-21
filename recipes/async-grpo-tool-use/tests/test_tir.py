"""Tool-integrated reasoning with one sandbox per episode: the pure parts, run locally."""

import json
import sys
import types
from pathlib import Path

import pytest
import yaml

from async_grpo_tool_use import dapo_data, run_gym

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gym_env"))
sys.path.insert(0, str(ROOT / "gym_env" / "math_sandbox_eval"))
import tir  # noqa: E402
import token_ids_model as tim  # noqa: E402


class FakeTokenizer:
    """One token per character; 0 is the end-of-text token."""

    all_special_ids = [0]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict):
        return [ord(c) for c in "".join(m["content"] for m in messages)]

    def encode(self, text, add_special_tokens):
        assert not add_special_tokens
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids if i)


def test_trailing_code_takes_only_a_block_that_ends_the_reply():
    assert tim.trailing_code("Let me compute.\n```python\nprint(1+1)\n```\n") == "print(1+1)\n"
    assert tim.trailing_code("```python\nx = 1\n```\n```output") == "x = 1\n"  # stop string kept by the engine
    assert tim.trailing_code("```python\na\n```\n```python\nb\n```") == "b\n"
    assert tim.trailing_code("```python\nprint(2)\n```\nSo the answer is \\boxed{2}.") is None
    assert tim.trailing_code("```python\nprint(2)") is None  # unclosed
    assert tim.trailing_code("```python\n\n```") is None
    assert tim.trailing_code("no code here") is None


def _chat(text, tokens, finish="stop"):
    message = {"role": "assistant", "content": text, "prompt_token_ids": [1],
               "generation_token_ids": tokens, "generation_log_probs": [-0.1] * len(tokens)}  # fmt: skip
    return {"choices": [{"index": 0, "finish_reason": finish, "message": message}]}


def test_reply_stopped_on_output_then_end_of_text_becomes_a_call():
    """The deployment drops the stop tokens and ends with end-of-text; that must not end the episode."""
    chat = tim.with_tool_call(_chat("```python\nprint(1)\n```", [5, 7, 0]), {0})
    message = chat["choices"][0]["message"]
    assert message["tool_calls"][0]["function"]["name"] == "execute_python"
    assert message["generation_token_ids"] == [5, 7] and len(message["generation_log_probs"]) == 2


def test_reply_ending_in_code_becomes_an_execute_python_call():
    chat = tim.with_tool_call(_chat("```python\nprint(6*7)\n```\n", [5, 6]), {0})
    choice = chat["choices"][0]
    (call,) = choice["message"]["tool_calls"]
    assert choice["finish_reason"] == "tool_calls" and call["function"]["name"] == "execute_python"
    assert json.loads(call["function"]["arguments"]) == {"code": "print(6*7)\n"}


@pytest.mark.parametrize(
    "text, tokens, finish",
    [
        ("```python\nprint(1)\n```\n", [5, 6], "length"),  # cut by the token limit
        ("The answer is \\boxed{1}.", [5, 6], "stop"),
    ],
)
def test_other_replies_stay_final(text, tokens, finish):
    assert "tool_calls" not in tim.with_tool_call(_chat(text, tokens, finish), {0})["choices"][0]["message"]


def test_later_turns_extend_the_previous_token_ids():
    """NeMo-RL requires each turn's prompt to start with the previous prompt and generation."""
    tok = FakeTokenizer()
    first = tim.prompt_token_ids([{"role": "user", "content": "Q"}], tok)
    generated = tok.encode("```python\nprint(42)\n```\n```output", False)
    messages = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "...", "prompt_token_ids": first, "generation_token_ids": generated,
         "generation_log_probs": [0.0] * len(generated), "tool_calls": []},
        {"role": "tool", "content": "42\n", "tool_call_id": "call_1"},
    ]  # fmt: skip
    prompt = tim.prompt_token_ids(messages, tok)
    assert prompt[: len(first) + len(generated)] == first + generated
    assert tok.decode(prompt[len(first) + len(generated) :]) == "\n42\n```\n"


def test_observation_opens_the_output_block_when_the_engine_dropped_the_stop_string():
    assert tim.observation("42", "```\n") == "```output\n42\n```\n"
    assert tim.observation("42", "```\n```output") == "\n42\n```\n"
    assert tim.observation("42", "print(1)\n```") == "\n```output\n42\n```\n"


def test_only_tool_results_can_follow_a_generated_turn():
    messages = [{"role": "assistant", "content": "x", "prompt_token_ids": [1], "generation_token_ids": [2]},
                {"role": "user", "content": "more?"}]  # fmt: skip
    with pytest.raises(ValueError):
        tim.prompt_token_ids(messages, FakeTokenizer())


def test_tir_request_stops_for_tool_output_and_fits_the_context():
    body = {"model": "m", "messages": [{"role": "user", "content": "abcd"}], "max_tokens": 2048,
            "tools": [{"type": "function", "function": {"name": "execute_python"}}]}  # fmt: skip
    request = tim.completion_request(body, FakeTokenizer(), max_model_len=10)
    assert request["stop"] == ["```output"] and request["prompt"] == [97, 98, 99, 100]
    assert request["max_tokens"] == 6
    full = dict(body, messages=[{"role": "user", "content": "x" * 12}])
    assert tim.completion_request(full, FakeTokenizer(), max_model_len=10)["max_tokens"] == 1  # engine rejects it
    plain = tim.completion_request({"messages": [{"role": "user", "content": "a"}], "max_tokens": 8}, FakeTokenizer())
    assert "stop" not in plain and plain["max_tokens"] == 8
    with pytest.raises(ValueError):
        tim.completion_request(
            dict(body, tools=[{"type": "function", "function": {"name": "search"}}]), FakeTokenizer()
        )


def test_output_truncation_keeps_head_and_tail():
    text = "".join(f"line {i}\n" for i in range(1000))
    cut = tir.truncate(text, 100)
    assert cut.startswith("line 0\n") and cut.endswith("line 999\n") and "characters cut" in cut
    assert len(cut) < 200
    assert tir.truncate("short", 100) == "short"


def test_rendered_output_shows_errors_and_timeouts():
    assert tir.render_output("42\n", "", 0, 15, 512) == "42\n"
    traceback = "Traceback (most recent call last):\n  File \"cell_1.py\", line 1\nNameError: name 'x' is not defined\n"
    assert tir.render_output("", traceback, 1, 15, 512) == "NameError: name 'x' is not defined"
    assert tir.render_output("partial\n", "", 124, 15, 512) == "partial\n\nTimeoutError: the code ran longer than 15 s"


def test_an_episode_that_stopped_on_a_tool_call_has_no_answer():
    final = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "\\boxed{7}"}]}
    call = {"type": "function_call", "name": "execute_python", "arguments": "{}", "call_id": "c"}
    result = {"type": "function_call_output", "call_id": "c", "output": "7"}
    assert tir.episode_answer([call, result, final]) == "\\boxed{7}"
    assert tir.episode_answer([final, call, result]) is None  # the agent hit max_steps
    assert tir.episode_answer([]) is None


def test_tir_config_limits_turns_and_defaults_to_serverless():
    config = yaml.safe_load((ROOT / "gym_env/math_sandbox_eval/configs/math_tir_sandbox.yaml").read_text())
    server = config["math_sandbox_eval"]["resources_servers"]["math_sandbox_eval"]
    assert server["entrypoint"] == "tir_app.py"
    assert server["sandbox"]["cwsandbox"]["placement"] == "${oc.env:SANDBOX_PLACEMENT,serverless}"
    assert (server["cpu"], server["memory_mib"]) == (1, 2048)
    for agent in ("math_tir_agent", "math_eval_agent"):
        assert config[agent]["responses_api_agents"]["simple_agent"]["max_steps"] == 5


def test_placement_kwargs(monkeypatch):
    stub = types.ModuleType("nemo_gym.sandbox")
    for name in ("SandboxCreateError", "SandboxExecResult", "SandboxHandle", "SandboxSpec", "SandboxStatus"):
        setattr(stub, name, type(name, (), {"STARTING": 0, "RUNNING": 1, "STOPPED": 2, "ERROR": 3, "UNKNOWN": 4}))
    stub.register_provider = lambda name, cls: None
    monkeypatch.setitem(sys.modules, "nemo_gym", types.ModuleType("nemo_gym"))
    monkeypatch.setitem(sys.modules, "nemo_gym.sandbox", stub)
    monkeypatch.delitem(sys.modules, "cwsandbox_provider", raising=False)
    import cwsandbox_provider

    assert cwsandbox_provider.placement_kwargs("serverless", "r") == {
        "placement_mode": "serverless", "placement_spillover": "strict"}  # fmt: skip
    assert cwsandbox_provider.placement_kwargs("cks", "r")["runner_ids"] == ["r"]
    spill = cwsandbox_provider.placement_kwargs("cks_then_serverless", "r")
    assert spill["placement_spillover"] == "cks_then_serverless" and spill["runner_ids"] == ["r"]
    assert cwsandbox_provider.CWSandboxProvider("r").placement == "cks"  # math_sandbox_eval keeps its runner
    with pytest.raises(ValueError):
        cwsandbox_provider.CWSandboxProvider("r", placement="anywhere")


def test_tir_rows_offer_the_tool_and_name_their_agent():
    row = {"prompt": [{"content": "What is 6*7?"}], "reward_model": {"ground_truth": 42}}
    out = dapo_data.gym_row(row)
    params = out["responses_create_params"]
    assert out["agent_ref"]["name"] == "math_tir_agent" and out["expected_answer"] == "42"
    assert params["input"][0] == {"role": "system", "content": dapo_data.TIR_SYSTEM_PROMPT}
    assert params["input"][1]["content"] == "What is 6*7?" and params["tools"][0]["name"] == "execute_python"


def test_gym_serves_the_training_agent_on_the_training_port():
    text = yaml.safe_load(run_gym.overrides_yaml("https://gw/v1", "m", "t", "r"))
    assert text["math_tir_agent"]["responses_api_agents"]["simple_agent"]["port"] == run_gym.TRAIN_AGENT_PORT
    assert text["policy_model"]["responses_api_models"]["vllm_model"]["max_model_len"] == 3072


def _message(text):
    return {"type": "message", "role": "assistant", "id": "msg_1", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}  # fmt: skip


def _call(code, call_id):
    return {"type": "function_call", "name": "execute_python", "arguments": json.dumps({"code": code}),
            "call_id": call_id, "id": call_id, "status": "completed"}  # fmt: skip


CODE = "for i in range(10, 100):\n    if 4641 % i == 0 and 10 <= 4641 // i <= 99:\n        print(i, 4641 // i, i + 4641 // i)\n        break\n"


def test_an_example_shows_each_turn_its_cell_and_the_sandbox_output():
    prompt = [{"role": "system", "content": dapo_data.TIR_SYSTEM_PROMPT},
              {"role": "user", "content": "The number 4641 can be expressed as the product of two 2-digit whole numbers. What is their sum?"}]  # fmt: skip
    output = [
        _message("We need two 2-digit factors of 4641. Let's search them.\n```python\n" + CODE + "```"),
        _call(CODE, "call_a"),
        {"type": "function_call_output", "call_id": "call_a", "output": "51 91 142\n"},
        _message("The factors are 51 and 91, so the sum is \\boxed{142}."),
    ]
    example = tir.episode_example(prompt, output, "142", 1.0, "142", "sbx-123")
    assert example["k"] == "example" and example["prompt"].startswith("The number 4641")
    first, final = example["turns"]
    assert first == {"text": "We need two 2-digit factors of 4641. Let's search them.", "code": CODE,
                     "cell": "/work/cell_1.py", "output": "51 91 142\n"}  # fmt: skip
    assert final == {"text": "The factors are 51 and 91, so the sum is \\boxed{142}."}
    assert (example["cells"], example["answer"], example["expected"], example["reward"], example["sandbox"]) == (
        1, "142", "142", 1.0, "sbx-123")  # fmt: skip


def test_an_example_of_a_long_episode_fits_one_log_line():
    reasoning = "Let me think about this carefully. " * 400
    code = "x = 1\n" * 2000
    output = []
    for i in range(5):
        output += [_message(reasoning + "\n```python\n" + code + "```"), _call(code, f"c{i}"),
                   {"type": "function_call_output", "call_id": f"c{i}", "output": "y" * 512}]  # fmt: skip
    example = tir.episode_example("Q" * 5000, output, "1", 0.0, None, None)
    assert example["cells"] == 5 and [t["cell"] for t in example["turns"]][-1] == "/work/cell_5.py"
    assert "characters cut" in example["turns"][0]["text"] and "characters cut" in example["turns"][0]["code"]
    assert len("@@metric " + json.dumps(example)) < 16_000


def test_an_example_of_code_full_of_escapes_still_fits():
    code = 'print("a\\n")\n' * 2000
    output = [
        _message("x"),
        _call(code, "c"),
        {"type": "function_call_output", "call_id": "c", "output": '"\n' * 300},
    ] * 5
    example = tir.episode_example('"' * 3000, output, "1", 0.0, None, None)
    assert len(json.dumps(example)) < 15_000 and example["cells"] == 5
