"""Role semantics for policy context and the canonical Weave dialogue."""

import contextlib
import copy
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_rollout_keeps_policy_simulator_and_tool_roles_distinct(monkeypatch) -> None:
    """A simulator reply must never be mislabeled as assistant or tool output."""
    from verl_taubench.agent import taubench_loop

    captured_spans = []

    @contextlib.contextmanager
    def capture_span(name, inputs, display_name=None, kind=None):
        output = {}
        captured_spans.append(
            {
                "name": name,
                "inputs": copy.deepcopy(inputs),
                "display_name": display_name,
                "kind": kind,
                "output": output,
            }
        )
        yield output

    monkeypatch.setattr(taubench_loop, "_weave_span", capture_span)

    class Tokenizer:
        chat_template = True

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
            return "\n".join(
                f"<{message['role']}>{message.get('content', '')}</{message['role']}>" for message in messages
            )

        def encode(self, text, add_special_tokens=False):
            return [ord(character) for character in text]

        def decode(self, token_ids, skip_special_tokens=True):
            return "".join(chr(int(token_id)) for token_id in token_ids)

    class Server:
        def __init__(self):
            self.responses = iter(
                [
                    '<tool_call>{"name":"tau_bench_step","arguments":'
                    '{"action_name":"lookup_order","action_kwargs":{"order_id":"O-1"}}}'
                    "</tool_call>",
                    "Could you confirm the replacement color?",
                    "Done — I changed it to white.",
                ]
            )

        async def generate(self, request_id, prompt_ids, sampling_params):
            token_ids = [ord(character) for character in next(self.responses)]
            return SimpleNamespace(
                token_ids=token_ids,
                log_probs=[-0.1] * len(token_ids),
                extra_fields={},
            )

    class StepTool:
        name = "tau_bench_step"
        owns_initial_turn = True
        tool_schemas = []

        def __init__(self):
            self.instance_registry = {}
            self.user_replies = iter(["White, please.", "###STOP###"])

        async def create(self, **kwargs):
            instance_id = "episode-1"
            self.instance_registry[instance_id] = {
                "system_prompt": "SYSTEM POLICY",
                "observation": "Please change my order.",
                "done": False,
            }
            return instance_id, SimpleNamespace(text="Please change my order.")

        async def execute(self, instance_id, parameters):
            state = self.instance_registry[instance_id]
            if parameters["action_name"] == "respond":
                observation = next(self.user_replies)
                state["done"] = observation == "###STOP###"
            else:
                observation = '{"order_id":"O-1","color":"black"}'
            return SimpleNamespace(text=observation), 0.0, {"done": state["done"]}

        async def calc_reward(self, instance_id):
            return 1.0

        async def release(self, instance_id):
            return None

    # Bypass the heavyweight real veRL base constructor while exercising the
    # shipped run/_run_episode implementation and message mutation paths.
    loop = object.__new__(taubench_loop.TauBenchAgentLoop)
    loop.server_manager = Server()
    loop.tokenizer = Tokenizer()
    loop.step_tool = StepTool()
    loop.max_turns = 3
    loop.response_length = 4096
    loop.continuous_token_builder = object()
    continuous_token_appends = []

    async def build_initial_tokens(messages, tools=None):
        return [1]

    async def merge_assistant_tokens(
        runtime_token_ids,
        assistant_token_ids,
        response_mask,
        response_logprobs=None,
        assistant_logprobs=None,
    ):
        if assistant_logprobs is not None and response_logprobs is None:
            raise ValueError("response_logprobs is required when assistant_logprobs is provided")
        return (
            SimpleNamespace(token_ids=runtime_token_ids + assistant_token_ids),
            response_mask + [1] * len(assistant_token_ids),
            (
                response_logprobs + assistant_logprobs
                if response_logprobs is not None and assistant_logprobs is not None
                else response_logprobs
            ),
        )

    async def merge_non_assistant_tokens(
        previous_messages,
        updated_messages,
        runtime_token_ids,
        response_mask,
        response_logprobs=None,
        tools=None,
    ):
        appended = updated_messages[len(previous_messages) :]
        continuous_token_appends.append([message["role"] for message in appended])
        return (
            SimpleNamespace(token_ids=runtime_token_ids + [0] * len(appended)),
            response_mask + [0] * len(appended),
            (
                response_logprobs + [0.0] * len(appended)
                if response_logprobs is not None
                else None
            ),
        )

    loop.ct_build_initial_tokens = build_initial_tokens
    loop.ct_merge_assistant_token = merge_assistant_tokens
    loop.ct_merge_non_assistant_msg = merge_non_assistant_tokens
    monkeypatch.setattr(taubench_loop, "VERL_AVAILABLE", True)

    output = await loop.run(
        {},
        extra_info={"domain": "retail", "task_split": "test", "task_index": 0},
    )

    assert output.response_logprobs is not None
    assert len(output.response_ids) == len(output.response_mask) == len(output.response_logprobs)
    assert continuous_token_appends == [["tool"], ["user"], ["user"]]

    policy_spans = [span for span in captured_spans if span["name"] == "policy.chat"]
    assert [[message["role"] for message in span["inputs"]["messages"]] for span in policy_spans] == [
        ["system", "user"],
        ["system", "user", "assistant", "tool"],
        ["system", "user", "assistant", "tool", "assistant", "user"],
    ]

    dialogue = next(span for span in captured_spans if span["name"] == "taubench.dialogue")
    assert dialogue["inputs"]["model"] == "policy-model"
    assert [message["role"] for message in dialogue["inputs"]["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert [message.get("content") for message in dialogue["inputs"]["messages"]] == [
        "SYSTEM POLICY",
        "Please change my order.",
        '<tool_call>{"name":"tau_bench_step","arguments":'
        '{"action_name":"lookup_order","action_kwargs":{"order_id":"O-1"}}}'
        "</tool_call>",
        '{"order_id":"O-1","color":"black"}',
        "Could you confirm the replacement color?",
        "White, please.",
        "Done — I changed it to white.",
        "###STOP###",
    ]
    assert dialogue["display_name"] == "rollout (policy = assistant, simulator = user)"
