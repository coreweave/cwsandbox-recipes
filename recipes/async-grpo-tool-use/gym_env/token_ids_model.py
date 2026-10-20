"""Gym model server for a Dedicated Inference deployment, with exact token IDs for training.

Training needs the prompt and generated token IDs and the log-probs of every generated token.
The deployment's gateway serves ``/v1/completions`` and ``/v1/chat/completions`` but not vLLM's
``/tokenize``, and it does not return prompt token IDs. So this server renders each chat request
with the model's tokenizer and sends the token IDs to ``/v1/completions``. The response gives
the generated token IDs (as ``token_id:<n>`` strings) and their log-probs.

Requests that offer the ``execute_python`` tool get tool-integrated reasoning (TIR) in
Qwen2.5-Math's format. The base model has no tool-call syntax, so this server plays the part of
vLLM's tool parser and chat template: generation stops at ```` ```output ````, a reply that ends
with a ```` ```python ```` block becomes an ``execute_python`` call, and the next request builds
its prompt from the token IDs of the previous turn plus the tool result in an ```` ```output ````
block. Building the prompt from token IDs, not by re-rendering the chat, keeps every turn's
prompt a prefix-extension of the last one, which NeMo-RL requires for multi-turn training.

``run_gym`` copies this file into Gym's ``vllm_model`` server, so it runs in that server's venv.
"""

import json
import math
import os
import re
import threading
import time
import uuid
from typing import Any, Optional

SAMPLING_KEYS = ("model", "temperature", "top_p", "seed", "presence_penalty", "frequency_penalty")
TOOL_NAME = "execute_python"
TOOL_STOP = "```output"
_HITS = [1] * 3  # and the first few that do
CODE_BLOCK = re.compile(r"```python[ \t]*\n(.*?)```", re.DOTALL)


class SamplerStats:
    """Request and token counts for the live dashboard (``DASHBOARD_METRICS=1``). Prints one
    ``@@metric`` line per second while there is traffic; the launcher's log stream carries it."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.inflight = 0
        self.done = 0
        self.prompt_tokens = 0
        self.gen_tokens = 0

    def start(self) -> None:
        threading.Thread(target=self._report, daemon=True).start()

    def begin(self) -> None:
        with self.lock:
            self.inflight += 1

    def end(self, usage: Optional[dict]) -> None:
        with self.lock:
            self.inflight -= 1
            self.done += 1
            self.prompt_tokens += (usage or {}).get("prompt_tokens") or 0
            self.gen_tokens += (usage or {}).get("completion_tokens") or 0

    def _report(self) -> None:
        last = None
        while True:
            time.sleep(1)
            with self.lock:
                row = {"k": "sampler", "inflight": self.inflight, "done": self.done,
                       "prompt_tokens": self.prompt_tokens, "gen_tokens": self.gen_tokens}  # fmt: skip
            if row != last:
                print("@@metric " + json.dumps({"ts": time.time(), **row}), flush=True)
                last = row


STATS = SamplerStats() if os.environ.get("DASHBOARD_METRICS") == "1" else None


def text_content(content: Any) -> Any:
    if isinstance(content, list):
        return "".join(part["text"] for part in content if part.get("type") in ("text", "input_text"))
    return content


def observation(output: str, tail: str) -> str:
    """The tool result as Qwen2.5-Math TIR writes it. ``tail`` is the decoded end of the turn:
    the ```` ```output ```` stop string is there unless the engine dropped its tokens."""
    # the deployment drops the stop string together with the newline before it
    opener = "" if TOOL_STOP in tail else TOOL_STOP if tail.endswith("\n") else "\n" + TOOL_STOP
    body = output.strip("\n")
    return f"{opener}\n{body}\n```\n"


def prompt_token_ids(messages: list[dict], tokenizer: Any, template_kwargs: Optional[dict] = None) -> list[int]:
    """The first turn renders the chat template. Later turns extend the last generated turn's
    token IDs with the tool results that follow it."""
    last = max((i for i, m in enumerate(messages) if m.get("prompt_token_ids") is not None), default=None)
    if last is None:
        chat = [{"role": m["role"], "content": text_content(m.get("content"))} for m in messages]
        rendered = tokenizer.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=True, return_dict=False, **(template_kwargs or {})
        )
        return list(rendered)
    turn = messages[last]
    prompt = [*turn["prompt_token_ids"], *turn["generation_token_ids"]]
    for message in messages[last + 1 :]:
        if message["role"] != "tool":
            raise ValueError(f"only tool results can follow a generated turn, got a {message['role']} message")
        text = observation(text_content(message.get("content")) or "", tokenizer.decode(prompt[-16:]))
        prompt += tokenizer.encode(text, add_special_tokens=False)
    return prompt


def completion_request(body: dict, tokenizer: Any, max_model_len: Optional[int] = None) -> dict:
    """A chat request -> a token-ID ``/v1/completions`` request."""
    if body.get("n", 1) != 1:
        raise ValueError("this model server supports one completion per request")
    tools = [tool["function"]["name"] for tool in body.get("tools") or []]
    if tools not in ([], [TOOL_NAME]):
        raise ValueError(f"this model server supports only the {TOOL_NAME} tool, got {tools}")
    prompt = prompt_token_ids(body["messages"], tokenizer, body.get("chat_template_kwargs"))
    request = {key: body[key] for key in SAMPLING_KEYS if body.get(key) is not None}
    stop = body.get("stop") or []
    stop = [stop] if isinstance(stop, str) else list(stop)
    if tools:
        stop.append(TOOL_STOP)
    if stop:
        request["stop"] = stop
    max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
    if max_model_len is not None:
        # Later turns have long prompts. With no room left, the engine's context-length error
        # makes Gym end the episode as truncated.
        max_tokens = max(1, min(max_tokens or max_model_len, max_model_len - len(prompt)))
    request.update(prompt=prompt, max_tokens=max_tokens, logprobs=0, return_tokens_as_token_ids=True)
    return request


def trailing_code(text: str) -> Optional[str]:
    """The code of the ```` ```python ```` block that ends ``text``, or None."""
    text = text.rstrip()
    if text.endswith(TOOL_STOP):
        text = text[: -len(TOOL_STOP)]
    blocks = list(CODE_BLOCK.finditer(text))
    if not blocks or text[blocks[-1].end() :].strip():
        return None
    code = blocks[-1].group(1)
    return code if code.strip() else None


def with_tool_call(chat: dict, special_ids: set[int]) -> dict:
    """Turn a reply that stopped right after a python block into an ``execute_python`` call.

    The deployment drops the ```` ```output ```` stop tokens and ends the reply with
    end-of-text instead, so that token cannot tell a stop from a real end. A reply whose text
    ends in a closed python block is a call either way (the TIR format always follows code with
    its output); the trailing end-of-text is dropped so the next turn continues the sequence."""
    choice = chat["choices"][0]
    message = choice["message"]
    generated = message["generation_token_ids"]
    code = trailing_code(message["content"] or "")
    if choice["finish_reason"] == "length" or not generated or code is None:
        return chat
    if generated[-1] in special_ids:
        message["generation_token_ids"] = generated[:-1]
        message["generation_log_probs"] = message["generation_log_probs"][:-1]
    message["tool_calls"] = [
        {
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": TOOL_NAME, "arguments": json.dumps({"code": code})},
        }
    ]
    choice["finish_reason"] = "tool_calls"
    if _HITS:
        _HITS.pop()
        print(f"[token_ids_model] code reply -> execute_python ({len(code)} chars)", flush=True)
    return chat


def chat_response(response: dict, prompt: list[int]) -> dict:
    """A ``/v1/completions`` response -> a chat response with the token IDs on the message."""
    (choice,) = response["choices"]
    logprobs = choice.get("logprobs") or {}
    pieces, probs = logprobs.get("tokens") or [], logprobs.get("token_logprobs") or []
    if len(pieces) != len(probs) or not all(p.startswith("token_id:") for p in pieces):
        raise ValueError("the deployment did not return generated token IDs with log-probs")
    if any(p is None or not math.isfinite(p) for p in probs):
        raise ValueError("the deployment returned a non-finite log-prob")
    return {
        "id": response["id"],
        "created": response["created"],
        "model": response["model"],
        "usage": response.get("usage"),
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": choice["finish_reason"],
                "message": {
                    "role": "assistant",
                    "content": choice["text"],
                    "prompt_token_ids": prompt,
                    "generation_token_ids": [int(p.removeprefix("token_id:")) for p in pieces],
                    "generation_log_probs": probs,
                },
            }
        ],
    }


def main() -> None:
    import nemo_gym  # noqa: F401 - puts the Gym tree on sys.path
    from nemo_gym.openai_utils import NeMoGymAsyncOpenAI, get_response_json
    from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig
    from transformers import AutoTokenizer

    class Config(VLLMModelConfig):
        tokenizer_revision: Optional[str] = None
        max_model_len: Optional[int] = None  # caps max_tokens so prompt + generation fit

    class Client(NeMoGymAsyncOpenAI):
        tokenizer: Any
        max_model_len: Optional[int] = None

        async def _complete(self, kwargs: dict) -> dict:
            body = completion_request(kwargs, self.tokenizer, self.max_model_len)
            request = {"url": f"{self.base_url}/completions", "json": body}
            response = await self._request(method="POST", **request)
            await self._raise_for_status(response, request)
            result = chat_response(await get_response_json(response), body["prompt"])
            if kwargs.get("tools"):
                result = with_tool_call(result, set(self.tokenizer.all_special_ids))
            return result

        async def create_chat_completion(self, **kwargs):
            if STATS is None:
                return await self._complete(kwargs)
            STATS.begin()
            usage = None
            try:
                result = await self._complete(kwargs)
                usage = result["usage"]
                return result
            finally:
                STATS.end(usage)

    class Model(VLLMModel):
        config: Config

        def _post_init(self) -> None:
            super()._post_init()
            tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer, revision=self.config.tokenizer_revision)
            self._clients = [
                Client(base_url=url, api_key=self.config.api_key, default_headers=self.config.default_headers,
                       tokenizer=tokenizer, max_model_len=self.config.max_model_len)
                for url in self.config.base_url
            ]  # fmt: skip

    if STATS is not None:
        STATS.start()
    Model.run_webserver()


if __name__ == "__main__":
    main()
