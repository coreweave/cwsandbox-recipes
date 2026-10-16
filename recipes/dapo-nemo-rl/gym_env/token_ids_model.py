"""Gym model server for a Dedicated Inference deployment, with exact token IDs for training.

Training needs the prompt and generated token IDs and the log-probs of every generated token.
The deployment's gateway serves ``/v1/completions`` and ``/v1/chat/completions`` but not vLLM's
``/tokenize``, and it does not return prompt token IDs. So this server renders each chat request
with the model's tokenizer and sends the token IDs to ``/v1/completions``. The response gives
the generated token IDs (as ``token_id:<n>`` strings) and their log-probs.

``run_gym`` copies this file into Gym's ``vllm_model`` server, so it runs in that server's venv.
"""

import math
from typing import Any, Optional

SAMPLING_KEYS = ("model", "temperature", "top_p", "seed", "stop", "presence_penalty", "frequency_penalty")


def completion_request(body: dict, tokenizer: Any) -> dict:
    """A text-only chat request -> a token-ID ``/v1/completions`` request."""
    if body.get("tools") or body.get("n", 1) != 1:
        raise ValueError("this model server supports one text-only completion per request")
    messages = []
    for item in body["messages"]:
        content = item.get("content")
        if isinstance(content, list):
            content = "".join(part["text"] for part in content if part.get("type") in ("text", "input_text"))
        messages.append({"role": item["role"], "content": content})
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
        **(body.get("chat_template_kwargs") or {}),
    )
    request = {key: body[key] for key in SAMPLING_KEYS if body.get(key) is not None}
    request.update(
        prompt=list(prompt),
        max_tokens=body.get("max_completion_tokens") or body.get("max_tokens"),
        logprobs=0,
        return_tokens_as_token_ids=True,
    )
    return request


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

    class Client(NeMoGymAsyncOpenAI):
        tokenizer: Any

        async def create_chat_completion(self, **kwargs):
            body = completion_request(kwargs, self.tokenizer)
            request = {"url": f"{self.base_url}/completions", "json": body}
            response = await self._request(method="POST", **request)
            await self._raise_for_status(response, request)
            return chat_response(await get_response_json(response), body["prompt"])

    class Model(VLLMModel):
        config: Config

        def _post_init(self) -> None:
            super()._post_init()
            tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer, revision=self.config.tokenizer_revision)
            self._clients = [
                Client(base_url=url, api_key=self.config.api_key, default_headers=self.config.default_headers,
                       tokenizer=tokenizer)
                for url in self.config.base_url
            ]  # fmt: skip

    Model.run_webserver()


if __name__ == "__main__":
    main()
