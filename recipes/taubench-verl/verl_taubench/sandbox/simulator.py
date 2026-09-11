"""Remote τ-bench user simulator.

The simulator is an LLM endpoint, not environment code, so it runs on an
inference service rather than inside the environment sandbox. The default is
W&B Inference (serverless, OpenAI-compatible, authenticated with the W&B API
key); any OpenAI-compatible endpoint works. It is the hidden cost centre of
τ-bench: every episode burns simulator tokens, and unlike the policy those
tokens are pure overhead.

Calls go through the ``openai`` SDK on purpose: Weave autopatches it, so every
simulator call appears in the rollout trace as an LLM span with the native
conversation view and token usage.

The prompt mirrors ``tau_bench.envs.user.LLMUserSimulationEnv`` so reward
semantics are unchanged -- in particular the simulator must be able to emit
``###STOP###`` to end an episode, which is the only non-``terminate_tool`` way a
τ-bench episode terminates.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Protocol, runtime_checkable

STOP_TOKEN = "###STOP###"

WANDB_INFERENCE_BASE_URL = "https://api.inference.wandb.ai/v1"

# Mirrors tau_bench's user simulator system prompt.
_SYSTEM_PROMPT = """You are a user interacting with an agent.

{instruction}

Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction.
- If the instruction goal is satisified, generate '###STOP###' as a standalone message without anything else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction.
"""


@runtime_checkable
class UserSimulator(Protocol):
    """What the tool needs from a user simulator."""

    async def start(self, episode_id: str, instruction: str) -> str:
        """Begin an episode; return the user's opening utterance."""
        ...

    async def respond(self, episode_id: str, agent_message: str) -> str:
        """Return the user's reply to ``agent_message``."""
        ...

    async def finish(self, episode_id: str) -> None:
        """Drop per-episode conversation state."""
        ...


def resolve_simulator_api_key(base_url: str, explicit: Optional[str] = None) -> Optional[str]:
    """Pick the API key that matches the endpoint.

    W&B Inference authenticates with the W&B API key; the OpenAI API with
    OPENAI_API_KEY; anything else (a self-hosted vLLM) takes
    TAUBENCH_SIMULATOR_API_KEY or falls back to OPENAI_API_KEY.
    """
    if explicit:
        return explicit
    if "inference.wandb.ai" in base_url:
        return os.environ.get("WANDB_API_KEY")
    if "api.openai.com" in base_url:
        return os.environ.get("OPENAI_API_KEY")
    return os.environ.get("TAUBENCH_SIMULATOR_API_KEY") or os.environ.get("OPENAI_API_KEY")


class OpenAICompatibleSimulator:
    """Talks to any OpenAI-compatible chat-completions endpoint via the openai SDK.

    That covers W&B Inference (the default), the OpenAI API, and a
    SkyServe-fronted vLLM deployment, so deployments can swap endpoints without
    touching the tool.
    """

    def __init__(
        self,
        base_url: str,
        *,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 256,
        timeout: float = 60.0,
        project: Optional[str] = None,
        http_client=None,
        concurrency: int = 8,
        max_retries: int = 6,
    ):
        import asyncio

        import openai

        # The SDK expects the /v1-style base URL; accept endpoints configured
        # either way.
        base = base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = base + "/v1"
        self.base_url = base
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        client_kwargs = {
            "base_url": base,
            # Self-hosted vLLM ignores auth but the SDK requires a key string.
            "api_key": resolve_simulator_api_key(base, api_key) or "EMPTY",
            "timeout": timeout,
            # The SDK backs off and retries 429s (honoring Retry-After); hosted
            # endpoints enforce per-user concurrency caps, so retries are load-
            # bearing during large rollout batches.
            "max_retries": int(max_retries),
        }
        if project is None and "inference.wandb.ai" in base:
            # W&B Inference attributes usage to entity/project via this header.
            entity = os.environ.get("WANDB_ENTITY")
            proj = os.environ.get("PROJECT_NAME") or os.environ.get("WANDB_PROJECT")
            if entity and proj:
                project = f"{entity}/{proj}"
        if project:
            client_kwargs["project"] = project
        if http_client is not None:
            client_kwargs["http_client"] = http_client
        self._client = openai.AsyncOpenAI(**client_kwargs)
        # Hundreds of episodes run concurrently per worker; without this cap the
        # burst instantly trips hosted per-user concurrency limits.
        self._semaphore = asyncio.Semaphore(max(1, int(concurrency)))
        self._conversations: Dict[str, List[Dict[str, str]]] = {}

    async def _complete(self, messages: List[Dict[str, str]]) -> str:
        import openai

        try:
            async with self._semaphore:
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
        except openai.APIStatusError as exc:
            raise RuntimeError(f"user simulator {exc.status_code}: {exc.message}") from exc
        except openai.APIError as exc:
            raise RuntimeError(f"user simulator error: {exc}") from exc
        try:
            content = response.choices[0].message.content
        except (IndexError, AttributeError) as exc:
            raise RuntimeError(f"malformed simulator response: {response}") from exc
        if content is None:
            raise RuntimeError(f"malformed simulator response: {response}")
        return str(content)

    async def start(self, episode_id: str, instruction: str) -> str:
        messages = [{"role": "system", "content": _SYSTEM_PROMPT.format(instruction=instruction)}]
        # tau_bench seeds the conversation with a fixed opener so the user speaks first.
        messages.append({"role": "user", "content": "Hi! How can I help you today?"})
        reply = await self._complete(messages)
        messages.append({"role": "assistant", "content": reply})
        self._conversations[episode_id] = messages
        return reply

    async def respond(self, episode_id: str, agent_message: str) -> str:
        messages = self._conversations.get(episode_id)
        if messages is None:
            raise KeyError(f"no simulator conversation for episode {episode_id!r}")
        messages.append({"role": "user", "content": agent_message})
        reply = await self._complete(messages)
        messages.append({"role": "assistant", "content": reply})
        return reply

    async def finish(self, episode_id: str) -> None:
        self._conversations.pop(episode_id, None)
