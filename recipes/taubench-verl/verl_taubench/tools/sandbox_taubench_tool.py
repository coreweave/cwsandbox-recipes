"""veRL ``BaseTool`` that runs τ-bench episodes in leased CoreWeave sandboxes.

This is the single integration point with veRL. It implements the wrapper tool
name and the ``create``/``execute``/``calc_reward``/``release`` contract that
``TauBenchAgentLoop`` drives.

Routing
-------
``execute`` splits the action stream across the two planes:

* **tool calls** -> the leased sandbox, which mutates its database and returns
  the tool output as the observation;
* **respond** -> recorded in the sandbox (needed for reward's ``task.outputs``
  check) *and* forwarded to the remote user simulator, whose reply becomes the
  observation.

An episode is done when the simulator emits ``###STOP###`` or a τ-bench
``terminate_tool`` fires -- matching upstream τ-bench semantics exactly.

This mirrors veRL's in-tree ``sandbox_fusion`` tool, which likewise assumes a
persistent server rather than a container per call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, Optional, Tuple

from verl_taubench._verl_compat import (
    VERL_AVAILABLE,
    BaseTool,
    OpenAIFunctionToolSchema,
    ToolResponse,
)
from verl_taubench.envs.taubench_env import build_system_prompt_from_parts
from verl_taubench.sandbox.client import HttpxTransport
from verl_taubench.sandbox.env_config import (
    resolve_backend_image,
    resolve_max_connections,
    resolve_placement_mode,
    resolve_placement_spillover,
    resolve_pool_size,
    resolve_runner_ids,
)
from verl_taubench.sandbox.pool import CwSandboxBackend, SandboxPool
from verl_taubench.sandbox.simulator import (
    STOP_TOKEN,
    OpenAICompatibleSimulator,
    UserSimulator,
)

logger = logging.getLogger(__name__)

_WRAPPER_TOOL_NAME = "tau_bench_step"
_RESPOND = "respond"


class SandboxTauBenchTool(BaseTool):
    """One τ-bench episode per leased sandbox."""

    # This tool drives a real user simulator, so it produces turn 0 itself. The
    # AgentLoop must use the simulator's opening utterance rather than the
    # dataset's ``raw_prompt``, which carries ``task.instruction`` -- the brief
    # the simulator is supposed to reveal gradually, not hand to the policy.
    owns_initial_turn = True

    _tool_schema_dict = {
        "type": "function",
        "function": {
            "name": _WRAPPER_TOOL_NAME,
            "description": (
                "Execute one action in the τ-bench environment. Use action_name='respond' "
                "to send a message to the user and continue the conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action_name": {
                        "type": "string",
                        "description": "Name of the τ-bench action/tool to call, or 'respond'.",
                    },
                    "action_kwargs": {
                        "type": "object",
                        "description": "Arguments for the action. For 'respond', use {'content': '...'}.",
                    },
                },
                "required": ["action_name", "action_kwargs"],
            },
        },
    }

    def __init__(
        self,
        config: dict,
        tool_schema: Optional[Any] = None,
        *,
        pool: Optional[SandboxPool] = None,
        simulator: Optional[UserSimulator] = None,
    ):
        config = config or {}
        self.instance_registry: Dict[str, Any] = {}
        self.tool_schemas: list = config.get("tool_schemas", [])
        self._spec_cache: Dict[str, str] = {}
        self.max_user_turns = int(config.get("max_user_turns", 30))

        # veRL constructs tools from `class_name` + `config` only, so the pool and
        # simulator MUST be buildable from config. The keyword arguments exist so
        # tests can inject fakes; they take precedence when supplied.
        self._pool_config = config
        self._owns_pools = pool is None
        self.pool = pool if pool is not None else self._build_pool(config)
        self.simulator = (
            simulator if simulator is not None else self._build_simulator(config)
        )
        # Domain/split the pool is pinned to, for validating dataset rows against.
        backend_cfg = config.get("backend") or {}
        self.domain = backend_cfg.get("domain")
        self.task_split = backend_cfg.get("task_split")
        self._default_pool_key = (self.domain, self.task_split)
        self._pools: Dict[Tuple[Optional[str], Optional[str]], SandboxPool] = {}
        if self.pool is not None:
            self._pools[self._default_pool_key] = self.pool
        self._prewarmed_pool_ids: set[int] = set()
        self._pools_lock = asyncio.Lock()
        self._closed = False

        if VERL_AVAILABLE:
            super().__init__(config, tool_schema)
        else:
            self.config = config
            self.tool_schema = tool_schema or self.get_openai_tool_schema()
            self.name = self.tool_schema.function.name  # type: ignore[union-attr]

    # -- construction from config -----------------------------------------

    @staticmethod
    def _build_pool(
        config: dict,
        *,
        domain: Optional[str] = None,
        task_split: Optional[str] = None,
    ) -> Optional[SandboxPool]:
        """Build a SandboxPool from the `backend:`/`pool:` config blocks."""
        backend_cfg = config.get("backend")
        if not backend_cfg:
            return None

        pool_cfg = dict(config.get("pool") or {})
        pool_size = resolve_pool_size(pool_cfg.pop("size", None))
        pool_cfg["size"] = pool_size

        config_max = pool_cfg.pop("max_connections", None)
        transport = HttpxTransport(
            timeout=float(pool_cfg.pop("transport_timeout_s", 30.0)),
            max_connections=resolve_max_connections(pool_size, config_max),
        )

        backend_cfg = dict(backend_cfg)
        if domain is not None:
            backend_cfg["domain"] = domain
        if task_split is not None:
            backend_cfg["task_split"] = task_split

        container_image, install_tau_bench = resolve_backend_image(
            backend_cfg.pop("container_image", "python:3.11")
        )
        backend_cfg["container_image"] = container_image
        backend_cfg["install_tau_bench"] = install_tau_bench
        placement_mode = resolve_placement_mode(backend_cfg.pop("placement_mode", None))
        backend_cfg["placement_mode"] = placement_mode
        backend_cfg["placement_spillover"] = resolve_placement_spillover(
            backend_cfg.pop("placement_spillover", None),
            placement_mode=placement_mode,
        )
        backend_cfg["runner_ids"] = resolve_runner_ids(backend_cfg.pop("runner_ids", None))

        backend = CwSandboxBackend(**backend_cfg)
        # veRL builds tools in AgentLoopWorker.__init__ on its async-actor thread.
        # Preload once through the cwsandbox compatibility guard before many
        # concurrent pool-provision tasks begin.
        backend.preload_sdk()
        return SandboxPool(backend, transport, **pool_cfg)

    @staticmethod
    def _build_simulator(config: dict) -> Optional[UserSimulator]:
        """Build the remote user simulator from the `simulator:` config block."""
        sim_cfg = dict(config.get("simulator") or {})
        if not sim_cfg:
            return None

        base_url = sim_cfg.pop("base_url", None)
        if not base_url:
            raise ValueError("simulator config requires 'base_url'")
        # Key selection matches the endpoint (W&B Inference -> WANDB_API_KEY,
        # OpenAI -> OPENAI_API_KEY, self-hosted -> TAUBENCH_SIMULATOR_API_KEY).
        return OpenAICompatibleSimulator(base_url, **sim_cfg)

    async def _pool_for(
        self,
        row_domain: Optional[str],
        row_split: Optional[str],
    ) -> SandboxPool:
        """Return the row's keyed pool, constructing owned non-default pools once."""
        if self._closed:
            raise RuntimeError("SandboxTauBenchTool is closed")
        if self.pool is None:
            raise RuntimeError(
                "SandboxTauBenchTool has no SandboxPool. Provide a `backend:` block in "
                "the tool config (see config/tool_config/taubench_sandbox_tool_config.yaml) "
                "or pass pool= explicitly."
            )

        if not self._owns_pools:
            if self.domain and row_domain and row_domain != self.domain:
                raise ValueError(
                    f"dataset row domain={row_domain!r} does not match the sandbox pool "
                    f"domain={self.domain!r}; run one pool per domain."
                )
            if self.task_split and row_split and row_split != self.task_split:
                raise ValueError(
                    f"dataset row task_split={row_split!r} does not match the sandbox pool "
                    f"task_split={self.task_split!r}."
                )
            return self.pool

        domain = row_domain or self.domain
        task_split = row_split or self.task_split
        if not domain or not task_split:
            raise ValueError(
                "domain and task_split are required to select a sandbox pool"
            )
        key = (str(domain), str(task_split))
        existing = self._pools.get(key)
        if existing is not None:
            return existing

        async with self._pools_lock:
            if self._closed:
                raise RuntimeError("SandboxTauBenchTool is closed")
            existing = self._pools.get(key)
            if existing is not None:
                return existing
            built = self._build_pool(
                self._pool_config,
                domain=key[0],
                task_split=key[1],
            )
            if built is None:
                raise RuntimeError("could not build sandbox pool from tool config")
            if self._closed:
                await built.aclose()
                raise RuntimeError("SandboxTauBenchTool is closed")
            self._pools[key] = built
            return built

    async def _ensure_prewarmed(self, pool: SandboxPool) -> None:
        """Fill each keyed warm pool once, on first use.

        Doing this lazily keeps the constructor synchronous (veRL builds tools
        outside an event loop) while still moving cold starts off the rollout
        critical path for every episode after the first batch.
        """
        pool_id = id(pool)
        if pool_id in self._prewarmed_pool_ids:
            return
        async with self._pools_lock:
            if self._closed:
                raise RuntimeError("SandboxTauBenchTool is closed")
            if pool_id in self._prewarmed_pool_ids:
                return
            await pool.prewarm()
            self._prewarmed_pool_ids.add(pool_id)

    def get_openai_tool_schema(self) -> Any:
        return OpenAIFunctionToolSchema.model_validate(self._tool_schema_dict)

    # -- lifecycle ---------------------------------------------------------

    async def create(
        self, instance_id: Optional[str] = None, **kwargs: Any
    ) -> Tuple[str, Any]:
        """Lease a sandbox, load the task, and seed the user simulator."""
        if self.simulator is None:
            raise RuntimeError(
                "SandboxTauBenchTool has no UserSimulator. Provide a `simulator:` block "
                "in the tool config or pass simulator= explicitly."
            )

        task_index = kwargs.get("task_index")
        if task_index is None:
            raise ValueError("task_index is required for deterministic training")

        row_domain = kwargs.get("domain")
        row_split = kwargs.get("task_split")
        pool = await self._pool_for(row_domain, row_split)
        await self._ensure_prewarmed(pool)
        instance_id = instance_id or str(uuid.uuid4())

        lease = await pool.acquire(int(task_index), episode_id=instance_id)
        try:
            instruction = lease.metadata.get("instruction", "")
            system_prompt = await self._system_prompt(lease)
            opening = await self.simulator.start(instance_id, instruction)
        except BaseException:
            # Never leak the lease if spec fetch or simulator seeding fails.
            await pool.release(lease, healthy=False)
            raise

        self.instance_registry[instance_id] = {
            "lease": lease,
            "pool": pool,
            "task_index": int(task_index),
            "observation": opening,
            "done": False,
            "reward": 0.0,
            "user_turns": 0,
            "instruction": instruction,
            # The AgentLoop reads this when the dataset supplies no raw_prompt.
            "system_prompt": system_prompt,
        }
        return instance_id, ToolResponse(text=opening)

    async def _system_prompt(self, lease: Any) -> str:
        """Build the policy system prompt from the sandbox's ``/spec``.

        Cached per domain: the wiki/rules/schema block is ~17 KB and constant, so
        refetching it per episode would be pure overhead.
        """
        domain = lease.metadata.get("domain") or "unknown"
        cached = self._spec_cache.get(domain)
        if cached is None:
            spec = await lease.client.spec()
            schemas = [
                {
                    "type": "function",
                    "function": {
                        "name": info["function"]["name"],
                        "description": info["function"]["description"],
                        "parameters": info["function"]["parameters"],
                    },
                }
                for info in spec.get("tools_info", [])
                if info.get("type") == "function"
            ]
            cached = build_system_prompt_from_parts(
                spec.get("wiki", ""), list(spec.get("rules", [])), schemas
            )
            self._spec_cache[domain] = cached
        return cached

    async def execute(
        self, instance_id: str, parameters: dict, **kwargs: Any
    ) -> Tuple[Any, float, dict]:
        state = self.instance_registry.get(instance_id)
        if state is None:
            return ToolResponse(text=f"Unknown instance_id {instance_id}"), 0.0, {"error": "unknown_instance"}
        if state["done"]:
            return ToolResponse(text="Episode already finished."), 0.0, {"error": "episode_done"}

        action_name = parameters.get("action_name", _RESPOND)
        action_kwargs = parameters.get("action_kwargs", {})
        if isinstance(action_kwargs, str):
            try:
                action_kwargs = json.loads(action_kwargs)
            except json.JSONDecodeError:
                action_kwargs = {"content": action_kwargs}
        if not isinstance(action_kwargs, dict):
            action_kwargs = {}

        lease = state["lease"]
        try:
            if action_name == _RESPOND:
                return await self._handle_respond(instance_id, state, lease, action_kwargs)
            return await self._handle_tool_call(state, lease, action_name, action_kwargs)
        except Exception as exc:
            logger.warning("episode %s failed on %s: %s", instance_id, action_name, exc)
            state["failed"] = True
            # End the episode. A dead sandbox will not recover, so continuing
            # would re-hit it once per remaining turn and append each failure to
            # the trajectory as observation tokens.
            state["done"] = True
            return (
                ToolResponse(text=f"Environment error: {exc}"),
                0.0,
                {"error": str(exc), "done": True},
            )

    async def _handle_respond(
        self, instance_id: str, state: dict, lease: Any, action_kwargs: dict
    ) -> Tuple[Any, float, dict]:
        content = action_kwargs.get("content")
        if content is None:
            return (
                ToolResponse(
                    text=(
                        "A respond action must include {'action_kwargs': {'content': '...'}}. "
                        "Please provide a response."
                    )
                ),
                0.0,
                {"error": "missing_respond_content"},
            )

        # Record in the sandbox first: calculate_reward scans env.actions for
        # respond actions to verify task.outputs. Skipping this silently zeroes
        # reward on every task that has required outputs.
        await lease.client.step(_RESPOND, {"content": content})

        user_reply = await self.simulator.respond(instance_id, str(content))
        state["user_turns"] += 1

        done = STOP_TOKEN in user_reply
        if state["user_turns"] >= self.max_user_turns:
            done = True
        state["observation"] = user_reply
        state["done"] = done

        reward = 0.0
        if done:
            reward = await self._finalize_reward(state, lease)

        return (
            ToolResponse(text=user_reply),
            reward,
            {"done": done, "action_name": _RESPOND, "source": "user"},
        )

    async def _handle_tool_call(
        self, state: dict, lease: Any, action_name: str, action_kwargs: dict
    ) -> Tuple[Any, float, dict]:
        result = await lease.client.step(action_name, action_kwargs)
        observation = result.get("observation", "")
        done = bool(result.get("done", False))
        reward = float(result.get("reward", 0.0))

        state["observation"] = observation
        state["done"] = done
        if done:
            # A terminate_tool fired; the sandbox already computed the reward.
            state["reward"] = reward

        return (
            ToolResponse(text=observation),
            reward,
            {"done": done, "action_name": action_name, "source": result.get("source", action_name)},
        )

    async def _finalize_reward(self, state: dict, lease: Any) -> float:
        """Ask the sandbox to diff its DB against the goal state."""
        payload = await lease.client.reward()
        reward = float(payload.get("reward", 0.0))
        state["reward"] = reward
        state["reward_info"] = payload.get("info")
        return reward

    async def calc_reward(self, instance_id: str, **kwargs: Any) -> float:
        state = self.instance_registry.get(instance_id)
        if state is None:
            return 0.0
        if state["done"] or state.get("reward"):
            return float(state["reward"])
        # Episode hit max_turns without terminating. τ-bench still defines a
        # reward for the current DB state, so ask for it rather than assuming 0.
        try:
            return await self._finalize_reward(state, state["lease"])
        except Exception as exc:
            logger.warning("reward fetch failed for %s: %s", instance_id, exc)
            return float(state.get("reward", 0.0))

    async def release(self, instance_id: str, **kwargs: Any) -> None:
        state = self.instance_registry.pop(instance_id, None)
        if state is None:
            return
        with_error = bool(state.get("failed"))
        try:
            await self.simulator.finish(instance_id)
        finally:
            # The episode mutated the DB and calculate_reward is destructive, so
            # the sandbox is reset on the next acquire(). Returning it healthy is
            # correct as long as the episode did not error.
            await state["pool"].release(state["lease"], healthy=not with_error)

    async def aclose(self) -> None:
        """Close each unique internally constructed pool exactly once."""
        async with self._pools_lock:
            if self._closed:
                return
            self._closed = True
            if not self._owns_pools:
                return
            pools: list[SandboxPool] = []
            seen: set[int] = set()
            for pool in self._pools.values():
                if id(pool) not in seen:
                    seen.add(id(pool))
                    pools.append(pool)

        if pools:
            results = await asyncio.gather(
                *(pool.aclose() for pool in pools),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result
