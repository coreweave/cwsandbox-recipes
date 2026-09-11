"""τ-bench environment factory and helpers.

This module wraps the original τ-bench `Env` from `sierra-research/tau-bench`
so that the rest of the project (veRL tools, AgentLoop, dataset preprocessing)
only ever interacts with a small, stable interface. It deliberately does NOT
import any veRL code.

Design notes:
- τ-bench is the reference environment. A different environment can replace
  the underlying backend while keeping the same surface area.
- The user simulator is part of τ-bench, so the only external calls are to
  the τ-bench env's `reset` and `step` methods.
- Agent actions are encoded as `Action(name=..., kwargs=...)`.
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional, Type

from tau_bench.envs.airline.env import MockAirlineDomainEnv
from tau_bench.envs.retail.env import MockRetailDomainEnv
from tau_bench.types import Action, EnvResponse, Task

# Map from domain name to the τ-bench Env class.
_DOMAIN_ENV_CLS: Dict[str, Type] = {
    "retail": MockRetailDomainEnv,
    "airline": MockAirlineDomainEnv,
}


def list_domains() -> List[str]:
    """Return the supported τ-bench domain names."""
    return list(_DOMAIN_ENV_CLS.keys())


def make_env(
    domain: str,
    task_index: Optional[int] = None,
    user_strategy: str = "llm",
    user_model: str = "gpt-4o-mini",
    user_provider: Optional[str] = "openai",
    **env_kwargs: Any,
) -> Any:
    """Create a τ-bench environment for the given domain.

    Args:
        domain: One of {"retail", "airline"}.
        task_index: Specific task to fix; random if None.
        user_strategy: τ-bench user simulator strategy ("llm" or "human").
        user_model: Model name for the LLM-based user simulator.
        user_provider: Provider string for the user simulator LLM.
        **env_kwargs: Domain-specific overrides forwarded to the Env class
            (e.g., task_split).

    Returns:
        An instantiated tau_bench.envs.Env subclass.
    """
    if domain not in _DOMAIN_ENV_CLS:
        raise ValueError(f"Unknown τ-bench domain: {domain}. Choose from {list_domains()}.")

    env_cls = _DOMAIN_ENV_CLS[domain]

    # ``task_index`` IS a documented constructor argument on both
    # ``MockRetailDomainEnv`` and ``MockAirlineDomainEnv``. When it is omitted,
    # τ-bench's ``Env.__init__`` picks a task with an INCLUSIVE
    # ``random.randint(0, len(tasks))`` and then indexes ``tasks[i]``, so it raises
    # ``IndexError`` with probability ``1/(n+1)``. Always construct with an explicit
    # in-range index so that never fires, then pin the task we actually want.
    constructor_kwargs = {
        "user_strategy": user_strategy,
        "user_model": user_model,
        "user_provider": user_provider,
        **env_kwargs,
    }
    constructor_kwargs["task_index"] = 0
    env = env_cls(**constructor_kwargs)

    if task_index is None:
        # Preserve the documented "random if None" behaviour, minus the off-by-one.
        task_index = random.randrange(len(env.tasks))
    elif not (0 <= task_index < len(env.tasks)):
        raise IndexError(
            f"task_index={task_index} out of range for {domain} "
            f"(tasks={len(env.tasks)})"
        )

    env.task_index = task_index
    env.task = env.tasks[task_index]

    return env


def reset_env(env: Any, task_index: Optional[int] = None) -> Dict[str, Any]:
    """Reset a τ-bench env and return a standard observation dict.

    If ``task_index`` is omitted, the env's current pinned task is preserved.
    This prevents ``reset_env(env)`` from silently re-randomising a task that
    was fixed by ``make_env(task_index=...)``.
    """
    if task_index is None and hasattr(env, "task_index"):
        task_index = env.task_index
    reset_response = env.reset(task_index=task_index)
    return {
        "observation": reset_response.observation,
        "task": reset_response.info.task,
        "source": reset_response.info.source or "user",
    }


def step_env(env: Any, action_name: str, action_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one action in a τ-bench env.

    Returns a dict with observation text, reward, done flag, and info.
    """
    action = Action(name=action_name, kwargs=action_kwargs)
    response: EnvResponse = env.step(action)
    return {
        "observation": response.observation,
        "reward": float(response.reward),
        "done": bool(response.done),
        "info": response.info,
        "source": response.info.source or action_name,
    }


def env_to_openai_tools(env: Any) -> List[Dict[str, Any]]:
    """Extract OpenAI-function tool schemas from a τ-bench environment.

    This is the schema that will be embedded in the LLM prompt and used by
    the veRL BaseTool subclasses during rollout.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool_info["function"]["name"],
                "description": tool_info["function"]["description"],
                "parameters": tool_info["function"]["parameters"],
            },
        }
        for tool_info in env.tools_info
        if tool_info.get("type") == "function"
    ]


def build_system_prompt_from_parts(
    wiki: str,
    rules: List[str],
    schemas: List[Dict[str, Any]],
) -> str:
    """Build the system prompt from raw parts.

    Split out from :func:`build_system_prompt` so the sandboxed tool, which has
    no local ``Env`` object and only the ``/spec`` payload fetched from inside
    the sandbox, produces a byte-identical prompt from the same code.
    """
    # Compact JSON saves >>> prompt budget while keeping the full machine-readable
    # schema intact; whitespace from ``indent=2`` accounts for more than 50% of
    # the JSON characters for the τ-bench tool list.
    schema_text = json.dumps(schemas, separators=(",", ":"))
    return (
        "You are a helpful assistant solving tasks in a simulated customer-service "
        "environment.\n\n"
        f"Environment rules:\n{wiki}\n\n"
        f"Rules:\n{chr(10).join(f'- {r}' for r in rules)}\n\n"
        "When you want to call a tool, output exactly:\n"
        "<tool_call>{\"action_name\": \"<tool_name>\", \"action_kwargs\": {...}}</tool_call>\n"
        "When you want to give the final answer to the user, output exactly:\n"
        "<tool_call>{\"action_name\": \"respond\", \"action_kwargs\": {\"content\": \"...\"}}</tool_call>\n\n"
        "Available tools:\n" + schema_text + "\n"
    )


def build_system_prompt(env: Any) -> str:
    """Build the LLM system prompt including wiki, rules, and tool schemas.

    The prompt instructs the model to emit tool calls wrapped in
    `<tool_call>{"action_name": ..., "action_kwargs": ...}</tool_call>` tags.

    Notes:
    - Rules and output format live *before* the schema dump so that right-side
      truncation keeps the protocol instructions intact.
    - The schema is compact JSON to stay well within a modest prompt budget.
    """
    return build_system_prompt_from_parts(env.wiki, list(env.rules), env_to_openai_tools(env))


def list_tasks(env: Any) -> List[Task]:
    """Return the task list from a τ-bench env."""
    return env.tasks
