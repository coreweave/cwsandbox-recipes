"""veRL symbols with stand-ins for environments where veRL is not installed.

The recipe's tool and agent-loop modules must stay importable on a laptop that
has no CUDA wheels, so every veRL symbol they subclass or instantiate is
imported here once and replaced with a minimal stand-in when veRL is absent.

A failed import is only treated as "veRL absent" when ``verl`` is genuinely not
installed. If ``verl`` is on the path but fails to import (a broken transitive
dependency), the error is re-raised: silently falling back to the stand-ins
would turn a broken install into a training run that produces wrong tokens.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
from typing import Any, Dict, List, Optional

__all__ = [
    "VERL_AVAILABLE",
    "AgentLoopBase",
    "AgentLoopMetrics",
    "AgentLoopOutput",
    "BaseTool",
    "OpenAIFunctionToolSchema",
    "ToolResponse",
    "register",
]


class _PlaceholderConfig:
    """Minimal attribute-bag so the stand-in AgentLoopBase mirrors the real one."""

    def __init__(self, **defaults: Any):
        # Store the defaults in an attribute unlikely to collide with user keys.
        object.__setattr__(self, "_defaults", defaults)

    def __getattr__(self, name: str) -> Any:
        # Reserved names should be looked up on the object itself; otherwise
        # ``hasattr`` and ``getattr(..., default)`` behave pathologically.
        if name in ("_defaults", "__copy__", "__deepcopy__", "__getstate__", "__setstate__"):
            return object.__getattribute__(self, name)
        defaults = object.__getattribute__(self, "_defaults")
        if name in defaults:
            return defaults[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        defaults = object.__getattribute__(self, "_defaults")
        defaults[name] = value

    def __repr__(self) -> str:
        return f"{type(self).__name__}({object.__getattribute__(self, '_defaults')!r})"

    def __copy__(self, memo: Optional[Dict[int, Any]] = None) -> "_PlaceholderConfig":
        return _PlaceholderConfig(**copy.copy(object.__getattribute__(self, "_defaults")))

    def __deepcopy__(self, memo: Optional[Dict[int, Any]] = None) -> "_PlaceholderConfig":
        return _PlaceholderConfig(**copy.deepcopy(object.__getattribute__(self, "_defaults"), memo))


try:
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopBase,
        AgentLoopMetrics,
        AgentLoopOutput,
        register,
    )
    from verl.tools.base_tool import BaseTool
    from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

    VERL_AVAILABLE = True
except ImportError:
    if importlib.util.find_spec("verl") is not None:
        raise

    VERL_AVAILABLE = False

    class AgentLoopBase:  # type: ignore[no-redef]
        """Stand-in base class used when veRL is not installed locally."""

        def __init__(
            self,
            trainer_config: Any,
            server_manager: Any,
            tokenizer: Any,
            processor: Any,
            dataset_cls: Any,
            data_config: Any,
            hf_model_type: Optional[str] = None,
            **kwargs: Any,
        ):
            # Capture enough state for unit tests to inspect the loop.
            self.config = getattr(trainer_config, "config", trainer_config)
            self.server_manager = server_manager
            self.tokenizer = tokenizer
            self.processor = processor
            self.dataset_cls = dataset_cls
            self.data_config = getattr(data_config, "config", data_config)
            self.hf_model_type = hf_model_type
            self.continuous_token_builder = None
            self.loop = None
            # The real base exposes ``rollout_config``; keep these defaults in
            # sync with the trainer config defaults we document.
            self.rollout_config = _PlaceholderConfig(response_length=4096, n=1, prompt_length=8192)

    class AgentLoopMetrics:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any):
            for key, value in kwargs.items():
                setattr(self, key, value)

    @dataclasses.dataclass
    class AgentLoopOutput:  # type: ignore[no-redef]
        prompt_ids: List[int]
        response_ids: List[int]
        response_mask: List[int]
        response_logprobs: Optional[List[float]] = None
        routed_experts: Any = None
        multi_modal_data: Optional[Dict[str, Any]] = None
        reward_score: Optional[float] = None
        num_turns: int = 0
        metrics: Any = dataclasses.field(default_factory=lambda: AgentLoopMetrics())
        extra_fields: Dict[str, Any] = dataclasses.field(default_factory=dict)
        mm_processor_kwargs: Optional[Dict[str, Any]] = None

        def as_dict(self) -> Dict[str, Any]:
            return dataclasses.asdict(self)

    def register(name: str):  # type: ignore[no-redef]
        def decorator(cls):
            cls._registered_agent_name = name
            return cls

        return decorator

    BaseTool = object  # type: ignore[misc,assignment]

    class _OpenAIFunctionSchema:
        def __init__(self, *, name: str, description: str, parameters: Dict[str, Any]):
            self.name = name
            self.description = description
            self.parameters = parameters

    class OpenAIFunctionToolSchema:  # type: ignore[no-redef]
        def __init__(self, data: Dict[str, Any]):
            function = data.get("function", {})
            self.type = data.get("type", "function")
            self.function = _OpenAIFunctionSchema(
                name=function.get("name", ""),
                description=function.get("description", ""),
                parameters=function.get("parameters", {}),
            )

        @classmethod
        def model_validate(cls, data: Dict[str, Any]) -> "OpenAIFunctionToolSchema":
            return cls(data)

        def model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            return {
                "type": self.type,
                "function": {
                    "name": self.function.name,
                    "description": self.function.description,
                    "parameters": self.function.parameters,
                },
            }

    class ToolResponse:  # type: ignore[no-redef]
        def __init__(self, text: Optional[str] = None, image: Any = None, video: Any = None):
            self.text = text
            self.image = image
            self.video = video
