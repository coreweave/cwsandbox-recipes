"""Multi-turn rollout loop for τ-bench inside veRL.

The loop implements the real ``verl.experimental.agent_loop.agent_loop.AgentLoopBase``
contract and uses veRL's Continuous Token builders so that the token sequence returned
for training is exactly the sequence the model was conditioned on during rollout.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from verl_taubench._verl_compat import (
    VERL_AVAILABLE,
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)
from verl_taubench.envs import taubench_env
from verl_taubench.sandbox.reporting import attach_sdk_reporter

try:
    # Optional rollout tracing (rollout.trace.backend=weave/mlflow/trackio).
    # The decorator is a no-op unless tracing is enabled by the trainer config.
    from verl.utils.rollout_trace import RolloutTraceConfig
    from verl.utils.rollout_trace import _trace_attributes as _verl_trace_attributes
    from verl.utils.rollout_trace import _trace_enabled as _verl_trace_enabled
except ImportError:  # pragma: no cover - veRL absent or too old for tracing
    RolloutTraceConfig = None  # type: ignore[assignment]
    _verl_trace_enabled = None
    _verl_trace_attributes = None


# weave.init runs once per AgentLoopWorker actor while wandb.init lives in the
# driver, so Weave never sees an ambient wandb.run to attach traces to. When the
# launcher forwards WANDB_RUN_ID, bind every worker's weave client to that run
# (re-bound whenever the training step advances so traces align to the metric
# step axis). https://docs.wandb.ai/weave/guides/tools/weave-in-workspaces
_weave_run_context = {"bound_step": (), "warned": False}


def _weave_trace_client():
    """The active weave client, or None when weave tracing is not configured."""
    if RolloutTraceConfig is None or RolloutTraceConfig.get_backend() != "weave":
        return None
    return RolloutTraceConfig.get_client()


def _coerce_step(value: Any) -> Optional[int]:
    """Best-effort int coercion for global-step values (numpy scalars included)."""
    if value is None:
        return None
    try:
        step = int(value)
    except (TypeError, ValueError):
        return None
    return step if step >= 0 else None


def _sync_weave_run_context(step: Optional[int] = None) -> None:
    run_id = os.environ.get("WANDB_RUN_ID")
    if not run_id:
        return
    client = _weave_trace_client()
    if client is None:
        return
    if not hasattr(client, "set_wandb_run_context"):
        if not _weave_run_context["warned"]:
            _weave_run_context["warned"] = True
            logging.getLogger(__name__).warning(
                "weave %s has no set_wandb_run_context; traces will not be "
                "linked to wandb run %s (need weave>=0.53)",
                getattr(__import__("weave"), "__version__", "?"),
                run_id,
            )
        return
    if step is None:
        # rollout_trace_attr sets these on the stock AgentLoopWorker path; the
        # v1 TQ worker dispatches trace=False so they stay unset there.
        attributes = _verl_trace_attributes.get() if _verl_trace_attributes is not None else None
        step = _coerce_step((attributes or {}).get("step"))
    if _weave_run_context["bound_step"] != (run_id, step):
        client.set_wandb_run_context(run_id=run_id, step=step)
        _weave_run_context["bound_step"] = (run_id, step)


@contextlib.contextmanager
def _ensure_trace_attributes(step: Optional[int], sample_index: Any = None):
    """Populate verl's trace-attributes contextvar when the dispatcher didn't.

    The v1 TQ worker calls ``_run_agent_loop`` with ``trace=False``, so
    ``rollout_trace_attr`` never records step/sample attributes and every weave
    span loses its step. Recreate the same attribute dict for the episode.
    """
    if (
        _verl_trace_attributes is None
        or RolloutTraceConfig is None
        or RolloutTraceConfig.get_backend() is None
        or _verl_trace_attributes.get() is not None
    ):
        yield
        return
    attributes: Dict[str, Any] = {
        "experiment_name": RolloutTraceConfig.get_instance().experiment_name,
    }
    if step is not None:
        attributes["step"] = step
    coerced_index = _coerce_step(sample_index)
    if coerced_index is not None:
        attributes["sample_index"] = coerced_index
    token = _verl_trace_attributes.set(attributes)
    try:
        yield
    finally:
        _verl_trace_attributes.reset(token)


@contextlib.contextmanager
def _weave_span(
    name: str,
    inputs: Dict[str, Any],
    display_name: Optional[str] = None,
    kind: Optional[str] = None,
):
    """Open a nested weave span; yields a dict to fill with the span's output.

    verl's ``rollout_trace_op`` creates calls without pushing them onto weave's
    call-context stack, so its spans never nest. Pushing here makes every span
    opened inside (autopatched simulator calls included) a child of this one,
    which is what turns a τ-bench episode into a readable multi-turn trace.

    ``kind`` (agent | llm | tool | chain | ...) sets the icon/color the trace
    tree renders for the span. A span whose inputs carry an OpenAI-style
    ``messages`` list and whose output is ChatCompletion-shaped gets Weave's
    conversation view.
    """
    client = _weave_trace_client()
    if client is None or (_verl_trace_enabled is not None and not _verl_trace_enabled.get()):
        yield {}
        return
    try:
        from weave.trace.context import call_context
    except ImportError:
        yield {}
        return
    attributes = dict((_verl_trace_attributes.get() if _verl_trace_attributes is not None else None) or {})
    if kind:
        attributes["kind"] = kind
    call = client.create_call(op=name, inputs=inputs, attributes=attributes or None, display_name=display_name)
    call_context.push_call(call)
    output: Dict[str, Any] = {}
    try:
        yield output
    except BaseException as exc:
        call_context.pop_call(call.id)
        client.finish_call(call, exception=exc)
        raise
    call_context.pop_call(call.id)
    client.finish_call(call, output=dict(output) or None)


@contextlib.contextmanager
def _force_rollout_trace_enabled():
    """Re-enable per-sample tracing that verl's transfer-queue worker disables.

    The v1 trainer's ``AgentLoopWorkerTQ.generate_sequences`` dispatches every
    sample with ``trace=False`` ("TODO(wuxibin): add trace support"), which sets
    verl's ``_trace_enabled`` context variable to False and turns every
    ``@rollout_trace_op`` span below into a no-op — even though the worker still
    initializes the trace backend (weave logs in, then never receives a call).
    When a backend is configured but the flag is off, flip it back on for the
    duration of this episode. The stock ``AgentLoopWorker`` path already runs
    with the flag on and is unaffected.
    """
    if (
        _verl_trace_enabled is None
        or RolloutTraceConfig is None
        or RolloutTraceConfig.get_backend() is None
        or _verl_trace_enabled.get()
    ):
        yield
        return
    token = _verl_trace_enabled.set(True)
    try:
        yield
    finally:
        _verl_trace_enabled.reset(token)

_WRAPPER_TOOL_NAME = "tau_bench_step"
_RESPOND_ACTION = "respond"

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)


def _normalize_kwargs_value(value: Any) -> Any:
    """Unwrap numpy scalar/array objects that Hydra/OmegaConf inject into kwargs.

    veRL supplies non-tensor batch columns as numpy object arrays (e.g.
    ``raw_prompt``).  We unwrap 0-d arrays to their scalar/list value while
    preserving normal Python containers so that downstream truthiness tests
    do not raise ``ValueError`` for object-ndarray inputs.
    """
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        # 0-d arrays wrap a single scalar/object (e.g. an ``extra_info`` dict).
        if value.ndim == 0:
            return _normalize_kwargs_value(value.item())
        # Every other array is a sequence and must stay a sequence. Notably a
        # 1-message ``raw_prompt`` has shape (1,); unwrapping it to the bare dict
        # would make ``[dict(m) for m in raw_prompt]`` iterate over dict keys.
        return [_normalize_kwargs_value(v) for v in value.tolist()]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return [_normalize_kwargs_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize_kwargs_value(v) for k, v in value.items()}
    return value


def _is_taubench_step_tool(tool: Any) -> bool:
    """Accept any tool implementing the τ-bench step contract.

    Structural rather than ``isinstance`` so a tool subclassed or swapped out
    downstream still resolves, as long as it honours the contract.
    """
    if getattr(tool, "name", None) != _WRAPPER_TOOL_NAME:
        return False
    return all(
        callable(getattr(tool, method, None))
        for method in ("create", "execute", "calc_reward", "release")
    )


def _extract_tool_create_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Find the step tool's ``create`` kwargs in the dataset row."""
    extra_info = _normalize_kwargs_value(kwargs.get("extra_info")) or {}
    if isinstance(extra_info, dict):
        tools_kwargs = extra_info.get("tools_kwargs", {})
        if isinstance(tools_kwargs, dict):
            wrapper = tools_kwargs.get(_WRAPPER_TOOL_NAME, {})
            create_kwargs = wrapper.get("create_kwargs", {})
            if create_kwargs:
                return dict(create_kwargs)
        # Fallback: extra_info directly stores domain/task_split/task_index.
        return {
            "domain": extra_info.get("domain", "retail"),
            "task_split": extra_info.get("task_split", "train"),
            "task_index": extra_info.get("task_index"),
        }
    return {"domain": "retail", "task_split": "train", "task_index": None}


def _parse_tool_call(text: str) -> Tuple[str, Dict[str, Any]]:
    """Extract action_name/action_kwargs from a model response.

    Supports:
      - <tool_call>{"name": "...", "arguments": {...}}</tool_call> (OpenAI schema)
      - <tool_call>{"action_name": "...", "action_kwargs": {...}}</tool_call> (our wrapper)
      - plain text -> respond action
    """
    match = _TOOL_CALL_RE.search(text)
    if match:
        payload = match.group(1)
    else:
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            payload = stripped
        else:
            return "respond", {"content": text}

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return "respond", {"content": text}

    if not isinstance(data, dict):
        return "respond", {"content": text}

    # OpenAI function-call style.
    if "name" in data:
        name = str(data["name"])
        arguments = data.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"content": arguments}
        if name == _WRAPPER_TOOL_NAME:
            return str(arguments.get("action_name", "respond")), dict(arguments.get("action_kwargs", {}))
        return name, dict(arguments) if isinstance(arguments, dict) else {}

    # Native wrapper schema.
    if "action_name" in data:
        return str(data["action_name"]), dict(data.get("action_kwargs", {}))

    return "respond", {"content": text}


@register("tau_bench_agent")
class TauBenchAgentLoop(AgentLoopBase):
    """AgentLoopBase implementation for τ-bench multi-turn episodes."""

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
        # If veRL is not installed we hit the stub base above, which does not accept
        # these arguments; in that case suppress the real signature.
        if VERL_AVAILABLE:
            super().__init__(
                trainer_config=trainer_config,
                server_manager=server_manager,
                tokenizer=tokenizer,
                processor=processor,
                dataset_cls=dataset_cls,
                data_config=data_config,
                hf_model_type=hf_model_type,
                **kwargs,
            )
        else:
            # Stub path: keep constructor compatible with tests that import the file.
            super().__init__(
                trainer_config=trainer_config,
                server_manager=server_manager,
                tokenizer=tokenizer,
                processor=processor,
                dataset_cls=dataset_cls,
                data_config=data_config,
                hf_model_type=hf_model_type,
            )
            self.tools = kwargs.get("tools", [])

        # ``tools`` is passed as ``ToolListWrap`` from AgentLoopWorker.
        tools = kwargs.get("tools")
        tool_list = getattr(tools, "tools", tools) or []
        self.step_tool: Optional[Any] = None
        for tool in tool_list:
            if _is_taubench_step_tool(tool):
                self.step_tool = tool
                break
        if self.step_tool is None:
            raise ValueError(
                "TauBenchAgentLoop requires a τ-bench step tool in `tools` "
                "(SandboxTauBenchTool)."
            )

        # Convenience aliases only used in unit-test / stub mode.
        self.llm_client = server_manager
        self.max_turns = int(kwargs.get("max_turns", 12))
        self.response_length = int(getattr(self.rollout_config, "response_length", 4096))

    async def run(self, sampling_params: Dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run one full τ-bench episode and return the token trajectory."""
        # The SDK reporter looks for wandb.run in this worker process. Attach
        # before provisioning so even the initial setup execs are recorded.
        await asyncio.to_thread(attach_sdk_reporter, getattr(self, "config", None))
        # The v1 TQ worker forwards the dataloader's global step per sample;
        # bind weave traces to the wandb run at that step.
        step = _coerce_step(kwargs.get("global_steps"))
        _sync_weave_run_context(step)
        tool_create_kwargs = _extract_tool_create_kwargs(kwargs)
        episode_inputs = {
            "domain": tool_create_kwargs.get("domain", "retail"),
            "task_split": tool_create_kwargs.get("task_split", "train"),
            "task_index": _normalize_kwargs_value(tool_create_kwargs.get("task_index")),
            "global_step": step,
        }
        with (
            _force_rollout_trace_enabled(),
            _ensure_trace_attributes(step, sample_index=kwargs.get("index")),
            _weave_span("taubench.episode", episode_inputs, kind="agent") as episode_output,
        ):
            output = await self._run_episode(sampling_params, **kwargs)
            episode_output["reward"] = output.reward_score
            episode_output["num_assistant_turns"] = output.num_turns
            return output

    async def _run_episode(self, sampling_params: Dict[str, Any], **kwargs) -> AgentLoopOutput:
        request_id = str(uuid.uuid4())

        tool_create_kwargs = _extract_tool_create_kwargs(kwargs)
        domain = tool_create_kwargs.get("domain", "retail")
        task_split = tool_create_kwargs.get("task_split", "train")
        task_index = tool_create_kwargs.get("task_index")

        instance_id: Optional[str] = None
        try:
            instance_id, create_response = await self.step_tool.create(
                domain=domain,
                task_split=task_split,
                task_index=task_index,
                user_strategy=kwargs.get("user_strategy", "llm"),
                user_model=kwargs.get("user_model", "gpt-4o-mini"),
                user_provider=kwargs.get("user_provider"),
            )
            state = self.step_tool.instance_registry[instance_id]

            # Prefer the raw OpenAI-format prompt from the dataset; fall back to
            # building the system prompt from the environment.
            raw_prompt = _normalize_kwargs_value(kwargs.get("raw_prompt")) or _normalize_kwargs_value(
                kwargs.get("prompt")
            )
            # Some tools own turn 0 themselves. The sandboxed tool drives a real
            # user simulator, whose opening utterance is the correct first user
            # message. The dataset's ``raw_prompt`` instead carries
            # ``task.instruction`` -- the simulator's *private* brief -- so using
            # it would both leak the hidden goal to the policy and waste the
            # simulator call we already paid for.
            tool_owns_initial_turn = bool(getattr(self.step_tool, "owns_initial_turn", False))
            if raw_prompt and not tool_owns_initial_turn:
                messages = [dict(msg) for msg in raw_prompt]
            else:
                # The in-process tool exposes a live ``env``; the sandboxed tool
                # has none locally and precomputes the prompt from ``/spec``.
                if state.get("system_prompt") is not None:
                    system_text = state["system_prompt"]
                else:
                    system_text = taubench_env.build_system_prompt(state["env"])
                initial_obs = state["observation"]
                messages = [
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": initial_obs},
                ]

            # Canonical rollout in the policy's frame. Keep semantic roles even
            # though the simulator is itself an LLM whose model-local trace has
            # an assistant output: policy=assistant, simulator=user, env=tool.
            dialogue: List[Dict[str, Any]] = [dict(message) for message in messages]

            # Tool schemas used by the chat template and for resolving tool names.
            openai_tools = self.step_tool.tool_schemas or []

            if VERL_AVAILABLE and self.continuous_token_builder is not None:
                prompt_ids = await self.ct_build_initial_tokens(messages, tools=openai_tools)
            else:
                prompt_ids = self._fallback_message_ids(messages)

            runtime_token_ids = list(prompt_ids)
            response_mask: List[int] = []
            response_logprobs: Optional[List[float]] = None
            # Policy-version bookkeeping (min/max_global_steps etc.) reported by
            # the rollout server per generate call; the v1 trainer reads it from
            # sample tags for staleness metrics.
            generation_extras: Dict[str, Any] = {}
            done = False
            num_assistant_turns = 0

            for turn in range(self.max_turns):
                if done:
                    break

                used = len(runtime_token_ids) - len(prompt_ids)
                with _weave_span(
                    "taubench.turn",
                    {"turn": turn, "response_tokens_used": used},
                    display_name=f"turn {turn}",
                    kind="chain",
                ) as turn_output:
                    remaining = max(1, self.response_length - used - 1)
                    generate_params = dict(sampling_params)
                    # vLLM/SGLang ``SamplingParams`` use ``max_tokens``. The
                    # HuggingFace-style ``max_new_tokens`` is dropped rather than sent
                    # alongside, because vLLM raises on unknown sampling fields.
                    generate_params.pop("max_new_tokens", None)
                    generate_params["max_tokens"] = remaining

                    # OpenAI-shaped inputs/output give the span Weave's
                    # conversation view; the policy itself speaks token ids.
                    with _weave_span(
                        "policy.chat",
                        {"messages": [dict(m) for m in messages], "model": self._policy_model_name()},
                        kind="llm",
                    ) as llm_output:
                        generation = await self._generate(
                            prompt_ids=runtime_token_ids,
                            request_id=request_id,
                            sampling_params=generate_params,
                        )
                        assistant_token_ids = generation["token_ids"]
                        assistant_logprobs = generation["log_probs"]
                        gen_extra = generation["extra_fields"]
                        assistant_text = (
                            self.tokenizer.decode(assistant_token_ids, skip_special_tokens=True)
                            if assistant_token_ids
                            else ""
                        )
                        if assistant_text:
                            llm_output["model"] = self._policy_model_name()
                            llm_output["choices"] = [
                                {
                                    "index": 0,
                                    "message": {"role": "assistant", "content": assistant_text},
                                    "finish_reason": "stop",
                                }
                            ]
                    if not assistant_token_ids:
                        break

                    if gen_extra:
                        if not generation_extras:
                            generation_extras.update(gen_extra)
                        else:
                            # Multi-turn: keep min_global_steps from the first turn,
                            # track the newest policy version seen (same semantics as
                            # verl's ToolAgentLoop).
                            max_steps = gen_extra.get("max_global_steps")
                            if max_steps is not None:
                                generation_extras["max_global_steps"] = max_steps

                    parsed_name, parsed_kwargs = _parse_tool_call(assistant_text)
                    turn_output["assistant_text"] = assistant_text
                    turn_output["action"] = {"name": parsed_name, "kwargs": parsed_kwargs}

                    # Append the assistant turn *in message space* before using CT builders.
                    assistant_message = {"role": "assistant", "content": assistant_text}
                    messages.append(assistant_message)
                    dialogue.append(dict(assistant_message))

                    if VERL_AVAILABLE and self.continuous_token_builder is not None:
                        # veRL's CT aligner needs an empty accumulator (rather
                        # than None) when the first assistant turn has rollout
                        # logprobs. Empty server logprob lists mean "absent".
                        merge_kwargs: Dict[str, Any] = {
                            "response_logprobs": (
                                response_logprobs
                                if response_logprobs or not assistant_logprobs
                                else []
                            )
                        }
                        if assistant_logprobs:
                            # Rollout logprobs flow: sampling_params["logprobs"] is set by
                            # verl when rollout.calculate_log_probs is on; the CT builder
                            # aligns them so the trainer gets `rollout_log_probs`.
                            merge_kwargs["assistant_logprobs"] = assistant_logprobs
                        assistant_merge, response_mask, response_logprobs = await self.ct_merge_assistant_token(
                            runtime_token_ids,
                            assistant_token_ids,
                            response_mask,
                            **merge_kwargs,
                        )
                        runtime_token_ids = list(assistant_merge.token_ids)
                    else:
                        runtime_token_ids.extend(assistant_token_ids)
                        response_mask.extend([1] * len(assistant_token_ids))
                        if assistant_logprobs is not None:
                            response_logprobs = (response_logprobs or []) + list(assistant_logprobs)

                    num_assistant_turns += 1

                    # Execute the action in the environment.
                    observation = await self._step_env(
                        action_name=parsed_name,
                        action_kwargs=parsed_kwargs,
                        instance_id=instance_id,
                    )
                    done = state["done"]
                    turn_output["observation"] = observation
                    turn_output["done"] = done

                    # τ-bench distinguishes conversation from environment I/O:
                    # a reply to `respond` is the next customer (`user`) turn;
                    # only an actual environment action produces a `tool` turn.
                    tool_call_id = f"{request_id}_{turn}"
                    if parsed_name == _RESPOND_ACTION:
                        observation_message = {"role": "user", "content": observation}
                    else:
                        observation_message = {
                            "role": "tool",
                            "name": parsed_name,
                            "content": observation,
                            "tool_call_id": tool_call_id,
                        }
                    messages_before_observation = list(messages)
                    messages.append(observation_message)
                    dialogue.append(dict(observation_message))

                    if VERL_AVAILABLE and self.continuous_token_builder is not None:
                        non_ass_merge, response_mask, response_logprobs = await self.ct_merge_non_assistant_msg(
                            messages_before_observation,
                            messages,
                            runtime_token_ids,
                            response_mask,
                            response_logprobs=response_logprobs,
                            tools=openai_tools,
                        )
                        runtime_token_ids = list(non_ass_merge.token_ids)
                    else:
                        # Fallback path used by syntax-check tests without veRL.
                        obs_ids = self._fallback_encode(observation)
                        runtime_token_ids.extend(obs_ids)
                        response_mask.extend([0] * len(obs_ids))
                        if response_logprobs is not None:
                            response_logprobs.extend([0.0] * len(obs_ids))

            extra_fields: Dict[str, Any] = dict(generation_extras)
            # The v1 trainer converts these tags with dtype=int unconditionally
            # (staleness metrics); -1 is verl's unknown-step convention for
            # servers that do not report policy versions.
            for key in ("min_global_steps", "max_global_steps"):
                if extra_fields.get(key) is None:
                    extra_fields[key] = -1

            # Final reward is the τ-bench episode reward.
            reward_score: Optional[float] = None
            if instance_id is not None:
                reward_score = await self.step_tool.calc_reward(instance_id)
                # Also record it in extra_fields so any custom reward function can see it.
                extra_fields["task_reward"] = reward_score

            # `response_ids` must contain every token after the initial prompt.
            response_ids = runtime_token_ids[len(prompt_ids):]

            # Truncate to the configured rollout response length to avoid tensor mismatches.
            if len(response_ids) > self.response_length:
                response_ids = response_ids[:self.response_length]
                response_mask = response_mask[:self.response_length]
                if response_logprobs:
                    response_logprobs = response_logprobs[:self.response_length]

            # Reconcile mask length to response length regardless of which path
            # produced it (fallback or Continuous-Token builder). Pad with 0 so we
            # never train on tokens without an explicit mask decision.
            if len(response_mask) < len(response_ids):
                response_mask.extend([0] * (len(response_ids) - len(response_mask)))
            elif len(response_mask) > len(response_ids):
                response_mask = response_mask[:len(response_ids)]

            # When the trainer asked for rollout logprobs (rollout.calculate_log_probs),
            # `rollout_log_probs` must exist for every sample in the batch or verl's
            # debug metrics KeyError. Keep them aligned 1:1 with response_ids; masked
            # (observation/padding) positions carry 0.0 and are excluded by the mask.
            if response_logprobs is None and sampling_params.get("logprobs"):
                response_logprobs = [0.0] * len(response_ids)
            if response_logprobs is not None:
                if len(response_logprobs) < len(response_ids):
                    response_logprobs.extend([0.0] * (len(response_ids) - len(response_logprobs)))
                elif len(response_logprobs) > len(response_ids):
                    response_logprobs = response_logprobs[:len(response_ids)]

            if dialogue:
                with _weave_span(
                    "taubench.dialogue",
                    {"messages": dialogue, "model": self._policy_model_name()},
                    display_name="rollout (policy = assistant, simulator = user)",
                    kind="chain",
                ) as dialogue_output:
                    dialogue_output["reward"] = reward_score
                    dialogue_output["num_assistant_turns"] = num_assistant_turns

            return AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_mask=response_mask,
                response_logprobs=response_logprobs,
                reward_score=reward_score,
                num_turns=num_assistant_turns,
                metrics=AgentLoopMetrics(),
                extra_fields=extra_fields,
            )
        finally:
            if instance_id is not None:
                await self.step_tool.release(instance_id)

    def _policy_model_name(self) -> str:
        try:
            return str(self.config.actor_rollout_ref.model.path)
        except Exception:
            return "policy-model"

    async def _generate(
        self,
        prompt_ids: List[int],
        request_id: str,
        sampling_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Thin async wrapper around the LLM server generate call.

        Newer veRL returns a ``TokenOutput`` model (``token_ids``, optional
        ``log_probs``, and ``extra_fields`` with policy-version bookkeeping)
        instead of a bare token-id list; normalize so callers can handle
        either shape. Tracing happens in the caller's ``policy.chat`` span,
        which shows the conversation as text instead of token ids.
        """
        result = await self.server_manager.generate(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
        )
        token_ids = getattr(result, "token_ids", result)
        log_probs = getattr(result, "log_probs", None)
        extra_fields = getattr(result, "extra_fields", None) or {}
        return {
            "token_ids": list(token_ids) if token_ids else [],
            "log_probs": list(log_probs) if log_probs is not None else None,
            "extra_fields": dict(extra_fields),
        }

    async def _step_env(
        self,
        action_name: str,
        action_kwargs: Dict[str, Any],
        instance_id: Optional[str],
    ) -> str:
        """Execute one τ-bench action; traced so Weave shows action → observation."""
        with _weave_span(
            "taubench.step_env",
            {"action_name": action_name, "action_kwargs": action_kwargs},
            display_name=action_name,
            kind="tool",
        ) as tool_output:
            tool_response, _step_reward, _ = await self.step_tool.execute(
                instance_id,
                {"action_name": action_name, "action_kwargs": action_kwargs},
            )
            observation = tool_response.text or ""
            tool_output["observation"] = observation
        return observation

    def _fallback_message_ids(self, messages: List[Dict[str, Any]]) -> List[int]:
        """CPU-only fallback tokenization used when veRL is not installed."""
        if self.tokenizer is None:
            raise RuntimeError("No tokenizer available in fallback mode")
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return self._fallback_encode(text)
        pieces = []
        for msg in messages:
            pieces.append(f"<{msg['role']}>\n{msg['content']}\n</{msg['role']}>")
        return self._fallback_encode("\n".join(pieces))

    def _fallback_encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)
