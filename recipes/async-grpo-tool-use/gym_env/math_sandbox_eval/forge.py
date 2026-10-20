"""CoreWeave Forge logging for the episode server (``tir_app.py``): Agent Lens traces of the
rollouts and Weave evaluations of the AIME-2024 validation rounds.

With ``WANDB_API_KEY`` set, the server logs to the W&B project ``WANDB_ENTITY/WANDB_PROJECT``:

* **Agent Lens**: one conversation per traced rollout. Its turn holds each model call to the
  Dedicated Inference deployment (``chat``), each ``execute_python`` call in the episode's
  sandbox (``execute_tool``), and the grade as attributes. Every evaluation rollout is traced,
  and a fraction ``AGENT_LENS_SAMPLE`` of the training rollouts (default 1/16).
* **Weave evaluation**: one per validation round, with a row per rollout (problem, response,
  extracted answer, expected answer, correct) and the accuracy. NeMo-RL sends a round's
  ``EVAL_ROWS`` rollouts at once; a round is logged when all of them are graded, or when no
  more arrive for ``EVAL_IDLE_S`` seconds. Rounds run every ``VAL_PERIOD`` steps and after the
  last step (``MAX_NUM_STEPS``), so round ``k`` is labeled with the step it ran after.

Spans are built after the rollout ends, from the Gym response and the times the server saw:
a model call runs from the end of the previous cell (or the start of the episode) to the start
of the next cell (or the grade).
"""

import json
import os
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import weave
from coreweave.forge.agentlens import tracing
from coreweave.forge.agentlens.tracing import LLM, Message, Tool, Usage

FORGE_URL = "https://forge.coreweave.com/wandb"
AGENT_NAME = "math-tir-agent"
TOOL_NAME = "execute_python"


@dataclass
class Cell:
    code: str
    output: str
    started: float  # epoch seconds
    ended: float


@dataclass
class ModelCall:
    text: str
    tool_call_ids: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


def at(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


def model_calls(output: list[dict[str, Any]]) -> list[ModelCall]:
    """Split a Gym response's output items into model calls: each model call's messages and
    function calls come before the ``function_call_output`` items that answer them."""
    calls: list[ModelCall] = []
    current: Optional[ModelCall] = None
    for item in output:
        kind = item.get("type")
        if kind == "function_call_output":
            current = None
            continue
        if current is None:
            current = ModelCall(text="")
            calls.append(current)
        if kind == "message":
            current.text += "".join(
                p.get("text", "") for p in item.get("content") or [] if p.get("type") == "output_text"
            )
        elif kind == "function_call":
            current.tool_call_ids.append(item.get("call_id", ""))
        if item.get("generation_token_ids"):
            current.input_tokens = len(item.get("prompt_token_ids") or [])
            current.output_tokens += len(item["generation_token_ids"])
    return calls


def episode_spans(output: list[dict[str, Any]], cells: list[Cell], started: float, ended: float, model: str) -> list:
    """The ``chat`` and ``execute_tool`` spans of one rollout, in order."""
    spans: list = []
    boundary = started
    remaining = list(cells)
    for call in model_calls(output):
        next_cell = remaining[0] if remaining and call.tool_call_ids else None
        spans.append(
            LLM(
                model=model,
                provider_name="coreweave",
                output_messages=[Message(role="assistant", content=call.text)],
                usage=Usage(input_tokens=call.input_tokens, output_tokens=call.output_tokens),
                started_at=at(boundary),
                ended_at=at(next_cell.started if next_cell else ended),
            )
        )
        for call_id in call.tool_call_ids:
            if not remaining:
                break  # a call the server never ran (bad arguments)
            cell = remaining.pop(0)
            spans.append(
                Tool(
                    name=TOOL_NAME,
                    arguments=json.dumps({"code": cell.code}),
                    result=cell.output,
                    tool_call_id=call_id,
                    started_at=at(cell.started),
                    ended_at=at(cell.ended),
                )
            )
            boundary = cell.ended
    return spans


def round_step(round_index: int, val_period: int, max_steps: int) -> int:
    """The training step that validation round ``round_index`` (1-based) ran after."""
    return min(round_index * val_period, max_steps)


@dataclass
class EvalRound:
    """Graded evaluation rollouts of the current validation round."""

    size: int
    idle_s: float
    rows: list[dict[str, Any]] = field(default_factory=list)
    last: float = 0.0

    def add(self, row: dict[str, Any], now: float) -> Optional[list[dict[str, Any]]]:
        self.rows.append(row)
        self.last = now
        return self.take() if len(self.rows) >= self.size else None

    def due(self, now: float) -> Optional[list[dict[str, Any]]]:
        return self.take() if self.rows and now - self.last > self.idle_s else None

    def take(self) -> list[dict[str, Any]]:
        rows, self.rows = self.rows, []
        return rows


class Forge:
    """Agent Lens traces and Weave evaluations for one run. ``from_env`` returns None without
    ``WANDB_API_KEY``."""

    def __init__(self, entity: str, project: str, run_tag: str, model: str, *, sample: float, eval_rows: int,
                 eval_idle_s: float, val_period: int, max_steps: int) -> None:  # fmt: skip
        self.project = f"{entity}/{project}"
        self.run_tag = run_tag
        self.model = model
        self.sample = sample
        self.val_period = val_period
        self.max_steps = max_steps
        self.round = EvalRound(eval_rows, eval_idle_s)
        self.rounds = 0
        self.lock = threading.Lock()  # the server adds rows from worker threads
        tracing.init(self.project, autopatch_integrations=False)
        weave.init(self.project)
        print(f"[forge] traces: {FORGE_URL}/{self.project}/weave/agents", flush=True)

    @classmethod
    def from_env(cls, environ: dict[str, str] = os.environ) -> Optional["Forge"]:
        if not environ.get("WANDB_API_KEY") or not environ.get("WANDB_ENTITY"):
            return None
        return cls(
            environ["WANDB_ENTITY"],
            environ.get("WANDB_PROJECT", "async-grpo-tool-use"),
            environ.get("RUN_TAG", ""),
            environ.get("MODEL_REPO", "model").split("/")[-1],
            sample=float(environ.get("AGENT_LENS_SAMPLE", 1 / 16)),
            eval_rows=int(environ.get("EVAL_ROWS", 120)),
            eval_idle_s=float(environ.get("EVAL_IDLE_S", 120)),
            val_period=int(environ.get("VAL_PERIOD", 10)),
            max_steps=int(environ.get("MAX_NUM_STEPS", 20)),
        )

    def wants(self, split: str) -> bool:
        return split == "eval" or random.random() < self.sample

    def trace(self, *, session_id: str, split: str, question: str, system: str, output: list[dict[str, Any]],
              cells: list[Cell], started: float, ended: float, answer: str, expected: str,
              extracted: Optional[str], reward: float) -> None:  # fmt: skip
        tracing.log_turn(
            conversation_id=f"{self.run_tag}-{session_id}",
            conversation_name=question[:120],
            agent_name=AGENT_NAME,
            model=self.model,
            messages=[Message(role="user", content=question)],
            output_messages=[Message(role="assistant", content=answer)],
            system_instructions=[system] if system else None,
            spans=episode_spans(output, cells, started, ended, self.model),
            started_at=at(started),
            ended_at=at(ended),
            attributes={
                "rl.run_tag": self.run_tag, "rl.split": split, "rl.reward": reward,
                "rl.expected_answer": expected, "rl.extracted_answer": extracted or "", "rl.code_cells": len(cells),
            },
        )  # fmt: skip

    def add_eval(self, row: dict[str, Any]) -> None:
        with self.lock:
            rows = self.round.add(row, time.monotonic())
            if rows:
                self.log_round(rows)

    def flush_idle(self) -> None:
        with self.lock:
            rows = self.round.due(time.monotonic())
            if rows:
                self.log_round(rows)

    def log_round(self, rows: list[dict[str, Any]]) -> None:
        self.rounds += 1
        step = round_step(self.rounds, self.val_period, self.max_steps)
        logger = weave.EvaluationLogger(
            name=f"aime-2024-step-{step}",
            model={"name": f"{self.model}-{self.run_tag}", "metadata": {"step": step}},
            dataset="aime-2024",
            eval_attributes={"run_tag": self.run_tag, "step": step},
        )
        for row in rows:
            logger.log_example(
                inputs={"problem": row["question"], "expected_answer": row["expected"]},
                output={"response": row["answer"], "extracted_answer": row["extracted"], "code_cells": row["cells"]},
                scores={"correct": row["reward"] > 0},
            )
        accuracy = sum(row["reward"] > 0 for row in rows) / len(rows)
        logger.log_summary({"accuracy": accuracy, "samples": len(rows), "step": step})
        print(f"[forge] evaluation step {step}: accuracy {accuracy:.4f} over {len(rows)} rollouts", flush=True)

    def shutdown(self) -> None:
        tracing.force_flush()
