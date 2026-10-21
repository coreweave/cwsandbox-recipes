"""Build the NeMo-Gym data files: DAPO-Math-17k for training, AIME-2024 for evaluation.

Both datasets repeat each problem many times, so rows are de-duplicated by prompt. Every row
asks for tool-integrated reasoning (Qwen2.5-Math's prompt) and offers the ``execute_python``
tool, and names the Gym agent that handles it:

* training rows: ``math_tir_agent``
* evaluation rows: ``math_eval_agent``, with ``split: eval`` so the episode server logs them as
  a Weave evaluation

Both agents run in ``math_tir_sandbox.yaml``, one sandbox per episode.

Usage: python -m async_grpo_tool_use.dapo_data --out-dir /work/data [--max-samples N]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

TRAIN_DATASET = "BytedTsinghua-SIA/DAPO-Math-17k"
EVAL_DATASET = "BytedTsinghua-SIA/AIME-2024"
TRAIN_AGENT = "math_tir_agent"
EVAL_AGENT = "math_eval_agent"
AIME_PROBLEMS = 30
EVAL_REPEATS = 4
# Qwen2.5-Math's tool-integrated reasoning prompt, from its model card.
TIR_SYSTEM_PROMPT = (
    "Please integrate natural language reasoning with programs to solve the problem above, "
    "and put your final answer within \\boxed{}."
)
# The model writes ```python blocks; gym_env/token_ids_model.py maps them to this tool.
TIR_TOOL = {
    "type": "function",
    "name": "execute_python",
    "description": "Run Python code in this episode's sandbox and return what it prints.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
    "strict": False,
}


def gym_row(row: Mapping[str, Any], agent: str = TRAIN_AGENT) -> dict[str, Any]:
    question = row["prompt"][0]["content"]
    messages = [{"role": "system", "content": TIR_SYSTEM_PROMPT}, {"role": "user", "content": question}]
    return {
        "agent_ref": {"type": "responses_api_agents", "name": agent},
        "responses_create_params": {"input": messages, "tools": [TIR_TOOL]},
        "question": question,
        "expected_answer": str(row["reward_model"]["ground_truth"]),
        "split": "eval" if agent == EVAL_AGENT else "train",
    }


def unique_rows(
    rows: Iterable[Mapping[str, Any]], limit: int | None = None, agent: str = TRAIN_AGENT
) -> Iterator[dict[str, Any]]:
    seen: set[str] = set()
    for row in rows:
        out = gym_row(row, agent)
        if out["question"] in seen:
            continue
        seen.add(out["question"])
        yield out
        if limit is not None and len(seen) >= limit:
            return


def write(rows: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    count = 0
    with tmp.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
            count += 1
    tmp.replace(path)
    return count


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def prepare(out_dir: Path, max_samples: int | None = None, eval_repeats: int = EVAL_REPEATS) -> tuple[Path, Path]:
    """Write train.jsonl and eval.jsonl. Eval repeats each AIME problem ``eval_repeats`` times."""
    from datasets import load_dataset

    train, evaluation = out_dir / "train.jsonl", out_dir / "eval.jsonl"
    if not train.is_file():
        rows = load_dataset(TRAIN_DATASET, split="train", streaming=True)
        count = write(unique_rows(rows, max_samples), train)
        print(f"[grpo] wrote {count} DAPO prompts to {train}", flush=True)
    if not evaluation.is_file():
        problems = list(unique_rows(load_dataset(EVAL_DATASET, split="train"), agent=EVAL_AGENT))
        count = write((row for row in problems for _ in range(eval_repeats)), evaluation)
        print(
            f"[grpo] wrote {len(problems)} AIME-2024 problems x {eval_repeats} to {evaluation} ({count} rows)",
            flush=True,
        )
    return train, evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--eval-repeats", type=int, default=EVAL_REPEATS)
    args = parser.parse_args()
    prepare(args.out_dir, args.max_samples, args.eval_repeats)


if __name__ == "__main__":
    main()
