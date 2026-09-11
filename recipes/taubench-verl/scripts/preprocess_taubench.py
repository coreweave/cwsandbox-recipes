"""Preprocess τ-bench tasks into veRL parquet datasets.

Each row in the output parquet contains:
  - prompt: chat messages (system + first user turn)
  - agent_name: "tau_bench_agent" (used by the custom AgentLoop)
  - reward_model: {"style": "rule", "ground_truth": <env ground truth>}
  - extra_info: episode metadata including domain, task_split, task_index,
    and tool create kwargs.

Usage:
  python scripts/preprocess_taubench.py \
      --domain retail \
      --task-split train \
      --local-save-dir ./data/processed
"""

import argparse
import os

import pandas as pd

from verl_taubench.envs import taubench_env


def build_prompt(system_text: str, user_instruction: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_instruction},
    ]


def process_task(domain: str, task_split: str, task_index: int) -> dict:
    env = taubench_env.make_env(
        domain=domain,
        task_split=task_split,
        task_index=task_index,
        user_strategy="human",
    )
    # reset() with the human user strategy still reads stdin, so we directly
    # use the task's instruction as the first user message.
    instruction = env.task.instruction
    system_text = taubench_env.build_system_prompt(env)

    prompt = build_prompt(system_text, instruction)
    return {
        "data_source": f"tau-bench-{domain}-{task_split}",
        "agent_name": "tau_bench_agent",
        "prompt": prompt,
        # Same as ``prompt``; AgentLoopBase postprocess reads ``raw_prompt``.
        "raw_prompt": prompt,
        "ability": "multi-turn-tool-use",
        "domain": domain,
        "task_split": task_split,
        "task_index": task_index,
        "reward_model": {
            "style": "rule",
            # τ-bench tasks don't expose a single string ground truth; store the
            # task outputs for use by the reward function.
            "ground_truth": " ".join(env.task.outputs),
        },
        "extra_info": {
            "domain": domain,
            "task_split": task_split,
            "task_index": task_index,
            "task_outputs": env.task.outputs,
            "instruction": env.task.instruction,
            # The AgentLoop passes these kwargs to the step tool's create().
            "tools_kwargs": {
                "tau_bench_step": {
                    "create_kwargs": {
                        "domain": domain,
                        "task_split": task_split,
                        "task_index": task_index,
                    },
                },
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="retail", choices=["retail", "airline"])
    parser.add_argument("--task-split", default="train", choices=["train", "test", "dev"])
    parser.add_argument("--local-save-dir", default="./data/processed")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1)
    args = parser.parse_args()

    # Use user_strategy="human" so preprocessing does not call any LLM API.
    env = taubench_env.make_env(
        domain=args.domain, task_split=args.task_split, user_strategy="human"
    )
    tasks = taubench_env.list_tasks(env)

    end_index = args.end_index if args.end_index >= 0 else len(tasks)
    selected_indices = range(args.start_index, min(end_index, len(tasks)))

    records = [process_task(args.domain, args.task_split, idx) for idx in selected_indices]
    df = pd.DataFrame(records)

    os.makedirs(args.local_save_dir, exist_ok=True)
    path = os.path.join(args.local_save_dir, f"{args.task_split}.parquet")
    df.to_parquet(path, index=False)
    print(f"Wrote {len(records)} tasks to {path}")


if __name__ == "__main__":
    main()
