"""Run NVIDIA NeMo RL's native GRPO loop with a CPU sandbox reward environment."""

import argparse
import json
import os
import sys
from pathlib import Path

import ray
import torch
from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.datasets import AllTaskProcessedDataset, load_response_dataset
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.processors import math_hf_data_processor
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl_sandbox.environment import SandboxRewardClient


@ray.remote(num_cpus=1)
class SandboxEnvironment(EnvironmentInterface):
    def __init__(self, api_key):
        os.environ["CWSANDBOX_API_KEY"] = api_key
        self.client = SandboxRewardClient()

    def step(self, message_log_batch, metadata):
        responses = [
            "".join(str(m["content"]) for m in messages if m["role"] == "assistant")
            for messages in message_log_batch
        ]
        rewards = self.client.score(responses, [m["ground_truth"] for m in metadata])
        print(
            json.dumps(
                {
                    "event": "cpu_sandbox_rewards",
                    "count": len(rewards),
                    "rewards": rewards,
                }
            ),
            flush=True,
        )
        return EnvironmentReturn(
            observations=[
                {"role": "environment", "content": "Episode complete."} for _ in rewards
            ],
            metadata=metadata,
            next_stop_strings=[None] * len(rewards),
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.ones(len(rewards)),
            answers=None,
        )

    def global_post_process_and_metrics(self, batch):
        batch["rewards"] *= batch["is_end"]
        return batch, {"reward": batch["rewards"].float().mean().item()}

    def shutdown(self):
        self.client.close()
        return {
            "event": "cpu_sandbox_stopped",
            "calls": self.client.calls,
            "sandbox_id": self.client.sandbox.sandbox_id,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(Path(__file__).with_name("native-grpo.yaml"))
    )
    args, overrides = parser.parse_known_args()
    OmegaConf.register_new_resolver("mul", lambda a, b: a * b, replace=True)
    config = load_config(args.config)
    config = parse_hydra_overrides(config, overrides) if overrides else config
    config = OmegaConf.to_container(config, resolve=True)
    data_path = Path(config["data"]["train_data_path"])
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(
        "".join(
            json.dumps(
                {
                    "input": f"Calculate {a} * {b}. Explain briefly, then end with the integer answer.",
                    "output": str(a * b),
                }
            )
            + "\n"
            for a in range(12, 20)
            for b in range(7, 11)
        )
    )

    # Keep the credential out of Ray's environment metadata for GPU workers.
    api_key = os.environ.pop("CWSANDBOX_API_KEY")
    # Always create a job-owned cluster inside this single-node allocation.
    runtime_env = dict(os.environ)
    runtime_env.pop("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", None)
    runtime_env.pop("RAY_ADDRESS", None)
    ray.init(
        address="local",
        log_to_driver=True,
        include_dashboard=False,
        runtime_env={"env_vars": runtime_env},
        num_cpus=int(os.environ.get("SLURM_CPUS_PER_TASK", "16")),
    )
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )
    raw = load_response_dataset(config["data"], config["grpo"]["seed"])
    task = TaskDataSpec(
        task_name=raw.task_name, prompt_file=None, system_prompt_file=None
    )
    processors = {raw.task_name: (task, math_hf_data_processor)}
    dataset = AllTaskProcessedDataset(
        raw.formatted_ds["train"],
        tokenizer,
        task,
        processors,
        max_seq_length=config["data"]["max_input_seq_length"],
    )
    environment = SandboxEnvironment.remote(api_key)
    del api_key
    environments = {raw.task_name: environment}
    policy = generation = logger = None
    try:
        (
            policy,
            generation,
            cluster,
            dataloader,
            val_dataloader,
            loss_fn,
            logger,
            checkpointer,
            state,
            config,
        ) = setup(config, tokenizer, dataset, None)
        grpo_train(
            policy,
            generation,
            dataloader,
            val_dataloader,
            tokenizer,
            loss_fn,
            environments,
            environments,
            logger,
            checkpointer,
            state,
            config,
        )
        print(
            json.dumps(
                {
                    "event": "native_grpo_completed",
                    "checkpoint_dir": config["checkpointing"]["checkpoint_dir"],
                }
            ),
            flush=True,
        )
    finally:
        active_error = sys.exc_info()[0] is not None
        try:
            shutdown_ack = ray.get(environment.shutdown.remote())
            print(json.dumps(shutdown_ack), flush=True)
        except Exception:
            if not active_error:
                raise
            print(
                "CPU sandbox cleanup also failed; its maximum lifetime remains bounded.",
                file=sys.stderr,
            )
        finally:
            # Release native worker groups before Ray and before the policy
            # destructor runs, including when training raises an exception.
            for worker in (generation, policy):
                if worker is not None:
                    worker.shutdown()
            if logger is not None:
                for backend in logger.loggers:
                    if hasattr(backend, "writer"):
                        backend.writer.close()
            ray.shutdown()


if __name__ == "__main__":
    main()
