"""Run a bounded DeepSWE sample with Pier's separate verifier lifecycle."""

import argparse
import asyncio
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pier.job import Job
from pier.models.job.config import DatasetConfig, JobConfig, RetryConfig
from pier.models.trial.config import AgentConfig, EnvironmentConfig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument(
        "--task", action="append", help="Task name; repeat to select several"
    )
    parser.add_argument("--n-tasks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="moonshotai/Kimi-K2.7-Code")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--agent-timeout", type=int, default=1800)
    parser.add_argument("--job-name")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Additional Pier trial executions after transient inference errors",
    )
    parser.add_argument("--agent", choices=["mini", "nop", "oracle"], default="mini")
    args = parser.parse_args(argv)
    for key in ["CWSANDBOX_API_KEY"] + (
        ["WANDB_API_KEY"] if args.agent == "mini" else []
    ):
        if not os.environ.get(key):
            parser.error(f"Set {key} in the environment")
    if min(args.n_tasks, args.steps, args.agent_timeout, args.concurrency) <= 0:
        parser.error(
            "Task count, step limit, timeout, and concurrency must be positive"
        )
    if args.max_retries < 0:
        parser.error("Maximum retries must be nonnegative")
    return args


def build_config(args):
    agent = (
        AgentConfig(
            import_path="host_agent:HostMiniSweAgent",
            model_name=args.model,
            override_timeout_sec=args.agent_timeout,
            kwargs={"step_limit": args.steps},
        )
        if args.agent == "mini"
        else AgentConfig(name=args.agent)
    )
    config = JobConfig(
        n_concurrent_trials=args.concurrency,
        retry=RetryConfig(
            max_retries=args.max_retries,
            include_exceptions={
                "RateLimitError",
                "APIConnectionError",
                "ServiceUnavailableError",
            },
            min_wait_sec=30,
            wait_multiplier=2,
            max_wait_sec=60,
        ),
        environment=EnvironmentConfig(
            import_path="cwsandbox_environment:CWSandboxEnvironment"
        ),
        agents=[agent],
        datasets=[
            DatasetConfig(
                path=args.tasks.resolve(),
                task_names=args.task,
                n_tasks=len(args.task) if args.task else args.n_tasks,
                sample_seed=args.seed,
            )
        ],
    )
    if args.job_name:
        config.job_name = args.job_name
    return config


async def preserve_failed_attempt(event):
    """Keep diagnostics before Pier deletes an attempt's directory for a retry."""
    if event.result is None or event.result.exception_info is None:
        return
    root = event.config.trials_dir
    source = root / event.config.trial_name
    destination = (
        root / "failed-attempts" / event.config.trial_name / str(event.result.id)
    )
    await asyncio.to_thread(shutil.copytree, source, destination)


async def execute(config):
    # Each synchronous agent step can wait for a remote shell operation that
    # also uses this pool. Leave room for both at the requested concurrency.
    asyncio.get_running_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=2 * config.n_concurrent_trials + 8)
    )
    job = await Job.create(config)
    job.on_trial_ended(preserve_failed_attempt)
    return await job.run()


def main():
    config = build_config(parse_args())
    result = asyncio.run(execute(config))
    print(f"Results: {config.jobs_dir / config.job_name / 'result.json'}")
    print(
        f"Finished: {result.stats.n_completed_trials}; "
        f"errors: {result.stats.n_errored_trials}; "
        f"cancelled: {result.stats.n_cancelled_trials}; "
        f"trial retries: {result.stats.n_retries}"
    )
    for trial in result.trial_results:
        if trial.exception_info:
            print(f"{trial.task_name}: {trial.exception_info.exception_type}")
        else:
            print(f"{trial.task_name}: {trial.verifier_result}")
    if result.stats.n_errored_trials or result.stats.n_cancelled_trials:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
