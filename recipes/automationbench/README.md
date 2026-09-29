# Evaluate models with AutomationBench

Run AutomationBench in CoreWeave CPU sandboxes with Harbor scheduling, per-task results, and bounded retries for capacity or inference-limit failures. The model runs at a configurable OpenAI-compatible Chat Completions endpoint. CoreWeave Serverless Inference is the example provider.

## Architecture

The local `run.py` script generates Harbor tasks and starts the job. Each concurrent trial gets one sandbox with 2 vCPUs and 4 GiB of memory. Inside it, the unchanged AutomationBench evaluator simulates business applications and scores the resulting state. You don't need a GPU or credentials for those simulated applications.

The recipe implements Harbor's `BaseAgent` and `BaseEnvironment` interfaces with public CWSandbox software development kit (SDK) calls. It pins Harbor 0.23.0 and AutomationBench 1.0.6 to Git revisions, and CWSandbox to 1.17.0. It doesn't require an upstream AutomationBench Harbor adapter.

## Prerequisites

Prepare the following:

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) and Git. The local project requires Python 3.13 or later. `uv` can install Python.
- A [CoreWeave API access token with sandbox access](https://docs.coreweave.com/products/sandboxes/get-started#choose-a-credential).
- Enough [concurrent sandbox quota](https://docs.coreweave.com/products/sandboxes/reference/limits-and-quotas) for your chosen concurrency, and resource policies that permit 2 vCPUs and 4 GiB per sandbox.
- A reachable inference endpoint with tool calling, an API key, and inference quota. The example requires [W&B Inference access](https://docs.wandb.ai/inference/prerequisites).
- Network access to the sandbox API, GitHub, Debian and Python package repositories, and your model endpoint.

The initial check creates one CPU sandbox. Sixteen concurrent trials request up to 32 vCPUs and 64 GiB in total. Sandbox resources and inference are billed separately. Estimate sandbox usage as the sum of trial durations multiplied by each sandbox's resources. Estimate inference usage from your model provider's token rates. Review [sandbox pricing](https://www.coreweave.com/pricing) and [inference usage limits](https://docs.wandb.ai/inference/usage-limits) before scaling.

## Run

Follow these steps to configure the recipe and evaluate tasks:

1. From this recipe directory, install the pinned dependencies and prepare configuration:

   ```bash
   uv sync --frozen
   cp .env.example .env
   ```

   In the `.env` file, set only the model endpoint, model name, and optional project. Don't save keys in the `.env` file. Load credentials from your secret manager into the process environment: `CWSANDBOX_API_KEY` for your CoreWeave sandbox credential and `MODEL_API_KEY` for your inference credential. The sandbox token stays on the host. The recipe injects the model key into the sandbox at creation.

   The example sets `MODEL_BASE_URL=https://api.inference.wandb.ai/v1` and `MODEL_NAME=openai/gpt-oss-120b`. For W&B Inference, set `MODEL_PROJECT` to `[WANDB-TEAM]/[WANDB-PROJECT]` using names you can access.

   For another provider, change the endpoint and model ID in the `.env` file, and load that provider's key into `MODEL_API_KEY` in the process environment. If the provider doesn't use the project header, remove `MODEL_PROJECT` from the `.env` file and unset any inherited `MODEL_PROJECT` environment variable before running the command. The endpoint must support Chat Completions and tool calls. `localhost` refers to the sandbox, not your machine.

2. To check setup, run one task:

   ```bash
   uv run --env-file .env python run.py \
     --task simple.email_sf_contact_phone_update \
     --concurrency 1 \
     --output results-smoke
   ```

   The output directory must be new. The script installs AutomationBench, runs the task, retrieves its results, and deletes the sandbox. A completed job prints `Harbor job finished; inspect job-result.json and archived attempts.` Exit status zero means no trials remain errored or missing. It doesn't mean the model passed the assertions.

3. After inspecting the setup result, evaluate a small subset with four concurrent trials:

   ```bash
   head -n 8 tasks-scored.txt > tasks-small.txt
   uv run --env-file .env python run.py \
     --tasks-file tasks-small.txt \
     --concurrency 4 \
     --output results-small
   ```

   Harbor schedules each task independently. Completed trials release slots for other tasks. Each new sandbox installs the evaluator, so per-task setup adds overhead.

4. When quota and inference capacity permit, run the scored set:

   ```bash
   uv run --env-file .env python run.py \
     --tasks-file tasks-scored.txt \
     --concurrency 16 \
     --output results-scored
   ```

   Sixteen is an example concurrency ceiling, not a validated throughput target. The `tasks-scored.txt` file contains 600 public tasks across six domains. The `tasks-simple.txt` file contains 200 separate baseline tasks. Don't mix baseline tasks into the scored-set denominator. The official leaderboard uses a separate private task set.

## Results and retries

The output directory contains the following files and directories:

| Path | Contents |
| --- | --- |
| `job-config.json` | Harbor configuration without credential values |
| `job-result.json` | Completed and errored trial statistics |
| `jobs/automationbench/` | Final trials, `agent/automationbench.json`, logs, and verifier rewards |
| `attempts/` | Archived attempts, including failures before retry |
| `sandbox-ids.jsonl` | IDs created by this job, for cleanup |

Native `passed` and `score` values become Harbor rewards named `pass` and `partial_credit`. Use native exports for timing and token usage. The adapter doesn't populate Harbor's aggregate agent token or cost fields. Protect logs and traces, which can contain task data and provider messages.

The agent uses the `api` toolset, 50 response steps, and the provider's default reasoning setting. Native whole-task autohealing is deactivated.

Harbor allows five retries for selected sandbox capacity, availability, startup-timeout, and inference-limit errors, waiting 30, 60, 120, 240, and 300 seconds. The policy removes `ApiUsageLimitError` from Harbor's default exclusions. The adapter recognizes terminal inference quota errors from log text. Unfamiliar provider formats need inspection. Individual model requests can still retry internally.

Assertion failures, refusals, malformed tool-call JSON, and agent execution timeouts don't trigger this retry policy. Persistent quota exhaustion remains a reported error after retries. Preserve attempt history rather than silently replacing model failures with better outcomes.

Model time can dominate a task even with few steps or output tokens. Compare `model_time_s` and `tool_time_s` and inspect the `agent/eval.log` file. Native error arrays can remain empty after an aborted rollout. Interrupted tasks without exported results are missing evaluations, not completed failures.

## Cleanup

To stop a foreground runner, press Ctrl+C and allow Harbor to exit. The job uses `delete=True` for teardown. After the runner stops, verify deletion of all recorded sandbox IDs:

```bash
uv run --env-file .env python cleanup.py results-smoke
```

Replace `results-smoke` with the output directory of the job you stopped. The helper deletes only recorded IDs and reports terminal states or `not_found`. It exits nonzero if any deletion or terminal-state check fails. Don't run it while the runner can create replacement sandboxes.

Each sandbox has a 1-hour lifetime cap. Agent execution has a 45-minute limit and setup has a separate 11-minute limit. The native evaluation command has a 40-minute limit. An agent execution timeout isn't retried by this policy.

## Test and compatibility

Run offline checks without credentials or cloud resources:

```bash
uv run python -m unittest -v
uv run ruff check .
uv run ruff format --check .
```

Earlier experiments verified the native evaluator in parallel sandboxes and Harbor retry scheduling with an injected quota failure followed by one real task. This recipe replaces the experimental environment hooks with a `BaseEnvironment` implementation. That refactor has offline coverage but hasn't been run against the live service. A full dataset at high Harbor concurrency hasn't been validated.

The optional `--inject-quota-once` flag enables a controlled live retry check when you choose to run one. It fails once before sandbox creation and then follows the normal retry path.
