# Run Browser Use in a sandbox

Run a [Browser Use](https://github.com/browser-use/browser-use) agent and headless Chromium in a CoreWeave sandbox, using W&B Serverless Inference. The agent reads the title and a link from example.com. The recipe checks its answer against the browser's document object model (DOM), saves a screenshot and history, and stops the sandbox.

Use this recipe to build browser-agent workflows with independent result verification and a separate sandbox for each task.

Each worker requests and limits 2 CPUs and 4 GiB of memory, with a 30-minute maximum lifetime. The recipe defaults to one worker and supports up to eight. Compute and inference are billed separately. If a worker runs for the full lifetime, budget up to one CPU-hour and two GiB-hours, plus model tokens. Normal completion stops compute earlier. See [sandbox billing](https://www.coreweave.com/pricing) and [inference usage limits](https://docs.wandb.ai/inference/usage-limits).

## Architecture

The workflow has three components:

- Your local Python process creates sandboxes, uploads the agent, downloads results, and stops compute.
- Each sandbox runs its own Browser Use agent and Chromium process, with a separate browser profile.
- Serverless Inference serves `Qwen/Qwen3.6-35B-A3B` through its OpenAI-compatible API.

Chromium runs directly in the sandbox. This recipe doesn't start a nested virtual machine or require `/dev/kvm`. Package installation runs as root. The agent and Chromium run as the unprivileged `browser` user with Chromium's process sandbox enabled. The agent launches Chromium explicitly and connects Browser Use through the Chrome DevTools Protocol (CDP) on loopback. This avoids Browser Use 0.13.10 adding `--no-sandbox` when it detects a container.

## Prerequisites

Before you begin, confirm that you have the following prerequisites:

- Python 3.12 or later, Git, and [`uv`](https://docs.astral.sh/uv/getting-started/installation/).
- A [W&B API key](https://wandb.ai/authorize), a W&B team, and access to [serverless sandboxes](https://docs.coreweave.com/products/sandboxes/get-started) and [Serverless Inference](https://docs.wandb.ai/inference/prerequisites).
- Quota for the requested workers. Eight workers request 16 CPUs and 32 GiB in total. No GPU is required.
- Outbound access from the sandbox to Debian package mirrors, PyPI, api.inference.wandb.ai, and example.com.

## Set up

To prepare the recipe, complete the following steps:

1. Clone the repository and install the locked local dependencies:

   ```bash
   git clone https://github.com/coreweave/cwsandbox-recipes.git
   cd cwsandbox-recipes/recipes/browser-use
   uv sync --locked
   ```

2. Copy the environment template, set your team and project, and load it in your terminal:

   ```bash
   cp .env.example .env
   # Edit .env before continuing.
   set -a
   source .env
   set +a
   ```

   Load `WANDB_API_KEY` into your terminal environment from your secret manager. Keep the key out of `.env` and source control.

   `WANDB_ENTITY` is your team slug, not your organization slug. The recipe explicitly selects W&B authentication, even if `CWSANDBOX_API_KEY` is set. For this local demo, it passes the key from process memory to the sandbox as `MODEL_API_KEY` for inference and uses `WANDB_ENTITY/WANDB_PROJECT` for inference usage tracking.

For application deployments, use [W&B secret references](https://docs.coreweave.com/products/sandboxes/secrets) to deliver the inference key to the sandbox. Code in the sandbox can read the injected key.

## Run and verify

To verify a single agent and then run agents concurrently, complete the following steps:

1. Run one agent:

   ```bash
   uv run --locked python run.py --output outputs/single
   ```

   Successful output ends with:

   ```text
   1/1 passed; artifacts: outputs/single
   ```

2. Inspect the verified answer and cleanup status:

   ```bash
   cat outputs/single/worker-0/result.json
   cat outputs/single/summary.json
   ```

   The `result.json` file must have `passed: true` and the title `Example Domain`. Its HTTPS link must be on iana.org or www.iana.org and exist in the observed page. Each worker in the `summary.json` file must have `stopped: true`. If any task fails verification or cleanup, the command returns nonzero.

3. Run independent agents concurrently:

   ```bash
   uv run --locked python run.py --workers 4 --output outputs/parallel-4
   ```

   A successful run reports `4/4 passed`. Use `--workers 8` to run eight sandboxes. Choose a new output directory for every run. The script doesn't overwrite an existing directory, which preserves sandbox IDs for cleanup.

## Inspect results

Each `worker-N` directory contains `result.json`, `history.json`, `final.png`, setup and agent logs, the sandbox ID, and a worker summary. Failed runs might have only some artifacts. The controller attempts to collect results before stopping compute. Browser profiles stay in the disposable sandbox.

The worker summary separates sandbox readiness, package installation, and agent execution time. `agent_seconds` includes browser startup and verification but excludes browser cleanup. `total_seconds` includes provisioning, installation, artifact transfer, and stopping the sandbox. Compare the same task, model, and timing boundaries when measuring concurrency. These values aren't a browser-agent benchmark score.

## Clean up

The controller attempts to stop every sandbox in a `finally` block, including after setup or agent failures. If the controller is interrupted or a stop request fails, keep the output directory and retry cleanup with the same W&B key and team:

```bash
uv run --locked python run.py --cleanup --output outputs/parallel-4
```

Cleanup targets only sandbox IDs recorded in that directory. It reports errors instead of assuming a failed lookup means compute stopped. The 30-minute lifetime bounds compute if the controller can't stop it. Files that weren't downloaded before the sandbox stops are lost.

## Troubleshoot

Use the following checks to diagnose common failures:

- **Creation or inference fails:** Check the selected team's access, credits, and quota. Inspect the worker summary and logs. If inference returns HTTP `429`, reduce `--workers`.
- **The model reports success but verification fails:** Inspect the `result.json`, `history.json`, and `final.png` files. Treat the independent check as the outcome, not the agent's success message.
- **Model output is truncated:** The recipe allows 8,192 output tokens per call. If you change the model or task, review its output budget and the agent's step and time limits in the `agent.py` file.
- **Chromium won't launch:** Inspect the agent logs. The recipe requires a runtime that permits Chromium's process sandbox for the unprivileged user. It doesn't automatically disable that protection.

## Adapt the agent

In the `agent.py` and `verify.py` files, change the task, output schema, allowed domains, and verifier together. The domain allowlist is a Browser Use navigation control, not a network firewall. Browser Use sends page content and screenshots to the model provider. Local histories can contain page content. Review those data flows before using authenticated or sensitive sites.

The `requirements.txt` file pins the sandbox's Python dependencies for Linux x86-64 and Python 3.12. The `uv.lock` file pins the local dependencies. The Debian base-image tag and Chromium packages can change. If you need reproducible system packages, build and pin your own image.

Run the local checks without creating sandboxes:

```bash
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
```
