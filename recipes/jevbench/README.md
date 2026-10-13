# JevBench on CoreWeave Sandbox

Run JevBench's public original dataset with its Python harness and scorer inside a CPU sandbox. Claude Haiku 4.5 generates decisions through Anthropic's API. No GPU or model server is required.

The recipe requests one serverless sandbox with 2 CPUs and 4 GiB memory. Budget about **$0.01 to $0.03 in sandbox compute** for a 72-task run, assuming 2 to 5 minutes of billable runtime. At the listed rates of $0.128 per CPU-hour and $0.0212 per GiB-hour, this configuration costs `(2 × $0.128 + 4 × $0.0212) = $0.3408/hour`. Runtime includes time waiting for model responses. Longer runs cost proportionally more. See [sandbox pricing](https://www.coreweave.com/forge-pricing/inference-agents) for rates and billing details.

Model inference is billed separately. A 72-task validation run recorded about 104 seconds from provisioning through cleanup and used about $0.05 of inference. This elapsed time includes controller overhead and is not a measurement of billable sandbox runtime. The default $2 inference ledger cap excludes sandbox compute charges. Check [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing) before running.

## What this recipe demonstrates

- Stage a pinned benchmark in a disposable sandbox.
- Run measurement and scoring remotely while using a hosted model.
- Retrieve predictions, raw responses, summary metrics, and provenance.
- Verify response hashes and stop the sandbox after success or failure.

This evaluates the 72 tasks in JevBench's **public original cohort**, or a smaller smoke sample. It does not reproduce the official held-out leaderboard or its current composite score.

## Architecture

```text
Local Python controller
  ├─ Fetch pinned public JevBench source with Git
  ├─ Create CPU sandbox and upload harness
  │    └─ JevBench → native Anthropic Messages API → JevBench scorer
  └─ Download results and stop sandbox
```

The [upstream revision](https://github.com/fstandhartinger/jevbench/tree/bb05a335bc809e61b20c0f745d25499a82b326fc) supplies tasks, prompts, output parsing, scoring, and budget accounting. `transport.py` translates the HTTP envelope to Anthropic's native structured-output API, preserving the upstream prompt and JSON schema. It preserves native requests and responses as evidence and does not repair probabilities. The upstream repository remains unchanged. The container image, `python:3.12-slim-bookworm`, uses a mutable tag, so Python patch versions and OS packages can change between runs. Pin `IMAGE` in `run.py` to an image digest if you need an identical container filesystem.

The sandbox receives only `ANTHROPIC_API_KEY`. The controller uses `CWSANDBOX_API_KEY` with explicit CoreWeave authentication. Sandbox compute is billed to the CoreWeave organization associated with that token. Public task prompts are sent to Anthropic; answer keys remain with the scorer in the sandbox.

## Prerequisites

Follow [Get started with serverless sandboxes](https://docs.coreweave.com/products/sandboxes/get-started) to configure access and run your first sandbox.

- Python 3.12 or later, [uv](https://docs.astral.sh/uv/getting-started/installation/), and Git.
- A [CoreWeave API access token](https://docs.coreweave.com/products/sandboxes/get-started) with sandbox access and quota for one 2-CPU, 4-GiB serverless sandbox.
- A funded [Anthropic API account](https://platform.claude.com/) with access to `claude-haiku-4-5-20251001`.
- Network access from your machine to GitHub and the sandbox API, and from the sandbox to `api.anthropic.com`.

## Set up

1. Clone the recipes repository and install the locked dependencies:

   ```bash
   git clone https://github.com/coreweave/cwsandbox-recipes.git
   cd cwsandbox-recipes/recipes/jevbench
   uv sync --locked
   ```

2. Copy the credential template and fill in both values in your editor:

   ```bash
   cp .env.example .env
   ```

   `run.py` loads `.env` from the recipe directory. Existing environment variables take precedence. The repository ignores `.env` and the default `outputs/` directory.

## Run

1. Run two tasks to check credentials, model access, execution, and artifact retrieval:

   ```bash
   uv run --locked python run.py --limit 2 --output outputs/smoke
   ```

   A successful run prints `"attempted": 2`, `"complete": true`, and `Sandbox stopped.`. A failed model request, an unparseable response, or an incomplete run returns a nonzero exit code. Incorrect model answers remain valid benchmark outcomes.

2. Run the full public original cohort:

   ```bash
   uv run --locked python run.py --output outputs/full
   ```

   Expected completion is 72 attempted tasks with zero request or response-parsing failures. Accuracy varies by model response. Every run requires a new output directory; the recipe refuses to overwrite existing results.

The default model is pinned to `claude-haiku-4-5-20251001`, with temperature 0 and a 4,096-token output limit. The runner is serial and doesn't retry model requests. Responses the adapter can't parse, such as truncated JSON, count as failed tasks. Three consecutive failed tasks stop the run. `--limit` accepts 1 through 72. `--budget-usd` sets the upstream inference ledger cap; reservations can stop a run before its actual spend reaches that cap. A budget-stopped run retains its partial results and exits nonzero. Failed requests may retain estimated reservations, so ledger totals are not invoices.

Changing the model requires updating `MODEL` and the input/output tariffs in `evaluate.py`, then validating its native structured-output support. The recipe supplies $1/M input and $5/M output for Haiku 4.5; derived cost excludes sandbox compute and may differ from your bill.

## Inspect results

```bash
cat outputs/full/output/verification.json
cat outputs/full/output/summary.json
```

| File | Contents |
| --- | --- |
| `run.json` | Source revision and hash, model, resources, limits, sandbox ID, and cleanup status |
| `source.tar.gz` | Pinned upstream source, public tasks, and upstream MIT license |
| `evaluation.log` | Harness output, summary, and verification result |
| `output/summary.json` | Accuracy, schema validity, calibration, latency, and estimated cost |
| `output/results.jsonl` | Per-task predictions, status, scores, usage, and raw-response hashes |
| `output/raw/` | Raw requests and responses, including the native Anthropic exchange |
| `output/tasks.jsonl` | Exact public cohort selected for this run |
| `output/manifest.json` | Upstream run settings and dataset hash |
| `output/ledger.jsonl` | Request reservations and settlements |
| `output/verification.json` | Coverage, failed requests, valid answers, correctness, and hash checks |

The two-task sample is for checking execution, not estimating model quality. The full public cohort can have been seen during training and is not interchangeable with the held-out benchmark. Latency includes the inference API round trip; this is not a CPU performance benchmark.

## Clean up

The controller retrieves available results, then calls `stop()` in a `finally` block. Check `"sandbox_stopped": true` in `run.json`. Each sandbox also has a 1,200-second maximum lifetime. The controller caps evaluation at 900 seconds and shortens it after slow setup to reserve 300 seconds for result retrieval and cleanup. At the listed rates, a sandbox left running for the full 20 minutes costs about $0.11 in compute, excluding inference. No persistent volumes or model deployments are created.

If the controller is forcibly terminated or cleanup fails, stop the recorded sandbox manually. Run from the recipe directory. The command loads `.env` if present and preserves exported credentials. Replace `[SANDBOX-ID]` with the ID in that run's `run.json`:

```bash
uv run --locked python - <<'PY'
from pathlib import Path

from cwsandbox import AuthStrategy, Sandbox
from dotenv import load_dotenv

load_dotenv(Path.cwd() / ".env")
sandbox = Sandbox.from_id(
    "[SANDBOX-ID]", auth=AuthStrategy.COREWEAVE_API_KEY
).result()
sandbox.stop(missing_ok=True).result()
PY
```

## Troubleshooting

- **Setup fails:** Check that both credentials are set, Git can reach GitHub, and the output directory does not exist.
- **HTTP 401, 403, or 429:** Inspect `evaluation.log` and `output/raw/`. Check provider access, credits, and rate limits before starting another run. The upstream runner stops on these statuses.
- **Partial run:** Inspect `output/verification.json` and the ledger. Budget stops, request failures, response-parsing failures, and timeouts are separate from model accuracy.
- **Artifact or cleanup error:** Inspect `artifact_error` or `cleanup_error` in `run.json`. Stop any remaining sandbox using the command above.

## Local checks

These checks do not provision sandboxes or call model APIs:

```bash
uv sync --locked --group dev
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest -q
```

JevBench is downloaded at runtime under its [MIT license](https://github.com/fstandhartinger/jevbench/blob/bb05a335bc809e61b20c0f745d25499a82b326fc/LICENSE). Its source and dataset are not vendored in this recipe.
