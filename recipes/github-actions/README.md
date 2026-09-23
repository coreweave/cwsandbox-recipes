# GitHub Actions checks in CoreWeave sandboxes

Run a public repository's tests in a fresh CoreWeave sandbox and return the result to GitHub Actions. An optional Anthropic review summarizes the diff and test output.

Each run requests **1 CPU and 2 GiB of memory**, with a **15-minute maximum lifetime**. No GPU or customer-managed cluster is required. Budget for up to 0.25 CPU-hours and 0.5 GiB-hours per run; typical test runs stop sooner. Usage is metered under your CoreWeave account. See [serverless billing information](https://docs.coreweave.com/products/sandboxes/get-started#choose-a-credential) for current charges. AI review adds Anthropic API usage.

## What this recipe demonstrates

- Keep orchestration and credentials on the CI runner while repository code executes remotely.
- Fetch a specific commit, verify the checkout, and optionally compute a merge-base diff.
- Preserve test failures and timeouts in machine-readable results.
- Stop the sandbox after successful and failed runs.

## Architecture

```text
Local terminal or GitHub Actions runner
  run.py + CoreWeave credential
    ├── CoreWeave sandbox: Git checkout → dependencies → tests
    ├── optional Anthropic API: bounded diff + test output → advisory review
    └── outputs/result.json + outputs/summary.md → Actions artifact and summary
```

The sandbox receives no CoreWeave, GitHub, or Anthropic credentials. Git fetches use public HTTPS access. The model receives data and has no tools. The orchestrator never executes model responses or repository commands locally.

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.11 or newer. `uv` can install Python for you.
- A [CoreWeave API access token](https://console.coreweave.com/tokens) with the `SANDBOX_USER` IAM action and serverless capacity for 1 CPU and 2 GiB of memory.
- A public GitHub repository for repository mode. The bundled sample needs no repository access.
- Optional: an [Anthropic API key](https://console.anthropic.com/settings/keys) for `--review`.

This recipe explicitly uses CoreWeave authentication. A W&B key is not a substitute for `CWSANDBOX_API_KEY`.

## Setup

From this directory:

```bash
uv sync --locked
```

If `CWSANDBOX_API_KEY` is already exported, use it as-is. Otherwise, copy the environment template:

```bash
cp .env.example .env
```

In `.env`, uncomment `CWSANDBOX_API_KEY` and replace its placeholder with your token. Uncomment and set `ANTHROPIC_API_KEY` only if you need it for `--review`. Leave variables you already export commented out, then load the file:

```bash
set -a
source .env
set +a
```

Keep `.env` out of version control. The script does not load it automatically.

## Run

1. Run the bundled sample:

   ```bash
   uv run --locked python run.py
   cat outputs/result.json
   ```

   Expect `Status: passed`, `returncode: 0`, and `stopped: true`. The sample runs two tests: addition and absence of orchestrator credentials in the sandbox environment.

2. Verify that a failed command fails the local process and still stops the sandbox:

   ```bash
   uv run --locked python run.py \
     --command 'python -c "raise SystemExit(7)"' \
     --output outputs/failure
   ```

   Expect exit status `1`; `outputs/failure/result.json` records `status: failed`, `returncode: 7`, and `stopped: true`.

3. Test a public repository at a fixed commit. This example installs MarkupSafe and pytest inside the sandbox:

   ```bash
   uv run --locked python run.py \
     --repo pallets/markupsafe \
     --head b2e4d9c7687be25695fffbe93a37622302b24fb1 \
     --command 'pip install . pytest==8.4.2 && python -m pytest -q' \
     --timeout 300 \
     --output outputs/repository
   ```

   Expect `Status: passed`. For your project, replace the repository, full commit SHA, and command. Combine dependency installation and testing with `&&` so installation failures fail the check. The image is `python:3.12-bookworm`; change `container_image` in `run.py` for other toolchains.

4. Optional: repeat the repository command with an advisory AI review:

   ```bash
   uv run --locked python run.py \
     --repo pallets/markupsafe \
     --head b2e4d9c7687be25695fffbe93a37622302b24fb1 \
     --base adf9f3d76f1579444a9fbadabace6c2315e79408 \
     --command 'pip install . pytest==8.4.2 && python -m pytest -q' \
     --timeout 300 --review \
     --output outputs/review
   ```

   Expect a `review` field and an **Advisory AI review** section in `summary.md`. Both revisions must be reachable in the public repository. The script fetches their history and uses `git diff BASE...HEAD`, so it does not assume a branch called `main` or a shallow-clone merge base. Large histories can exceed the 120-second fetch timeout.

Review sends up to 24,000 characters each of the diff, stdout, and stderr to Anthropic. Review text is untrusted advisory output: inspect it before acting. A review API failure is recorded as `review_error` and leaves the test result unchanged. A test or infrastructure failure exits `1`; success exits `0`.

## Add to GitHub Actions

1. From the root of your public repository, copy only the runtime files. Replace `[RECIPES-CHECKOUT]` with the path to your `cwsandbox-recipes` clone:

   ```bash
   recipe_dir="[RECIPES-CHECKOUT]/recipes/github-actions"
   mkdir -p ci/sandbox .github/workflows
   cp "$recipe_dir/run.py" "$recipe_dir/pyproject.toml" \
     "$recipe_dir/uv.lock" "$recipe_dir/.gitignore" ci/sandbox/
   cp "$recipe_dir/workflow.yml" .github/workflows/sandbox-ci.yml
   ```

   Keep the recipe's `tests/` and `sample/` directories out of your project so `pytest` doesn't collect them. This runtime-only copy supports repository mode (`--repo` and `--head`); run the bundled sample from the original recipes checkout.
2. Review `.github/workflows/sandbox-ci.yml`.
3. Replace the workflow's `--command` with your project's dependency installation and test command. Its default uses Python's standard-library `unittest` runner.
4. Create a repository Actions secret named `CWSANDBOX_API_KEY` containing the CoreWeave token.
5. Merge the orchestration files and workflow into the PR's base branch before opening a test PR. The workflow deliberately reads these files from `github.event.pull_request.base.sha`.
6. Open or update a same-repository PR. Inspect the **Sandbox CI** check's job summary and download the **sandbox-ci-results** artifact.

The workflow runs on `opened`, `synchronize`, and `reopened` pull request events. It skips fork PRs, Dependabot PRs, and private repositories. GitHub does not supply repository secrets to fork or Dependabot pull request workflows. A skipped job satisfies a required status check, so requiring this check does not enforce tests for the skipped PRs. See [GitHub required-check behavior](https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks#handling-skipped-but-required-checks). Use this template for trusted repository contributors. Do not enable it for fork code by switching to `pull_request_target`.

The supplied workflow checks out `run.py` and its dependency files from the base commit. The workflow definition itself comes from the PR merge ref: a same-repository PR can change the workflow, test command, checkout ref, and use of secrets. The base checkout is not protection against a collaborator who can change workflows. With the supplied workflow, the PR head is fetched and executed inside the sandbox. The workflow tests the head commit itself, not GitHub's synthetic merge commit. It requests `contents: read`, disables persisted checkout credentials, and does not post PR comments.

To enable AI review in the workflow, add an `ANTHROPIC_API_KEY` repository secret, expose it only on the **Run tests in a sandbox** step, and add `--review` to that step's command. The default review model is `claude-sonnet-5`, with thinking disabled to reserve the output budget for advisory text. Optionally set `SANDBOX_REVIEW_MODEL` to a model your account can use; this recipe does not read or set `ANTHROPIC_MODEL`.

## Results and limits

`result.json` includes the tested SHA, sandbox ID, test return code, the last 24,000 characters each of stdout and stderr, and cleanup outcome. `summary.md` escapes returned text for display. In Actions, the same summary is appended to `GITHUB_STEP_SUMMARY`; artifacts upload even when tests fail and expire after seven days. Test output is not printed to the runner console, preventing returned `::error::` or other workflow commands from being interpreted there. On failure in Actions, the script emits a fixed-text error annotation directing you to the summary and artifact.

The test command times out after 120 seconds by default (`--timeout` accepts 1–300). Setup commands each have a 120-second timeout, startup polling allows 180 seconds, and the sandbox lifetime is capped at 900 seconds. The runner job has a 20-minute timeout. These are separate limits.

This example buffers SDK command results before truncating stored output. A timed-out command is recorded as `status: error` with `error: SandboxTimeoutError`; partial command output is not captured. Use it for small repositories and bounded-output test suites. It does not install Git LFS objects, initialize submodules, provide private package credentials, or enforce a sandbox network allowlist. Tests can access the network allowed by serverless policy. An attacker can falsify their own test output; a passing check is not proof that code is safe.

## Cleanup

`run.py` calls `stop(missing_ok=True)` in `finally`, including after test failures and timeouts. The 15-minute maximum lifetime bounds execution if the orchestrator is forcibly killed before cleanup. No snapshots, volumes, or runners are created.

If `cleanup_error` is present, use the saved sandbox ID to delete it with the same CoreWeave credential:

```bash
uv run --locked python - <<'PY'
import json
from cwsandbox import AuthStrategy, Sandbox

with open("outputs/result.json") as stream:
    sandbox_id = json.load(stream)["sandbox_id"]
Sandbox.delete(
    sandbox_id, auth=AuthStrategy.COREWEAVE_API_KEY, missing_ok=True
).result()
PY
```

Adjust the result path if you used `--output`. Keep the result files you need, then remove the local `outputs/` directory. Remove the workflow or disable it when you no longer want PRs to provision sandboxes.

## Development

These checks run without cloud credentials or sandbox creation:

```bash
uv sync --locked
uv run python -m pytest -q
uv run ruff check .
uv run ruff format --check .
```
