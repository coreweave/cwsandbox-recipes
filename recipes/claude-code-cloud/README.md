# Claude Code cloud sessions on CoreWeave Sandboxes

Give Claude a task in the [Claude Code web app](https://claude.ai/code). This recipe deploys an orchestrator in a CoreWeave sandbox that creates a worker sandbox for each queued session. Your laptop can disconnect after deployment. No `cws-agent` installation is required.

For architecture, configuration, lifecycle limits, and troubleshooting, see the [full setup guide](https://docs.coreweave.com/products/sandboxes/agents/claude-code-cloud).

## Prerequisites

Before you begin, make sure you have the following:

- A Claude Team or Enterprise organization and a connected GitHub test repository. An Owner must [enable and create a self-hosted environment](https://code.claude.com/docs/en/self-hosted-environments-quickstart). Keep its environment key and `ccpool_...` ID.
- A [CoreWeave API access token with sandbox permissions](https://docs.coreweave.com/products/sandboxes/placement#choose-and-get-credentials) and serverless quota. The orchestrator requests 1 CPU and 2 GiB. Each worker requests 2 CPUs and 4 GiB. The default limit is four workers.
- A Linux or macOS computer with Git, Python 3.11 or later, and [`uv`](https://docs.astral.sh/uv/getting-started/installation/). A local Claude installation is optional.

The orchestrator remains billable while idle. CoreWeave charges and Claude usage are separate.

## Quickstart

Deploy one orchestrator per Claude environment, verify a task, then stop the deployment.

### Configure credentials

Install the recipe:

```bash
git clone https://github.com/coreweave/cwsandbox-recipes.git
cd cwsandbox-recipes/recipes/claude-code-cloud
uv sync --locked
cp .env.example .env
chmod 600 .env
```

In the `.env` file, set `CWSANDBOX_API_KEY` to your CoreWeave token and `SELF_HOSTED_RUNNER_ENVIRONMENT_SECRET` to the Claude environment key. Keep this file private and out of version control. Existing environment variables take precedence.

### Deploy the orchestrator

Replace `[ENVIRONMENT-ID]` with your `ccpool_...` ID:

```bash
uv run cloud.py deploy --environment '[ENVIRONMENT-ID]'
```

Wait for `Connected`. If setup times out, inspect `uv run cloud.py status` and `uv run cloud.py logs --bootstrap` before retrying. Resources may still be running.

Keep the `.deployment.json` file to manage the deployment. Run subsequent commands from this recipe directory. By default, the orchestrator expires after 24 hours and isn't automatically replaced.

### Send a task and verify it

In the Claude Code web app, select your organization, repository, and self-hosted environment. To create the verification file, send exactly this prompt:

```text
Use Bash to write exactly CWS_CLOUD_OK followed by a newline to
/tmp/claude-cloud-proof.txt, then read it back and report the current
directory. Do not modify repository files, commit, or push.
```

List workers:

```bash
uv run cloud.py status
```

After Claude finishes, replace `[WORKER-ID]` with the sandbox ID from a `Worker:` line:

```bash
uv run cloud.py read --worker '[WORKER-ID]' --path /tmp/claude-cloud-proof.txt
```

Expect `CWS_CLOUD_OK`. Read it before the worker's default 15-minute idle timeout. If you sent a different task, read a file that task created instead.

### Stop the deployment

Save any results you want to keep, then stop the orchestrator and its workers:

```bash
uv run cloud.py stop
```

This interrupts running tasks, discards sandbox files, and removes `.deployment.json`. The Claude environment remains available. The recipe configures no snapshots or automatic pushes.

## Validate changes

Offline checks provision no resources:

```bash
uv run pytest -q
uv run ruff check cloud.py entrypoint.py tests
uv run ruff format --check cloud.py entrypoint.py tests
bash -n bootstrap.sh
```

Repeat the quickstart for a live test. Confirm the proof file, inspect logs, and run `stop`. To test idle cleanup, deploy with `--idle-minutes 1` and complete the proof task. Wait for its worker to disappear from `status` before stopping the orchestrator.

After changing dependencies, regenerate the lockfile and the orchestrator's installation requirements:

```bash
uv lock
uv export --frozen --no-dev --no-emit-project --no-hashes --output-file requirements.txt
```

For additional runner diagnostics, see [Anthropic's end-to-end testing guide](https://code.claude.com/docs/en/self-hosted-environments-testing).
