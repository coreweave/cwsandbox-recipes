# Investigate an inference incident with OpenAI Agents API

Run a coordinator and three built-in OpenAI subagents against synthetic service
telemetry in one CoreWeave sandbox. Download their analysis and verify both the
metrics and evidence of delegation.

This original example examines a latency and error regression after a fictional
inference-service deployment. It uses no customer data, internal service details,
or copied provider-example dataset.

## Investigation overview

| Agent | Investigation | Owned outputs |
| --- | --- | --- |
| Latency specialist | Compare request latency across regions and periods | `latency.py`, `latency.json` |
| Error specialist | Compare failures across regions and periods | `errors.py`, `errors.json` |
| Capacity specialist | Compare utilization and queueing with deployment timing | `capacity.py`, `capacity.json` |
| Coordinator | Combine the evidence and answer a follow-up | `incident.md`, `followup.md` |

The dataset contains two regions, `east` and `west`. At 15:30 UTC, `west` moves
from `deploy-a` to `deploy-b`. The `east` region remains on `deploy-a`. Investigate what changed,
which hypotheses fit the observations, and what evidence is still missing.
Don't present temporal association as proof of causation.

## Architecture

The recipe enables three concurrent subagents. The harness supplies delegation
tools. Each specialist writes separate files. All tool execution uses the same
sandbox. This demonstrates delegation within one execution environment, not a
fleet of sandboxes or a distributed inference benchmark.

The verifier checks actual agent activity. It reports overlapping subagent turns
only when the API supplies timestamps that establish overlap. This is evidence of
concurrent agent work, not simultaneous shell processes or measured speedup.

## Prerequisites and cost

Meet these requirements before running the recipe:

- Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).
- A [sandbox credential](https://docs.coreweave.com/products/sandboxes/get-started#choose-a-credential):
  a CoreWeave API access token with the `SANDBOX_USER` action, or a W&B API key.
- Capacity for one sandbox: 2 CPUs and 4 GiB memory. No GPU is required.
- An OpenAI project with Agents API access and access to the selected model.
- An OpenAI application key and a separate
  [environment key](https://platform.openai.com/agents?tab=environments&environment_view=keys).

Live runs consume OpenAI tokens and metered CoreWeave sandbox compute for a coordinator
and up to three concurrent subagents. OpenAI model charges and any applicable sandbox
charges are separate. Check the current [sandbox billing terms](https://docs.coreweave.com/products/sandboxes/get-started).

The sandbox lifetime is capped at 20 minutes.
The overall work deadline is 10 minutes, covering setup, both turns, and artifact
collection. Cleanup runs afterward.

These are duration bounds, not a
dollar spending cap. Actual model usage depends on the agent's work. The recipe
requests no GPU, persistent volume, or external storage bucket.

## Set up the recipe

Install the recipe's pinned dependencies and create its local credential file.

From the repository root, install the dependencies and copy the credential template:

```bash
cd recipes/openai-agents-api
uv sync --locked
cp .env.example .env
```

Choose the sandbox credential and complete the `.env` file. Keep it local and
gitignored. Use `CWSANDBOX_API_KEY` with the default `--sandbox-auth coreweave`, or
`WANDB_API_KEY` with `--sandbox-auth wandb`. Leave the unused sandbox key empty.
Both modes require the two OpenAI keys. You can also export these variables and
omit `--env-file` entirely.

When you pass `--env-file`, values defined in that file override exported values
for both `run` and `cleanup`, including the sandbox keys. Blank values also
override exports and fail preflight if required. Variables omitted from the file
retain their exported values. Without `--env-file`, only exported variables are used.

The application key requires `api.agents.read`, `api.agents.write`, and
`api.responses.write`. The executor key must match the session's organization,
project, and user or service-account ownership.

The recipe application uploads only the `TASK.md` file and the three fixture files. The application keys,
verifier, dependency files, and local result directories stay on your machine.

## Check the recipe locally

Check the application and verifier before provisioning resources for the investigation.

```bash
uv run pytest -q
uv run python run.py --help
uv run python verify.py --help
```

These checks exercise the fixture calculations, evidence validation, and application
failure handling without provisioning resources. They don't establish live API
compatibility or demonstrate model delegation.

## Run the investigation

Use a new output directory for every run.

```bash
uv run python run.py run --env-file .env --output-dir runs/incident-001
```

For W&B authentication, select it explicitly:

```bash
uv run python run.py run --sandbox-auth wandb --env-file .env --output-dir runs/incident-wandb-001
```

The application records the authentication mode in `run.json`. Cleanup uses that
saved mode, even when both sandbox keys are available.

To request CKS placement with CoreWeave authentication, run:

```bash
uv run python run.py run --sandbox-auth coreweave --sandbox-mode cks --env-file .env --output-dir runs/incident-cks-001
```

CKS requires an eligible runner for the selected credentials. W&B authentication
with CKS placement hasn't been validated by this recipe.
Use `--sandbox-mode serverless` to request serverless placement. If omitted,
placement follows the SDK and service defaults. Placement uses strict mode,
so a CKS request cannot fall back to serverless. The selected mode is saved in `run.json`.

The application creates the API session and sandbox, starts the executor, uploads
the fixtures, and confirms the connection before submitting `TASK.md`. After the
first turn, it asks a follow-up in the same session.

Bootstrap installs `@openai/codex@0.155.0-alpha.6`, the prerelease executor version
used in the validated live runs. Keep the exact pin for reproducibility. Before
changing it, verify `codex exec-server` support and rerun the full investigation,
verification, and cleanup flow.

It saves artifacts and activity evidence locally, then attempts to delete the API session and stop the
sandbox, including when a run fails.

To select another model available to your project, pass `--model MODEL`.
The recipe defaults to `gpt-6-astra`. Changing models may affect delegation and output
quality. Run the verifier again.

## Verify the result

Check the downloaded artifacts and API records to establish what the run demonstrated.

```bash
uv run python verify.py --output-dir runs/incident-001 --require-overlap
```

A successful demonstration meets all of the following criteria:

- Metrics matching independently calculated results from the synthetic fixtures.
- The required specialist scripts, metric files, and coordinator reports.
- API-completed Python command records attributed to three distinct built-in subagents.
- Completed coordinator turns for the investigation and follow-up.
- Timestamp evidence that at least two subagent turns overlapped.

Turn timestamps have whole-second precision. Sub-second overlap may not be visible.
If timestamps are absent or no overlap is established, the strict command fails
even if the report is correct. Rerun verification without `--require-overlap` to
inspect a result that demonstrates delegation but not verified parallelism. Don't relabel that result as a successful parallel run.

The verifier links each script command to a completed turn and its subagent ID.
It doesn't require plaintext task instructions, which the API can omit. Commands
must have API status `completed`; an explicit nonzero exit code fails validation.
When exit codes are absent, the result reports them as unknown. To additionally
require an observed zero exit code for every specialist, add `--require-exit-codes`.
That stronger check fails if the API omits exit codes, even when the artifacts,
completed command records, and overlap checks pass.

The verifier runs locally against supplied data and downloaded evidence. It doesn't execute the agent's generated scripts on your computer. Read the report to
assess its reasoning: numerical validation can't establish that every narrative
claim or recommended action is justified.

## Results

```text
runs/incident-001/
  run.json                  # resource journal and cleanup status
  events.jsonl              # captured API activity
  subagents.json            # saved child-agent records
  turns.json                # saved turn records
  items.json                # saved root and child activity
  output/
    latency.py
    latency.json
    errors.py
    errors.json
    capacity.py
    capacity.json
    incident.md
    followup.md
```

Expect the analysis to identify the regression in `west` after `deploy-b`, compare
it with the stable `east` region, and distinguish correlation from causation.
The follow-up should propose a bounded validation experiment with evidence needed
to accept or reject the deployment hypothesis. All actions remain proposals.
This recipe doesn't modify a real service.

The exact prose varies between runs. When sharing a run for review, keep `events.jsonl`, `turns.json`,
`items.json`, `subagents.json`, and `run.json` with the artifacts.
Inspect files before publishing them. Live records are gitignored.

## Cleanup and recovery

Normal execution attempts both API-session deletion and sandbox stop. Review
the `run.json` file for their individual outcomes. Confirm `session_deleted` and
`sandbox_stopped` are `true` and `cleanup_errors` is empty. An artifact directory
alone doesn't prove cleanup succeeded.

If the process was interrupted or cleanup failed, clean up the recorded resources:

```bash
uv run python run.py cleanup --env-file .env --output-dir runs/incident-001
```

This targets the resources recorded for that run. Supply the sandbox credential
for the saved authentication mode and `OPENAI_API_KEY`. It doesn't delete local
artifacts. Check the journal again afterward.

After a forced process stop during resource creation, run cleanup and inspect
provider resources. Cleanup can discover a sandbox by the unique tag saved before
creation. It can't delete an API session whose ID was never saved. The sandbox's maximum
lifetime limits compute duration, but doesn't delete an orphaned API session.

The recipe creates no snapshots, stored agents, volumes, buckets, or webhook controllers.
The follow-up runs before cleanup. This example can't resume a deleted session.

## Troubleshooting

| Symptom | Next step |
| --- | --- |
| Missing credentials | Complete the `.env` file with the selected sandbox key and both OpenAI keys, then pass `--env-file .env` |
| OpenAI reports a usage or billing limit | Check the OpenAI organization and project billing limits before retrying. Successful sandbox provisioning does not establish model access |
| Sandbox bootstrap or executor launch fails | Read the exception's stage, exit code, and redacted stdout/stderr tails. Bootstrap failures can occur before `executor.log` exists |
| Executor fails to connect | Check environment-key ownership and outbound network access. Inspect the saved `executor.log` if available; it can be empty and is not proof of a successful connection |
| No eligible runner is available | Check runner availability and credential access for the requested placement. CKS requests don't fall back to serverless |
| Timed-out or interrupted run | Read the `run.json` file. Run cleanup before starting a new run |
| Wrong metrics | Compare the relevant specialist's script with the fixture schema in the `TASK.md` file |
| Missing subagent evidence | Inspect saved events, child turns, and command items. A prompt isn't evidence |
| No completed Python execution attributed to a specialist | Inspect command items. The verifier accepts only direct script commands or one shell wrapper. A compound command such as `cd /workspace/project && python3 output/latency.py` can fail verification despite executing the script |
| Ambiguous specialist attribution | Check which subagent owns each script command. Each specialist must map to one distinct subagent ID |
| Zero exit codes could not be established | Inspect `exit_code_evidence`. The optional `--require-exit-codes` check fails when numeric exit codes are unavailable |
| No verified overlap | Inspect actual turn timestamps. Concurrency configuration is only a limit |

For broader recovery policies, see OpenAI's
[sandbox lifecycle guide](https://developers.openai.com/api/docs/guides/agents-api/environments/lifecycle).

## Sources and scope

The workflow and fixtures were written for this recipe. Provider guides informed
the documentation structure, not the incident content. The API contract follows
OpenAI's [self-hosted guide](https://developers.openai.com/api/docs/guides/agents-api/environments/self-hosted)
and [multi-agent guide](https://developers.openai.com/api/docs/guides/agents-api/multi-agent).

This recipe uses application-managed provisioning. It includes no web UI,
webhook deployment, Agents SDK (software development kit) provider, or ChatGPT or Codex app integration.
