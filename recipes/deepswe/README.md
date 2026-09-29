# Run DeepSWE v1.1 on CoreWeave Sandbox

Evaluate a coding agent on a small DeepSWE v1.1 sample. CoreWeave Sandbox
provides the environments where the agent edits code and the benchmark grades
its patches. Pier coordinates the tasks and separate verifier workflow.

The example runs mini-swe-agent with `moonshotai/Kimi-K2.7-Code` through W&B
Serverless Inference. This is the sample's model provider, not a requirement of
DeepSWE or Sandbox. The supplied recipe fixes the W&B endpoint and credential
handling in `host_agent.py`. The `--model` option selects another model on that
endpoint; another provider requires changes to the model configuration and
authentication.

The local Python process calls the model and sends its shell commands to an agent
sandbox. After the agent finishes, Pier collects its committed changes, stops that
sandbox, and applies the patch in a fresh verifier sandbox. Only the verifier
receives the held-out tests. Both sandboxes deny outbound network access. The
inference and sandbox API credentials stay on the local runner.

This is an integration sample, not a leaderboard reproduction. It uses a host-side
adapter for mini-swe-agent, a 160-step limit, and a 30-minute agent timeout per
task. Upstream tasks allow longer runs. The adapter uses the task's CPU and memory
settings, but doesn't enforce its `storage_mb` setting.
Shell commands have a 120-second timeout. A command timeout is returned to the
agent as an observation so it can continue; a stopped sandbox remains a run error.

## Prerequisites

This recipe requires the following:

- Python 3.12 or later, [`uv`](https://docs.astral.sh/uv/getting-started/installation/), and Git.
- A [CoreWeave API access token](https://docs.coreweave.com/products/sandboxes/placement#coreweave-api-access-token)
  with the Sandbox User role and serverless sandbox access. The selected tasks each request 2 CPUs and 8 GiB
  of memory. You don't need a GPU or local Docker installation.
- A [W&B API key](https://wandb.ai/authorize) with access to
  Serverless Inference.
- Outbound access from your local runner to the Sandbox API and
  `https://api.inference.wandb.ai/v1`.

Pier manages concurrency and trial retries. By default, the sample runs one task
at a time. Each task uses an agent sandbox followed by a verifier sandbox.

Sandbox runtime is metered, and inference uses model tokens. Check the current
[sandbox billing terms](https://docs.coreweave.com/products/sandboxes/placement)
and [inference rates](https://docs.wandb.ai/inference/usage-limits/) for your account.
Step and time limits bound the run, but aren't a dollar budget. Token usage is
saved locally but can be incomplete: responses rejected for missing tool calls
can consume tokens without retaining their usage in the trajectory. Treat costs
calculated from saved counts as partial estimates. W&B billing is the source for
actual inference charges.

For cost planning, multiply uncached input, cached input, and output token counts
by the corresponding [model rates](https://site.wandb.ai/inference-model/moonshot-ai-kimi-k2-7-code/).
The saved input count includes cached tokens, so subtract cached tokens before
applying the uncached input rate. Preflight requests and sandbox charges are separate.

## 1. Install the recipe

From this directory, install the locked dependencies and fetch the tested task
revision:

```bash
uv sync --locked
git clone https://github.com/datacurve-ai/deep-swe.git
git -C deep-swe checkout 0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea
```

The lockfile pins Pier 0.3.1, mini-swe-agent 2.2.6, and `cwsandbox` 1.14.2. This
DeepSWE revision uses v1.1 task images and `[[verifier.collect]]` hooks.

## 2. Configure the example model and credentials

Export both keys from your secret manager, or use hidden shell prompts.
In Bash, run these prompts:

```bash
read -r -s -p 'CoreWeave Sandbox API key: ' CWSANDBOX_API_KEY
echo
export CWSANDBOX_API_KEY
read -r -s -p 'W&B API key: ' WANDB_API_KEY
echo
export WANDB_API_KEY
export MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=3
export MSWEA_SILENT_STARTUP=1
```

The recipe explicitly selects CoreWeave authentication
for sandboxes and uses the W&B key only for inference. It doesn't load the `.env` file
automatically. The `.env.example` file lists the available variables.

Before you start a task, check that you can access the selected model:

```bash
uv run --locked python - <<'PY'
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.inference.wandb.ai/v1",
    api_key=os.environ["WANDB_API_KEY"],
    timeout=60,
)
response = client.chat.completions.create(
    model="moonshotai/Kimi-K2.7-Code",
    messages=[{"role": "user", "content": "Respond with exactly OK."}],
    max_tokens=64,
)
print(response.choices[0].message.content)
PY
```

A successful request returns a short response such as `OK`. Before you run the
sample, resolve any authentication, model-access, or credit errors.

## 3. Run the sample

For an initial integration check, run one task:

```bash
uv run --locked python run.py \
  --tasks deep-swe/tasks --task tomlkit-toml-table-converters \
  --steps 160 --agent-timeout 1800 --job-name kimi-smoke
```

A one-task run can still use most of the step or time allowance. A completed
agent-and-verifier workflow validates the integration even when the model doesn't
solve the task.

To expand the sample, run both tasks with a new job name:

```bash
uv run --locked python run.py \
  --tasks deep-swe/tasks \
  --task abs-stepped-slices \
  --task tomlkit-toml-table-converters \
  --job-name kimi-sample
```

The runner reports sandbox creation, completed model steps, and sandbox shutdown.
The agent works on the original task prompt and must commit its changes: the
upstream collection hook extracts the difference between the base commit and
`HEAD`. Uncommitted edits aren't part of the submitted patch.

Use a new job name for a fresh run. Pier can resume an existing job directory.
Reusing a name isn't a clean rerun.

To select a deterministic random sample instead, omit `--task`:

```bash
uv run --locked python run.py \
  --tasks deep-swe/tasks --n-tasks 2 --seed 0 \
  --model moonshotai/Kimi-K2.7-Code --steps 160 --agent-timeout 1800 \
  --job-name kimi-random-sample
```

Other task images and verifier layouts may have requirements beyond the two-task
sample. The adapter rejects verifier Dockerfiles it can't reproduce. Model
availability can change. `client.models.list()` lists models available through
the inference endpoint.

### Configure concurrency and retries

Use `--concurrency` to set Pier's maximum number of concurrent trials. Start with
one and verify the complete agent-and-verifier workflow. Then increase concurrency
gradually within your inference provider's allowance. A successful model-access check doesn't
validate capacity for concurrent requests. To run the two sample tasks concurrently,
use this example:

```bash
uv run --locked python run.py \
  --tasks deep-swe/tasks \
  --task abs-stepped-slices --task tomlkit-toml-table-converters \
  --concurrency 2 --max-retries 2 --job-name kimi-parallel-sample
```

The recipe uses Pier's `Job` and `RetryConfig` to schedule and retry trials.
`--max-retries 2`, the default, allows two additional executions after a
`RateLimitError`, `APIConnectionError`, or `ServiceUnavailableError`. Pier waits
30 seconds before the first retry and 60 seconds before the second.

Each retry starts a fresh agent attempt and can incur additional inference and sandbox
charges. A retried trial creates another agent sandbox and, if it reaches grading,
a verifier sandbox. The total number of sandboxes created can exceed `--concurrency`,
which limits active trials. Step and agent-time limits apply separately to each
execution. To disable trial retries, set `--max-retries 0`.

Authentication errors, other unlisted exceptions, and low verifier scores don't
trigger automatic retries. `SandboxFailedError` isn't included in this inference
retry policy. Inspect sandbox startup errors separately from inference throttling.

The recipe preserves unsuccessful attempts under
`jobs/[JOB-NAME]/failed-attempts/[TRIAL-NAME]/[ATTEMPT-ID]/` through a Pier completion
hook, because Pier 0.3.1 deletes the trial directory before retrying it. The normal
trial directory and job summary contain the final execution's result.

Pier 0.3.1 limits whole trials. This recipe has no separate inference-request
concurrency limit or automatic concurrency adjustment. Sandbox capacity and
inference capacity are independent. When inference is the bottleneck, lower
`--concurrency` and account for load from other jobs using the same account.

### Recover from inference throttling

An HTTP `429` response with `rate_limit_exceeded` and `concurrency limit reached for requests` means the
inference provider rejected a model request. W&B applies concurrency limits per
project and per user. Other jobs using the same account can consume that allowance,
even when this recipe runs one task at a time.

The example's `MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=3` setting allows up to three
mini-swe-agent attempts per failed model query, including the initial attempt.
These request retries continue the same agent conversation if a request succeeds.
The model client can also retry within an attempt, so this setting isn't a limit
on physical API requests. The agent timeout includes time spent on request retries.
If request retries are exhausted with an eligible exception, Pier applies the
trial retry policy in [Configure concurrency and retries](#configure-concurrency-and-retries).

A job can recover some throttled trials while others exhaust their retries. Pier's
backoff doesn't lower concurrency, so another burst of retries can hit the same
limit. If retries are exhausted, the trial retains its final exception. The job
then continues with other tasks, and the runner exits with a nonzero status. These are
failed trials, not skipped tasks or model scores.

If a trial still fails after its retry allowance, follow these steps:

1. To distinguish throttling from an authentication, credit, or other service
   error, read its `exception_info` field in the `result.json` file.
2. Reduce `--concurrency` and simultaneous inference requests from other jobs.
   Increasing retries alone doesn't resolve sustained overload. For a higher
   allowance, contact support through the [inference limits documentation](https://docs.wandb.ai/inference/usage-limits/#concurrency-limits).
3. After you resolve the limit, rerun the affected task with a new job name:

```bash
uv run --locked python run.py \
  --tasks deep-swe/tasks --task tomlkit-toml-table-converters \
  --concurrency 1 --max-retries 2 --job-name kimi-throttle-retry
```

Use an unused job name and repeat the `--task` option to select other affected tasks. This
starts new agent attempts. It doesn't resume their conversations or regrade saved
patches. Preserve the original job directory to compare the results.

When the final error differs from the first failure, inspect the `failed-attempts/` directory.
A trial that was throttled can encounter a sandbox startup failure on its next execution.

Pier also supports retrying only trials with a selected error in an existing job.
The following command deletes matching trial directories and uses the job's saved
concurrency and retry settings. It doesn't run the recipe's diagnostic-preservation
hook, and it doesn't reduce concurrency. Before you run this command, back up the
job directory. Set `PYTHONPATH=.` so Pier can import the recipe's local adapters.

```bash
PYTHONPATH=. uv run --locked pier job resume -p jobs/kimi-parallel-sample -f RateLimitError
```

If you need to lower concurrency or preserve each new unsuccessful attempt
automatically, use a fresh recipe run.

## 4. Inspect the results

Results are under `jobs/[JOB-NAME]/`, where `[JOB-NAME]` is the job name. Each task directory contains:

| File | Purpose |
| --- | --- |
| `result.json` | Trial status, timings, token counts, and verifier result |
| `agent/mini.trajectory.json` | mini-swe-agent conversation and tool calls |
| `agent/trajectory.json` | Conversation in Pier's Agent Trajectory Interchange Format (ATIF) |
| `artifacts/model.patch` | Changes collected from the agent's commits |
| `verifier/reward.json` | Binary reward and partial scores |
| `verifier/ctrf.json` | Structured test results |
| `verifier/test-stdout.txt` | Verifier output |

In the top-level `jobs/[JOB-NAME]/result.json` file, `stats.n_completed_trials`
counts finished trials, including errors. `stats.n_errored_trials` counts trials
whose final execution ended with an exception, and `stats.n_retries` counts extra
trial executions. A recovered trial can therefore have archived errors while its
final result has no exception. Retry executions aren't additional benchmark samples.

Before you interpret scores, check each trial's `exception_info` and
`verifier_result`. A job can display an aggregate mean of `0` when verifier startup
failed and no patch was graded. A completed trial with reward `0` is a model failure on the
task, not necessarily a broken integration. An exception, missing reward, or reward of `-1` is a
run error. A two-task result doesn't estimate performance across all 113 tasks.

Inspect the held-out results even when the agent reports that its own tests pass.
In `reward.json`, `f2p_passed` and `f2p_total` count new-behavior checks, and
`p2p_passed` and `p2p_total` count existing checks. A high `partial` score can still
have binary `reward: 0` if any required check fails.

If verifier startup fails after patch collection, preserve `artifacts/model.patch`
and the original error result. This recipe has no supported verifier-only retry
command. Starting a new model run produces a new attempt, not a regrade. Any
separate regrade must use a byte-identical patch and matching task revision, save
its result separately, and keep the original infrastructure error for provenance.

For a verifier control, run the same task with no edits and then with its reference
solution. These controls don't call the model and must remain separate from
model results:

```bash
uv run --locked python run.py \
  --tasks deep-swe/tasks --task abs-stepped-slices \
  --agent nop --job-name nop-control
uv run --locked python run.py \
  --tasks deep-swe/tasks --task abs-stepped-slices \
  --agent oracle --job-name oracle-control
```

The expected binary rewards are `0` for `nop` and `1` for `oracle`.

## 5. Clean up

Pier stops agent and verifier sandboxes on normal completion and during error
cleanup. Each sandbox also has a 90-minute maximum lifetime. If the local runner
is stopped before cleanup finishes, replace `[SANDBOX-ID]` with the sandbox ID from its logs:

```bash
uv run --locked python - <<'PY'
from cwsandbox import AuthStrategy, Sandbox

sandbox = Sandbox.from_id(
    "[SANDBOX-ID]", auth=AuthStrategy.COREWEAVE_API_KEY
).result()
sandbox.stop(missing_ok=True).result()
PY
```

If you need the patch or test reports, keep the job directory. This recipe doesn't
create a GPU deployment, model endpoint, or persistent volume.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Inference returns an authentication or permission error | Check the W&B key, model access, and inference credits. Sandbox credentials are separate. |
| Inference returns HTTP `429` or `RateLimitError` | Read the provider's error message. For concurrency errors, reduce simultaneous model requests and follow [Recover from inference throttling](#recover-from-inference-throttling). If request and trial retries are exhausted, the final exception remains a run error, not a score. |
| Sandbox creation fails | Check CoreWeave authentication, serverless access, and CPU and memory quotas. |
| Sandbox fails before the first model step | Check the sandbox's status for an image-pull failure. Retry with a new job name after a transient registry or platform network failure. This is an infrastructure error, not a model score. |
| Verifier startup fails after the agent submits | Preserve the patch and error result. An aggregate mean of `0` is not a verified score. See the regrade limitations above before starting another model run. |
| Model response has no tool call | The agent can recover through format-error feedback within its step and time limits. Inspect its final status and held-out results. Saved token usage can omit the rejected response. |
| `Unsupported verifier Dockerfile` | Use the pinned dataset revision and a supported task. Additional Dockerfile operations need adapter support. |
| Empty `model.patch` file | Check the trajectory for committed changes. The upstream hook collects commits only. |
| `LimitsExceeded` or agent timeout | Inspect the saved trajectory. Before you increase `--steps` or `--agent-timeout`, review runtime and token usage. |
| Dependency installation fails inside a task | Task sandboxes have no outbound network. Use the upstream prebuilt dependencies or investigate the task image. |

## Sources and local checks

The following sources describe the task format and dependencies:

- [DeepSWE task format and separate verification](https://github.com/datacurve-ai/deep-swe).
- [Pier](https://pypi.org/project/datacurve-pier/0.3.1/)
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)
- [Sandbox Python software development kit (SDK)](https://github.com/coreweave/cwsandbox-client).
- [Serverless Inference](https://docs.wandb.ai/inference)

Run the local tests and lint checks without cloud credentials:

```bash
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
```
