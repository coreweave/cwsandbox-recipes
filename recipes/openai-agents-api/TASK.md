# Investigate a synthetic inference incident

All inputs in `fixtures/` are invented for this example. They contain no customer
data or production telemetry. Work from these files only; do not use the network.
Keep inputs unchanged. Use Python's standard library.

Create three real subagents before waiting for their results. Put exactly one of
the following markers in each delegated task: `SPECIALIST=latency`,
`SPECIALIST=errors`, `SPECIALIST=capacity`. Give each subagent only its own output
files to write. Do not substitute three root-agent analyses for delegation.

Each specialist must write `output/<specialist>.py`, run it with
`python3 output/<specialist>.py` as a separate shell invocation, and produce
`output/<specialist>.json`. Run that exact command without extra shell statements
so API history can attribute its exit status to this script.
Scripts must read the supplied fixtures and calculate their results. Do not
hard-code metrics. All paths are relative to the workspace root. Scripts should
be rerunnable and must not modify another specialist's files.

## Shared calculation rules

Read the cutoff from `fixtures/deployments.json`. A timestamp strictly before the
cutoff belongs to `before`; all others belong to `after`. Calculate separate
groups for each `(window, region)`, sorted lexicographically by window then region
(`after` before `before`, and `east` before `west`). Include failed
requests in latency metrics. Use nearest-rank percentiles: sort values and select
the one-based rank `ceil(percentile * count)`. Round averages and fractions to
six decimal places. Error means HTTP status code at least 500.

Each JSON object has exactly these top-level fields:

```json
{"specialist": "latency", "groups": [], "assessment": "correlation_only"}
```

Use the relevant specialist name. Each group must have exactly the fields listed
below, with numeric metrics as JSON numbers:

| Specialist | Input | Group fields |
|---|---|---|
| latency | `fixtures/requests.csv` | `window`, `region`, `count`, `p50_ms`, `p95_ms` |
| errors | `fixtures/requests.csv` | `window`, `region`, `count`, `error_count`, `error_rate` |
| capacity | `fixtures/capacity.csv` and `fixtures/deployments.json` | `window`, `region`, `samples`, `mean_utilization_pct`, `max_queue_depth`, `min_replicas`, `max_replicas` |

`error_rate` is a fraction from 0 to 1. The capacity specialist must also inspect
the deployment timing and region and explain its relationship to capacity in
the message it returns to the parent. These observations establish association,
not causation; hence the fixed `assessment` value.

## Parent synthesis

Wait for all three specialists, read their JSON and messages, and write
`output/incident.md`. Identify the affected region and deployment, cite the three
JSON filenames and the deployment fixture, compare the two windows, and explain
what the data does and does not establish. Explicitly say that correlation does
not prove causation. Include a practical next check and a limitation of these
synthetic inputs. Do not claim that configured concurrency proves parallel
execution; API history will be checked separately.

Keep the six specialist artifacts and the report for a follow-up in this same
session. Do not create `output/followup.md` until the follow-up message arrives.

The follow-up will ask you to write `output/followup.md` using the existing files,
with references to `incident.md` and at least one specialist JSON file. Retain
the synthetic-data and correlation limitations in that answer. Propose a bounded
canary experiment to test the deployment hypothesis, with measurable acceptance
and rejection criteria. Label it as a proposal only; do not take any action on a
real service.
