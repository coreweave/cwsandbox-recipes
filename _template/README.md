# Recipe name

One sentence: what you will build and where CoreWeave Sandboxes fit.

## What this recipe demonstrates

- Bullet the 3-5 things a reader learns by running it.

## Architecture

A short diagram or list showing each component and whether it runs in a
sandbox, in a pod, or outside CoreWeave.

## Prerequisites

- Accounts, CLI tools, and API keys, each with a link to where to get them.
- GPU/CPU and quota requirements, including whether pool sizes are per worker or global.
- An existing, accessible bucket if the recipe uploads checkpoints or other outputs.

## Setup

```bash
cp .env.example .env
# fill in .env; every variable is documented inline
uv venv
source .venv/bin/activate
uv pip install -e .
set -a; source .env; set +a
```

## Run

Numbered, copy-pasteable steps. Show expected output after each command so
the reader knows it worked. Make placement and authentication explicit; separate
sandbox and pod-based launch paths, including their CLI prerequisites.

## Results

Where to look when it works: dashboards, run pages, output files. Explain how to
recover uploads without rerunning the workload if outputs remain local.

## Cleanup

Explicit teardown for every billable resource this recipe creates. Verify output
persistence first and scope cleanup to one run when other runs may be active.
