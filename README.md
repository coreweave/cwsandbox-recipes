# CoreWeave Sandbox Recipes

Practical, runnable examples of AI workloads on [CoreWeave Sandboxes](https://docs.coreweave.com/products/sandboxes).

Sandboxes are isolated compute environments with serverless and CoreWeave Kubernetes Service (CKS) placement options. Create CPU or GPU workloads from Python with the `cwsandbox` SDK, not the retired `wandb[sandbox]` wrapper. See the [get started guide](https://docs.coreweave.com/products/sandboxes/get-started).

## Recipes

| Recipe | What it shows | Hardware |
|---|---|---|
| [`tailcat`](recipes/tailcat/) | Connect Serverless Sandboxes to a private CKS Service with Tailcat, including reverse connections and a round trip. | CPU only |
| [`github-actions`](recipes/github-actions/) | Run public repository tests in disposable sandboxes from GitHub Actions, with optional advisory AI review. | CPU only |
| [`warm-pool`](recipes/warm-pool/) | Reduce startup latency for concurrent workloads with ready sandboxes and automatic replenishment. Includes a live comparison, isolation checks, and snapshot guidance. | CPU only |
| [`openai-agents-api`](recipes/openai-agents-api/) | Investigate synthetic inference telemetry with an OpenAI Agent coordinator and three subagents; verify metrics, delegation, turn overlap, and cleanup. | CPU only |
| [`taubench-verl`](recipes/taubench-verl/) | RL training (GRPO) of a tool-using agent on tau-bench with veRL. GPU sandbox or optional SkyPilot/CKS trainer; per-worker CPU sandbox pools, serverless by default. W&B Models metrics, optional Weave traces, and checkpoint reference artifacts. | 8x H100 + CPU pools |

## Getting started

1. Pick a recipe and open its README.
2. Copy its `.env.example` to `.env` and fill it in. Every variable is documented inline.
3. Follow the recipe's run steps.

Each recipe is self-contained: its own dependency declarations, config, and tests. Follow its installation steps and configure credentials and placement before running it.

## Cost and safety

Recipes provision billable cloud resources and may call paid APIs. Read a recipe's resource requests and cleanup section before running it; pool sizes may be per worker, not global. Use an existing bucket for checkpoint uploads and verify persistence before teardown. Keep credentials in `.env` (gitignored) or a secret store, never in source files.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Start a new recipe from [`_template/`](_template/).

## License

Apache 2.0. See [LICENSE](LICENSE).
