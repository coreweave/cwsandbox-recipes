# CoreWeave Sandbox Recipes

Practical, runnable examples of AI workloads on [CoreWeave Sandboxes](https://docs.coreweave.com/products/sandboxes).

Sandboxes are isolated compute environments with serverless and CoreWeave Kubernetes Service (CKS) placement options. Create CPU or GPU workloads from Python with the `cwsandbox` SDK, not the retired `wandb[sandbox]` wrapper. See the [get started guide](https://docs.coreweave.com/products/sandboxes/get-started).

## Recipes

| Recipe | What it shows | Hardware |
|---|---|---|
| [`claude-code-cloud`](recipes/claude-code-cloud/) | Run Claude Code web sessions in on-demand worker sandboxes, with a sandbox-hosted orchestrator. | CPU only |
| [`k3s-private-cluster`](recipes/k3s-private-cluster/) | Build a private Kubernetes cluster across sandboxes; scale workers with configurable CPU/memory and snapshot/restore the control plane. | CPU only |
| [`tailcat`](recipes/tailcat/) | Connect Serverless Sandboxes to a private CKS Service with Tailcat, including reverse connections and a round trip. | CPU only |
| [`github-actions`](recipes/github-actions/) | Run public repository tests in disposable sandboxes from GitHub Actions, with optional advisory AI review. | CPU only |
| [`deepswe`](recipes/deepswe/) | Evaluate a coding agent on DeepSWE v1.1 with separate task and verifier sandboxes. | CPU only |
| [`warm-pool`](recipes/warm-pool/) | Reduce startup latency for concurrent workloads with ready sandboxes and automatic replenishment. Includes a live comparison, isolation checks, and snapshot guidance. | CPU only |
| [`openai-agents-api`](recipes/openai-agents-api/) | Investigate synthetic inference telemetry with an OpenAI Agent coordinator and three subagents; verify metrics, delegation, turn overlap, and cleanup. | CPU only |
| [`dapo-nemo-rl`](recipes/dapo-nemo-rl/) | DAPO RL on Qwen2.5-Math-7B: NeMo-RL v0.7.0 on one SUNK node, rollouts from a CoreWeave Dedicated Inference deployment that gets new weights by hot-load, NeMo-Gym v0.6.0 in a CPU sandbox that grades evaluation and calibration in more CPU sandboxes. | SUNK 8x H100 node; 2 x H100 inference; CPU sandboxes |
| [`async-grpo-tool-use`](recipes/async-grpo-tool-use/) | Async GRPO with Python tool use on Qwen2.5-Math-7B, based on `dapo-nemo-rl`: NeMo-RL v0.7.0 on one SUNK node, background hot-load into Dedicated Inference, one CPU sandbox per rollout to run its code. One command prints a shareable live dashboard hosted in a sandbox, with W&B run, Agent Lens traces and Weave evaluations. | SUNK 8x H100 node; 2 x H100 inference; CPU sandboxes |
| [`taubench-verl`](recipes/taubench-verl/) | RL training (GRPO) of a tool-using agent on tau-bench with veRL. GPU sandbox or optional SkyPilot/CKS trainer; per-worker CPU sandbox pools, serverless by default. W&B Models metrics, optional Weave traces, and checkpoint reference artifacts. | 8x H100 + CPU pools |

## Getting started

1. Pick a recipe and open its README.
2. Configure credentials as described in the recipe's README. Its `.env.example` documents each variable.
3. Follow the recipe's run steps.

Each recipe is self-contained: its own dependency declarations, config, and tests. Follow its installation steps and configure credentials and placement before running it.

## Cost and safety

Recipes provision billable cloud resources and may call paid APIs. Read a recipe's resource requests and cleanup section before running it; pool sizes may be per worker, not global. Use an existing bucket for checkpoint uploads and verify persistence before teardown. Keep credentials in `.env` (gitignored) or a secret store, never in source files.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Start a new recipe from [`_template/`](_template/).

## License

Apache 2.0. See [LICENSE](LICENSE).
