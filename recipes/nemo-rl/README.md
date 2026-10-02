# Post-train with NeMo RL and CoreWeave Sandboxes

Run reinforcement learning with a CPU sandbox as the reward environment and a model with fewer than eight billion parameters. Run training and rollout generation in a GPU sandbox or SUNK. The Serverless Inference section explains the current bring-your-own-weights limitation for this small-model example.

This recipe is for machine learning (ML) engineers familiar with Python and reinforcement learning who want to verify the connection between sandbox rewards and policy updates.

The example uses arithmetic tasks so you can inspect the reward computation. Each response receives a correctness reward plus a small brevity reward. The brevity term gives the smoke run a signal when sampled answers have equal correctness. The result is an integration test, not a reasoning benchmark.

## Components

| Component | Role |
| --- | --- |
| `nemo_rl_sandbox/environment.py` | Creates a CPU sandbox, sends response batches to the verifier, and stops the sandbox. |
| `nemo_rl_sandbox/grader.py` | Computes the deterministic arithmetic and brevity rewards. |
| `sunk/native_grpo.py` | Connects the sandbox environment to the standard NeMo RL group relative policy optimization (GRPO) trainer. |
| `scripts/train_smoke.py` | Runs a compact single-GPU low-rank adaptation (LoRA) loop with NeMo RL loss and advantage utilities. |
| `serverless/checkpoints.py` | Publishes parameter-efficient fine-tuning (PEFT) adapters and retrieves serverless rollout tokens and log probabilities. |

The standard trainer and compact driver are separate implementations. The compact driver uses NeMo RL's policy-gradient loss, Transformers generation, and a PyTorch optimizer. It doesn't exercise NeMo RL's distributed trainer. The SUNK path uses the standard trainer with DTensor policy updates and colocated vLLM generation.

## Requirements

Before you begin, prepare the following:

- A CoreWeave account with sandbox access and CPU quota.
- A CoreWeave API access token in `CWSANDBOX_API_KEY`.
- Python and [`uv`](https://docs.astral.sh/uv/) on your launcher machine.
- GPU capacity for the selected training path.
- Network access to download public model weights and packages, and reach the Sandbox API.

CPU sandbox authentication is explicitly CoreWeave. The optional Serverless Inference transport helpers use a separate Forge API key and an explicitly selected entity and project.

The runs consume GPU time, CPU sandbox time, and, for Serverless Inference, inference tokens and artifact storage. Use the current [CoreWeave pricing](https://www.coreweave.com/pricing) and [Forge pricing](https://www.coreweave.com/forge-pricing) for your account. Each launcher defines resource requests and time limits. A time limit caps resource lifetime, not a promised completion time.

## Set up the recipe

Install the launcher's dependencies and set the sandbox credential before starting a training run.

Run from this directory. Replace `[COREWEAVE-API-ACCESS-TOKEN]` with your CoreWeave API access token:

```bash
uv sync --locked --python 3.12
export CWSANDBOX_API_KEY="[COREWEAVE-API-ACCESS-TOKEN]"
```

The launcher creates the CPU reward sandbox on serverless capacity. The verifier receives response text as a JavaScript Object Notation (JSON) argument and parses it as data.

## Run in a GPU sandbox

From your workstation, run the compact smoke test:

```bash
uv run python scripts/launch_sandbox.py --steps 2 --output results/sandbox
```

The launcher requests one serverless GPU, eight CPUs, and 32 GiB of host memory, with a one-hour maximum lifetime. The tested hardware is NVIDIA RTX PRO 6000 Blackwell Server Edition. The model is `Qwen/Qwen2.5-0.5B-Instruct`, with a rank-8 LoRA adapter. The launcher uploads the recipe, installs the pinned dependencies, runs two optimizer steps, downloads the checkpoint, and stops its GPU sandbox. Each step generates fresh responses from the current policy.

The compact driver loads NeMo RL's loss and advantage code at commit `b1c86a816c5e2b4ca41ece193624a38dc62e6fdf` (v0.3.0). It uses source imports for those components, rather than installing or launching the full NeMo RL framework. The GPU image provides PyTorch 2.7.1 with CUDA 12.8. See the `scripts/requirements-smoke.txt` file for the Python package pins.

The installer downloads a commit-pinned source archive and verifies its SHA-256 checksum, without apt or Git. Each download attempt has a two-minute deadline, with up to three attempts. The launcher limits the full installation stage to 15 minutes.

Inspect the results:

```bash
cat results/sandbox/summary.json
cat results/sandbox/metrics.json
tar -tzf results/sandbox/result.tar.gz
```

The summary identifies the backend as `custom-driver-nemo-rl-loss`. Verify two optimizer steps and a positive `parameter_delta_l2`. The archive contains the adapter and tokenizer files. When you reload a PEFT adapter, keep the corresponding base model. The adapter alone isn't a full-model checkpoint.

## Run the standard trainer on SUNK

Use an existing SUNK cluster, a shared working directory, and a GPU partition with Pyxis container support. The batch script requests one GPU, 16 CPUs, 96 GiB of host memory, and 30 minutes. The model is `Qwen/Qwen2.5-0.5B-Instruct`. The trainer and sampler share the allocated GPU.

For a new cluster, follow [SUNK Self-Service with the SunkCluster CR](https://docs.coreweave.com/products/sunk/tutorials/get-started-with-sunkcluster-cr). The `sunk/sunkcluster.example.yaml` manifest illustrates the cluster shape. Complete the documented quota, custom resource definition (CRD) enablement, user provisioning, and SSH prerequisites before applying it. The manifest is a setup example, not a prerequisite for testing against an existing cluster.

Copy the recipe to shared storage and connect to your SUNK login node. From the recipe directory, submit the job. Replace `[COREWEAVE-API-ACCESS-TOKEN]` with your CoreWeave API access token:

```bash
export RECIPE_DIR="$PWD"
export RESULTS_DIR="$PWD/results/sunk-$(date +%Y%m%d-%H%M%S)"
export CWSANDBOX_API_KEY="[COREWEAVE-API-ACCESS-TOKEN]"
mkdir -p "$RESULTS_DIR"
sbatch --output="$RESULTS_DIR/slurm-%j.log" sunk/train.sbatch
```

If your cluster requires a partition or Slurm account, add `--partition="[PARTITION]"` or `--account="[ACCOUNT]"` to `sbatch`. Replace `[PARTITION]` with your partition and `[ACCOUNT]` with your Slurm account. The container image must be downloadable from the compute node. Allow time for its first pull, or use a pre-pulled image with `CONTAINER_IMAGE`.

The driver runs inside the same Slurm allocation as the trainer and sampler. Keeping these together avoids a separate remote Ray control plane. The reward environment runs in a CPU sandbox.

After the job completes, inspect `RESULTS_DIR/verification.json`. It records gradient norms for both updates, the number of changed tensors between checkpoints, and whether the exported policy reloaded successfully with finite logits. The full Hugging Face checkpoint is in `RESULTS_DIR/hf-policy/`. This path was tested on one NVIDIA H100 GPU.

## Serverless Inference: bring-your-own-weights limitation

Serverless Inference supports customer LoRA adapters on [selected hosted base models](https://docs.coreweave.com/products/inference/serverless/lora#supported-base-models). It doesn't accept arbitrary full-model weights or provision a new serving base when you upload an adapter.

`Qwen/Qwen2.5-0.5B-Instruct` isn't a supported base, and the current supported-adapter catalog has no model with fewer than eight billion total parameters. Consequently, this recipe doesn't provide a runnable Serverless Inference training path for its small model. This is a bring-your-own-weights and catalog limitation, independent of GPU memory or available GPU types.

An adapter upload can succeed even when its base model is unsupported. Inference for the uploaded 0.5B adapter returned HTTP 404 with `The requested resource was not found.` If you see this error, check the base model against the supported-base list; upload success alone doesn't establish serving support.

Use the GPU sandbox or SUNK path to train and sample this model. A fixed catalog model wouldn't sample the updated policy.

The `serverless/checkpoints.py` reference helpers preserve the integration boundary for a future supported base. They publish PEFT adapters and retrieve exact-token rollouts; they don't implement a trainer or add an unavailable model to the serving catalog. If the chosen base becomes supported, connect your GPU trainer or a sandbox trainer with this sequence:

1. Export a PEFT adapter with a supported rank and matching base model.
2. Upload it as a `lora` artifact in Weights & Biases in `coreweave-us`, with the exact base ID in `wandb.base_model`.
3. Wait for upload completion and use its immutable `:vN` URI for an entire rollout batch.
4. Retrieve exact sampled tokens and behavior-policy log probabilities.
5. Compute CPU sandbox rewards and update the adapter on the trainer.
6. Publish a new adapter version and select it for the next batch.

The transport helper accepts explicit prompt token IDs for `/v1/completions` and retrieves output IDs through the documented [vLLM `return_tokens_as_token_ids` extension](https://docs.vllm.ai/en/v0.19.0/serving/openai_compatible_server/#completions-api). It rejects missing tokens, misaligned or nonfinite log probabilities, and mutable aliases such as `:latest`. Some serving backends return the canonical base model ID in `response.model`; the requested immutable artifact URI selects the adapter version.

Serving compatibility varies by model. Check the [supported-base list](https://docs.coreweave.com/products/inference/serverless/lora#supported-base-models), [model lifecycle](https://docs.coreweave.com/products/inference/serverless/lifecycle), and exact-token contract before adding a trainer. Publishing an artifact alone doesn't prove the serving backend applied the adapter.

## Verify the result

A successful smoke run must include CPU sandbox rewards, a nonzero policy update, saved weights, and successful checkpoint reload. Two training iterations establish integration behavior, not improved model quality.

## Clean up

Copy results from ephemeral compute before stopping a GPU sandbox.

The GPU launcher stops its sandbox on normal completion, errors, Ctrl-C (`SIGINT`), and `SIGTERM`. It records the GPU sandbox ID in `results/sandbox/sandbox-id.txt` before uploading files and records confirmed shutdown as `"stopped": true` in `results/sandbox/cleanup.json`. Choose a new, empty output directory for each run.

If the launcher is killed without a cleanup opportunity, the machine loses connectivity, or shutdown fails, stop the GPU sandbox manually. From the recipe directory, run:

```bash
uv run python scripts/stop_sandbox.py "$(cat results/sandbox/sandbox-id.txt)"
```

The CPU reward sandbox is a separate resource. Its client stops it when training completes or unwinds through an exception, but forcibly stopping the GPU sandbox or Slurm job can prevent that cleanup. Stop a remaining CPU sandbox with the same helper, using the `sandbox_id` in the training log's `cpu_sandbox_created` event. Each reward sandbox has a one-hour maximum lifetime as a fallback.

On SUNK, a successful cleanup prints an event named `cpu_sandbox_stopped` in the job log after the driver receives confirmation. Wait for the job to finish, or cancel your job with `scancel [JOB-ID]`, replacing `[JOB-ID]` with your Slurm job ID. Confirm CPU sandbox shutdown after cancellation. Keep shared checkpoints until you've copied them to persistent storage.
