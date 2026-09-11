#!/usr/bin/env bash
# scripts/train_grpo.sh
#
# Single-node GPU sandbox main workload for veRL GRPO training on τ-bench.
# Starts one local Ray head on port 6379 (dashboard 8265), ensures datasets
# exist, then invokes the veRL trainer in-process. Hydra overrides may be
# passed on the command line.
#
# Usage:
#   bash scripts/train_grpo.sh [hydra_overrides...]

set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift || true
fi

# ---------------------------------------------------------------------------
# Fail fast if required environment variables are missing.
# ---------------------------------------------------------------------------
: "${WANDB_API_KEY:?Set WANDB_API_KEY (do not commit it to git)}"
: "${HF_TOKEN:?Set HF_TOKEN for model/tokenizer download}"

# CoreWeave AI Object Storage for checkpoints. The recipe's veRL trainer hook
# uploads every scheduled save and logs its URI as a W&B reference artifact.
# scripts/upload_checkpoints.py remains an idempotent post-training recovery.
# Public CAOS endpoint. Sandboxes cannot reach the node-local LOTA proxy
# (http://cwlota.com): sandbox egress only routes to the internet. Pods with
# normal node networking (the SkyPilot track) may set CW_ENDPOINT to LOTA.
export CW_ENDPOINT="${CW_ENDPOINT:-https://cwobject.com}"
export CW_BUCKET="${CW_BUCKET:-verl-taubench-checkpoints}"
# Where verl writes checkpoints. Point at node-local scratch (e.g.
# /tmp/checkpoints) to avoid filling the shared NFS home; pair with the
# object-storage upload so the checkpoint survives the job.
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"

DATA_DIR="${DATA_DIR:-./data/processed}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen2.5-7b_grpo}"
PROJECT_NAME="${PROJECT_NAME:-verl-taubench-coreweave}"
N_GPUS="${N_GPUS:-8}"
TAUBENCH_DOMAIN="${TAUBENCH_DOMAIN:-retail}"
TAUBENCH_TEST_END_INDEX="${TAUBENCH_TEST_END_INDEX:-50}"
# Optional rollout tracing: weave, mlflow, or trackio. Weave traces every
# sampled episode (per-turn generate + env step) under WANDB_ENTITY/PROJECT_NAME
# (the same place metrics go) using the WANDB_API_KEY credentials.
TRACE_BACKEND="${TRACE_BACKEND:-}"

# The GPU-sandbox launcher supplies this up front. Generate one for direct and
# SkyPilot launches so live uploads and any recovery refer to the same run.
if [[ -z "${WANDB_RUN_ID:-}" ]]; then
    WANDB_RUN_ID="$(python -c 'from wandb.sdk.lib.runid import generate_id; print(generate_id())')"
fi
export WANDB_RUN_ID
# Session reporters run in Ray workers and join this same online run. Preserve
# explicitly offline/disabled launches; the trainer remains the primary writer.
case "${WANDB_MODE:-online}" in
    online|shared) export WANDB_MODE=shared ;;
esac

# veRL imports external modules before resolving trainer.v1.trainer_mode.
CHECKPOINT_TRAINER_MODULE="verl_taubench.trainer.checkpointing"
case ",${VERL_USE_EXTERNAL_MODULES:-}," in
    *",${CHECKPOINT_TRAINER_MODULE},"*) ;;
    ",,") export VERL_USE_EXTERNAL_MODULES="${CHECKPOINT_TRAINER_MODULE}" ;;
    *) export VERL_USE_EXTERNAL_MODULES="${VERL_USE_EXTERNAL_MODULES},${CHECKPOINT_TRAINER_MODULE}" ;;
esac

VERL_RAY_PORT=6379
DASHBOARD_PORT=8265

# Our Ray session gets a dedicated temp dir so cleanup can target exactly the
# processes that carry it in argv. A global `ray stop --force` would also kill
# SkyPilot's runtime Ray on pods, bricking the cluster for later jobs.
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-taubench}"

_stop_ray() {
    pkill -f "${RAY_TMPDIR}" 2>/dev/null || true
}

_cleanup() {
    local trainer_status=$?
    set +e
    _stop_ray
    if [[ -n "${TAUBENCH_ENV_TAG:-}" ]]; then
        python -m verl_taubench.sandbox.cleanup --tag "${TAUBENCH_ENV_TAG}" || {
            echo "Warning: CPU sandbox cleanup failed for tag ${TAUBENCH_ENV_TAG}" >&2
        }
    fi
    return "${trainer_status}"
}

# ---------------------------------------------------------------------------
# Install the mounted project unless the image already contains it.
# The CUDA base image supplies vLLM/torch; do not install the vllm extra here.
# ---------------------------------------------------------------------------
if [[ "${SKIP_PROJECT_INSTALL:-0}" != "1" ]]; then
    echo "[taubench] installing mounted project (sandbox extra)..."
    # cwsandbox >=1.9 needs cryptography>=42; the training image ships a
    # distro-installed copy pip cannot uninstall (no RECORD file), so
    # overwrite it in place instead of upgrading through an uninstall.
    python -m pip install --quiet --ignore-installed "cryptography>=42.0.0"
    python -m pip install -e ".[sandbox]"
fi

# The upload hook starts inside training, so boto3 must exist before the first
# checkpoint rather than being installed only after the trainer exits.
if [[ -n "${CW_ACCESS_KEY:-}" && -n "${CW_SECRET_KEY:-}" ]]; then
    python -c "import boto3" 2>/dev/null || python -m pip install -q boto3
fi

# ---------------------------------------------------------------------------
# Ensure train/test parquet datasets exist. data/ is not mounted into sandboxes.
# ---------------------------------------------------------------------------
mkdir -p "${DATA_DIR}"

if [[ ! -f "${DATA_DIR}/train.parquet" ]]; then
    echo "[taubench] generating train dataset (${TAUBENCH_DOMAIN})..."
    python scripts/preprocess_taubench.py \
        --domain "${TAUBENCH_DOMAIN}" \
        --task-split train \
        --local-save-dir "${DATA_DIR}"
fi

if [[ ! -f "${DATA_DIR}/test.parquet" ]]; then
    echo "[taubench] generating test dataset (${TAUBENCH_DOMAIN}, end-index=${TAUBENCH_TEST_END_INDEX})..."
    python scripts/preprocess_taubench.py \
        --domain "${TAUBENCH_DOMAIN}" \
        --task-split test \
        --end-index "${TAUBENCH_TEST_END_INDEX}" \
        --local-save-dir "${DATA_DIR}"
fi

# ---------------------------------------------------------------------------
# Trainer command. Ray bootstrap stays outside Hydra; never ray.init(address=auto).
# ---------------------------------------------------------------------------
TRAINER_CMD=(
    python -m verl.trainer.main_ppo
    --config-path "${PWD}/verl_taubench/trainer"
    --config-name grpo_trainer
    "data.train_files=${DATA_DIR}/train.parquet"
    "data.val_files=${DATA_DIR}/test.parquet"
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "actor_rollout_ref.rollout.data_parallel_size=${N_GPUS}"
    "trainer.n_gpus_per_node=${N_GPUS}"
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${EXPERIMENT_NAME}"
    "trainer.default_local_dir=${CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}"
)

if [[ -n "${TRACE_BACKEND}" ]]; then
    if [[ "${TRACE_BACKEND}" == "weave" ]]; then
        python -c "import weave" 2>/dev/null || {
            echo "Installing weave for rollout tracing..."
            python -m pip install -q weave
        }
    fi
    TRAINER_CMD+=(
        "actor_rollout_ref.rollout.trace.backend=${TRACE_BACKEND}"
        "actor_rollout_ref.rollout.trace.token2text=True"
    )
    if [[ -n "${WANDB_ENTITY:-}" ]]; then
        # weave.init ignores WANDB_ENTITY: a bare project name lands under the
        # API key's default entity instead of the team the metrics go to.
        TRAINER_CMD+=(
            "actor_rollout_ref.rollout.trace.project_name=${WANDB_ENTITY}/${PROJECT_NAME}"
        )
    fi
fi

TRAINER_CMD+=("$@")

if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "Dry-run; would execute:"
    echo "ray start --head --port=${VERL_RAY_PORT} --dashboard-port=${DASHBOARD_PORT}"
    printf '%q ' "${TRAINER_CMD[@]}"
    echo
    exit 0
fi

echo "[taubench] starting Ray head on port ${VERL_RAY_PORT}..."
# Reap a stale session from a previous job on the same machine, then start.
_stop_ray
ray start \
    --head \
    --temp-dir="${RAY_TMPDIR}" \
    --port="${VERL_RAY_PORT}" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${DASHBOARD_PORT}" \
    --disable-usage-stats

trap _cleanup EXIT

echo "[taubench] Ray head up; launching veRL GRPO trainer (epochs, batch and model come from grpo_trainer.yaml plus argv overrides)"
"${TRAINER_CMD[@]}"

# Idempotent recovery. In the normal path all objects and reference markers
# already exist, so this skips the bytes and does not reopen the W&B run.
LOCAL_CKPT_PATH="${CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}"
if [[ -n "${CW_ACCESS_KEY:-}" && -n "${CW_SECRET_KEY:-}" && -d "${LOCAL_CKPT_PATH}" ]]; then
    echo "[taubench] verifying checkpoint persistence at s3://${CW_BUCKET} via ${CW_ENDPOINT}..."
    python scripts/upload_checkpoints.py \
        --local-dir "${LOCAL_CKPT_PATH}" \
        --prefix "checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}" \
        --project "${PROJECT_NAME}" \
        --experiment "${EXPERIMENT_NAME}" \
        --wandb-run-id "${WANDB_RUN_ID}"
else
    echo "[taubench] CW_ACCESS_KEY/CW_SECRET_KEY not set (or no checkpoint dir);" \
         "checkpoints remain local at ${LOCAL_CKPT_PATH}"
fi
