#!/usr/bin/env bash
#SBATCH --job-name=nemo-rl-sandbox
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:30:00
set -euo pipefail

: "${CWSANDBOX_API_KEY:?Export your CoreWeave sandbox API key before submitting}"
: "${RECIPE_DIR:?Set RECIPE_DIR to this recipe on SUNK shared storage}"
: "${RESULTS_DIR:?Set RESULTS_DIR to a new directory on SUNK shared storage}"
# NVIDIA NeMo RL v0.5.0, linux/amd64. Override for another architecture or a
# pre-imported Enroot squashfs image on shared storage.
CONTAINER_IMAGE=${CONTAINER_IMAGE:-nvcr.io#nvidia/nemo-rl@sha256:6d2fe8d955c32a69b3566cb7fd3922f8a2b71d0ceef96b5448391ac9518c9241}
mkdir -p "$RESULTS_DIR"
if [[ -e "$RESULTS_DIR/checkpoints" || -e "$RESULTS_DIR/verification.json" ]]; then
  echo "Choose a new RESULTS_DIR for this two-step validation run." >&2
  exit 1
fi

srun --container-image="$CONTAINER_IMAGE" \
  --no-container-mount-home \
  --container-mounts="$RECIPE_DIR:/recipe,$RESULTS_DIR:/results" \
  --container-workdir=/opt/nemo-rl \
  --container-env=CWSANDBOX_API_KEY \
  bash /recipe/sunk/container-train.sh
