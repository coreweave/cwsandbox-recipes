#!/usr/bin/env bash
set -euo pipefail
export PATH="/opt/nemo_rl_venv/bin:/root/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
export PYTHONPATH="/recipe:/opt/nemo-rl${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/results/huggingface
export TOKENIZERS_PARALLELISM=false
cd /opt/nemo-rl
uv pip install --python /opt/nemo_rl_venv/bin/python cwsandbox==1.17.0
uv run --no-sync /recipe/sunk/native_grpo.py
uv run --no-sync /recipe/sunk/verify_checkpoint.py
