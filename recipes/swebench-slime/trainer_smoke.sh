#!/bin/bash
# Runs inside the GPU sandbox. Prepares Qwen2.5-0.5B-Instruct and dapo-math-17k,
# converts the checkpoint to Megatron torch_dist, then runs slime's own
# single-GPU colocated GRPO smoke script (one rollout -> reward -> update ->
# weight-sync cycle).
set -euo pipefail

cd /root/slime
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir /root/Qwen2.5-0.5B-Instruct > /dev/null
hf download zhuzilin/dapo-math-17k --repo-type dataset --local-dir /root/dapo-math-17k > /dev/null
echo "[ok] model and dataset downloaded"

source scripts/models/qwen2.5-0.5B.sh
PYTHONPATH=/root/Megatron-LM python3 tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" \
  --hf-checkpoint /root/Qwen2.5-0.5B-Instruct \
  --save /root/Qwen2.5-0.5B-Instruct_torch_dist > /root/convert.log 2>&1 \
  || { tail -30 /root/convert.log; exit 1; }
echo "[ok] checkpoint converted to torch_dist"

# The upstream script was written for a source build with Megatron-LM under
# /root/src; the slime image has it at /root/Megatron-LM. Keep the copy in
# scripts/ because it sources models/*.sh relative to its own directory.
sed 's#/root/src/Megatron-LM#/root/Megatron-LM#' scripts/run-qwen2.5-0.5B-gb10-smoke.sh \
  > scripts/run-qwen2.5-0.5B-sandbox-smoke.sh
set +e
bash scripts/run-qwen2.5-0.5B-sandbox-smoke.sh > /root/smoke.log 2>&1
rc=$?
set -e
grep -oE "(rollout 0|step 0): \{[^}]*\}" /root/smoke.log || true
grep -E "Job '.*' (succeeded|failed)" /root/smoke.log || tail -60 /root/smoke.log
exit $rc
