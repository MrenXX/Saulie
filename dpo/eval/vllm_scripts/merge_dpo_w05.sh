#!/bin/bash
# Cat-merge DPO v1.5 trial-4 @ dpo_weight=0.5 for vLLM deploy.
#
#   bash dpo/eval/vllm_scripts/merge_dpo_w05.sh

set -euo pipefail

REPO="/root/saulie"
PYTHON="${SAULIE_PYTHON:-/root/miniconda3/envs/saulgman/bin/python}"
DPO_ADAPTER="${REPO}/dpo/train/models/steering-dpo-v1.5/optuna-run-20260602-052732/trial-4/best_adapter"
OUTPUT="${REPO}/dpo/train/models/steering-dpo-v1.5/optuna-run-20260602-052732/trial-4/sft_dpo_cat_w05"

cd "$REPO"
"$PYTHON" dpo/train/merge_sft_dpo_lora.py \
  --dpo-adapter "$DPO_ADAPTER" \
  --dpo-weight 0.5 \
  --output "$OUTPUT" \
  --check-logps

echo ""
echo "Merged adapter: $OUTPUT"
echo "Deploy with: bash dpo/eval/vllm_scripts/deploy_dpo_w05.sh"
