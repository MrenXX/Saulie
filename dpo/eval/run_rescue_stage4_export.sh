#!/usr/bin/env bash
# Stage 4 — export weighted cat adapters (after HF smoke passes).
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate saulgman
cd /root/saulie

export_cat() {
  local adapter=$1 w=$2 out=$3
  echo "=== export w=${w} -> ${out} ==="
  python dpo/train/merge_sft_dpo_lora.py \
    --dpo-adapter "$adapter" \
    --dpo-weight "$w" \
    --output "$out" \
    --check-logps
}

T29=dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-29/best_adapter
T10=dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-10/best_adapter
T1=dpo/train/models/steering-dpo-v1.0/optuna-run-20260523-041252/trial-1/best_adapter

export_cat "$T29" 0.25 \
  dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-29/sft_dpo_cat_w025
export_cat "$T29" 0.50 \
  dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-29/sft_dpo_cat_w050
export_cat "$T10" 0.50 \
  dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-10/sft_dpo_cat_w050
export_cat "$T1" 0.50 \
  dpo/train/models/steering-dpo-v1.0/optuna-run-20260523-041252/trial-1/sft_dpo_cat_w050
