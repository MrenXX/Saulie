#!/usr/bin/env bash
# Stage 1 from DPO_POLICY_RESCUE_PLAN.md — HF greedy policy-stack grid.
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate saulgman
cd /root/saulie

declare -A TRIALS
TRIALS[trial29]=dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-29/best_adapter
TRIALS[trial10]=dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-10/best_adapter
TRIALS[trial1_v10]=dpo/train/models/steering-dpo-v1.0/optuna-run-20260523-041252/trial-1/best_adapter

for TRIAL in trial29 trial10 trial1_v10; do
  for W in 0.00 0.10 0.25 0.50 0.75 1.00; do
    OUT="dpo/eval/smoke_${TRIAL}_w${W}_greedy.json"
    echo "=== ${TRIAL} weight=${W} -> ${OUT} ==="
    python dpo/train/smoke_policy_stack_hf.py \
      --dpo-adapter "${TRIALS[$TRIAL]}" \
      --adapter-mode policy \
      --dpo-weight "$W" \
      --decode greedy \
      --skeleton-ids eval_A4_001,eval_B8_001 \
      --output "$OUT"
  done
done

echo "=== Scoring ==="
python dpo/eval/score_rescue_smoke.py dpo/eval/smoke_*_greedy.json
