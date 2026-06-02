#!/usr/bin/env bash
# Stage 2 — sampled decode for greedy-passing weights (focused subset).
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate saulgman
cd /root/saulie

SKELS="eval_A4_001,eval_B8_001,eval_C6_001,eval_D6_001"

run() {
  local trial=$1 adapter=$2 w=$3
  local out="dpo/eval/smoke_${trial}_w${w}_sample4.json"
  echo "=== ${trial} w=${w} sample -> ${out} ==="
  python dpo/train/smoke_policy_stack_hf.py \
    --dpo-adapter "$adapter" \
    --adapter-mode policy \
    --dpo-weight "$w" \
    --decode sample \
    --skeleton-ids "$SKELS" \
    --output "$out"
}

T29=dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-29/best_adapter
T10=dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-10/best_adapter
T1=dpo/train/models/steering-dpo-v1.0/optuna-run-20260523-041252/trial-1/best_adapter

# trial-29: all greedy passes except w=1.00 — sample key rescue weights
for W in 0.10 0.25 0.50 0.75; do
  run trial29 "$T29" "$W"
done

# trial-10 / v1.0 trial-1: representative weights
run trial10 "$T10" 0.50
run trial10 "$T10" 1.00
run trial1_v10 "$T1" 0.50
run trial1_v10 "$T1" 1.00

python dpo/eval/score_rescue_smoke.py dpo/eval/smoke_*_sample4.json
