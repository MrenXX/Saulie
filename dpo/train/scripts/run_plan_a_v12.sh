#!/usr/bin/env bash
# Plan A Step 2: two fixed v1.2 rescue trials (sequential, 1 worker).
# Usage: bash dpo/train/scripts/run_plan_a_v12.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate saulgman

echo "=== Preflight ==="
python dpo/train/scripts/preflight_plan_a_v12.py

STAMP=$(date +%Y%m%d-%H%M%S)
RUN_DIR="/root/saulie/dpo/train/models/steering-dpo-v1.2/optuna-run-${STAMP}"
mkdir -p "$RUN_DIR"

echo "=== Launch v1.2 study (run_dir=$RUN_DIR) ==="
python dpo/train/train_dpo.py --optuna --study-version v1.2 \
  --parallel-workers 1 \
  --target-complete-trials 2 \
  --max-attempted-trials 4 \
  --run-dir "$RUN_DIR" \
  --study-storage "$RUN_DIR/optuna_study.db" \
  --study-name steering-dpo-v1.2-plan-a-seed42

echo ""
echo "Run dir: $RUN_DIR"
echo "MLflow:  mlflow ui --backend-store-uri file:///root/saulie/dpo/train/mlruns --port 5001"
echo "Chat:    python dpo/eval/chat_policy_stack.py --dpo-adapter $RUN_DIR/trial-N/best_adapter"
