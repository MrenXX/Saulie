#!/usr/bin/env bash
# Plan B v1.4: same 4 fixed configs as v1.3, unmerged BnB+SFT+DPO stack.
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate saulgman
cd /root/saulie

python dpo/train/scripts/preflight_plan_b_v14.py

STAMP=$(date +%Y%m%d-%H%M%S)
RUN="dpo/train/models/steering-dpo-v1.4/optuna-run-${STAMP}"
STUDY_NAME="steering-dpo-v1.4-plan-b-seed42"
PIPE_LOG="${RUN}/pipeline.log"

mkdir -p "$RUN"
echo "$RUN" > dpo/train/models/steering-dpo-v1.4/LATEST_RUN_DIR.txt

exec > >(tee -a "$PIPE_LOG") 2>&1
echo "=== Plan B v1.4 train start $(date -Is) ==="
echo "RUN=$RUN"
echo "MLflow experiment: steering-dpo-v1.4"
echo "Stack: BnB8(base) + frozen SFT LoRA + trainable DPO LoRA"

babysit_loop() {
  echo "Babysit waiting for optuna DB..."
  for _ in $(seq 1 120); do
    if [[ -f "${RUN}/optuna_study.db" ]]; then
      break
    fi
    sleep 5
  done
  if [[ ! -f "${RUN}/optuna_study.db" ]]; then
    echo "ERROR: optuna_study.db never appeared"
    return 1
  fi
  python dpo/train/scripts/babysit_study.py \
    --run-dir "$RUN" \
    --study-name "$STUDY_NAME" \
    --target-complete 4 \
    --poll-min 5
}

babysit_loop &
BABYSIT_PID=$!

cleanup() {
  kill "$BABYSIT_PID" 2>/dev/null || true
}
trap cleanup EXIT

python dpo/train/train_dpo.py --optuna --study-version v1.4 \
  --run-dir "$RUN" \
  --parallel-workers 1 \
  --target-complete-trials 4 \
  --max-attempted-trials 6 \
  --study-name "$STUDY_NAME"

wait "$BABYSIT_PID" 2>/dev/null || true
trap - EXIT

echo "=== Training finished $(date -Is) ==="
if [[ ! -f "${RUN}/trial_summary.json" ]]; then
  echo "ERROR: missing trial_summary.json"
  exit 1
fi
python - <<'PY' "$RUN"
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
s = json.loads((run / "trial_summary.json").read_text())
print("target_reached:", s.get("target_reached"))
print("counts:", s.get("counts"))
PY
echo "REPL: python dpo/eval/chat_policy_stack.py --dpo-adapter \"\$RUN/trial-N/best_adapter\""
echo "RUN=$RUN"
