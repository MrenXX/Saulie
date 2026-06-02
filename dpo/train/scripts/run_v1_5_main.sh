#!/usr/bin/env bash
# v1.5 main study: 12 fixed + 8 TPE, 2 workers, babysit until trial_summary_v1.5.json.
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate saulgman
cd /root/saulie

python dpo/train/scripts/preflight_v1_5.py

STAMP=$(date +%Y%m%d-%H%M%S)
RUN="dpo/train/models/steering-dpo-v1.5/optuna-run-${STAMP}"
STUDY_NAME="steering-dpo-v1.5-hybrid-seed42"
PIPE_LOG="${RUN}/pipeline.log"
mkdir -p "$RUN"
echo "$RUN" > dpo/train/models/steering-dpo-v1.5/LATEST_RUN_DIR.txt

exec > >(tee -a "$PIPE_LOG") 2>&1
echo "=== v1.5 MAIN start $(date -Is) ==="
echo "RUN=$RUN"

python dpo/train/train_dpo.py --optuna --study-version v1.5 \
  --parallel-workers 2 \
  --run-dir="$RUN" \
  --study-storage="${RUN}/optuna_study.db" \
  --study-name="$STUDY_NAME" \
  --target-complete-trials 20 \
  --max-attempted-trials 60 &
TRAIN_PID=$!

python dpo/train/scripts/babysit_study.py \
  --run-dir "$RUN" \
  --study-name "$STUDY_NAME" \
  --study-version v1.5 \
  --target-complete 20 \
  --poll-min 10 &
BABYSIT_PID=$!
echo "$BABYSIT_PID" > "${RUN}/babysit.pid"

wait "$TRAIN_PID"
TRAIN_EXIT=$?
kill "$BABYSIT_PID" 2>/dev/null || true

if [[ "$TRAIN_EXIT" -ne 0 ]]; then
  echo "ERROR: train_dpo exited $TRAIN_EXIT"
  exit "$TRAIN_EXIT"
fi

SUMMARY="${RUN}/trial_summary_v1.5.json"
python - <<'PY' "$SUMMARY"
import json, sys
from pathlib import Path
s = json.loads(Path(sys.argv[1]).read_text())
print("target_reached", s.get("target_reached"))
print("COMPLETE", s.get("counts", {}).get("COMPLETE"))
print("best_trial", s.get("best_trial"), "survival", s.get("best_v1_5_survival_score"))
PY
echo "=== v1.5 MAIN done $(date -Is) ==="
echo "Summary: $SUMMARY"
