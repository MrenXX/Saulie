#!/usr/bin/env bash
# v1.5 smoke: preflight + 2-trial parallel Optuna with babysit.
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate saulgman
cd /root/saulie

python dpo/train/scripts/preflight_v1_5.py

STAMP=$(date +%Y%m%d-%H%M%S)
RUN="dpo/train/models/steering-dpo-v1.5/optuna-run-${STAMP}-smoke"
STUDY_NAME="steering-dpo-v1.5-hybrid-seed42"
PIPE_LOG="${RUN}/pipeline.log"
mkdir -p "$RUN"
echo "$RUN" > dpo/train/models/steering-dpo-v1.5/LATEST_SMOKE_RUN_DIR.txt

exec > >(tee -a "$PIPE_LOG") 2>&1
echo "=== v1.5 smoke start $(date -Is) ==="
echo "RUN=$RUN"

python dpo/train/train_dpo.py --optuna --study-version v1.5 --optuna-smoke \
  --parallel-workers 2 \
  --run-dir="$RUN" \
  --study-storage="${RUN}/optuna_study.db" \
  --study-name="$STUDY_NAME" \
  --target-complete-trials 2 \
  --max-attempted-trials 12 &
TRAIN_PID=$!

python dpo/train/scripts/babysit_study.py \
  --run-dir "$RUN" \
  --study-name "$STUDY_NAME" \
  --study-version v1.5 \
  --target-complete 2 \
  --poll-min 5 &
BABYSIT_PID=$!
echo "$BABYSIT_PID" > "${RUN}/babysit.pid"

wait "$TRAIN_PID"
TRAIN_EXIT=$?

if [[ -f "${RUN}/babysit.pid" ]]; then
  kill "$BABYSIT_PID" 2>/dev/null || true
fi

if [[ "$TRAIN_EXIT" -ne 0 ]]; then
  echo "ERROR: train_dpo exited $TRAIN_EXIT"
  exit "$TRAIN_EXIT"
fi

SUMMARY="${RUN}/trial_summary_v1.5.json"
python - <<'PY' "$SUMMARY"
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
s = json.loads(p.read_text())
print("target_reached", s.get("target_reached"))
print("COMPLETE", s.get("counts", {}).get("COMPLETE"))
print("best_trial", s.get("best_trial"), "best_v1_5", s.get("best_v1_5_survival_score"))
for t in s.get("trials", []):
    if t.get("state") == "COMPLETE":
        print(
            f"  #{t['trial_number']} survival={t.get('v1_5_survival_score')} "
            f"acc={t.get('eval_rewards_accuracy')} rpo={t.get('rpo_alpha')} "
            f"fixed={t.get('fixed_or_sampled')} id={t.get('fixed_id')}"
        )
PY

echo "=== v1.5 smoke done $(date -Is) ==="
echo "Summary: $SUMMARY"
echo "Copy: dpo/study_results/trial_summary_v1.5.json"
