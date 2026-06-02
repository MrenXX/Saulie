#!/usr/bin/env bash
# Plan B v1.3: train -> babysit -> 10-skeleton smoke -> fluency heuristics -> summary JSON
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate saulgman
cd /root/saulie

STAMP=$(date +%Y%m%d-%H%M%S)
RUN="dpo/train/models/steering-dpo-v1.3-sft-merged/optuna-run-${STAMP}"
STUDY_NAME="steering-dpo-v1.3-sft-merged-plan-b-seed42"
PIPE_LOG="${RUN}/pipeline.log"
JSONL="dpo/eval/plan_b_part1_w1_10skel_sample.jsonl"
FLUENCY_JSON="dpo/eval/plan_b_part1_fluency_score.json"

mkdir -p "$RUN"
echo "$RUN" > dpo/train/models/steering-dpo-v1.3-sft-merged/LATEST_RUN_DIR.txt

exec > >(tee -a "$PIPE_LOG") 2>&1
echo "=== Plan B v1.3 pipeline start $(date -Is) ==="
echo "RUN=$RUN"
echo "MLflow experiment: steering-dpo-v1.3-sft-merged"

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

python dpo/train/train_dpo.py --optuna --study-version v1.3 \
  --run-dir "$RUN" \
  --parallel-workers 1 \
  --target-complete-trials 4 \
  --max-attempted-trials 6

wait "$BABYSIT_PID" 2>/dev/null || true
trap - EXIT

echo "=== Training finished $(date -Is) ==="
if [[ ! -f "${RUN}/trial_summary.json" ]]; then
  echo "ERROR: missing trial_summary.json"
  exit 1
fi

echo "=== Stage A: 10-skeleton smoke ==="
python dpo/eval/run_plan_b_part1_hf_smoke.py \
  --run-dir "$RUN" \
  --output "$JSONL"

echo "=== Stage A: fluency heuristics ==="
set +e
python dpo/eval/score_plan_b_fluency.py --jsonl "$JSONL"
SCORE_EXIT=$?
set -e

python - <<'PY' "$JSONL" "$FLUENCY_JSON" "$SCORE_EXIT"
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
from dpo.eval.score_plan_b_fluency import score_jsonl

jsonl, out, score_exit = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
r = score_jsonl(jsonl)
r["heuristic_gate_exit_code"] = score_exit
out.write_text(json.dumps(r, indent=2), encoding="utf-8")
print(f"Wrote {out}")
PY

echo "=== Pipeline complete $(date -Is) ==="
echo "RUN=$RUN"
echo "JSONL=$JSONL"
echo "FLUENCY=$FLUENCY_JSON"
echo "tail -f ${RUN}/worker_0.log"
