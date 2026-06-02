#!/usr/bin/env bash
# Restart Optuna workers for an in-flight v1.5 run (picks up code fixes).
set -euo pipefail
RUN="${1:?usage: restart_v15_workers.sh RUN_DIR}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate saulgman
cd /root/saulie

STUDY_NAME="${STUDY_NAME:-steering-dpo-v1.5-hybrid-seed42}"
RUN_ABS="$(readlink -f "$RUN")"

echo "Restarting workers for $RUN_ABS"
pkill -f "${RUN_ABS}.*optuna-worker" 2>/dev/null || true
sleep 5

python - <<PY
import optuna
from pathlib import Path
from dpo.train.optuna_parallel import V1_5_FIXED_TRIALS, enqueue_v1_5_fixed_trials, params_key
from dpo.train.optuna_parallel import OptunaRunConfig
from optuna.trial import TrialState

run = Path("$RUN_ABS")
storage = f"sqlite:///{(run/'optuna_study.db').resolve()}"
study = optuna.load_study(study_name="$STUDY_NAME", storage=storage)
try:
    optuna.storages.fail_stale_trials(study)
    print("fail_stale_trials: ok")
except Exception as e:
    print("fail_stale_trials:", e)

cfg = OptunaRunConfig(
    run_dir=run,
    study_storage=run / "optuna_study.db",
    study_name="$STUDY_NAME",
    target_complete_trials=20,
    max_attempted_trials=60,
    parallel_workers=2,
    study_version="v1.5",
    experiment_name="steering-dpo-v1.5",
)
# Re-enqueue fixed controls if they failed and are not already waiting/complete
for fixed in V1_5_FIXED_TRIALS[:2]:
    key = params_key(fixed)
    ok = False
    for t in study.trials:
        if t.state == TrialState.COMPLETE and params_key(dict(t.params) or {}) == key:
            ok = True
            break
        ua = t.user_attrs or {}
        if t.state == TrialState.COMPLETE and ua.get("trial_params"):
            if params_key(ua["trial_params"]) == key:
                ok = True
                break
    if not ok:
        study.enqueue_trial(fixed)
        print("re-enqueued", key)
n = enqueue_v1_5_fixed_trials(study, cfg)
print("enqueue_v1_5_fixed_trials added", n)
PY

for wid in 0 1; do
  nohup python dpo/train/train_dpo.py --optuna --optuna-worker --worker-id="$wid" \
    --run-dir="$RUN_ABS" \
    --study-storage="${RUN_ABS}/optuna_study.db" \
    --study-name="$STUDY_NAME" \
    --target-complete-trials=20 \
    --max-attempted-trials=60 \
    --study-version=v1.5 \
    --experiment-name=steering-dpo-v1.5 \
    >> "${RUN_ABS}/worker_${wid}.log" 2>&1 &
  echo "worker $wid pid $!"
done
echo "Workers restarted."
