#!/usr/bin/env bash
# Resume main Optuna study + tmux: launcher | worker_0 | worker_1
set -euo pipefail
RUN=/root/saulie/dpo/train/models/steering-dpo-v1.0/optuna-run-20260523-041252
REPO=/root/saulie

pkill -f "train_dpo.py --optuna" 2>/dev/null || true
sleep 2

source /root/miniconda3/etc/profile.d/conda.sh
conda activate saulgman
cd "$REPO"

nohup python dpo/train/train_dpo.py --optuna --parallel-workers 2 --target-complete-trials 20 \
  --run-dir="$RUN" \
  --study-storage="$RUN/optuna_study.db" \
  --study-name=steering-dpo-v1.0-v4-seed42 \
  --dummy-report="$REPO/dpo/train/models/steering-dpo-v1.0/dummy-run/dummy_report.json" \
  >> /tmp/dpo_main_resume_stdout.log 2>&1 &
echo "Launcher PID=$!"

sleep 3
tmux kill-session -t dpo_main 2>/dev/null || true
tmux new-session -d -s dpo_main
tmux send-keys -t dpo_main "tail -f '$RUN/launcher.log'" C-m
tmux split-window -h -t dpo_main
tmux send-keys -t dpo_main "echo '=== worker_0 ===' && tail -f '$RUN/worker_0.log'" C-m
tmux split-window -v -t dpo_main
tmux send-keys -t dpo_main "echo '=== worker_1 ===' && tail -f '$RUN/worker_1.log'" C-m

echo ""
echo "Resume started. Attach:  tmux attach -t dpo_main"
echo "MONITOR.txt: $RUN/MONITOR.txt"
echo "Detach: Ctrl+b then d"
