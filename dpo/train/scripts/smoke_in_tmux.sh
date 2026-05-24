#!/usr/bin/env bash
# Start Optuna smoke (2 workers, 2 complete) in tmux with live log panes.
# Attach: tmux attach -t agent_monitor
set -euo pipefail
cd /root/saulie
source /root/miniconda3/etc/profile.d/conda.sh
conda activate saulgman

tmux kill-session -t agent_monitor 2>/dev/null || true
tmux new-session -d -s agent_monitor -c /root/saulie

tmux send-keys -t agent_monitor \
  "source /root/miniconda3/etc/profile.d/conda.sh && conda activate saulgman && python dpo/train/train_dpo.py --optuna --optuna-smoke --parallel-workers 2" C-m

tmux split-window -h -t agent_monitor
tmux send-keys -t agent_monitor \
  'sleep 20; RUN=$(ls -td /root/saulie/dpo/train/models/steering-dpo-v1.0/optuna-run-* | head -1); echo "=== $RUN/worker_0.log ==="; tail -f "$RUN/worker_0.log"' C-m

tmux split-window -v -t agent_monitor:0.1
tmux send-keys -t agent_monitor \
  'sleep 20; RUN=$(ls -td /root/saulie/dpo/train/models/steering-dpo-v1.0/optuna-run-* | head -1); tail -f "$RUN/worker_1.log"' C-m

tmux select-pane -t agent_monitor:0.0
echo "Smoke started in tmux session 'agent_monitor'"
echo "Attach: tmux attach -t agent_monitor"
