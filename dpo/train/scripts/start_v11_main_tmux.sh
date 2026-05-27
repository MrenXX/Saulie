#!/usr/bin/env bash
# Official DPO Optuna v1.1 study in tmux (launcher + log tails).
set -euo pipefail
REPO=/root/saulie
SESSION=dpo_v11

cd "$REPO"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate saulgman

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -c "$REPO"

# Pane 0: launcher (foreground)
tmux send-keys -t "$SESSION" \
  "source /root/miniconda3/etc/profile.d/conda.sh && conda activate saulgman && \
python dpo/train/train_dpo.py --optuna --study-version v1.1 --parallel-workers 2" C-m

# Extra windows for logs (after run dir exists)
sleep 12
RUN=$(ls -td "$REPO/dpo/train/models/steering-dpo-v1.1"/optuna-run-* 2>/dev/null | head -1)
if [[ -n "${RUN:-}" ]]; then
  tmux new-window -t "$SESSION" -n launch "tail -f '$RUN/launcher.log'"
  tmux new-window -t "$SESSION" -n w0 "tail -f '$RUN/worker_0.log'"
  tmux new-window -t "$SESSION" -n w1 "tail -f '$RUN/worker_1.log'"
  echo "RUN=$RUN"
fi

tmux select-window -t "$SESSION:0"
echo "SESSION=$SESSION"
