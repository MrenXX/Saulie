#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-rag-monitor}"
TAIL_LINES="${TAIL_LINES:-200}"

C1="eval_deploy_qwenie"
C2="tensorrt_bge-m3"
C3="qdrant_index"

need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing required command: $1" >&2; exit 1; }; }
need tmux
need docker

log_runner() {
  local name="$1"
  # Run docker logs; if it fails (missing container, not started, etc), keep an interactive shell open.
  echo "bash -lc 'echo \"==> logs: $name (tail=$TAIL_LINES)\"; docker logs --timestamps --follow --tail $TAIL_LINES \"$name\" || { echo; echo \"[!] Could not follow logs for $name\"; echo \"    Check name with: docker ps -a --format \\\"{{.Names}}\\\"\"; echo \"    Then re-run the script (or restart the container).\"; exec bash; }'"
}

gpu_monitor() {
  echo "bash -lc 'if command -v gpustat >/dev/null 2>&1; then exec gpustat -cpuP -i 1; else echo \"[!] gpustat not installed; using docker stats\"; echo \"    Install with: pip install gpustat\"; exec docker stats; fi'"
}

# If session exists, just go to it (switch if already in tmux, attach otherwise)
if tmux has-session -t "$SESSION" 2>/dev/null; then
  if [[ -n "${TMUX-}" ]]; then
    exec tmux switch-client -t "$SESSION"
  else
    exec tmux attach -t "$SESSION"
  fi
fi

# Create session + window (don't assume index 0/1; use window name)
tmux new-session -d -s "$SESSION" -n monitor
WIN="$SESSION:monitor"

# Niceties
# tmux set -t "$SESSION" -g mouse on
tmux set -t "$SESSION" -g remain-on-exit on
tmux set -t "$SESSION" -g pane-border-status top
tmux set -t "$SESSION" -g pane-border-format ' #{pane_index}: #{pane_title} '
tmux set -t "$SESSION" -g status-left "#[fg=green]$SESSION#[default] "
tmux set -t "$SESSION" -g status-right "#[fg=cyan]%Y-%m-%d %H:%M#[default]"

# Get the initial pane id reliably
P0="$(tmux list-panes -t "$WIN" -F '#{pane_id}' | head -n1)"

# Pane 0: C1 logs
tmux select-pane -t "$P0" -T "logs: $C1"
tmux respawn-pane -k -t "$P0" "$(log_runner "$C1")"

# Pane 1: split right (C2 logs)
P1="$(tmux split-window -h -t "$P0" -P -F '#{pane_id}' "$(log_runner "$C2")")"
tmux select-pane -t "$P1" -T "logs: $C2"

# Pane 2: split bottom of left (C3 logs)
P2="$(tmux split-window -v -t "$P0" -P -F '#{pane_id}' "$(log_runner "$C3")")"
tmux select-pane -t "$P2" -T "logs: $C3"

# Pane 3: split bottom of right (gpustat / docker stats)
P3="$(tmux split-window -v -t "$P1" -P -F '#{pane_id}' "$(gpu_monitor)")"
tmux select-pane -t "$P3" -T "gpustat / docker stats"

tmux select-layout -t "$WIN" tiled

# Enter session
if [[ -n "${TMUX-}" ]]; then
  exec tmux switch-client -t "$SESSION"
else
  exec tmux attach -t "$SESSION"
fi