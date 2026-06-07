#!/usr/bin/env bash
# Stop the full Saulie stack started by start_saulie.sh.
#
# Usage:
#   bash stop_saulie.sh              # stop agent, ngrok, containers, tmux session
#   bash stop_saulie.sh --keep-tmux  # stop services but leave dashboard session

set -euo pipefail

REPO="/root/saulie"
SESSION="saulie"
KEEP_TMUX=false

AGENT_PORT="${AGENT_PORT:-9000}"
VLLM_CONTAINER="eval_deploy_qwenie"
BGE_CONTAINER="tensorrt_bge-m3"
QDRANT_CONTAINER="qdrant_index"
NGINX_CONTAINER="nginx"

AGENT_PIDFILE="${REPO}/.agent_api.pid"

G=$'\033[0;32m'
Y=$'\033[1;33m'
B=$'\033[0;34m'
N=$'\033[0m'

log()  { printf "%s[*]%s %s\n" "$B" "$N" "$*"; }
ok()   { printf "%s[ok]%s %s\n" "$G" "$N" "$*"; }
skip() { printf "%s[-]%s %s\n" "$Y" "$N" "$*"; }

for arg in "$@"; do
  case "$arg" in
    --keep-tmux) KEEP_TMUX=true ;;
    -h|--help)
      sed -n '2,8p' "$0"
      exit 0
      ;;
  esac
done

need() {
  command -v "$1" >/dev/null 2>&1 || {
    printf "Missing required command: %s\n" "$1" >&2
    exit 1
  }
}

need docker

agent_running() {
  curl -sf --max-time 2 "http://127.0.0.1:${AGENT_PORT}/health" >/dev/null 2>&1
}

stop_agent() {
  local pid=""
  if [[ -f "$AGENT_PIDFILE" ]]; then
    pid="$(cat "$AGENT_PIDFILE" 2>/dev/null || true)"
  fi

  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    log "Stopping agent API (pid $pid)..."
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 15); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      log "Agent did not exit gracefully — sending SIGKILL"
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$AGENT_PIDFILE"
    ok "Agent API stopped"
    return 0
  fi

  if agent_running || pgrep -f "agent_chat_api.py api" >/dev/null 2>&1; then
    log "Stopping agent API (fallback pkill)..."
    pkill -f "agent_chat_api.py api" 2>/dev/null || true
    sleep 1
    rm -f "$AGENT_PIDFILE"
    ok "Agent API stopped"
    return 0
  fi

  skip "Agent API not running"
  rm -f "$AGENT_PIDFILE"
}

stop_ngrok() {
  if pgrep -x ngrok >/dev/null 2>&1; then
    log "Stopping ngrok..."
    pkill -x ngrok 2>/dev/null || true
    sleep 1
    ok "ngrok stopped"
    return 0
  fi
  skip "ngrok not running"
}

docker_running() {
  docker ps --format '{{.Names}}' | grep -qx "$1"
}

stop_container() {
  local name="$1"
  if docker_running "$name"; then
    log "Stopping container $name..."
    docker stop "$name" >/dev/null
    ok "Container $name stopped"
    return 0
  fi
  skip "Container $name not running"
}

stop_tmux() {
  if [[ "$KEEP_TMUX" == true ]]; then
    skip "Keeping tmux session '$SESSION' (--keep-tmux)"
    return 0
  fi
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    log "Killing tmux session '$SESSION'..."
    tmux kill-session -t "$SESSION"
    ok "Tmux session '$SESSION' closed"
    return 0
  fi
  skip "Tmux session '$SESSION' not running"
}

printf "\n%sStopping Saulie stack%s\n\n" "$Y" "$N"

stop_agent
stop_ngrok
stop_container "$NGINX_CONTAINER"
stop_container "$VLLM_CONTAINER"
stop_container "$BGE_CONTAINER"
stop_container "$QDRANT_CONTAINER"
stop_tmux

printf "\n%sAll services shut down.%s\n" "$G" "$N"
printf "Start again with: bash %s/start_saulie.sh\n\n" "$REPO"
