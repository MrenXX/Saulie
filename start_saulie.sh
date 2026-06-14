#!/usr/bin/env bash
# Start the full Saulie stack (RAG + vLLM + agent API + nginx + ngrok)
# and open a tmux dashboard for live monitoring.
#
# Usage:
#   bash start_saulie.sh          # start anything missing, open dashboard
#   bash start_saulie.sh --attach # skip startup, attach dashboard only
#
# Tmux prefix: Ctrl+B (also Ctrl+S per ~/.tmux.conf). Detach: Ctrl+B then D

set -euo pipefail

REPO="/root/saulie"
SESSION="saulie"
ATTACH_ONLY=false

PYTHON="${SAULIE_PYTHON:-/root/miniconda3/envs/saulgman/bin/python}"
AGENT_PORT="${AGENT_PORT:-9000}"
VLLM_PORT="${VLLM_PORT:-8000}"
NGINX_PORT="${NGINX_PORT:-8080}"
EMBED_PORT="${EMBED_PORT:-8888}"
QDRANT_PORT="${QDRANT_PORT:-1234}"
TAIL_LINES="${TAIL_LINES:-150}"

VLLM_CONTAINER="eval_deploy_qwenie"
BGE_CONTAINER="tensorrt_bge-m3"
QDRANT_CONTAINER="qdrant_index"
NGINX_CONTAINER="nginx"

AGENT_LOG="${REPO}/saulie_chat.log"
AGENT_STDLOG="${REPO}/agent_api.log"
AGENT_PIDFILE="${REPO}/.agent_api.pid"
NGROK_LOG="${REPO}/ngrok.log"
DEPLOY_SCRIPT="${SAULIE_DEPLOY_SCRIPT:-${REPO}/dpo/eval/vllm_scripts/deploy_finalist_pick.sh}"
SAULIE_MODEL="${SAULIE_MODEL:-dpo-v15-trial-4}"
SAULIE_PROMPT="${SAULIE_PROMPT:-compressed}"
SAULIE_NO_DASHBOARD="${SAULIE_NO_DASHBOARD:-0}"

# --- colors ---
G=$'\033[0;32m'
Y=$'\033[1;33m'
R=$'\033[0;31m'
B=$'\033[0;34m'
N=$'\033[0m'

log()  { printf "%s[*]%s %s\n" "$B" "$N" "$*"; }
ok()   { printf "%s[ok]%s %s\n" "$G" "$N" "$*"; }
warn() { printf "%s[!]%s %s\n" "$Y" "$N" "$*"; }
err()  { printf "%s[x]%s %s\n" "$R" "$N" "$*" >&2; }

for arg in "$@"; do
  case "$arg" in
    --attach|-a) ATTACH_ONLY=true ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
  esac
done

need() {
  command -v "$1" >/dev/null 2>&1 || {
    err "Missing required command: $1"
    exit 1
  }
}

need tmux
need curl
need docker

if [[ -f "${REPO}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO}/.env"
  set +a
fi

export QDRANT_COLLECTION="${QDRANT_COLLECTION:-amazon_products_v2}"
export FUSION_METHOD="${FUSION_METHOD:-rrf}"
NGINX_API_KEY="${NGINX_API_KEY:-secret}"
VLLM_API_KEY="${VLLM_API_KEY:-dipshit}"
LLM_API_KEY="${LLM_API_KEY:-dipshit}"
export NGINX_API_KEY VLLM_API_KEY LLM_API_KEY

prepare_nginx_conf() {
  local src="${REPO}/nginx/nginx.conf"
  local dst="${REPO}/nginx/nginx.runtime.conf"
  if [[ ! -f "$src" ]]; then
    err "Missing $src"
    return 1
  fi
  # Substitute NGINX_API_KEY from .env into the Bearer map (placeholder __NGINX_API_KEY__)
  sed "s|__NGINX_API_KEY__|${NGINX_API_KEY}|g" "$src" >"$dst"
}

http_ok() {
  curl -sf --max-time 3 "$1" >/dev/null 2>&1
}

embed_ok() {
  curl -sf --max-time 5 \
    -X POST "http://127.0.0.1:${EMBED_PORT}/embed" \
    -H "Content-Type: application/json" \
    -d '{"text":"healthcheck"}' >/dev/null 2>&1
}

docker_running() {
  docker ps --format '{{.Names}}' | grep -qx "$1"
}

docker_exists() {
  docker ps -a --format '{{.Names}}' | grep -qx "$1"
}

wait_for() {
  local label="$1" url="$2" timeout="${3:-120}"
  local i=0
  while (( i < timeout )); do
    if http_ok "$url"; then
      ok "$label ready"
      return 0
    fi
    if [[ -n "${4:-}" ]] && ! docker_running "$4"; then
      err "$label container stopped while waiting ($url)"
      docker logs "$4" --tail 40 2>&1 || true
      return 1
    fi
    sleep 2
    (( i += 2 )) || true
  done
  err "$label did not become ready within ${timeout}s ($url)"
  return 1
}

redeploy_vllm() {
  if [[ ! -f "$DEPLOY_SCRIPT" ]]; then
    err "vLLM redeploy failed — script not found: $DEPLOY_SCRIPT"
    return 1
  fi
  warn "Recreating vLLM container via deploy script (may take a few minutes)..."
  bash "$DEPLOY_SCRIPT"
}

ensure_vllm() {
  if docker_running "$VLLM_CONTAINER" && http_ok "http://127.0.0.1:${VLLM_PORT}/health"; then
    ok "vLLM already healthy on :${VLLM_PORT}"
    return 0
  fi

  if docker_exists "$VLLM_CONTAINER"; then
    if ! docker_running "$VLLM_CONTAINER"; then
      log "Starting existing vLLM container..."
      if ! docker start "$VLLM_CONTAINER" 2>/dev/null; then
        warn "docker start failed — removing broken container and redeploying"
        docker rm -f "$VLLM_CONTAINER" 2>/dev/null || true
        redeploy_vllm
        return $?
      fi
      sleep 3
      if ! docker_running "$VLLM_CONTAINER"; then
        warn "vLLM container exited immediately — redeploying"
        docker logs "$VLLM_CONTAINER" --tail 30 2>&1 || true
        docker rm -f "$VLLM_CONTAINER" 2>/dev/null || true
        redeploy_vllm
        return $?
      fi
    fi
    wait_for "vLLM" "http://127.0.0.1:${VLLM_PORT}/health" 180 "$VLLM_CONTAINER" || {
      warn "vLLM health check timed out — redeploying"
      docker rm -f "$VLLM_CONTAINER" 2>/dev/null || true
      redeploy_vllm
      return $?
    }
    return 0
  fi

  redeploy_vllm
}

wait_for_embed() {
  local timeout="${1:-90}"
  local i=0
  while (( i < timeout )); do
    if embed_ok; then
      ok "BGE embed server ready"
      return 0
    fi
    sleep 2
    (( i += 2 )) || true
  done
  err "BGE embed server not ready on :${EMBED_PORT}"
  return 1
}

agent_running() {
  http_ok "http://127.0.0.1:${AGENT_PORT}/health"
}

stop_agent_quick() {
  local pid=""
  pid="$(cat "$AGENT_PIDFILE" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 10); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$pid" 2>/dev/null || true
  elif pgrep -f "agent_chat_api.py api" >/dev/null 2>&1; then
    pkill -f "agent_chat_api.py api" 2>/dev/null || true
    sleep 1
  fi
  rm -f "$AGENT_PIDFILE"
}

start_agent() {
  if [[ ! -x "$PYTHON" ]]; then
    err "Python not found: $PYTHON (set SAULIE_PYTHON)"
    return 1
  fi

  if agent_running || [[ -f "$AGENT_PIDFILE" ]]; then
    log "Restarting agent API (RAG: ${QDRANT_COLLECTION} / ${FUSION_METHOD})..."
    stop_agent_quick
  fi

  log "Starting agent API on :${AGENT_PORT} (model=${SAULIE_MODEL} prompt=${SAULIE_PROMPT})..."
  mkdir -p "$REPO"
  touch "$AGENT_LOG" "$AGENT_STDLOG"
  (
    cd "$REPO"
    nohup env QDRANT_COLLECTION="$QDRANT_COLLECTION" FUSION_METHOD="$FUSION_METHOD" \
      MODEL_NAME="$SAULIE_MODEL" SAULIE_PROMPT="$SAULIE_PROMPT" \
      LLM_API_KEY="$LLM_API_KEY" AGENT_HOST="127.0.0.1" \
      "$PYTHON" agent_chat_api.py api >>"$AGENT_STDLOG" 2>&1 &
    echo $! >"$AGENT_PIDFILE"
  )
  wait_for "Agent API" "http://127.0.0.1:${AGENT_PORT}/health" 45
}

start_ngrok() {
  if curl -sf --max-time 2 "http://127.0.0.1:4040/api/tunnels" >/dev/null 2>&1; then
    ok "ngrok already running"
    return 0
  fi

  if ! command -v ngrok >/dev/null 2>&1; then
    warn "ngrok not installed — skipping public tunnel"
    return 0
  fi

  if [[ -z "${NGROK_KEY:-}" ]]; then
    warn "NGROK_KEY not set in ${REPO}/.env — skipping ngrok"
    return 0
  fi

  log "Starting ngrok → :${NGINX_PORT}..."
  ngrok config add-authtoken "$NGROK_KEY" >/dev/null 2>&1 || true
  nohup ngrok http "$NGINX_PORT" --log=stdout >>"$NGROK_LOG" 2>&1 &
  sleep 2
  ok "ngrok started (URL shown in dashboard status pane)"
}

ensure_docker_container() {
  local name="$1"
  if docker_running "$name"; then
    ok "Container $name already running"
    return 0
  fi
  if docker_exists "$name"; then
    log "Starting container $name..."
    if docker start "$name" >/dev/null 2>&1 && docker_running "$name"; then
      ok "Container $name started"
      return 0
    fi
    err "Container $name failed to start — check: docker logs $name"
    return 1
  fi
  warn "Container $name not found — create/deploy it first"
  return 1
}

ensure_nginx() {
  if docker_running "$NGINX_CONTAINER"; then
    ok "Container $NGINX_CONTAINER already running"
    return 0
  fi

  if docker_exists "$NGINX_CONTAINER"; then
    log "Starting existing nginx container..."
    if docker start "$NGINX_CONTAINER" >/dev/null 2>&1 && sleep 1 && docker_running "$NGINX_CONTAINER"; then
      ok "Container $NGINX_CONTAINER started"
      return 0
    fi
    warn "nginx container has stale/broken bind mounts — recreating..."
    docker rm -f "$NGINX_CONTAINER" >/dev/null 2>&1 || true
  fi

  if [[ ! -f "${REPO}/nginx/nginx.conf" ]]; then
    err "Missing ${REPO}/nginx/nginx.conf — cannot create nginx container"
    return 1
  fi

  prepare_nginx_conf || return 1

  log "Creating nginx container via docker compose..."
  docker compose -f "${REPO}/nginx/docker-compose.nginx.yml" up -d
  sleep 1
  if docker_running "$NGINX_CONTAINER"; then
    ok "Container $NGINX_CONTAINER created and running"
    return 0
  fi
  err "nginx failed to start after recreate"
  return 1
}

ensure_bge() {
  ensure_docker_container "$BGE_CONTAINER" || return 0
  if embed_ok; then
    ok "BGE embed server already responding"
    return 0
  fi
  log "Starting serve.py inside $BGE_CONTAINER..."
  docker exec -d "$BGE_CONTAINER" sh -c "python -u serve.py >> /proc/1/fd/1 2>&1" || true
  wait_for_embed 90 || warn "BGE may still be warming up — check rag pane"
}

ngrok_public_url() {
  curl -sf --max-time 2 "http://127.0.0.1:4040/api/tunnels" 2>/dev/null \
    | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)
for t in data.get('tunnels', []):
    if t.get('proto') == 'https':
        print(t.get('public_url', ''))
        break
" 2>/dev/null || true
}

attach_session() {
  if [[ -n "${TMUX-}" ]]; then
    exec tmux switch-client -t "$SESSION"
  else
    exec tmux attach -t "$SESSION"
  fi
}

log_runner() {
  local name="$1"
  printf "bash -lc 'echo \"==> docker logs: %s\"; docker logs --timestamps --follow --tail %s \"%s\" 2>&1 || { echo; echo \"[!] Could not follow %s\"; exec bash; }'" \
    "$name" "$TAIL_LINES" "$name" "$name"
}

file_tail_runner() {
  local label="$1" file="$2"
  printf "bash -lc 'echo \"==> %s\"; touch \"%s\"; tail -n %s -F \"%s\"'" \
    "$label" "$file" "$TAIL_LINES" "$file"
}

SAULIE_DIR="${REPO}/.saulie"
STATUS_SCRIPT="${SAULIE_DIR}/status_loop.sh"

write_status_script() {
  mkdir -p "$SAULIE_DIR"
  cat >"$STATUS_SCRIPT" <<EOF
#!/usr/bin/env bash
while true; do
  clear
  echo "══════════════════════════════════════════════════════════════"
  echo " SAULIE STACK — \$(date '+%Y-%m-%d %H:%M:%S')"
  echo "══════════════════════════════════════════════════════════════"
  printf " vLLM (:${VLLM_PORT})     : "; curl -sf --max-time 2 "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null && echo "UP" || echo "DOWN"
  printf " Agent (:${AGENT_PORT})   : "; curl -sf --max-time 2 "http://127.0.0.1:${AGENT_PORT}/health" >/dev/null && echo "UP" || echo "DOWN"
  printf " nginx (:${NGINX_PORT})   : "; code=\$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 -X POST "http://127.0.0.1:${NGINX_PORT}/v1/chat/completions" -H "Content-Type: application/json" -d '{}'); [[ "\$code" == "401" ]] && echo "UP" || echo "DOWN"
  printf " BGE (:${EMBED_PORT})     : "; curl -sf --max-time 3 -X POST "http://127.0.0.1:${EMBED_PORT}/embed" -H "Content-Type: application/json" -d '{"text":"ping"}' >/dev/null && echo "UP" || echo "DOWN"
  printf " Qdrant (:${QDRANT_PORT}) : "; curl -sf --max-time 2 "http://127.0.0.1:${QDRANT_PORT}/" >/dev/null && echo "UP" || echo "DOWN"
  echo "──────────────────────────────────────────────────────────────"
  echo " Local agent : http://127.0.0.1:${AGENT_PORT}/health  (localhost only)"
  echo " nginx proxy : http://127.0.0.1:${NGINX_PORT}/v1/chat/completions  (Bearer ${NGINX_API_KEY})"
  URL=\$(curl -sf --max-time 2 http://127.0.0.1:4040/api/tunnels 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(next((t['public_url'] for t in d.get('tunnels',[]) if t.get('proto')=='https'), 'ngrok not running'))" 2>/dev/null || echo "ngrok not running")
  echo " Public URL  : \$URL"
  echo "──────────────────────────────────────────────────────────────"
  echo " tmux: Ctrl+B D detach | Ctrl+B [ scroll | windows 1-4 at bottom"
  echo "══════════════════════════════════════════════════════════════"
  sleep 3
done
EOF
  chmod +x "$STATUS_SCRIPT"
}

status_runner() {
  echo "bash -lc '${STATUS_SCRIPT}'"
}

gpu_runner() {
  echo "bash -lc 'if command -v gpustat >/dev/null 2>&1; then exec gpustat -cpuP -i 2; else echo \"[!] gpustat not installed\"; exec docker stats --no-trunc; fi'"
}

nvitop_runner() {
  echo "bash -lc 'NVITOP=\$(command -v nvitop || true); [[ -z \"\$NVITOP\" && -x /root/miniconda3/envs/saulgman/bin/nvitop ]] && NVITOP=/root/miniconda3/envs/saulgman/bin/nvitop; if [[ -n \"\$NVITOP\" ]]; then exec \"\$NVITOP\"; else echo \"[!] nvitop not installed — pip install nvitop\"; exec bash; fi'"
}

create_tmux_dashboard() {
  log "Creating tmux dashboard session '$SESSION'..."
  write_status_script

  tmux new-session -d -s "$SESSION" -n status -c "$REPO"
  tmux set -t "$SESSION" -g remain-on-exit on
  tmux set -t "$SESSION" -g pane-border-status top
  tmux set -t "$SESSION" -g pane-border-format ' #{pane_index}: #{pane_title} '
  tmux set -t "$SESSION" -g status-left "#[fg=green]${SESSION}#[default] "
  tmux set -t "$SESSION" -g status-right "#[fg=cyan]%Y-%m-%d %H:%M#[default]"

  # Window 1 — status + GPU
  WIN="${SESSION}:status"
  P0="$(tmux list-panes -t "$WIN" -F '#{pane_id}' | head -n1)"
  tmux select-pane -t "$P0" -T "stack status"
  tmux respawn-pane -k -t "$P0" "$(status_runner)"

  P1="$(tmux split-window -v -t "$P0" -P -F '#{pane_id}' "$(gpu_runner)")"
  tmux select-pane -t "$P1" -T "GPU / docker stats"
  tmux resize-pane -t "$P0" -y 20

  # Window 2 — vLLM logs + nvitop
  tmux new-window -t "$SESSION" -n vllm -c "$REPO"
  P_VLLM="$(tmux list-panes -t "${SESSION}:vllm" -F '#{pane_id}' | head -n1)"
  tmux select-pane -t "$P_VLLM" -T "vLLM: ${VLLM_CONTAINER}"
  tmux respawn-pane -k -t "$P_VLLM" "$(log_runner "$VLLM_CONTAINER")"
  P_NVITOP="$(tmux split-window -h -t "$P_VLLM" -P -F '#{pane_id}' "$(nvitop_runner)")"
  tmux select-pane -t "$P_NVITOP" -T "nvitop"
  tmux select-pane -t "$P_VLLM"

  # Window 3 — RAG (BGE + Qdrant)
  tmux new-window -t "$SESSION" -n rag -c "$REPO"
  P_BGE="$(tmux list-panes -t "${SESSION}:rag" -F '#{pane_id}' | head -n1)"
  tmux select-pane -t "$P_BGE" -T "BGE: ${BGE_CONTAINER}"
  tmux respawn-pane -k -t "$P_BGE" "$(log_runner "$BGE_CONTAINER")"
  P_QD="$(tmux split-window -h -t "$P_BGE" -P -F '#{pane_id}' "$(log_runner "$QDRANT_CONTAINER")")"
  tmux select-pane -t "$P_QD" -T "Qdrant: ${QDRANT_CONTAINER}"

  # Window 4 — agent / edge
  tmux new-window -t "$SESSION" -n agent -c "$REPO"
  P_API="$(tmux list-panes -t "${SESSION}:agent" -F '#{pane_id}' | head -n1)"
  tmux select-pane -t "$P_API" -T "agent uvicorn"
  tmux respawn-pane -k -t "$P_API" "$(file_tail_runner "agent stdout: agent_api.log" "$AGENT_STDLOG")"
  P_CHAT="$(tmux split-window -v -t "$P_API" -P -F '#{pane_id}' "$(file_tail_runner "agent app log: saulie_chat.log" "$AGENT_LOG")")"
  tmux select-pane -t "$P_CHAT" -T "agent app log"
  P_NGX="$(tmux split-window -h -t "$P_API" -P -F '#{pane_id}' "$(log_runner "$NGINX_CONTAINER")")"
  tmux select-pane -t "$P_NGX" -T "nginx"
  P_NGROK="$(tmux split-window -v -t "$P_NGX" -P -F '#{pane_id}' "$(file_tail_runner "ngrok log" "$NGROK_LOG")")"
  tmux select-pane -t "$P_NGROK" -T "ngrok"
  tmux select-layout -t "${SESSION}:agent" tiled

  tmux select-window -t "${SESSION}:status"
  ok "Dashboard created"
}

open_dashboard() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    ok "Tmux session '$SESSION' already exists — attaching"
  else
    create_tmux_dashboard
  fi
  ok "Dashboard ready — Ctrl+B D to detach, re-run: bash start_saulie.sh"
  attach_session
}

# --- main ---

if [[ "$ATTACH_ONLY" == true ]]; then
  tmux has-session -t "$SESSION" 2>/dev/null || {
    err "No tmux session '$SESSION'. Run without --attach first."
    exit 1
  }
  open_dashboard
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  warn "Tmux session '$SESSION' already running — will attach after service checks"
fi

printf "\n%sStarting Saulie stack%s\n\n" "$G" "$N"
log "RAG defaults: QDRANT_COLLECTION=${QDRANT_COLLECTION} FUSION_METHOD=${FUSION_METHOD}"
log "Model: ${SAULIE_MODEL}  Prompt: ${SAULIE_PROMPT}  Deploy: ${DEPLOY_SCRIPT}"

ensure_docker_container "$QDRANT_CONTAINER" || true
ensure_bge
ensure_vllm || warn "vLLM not ready — agent will start but LLM calls may fail"
ensure_nginx || warn "nginx not ready — use agent directly on :${AGENT_PORT}"
start_agent || warn "Agent failed to start — check agent window in dashboard"
start_ngrok

PUBLIC_URL="$(ngrok_public_url)"
if [[ -n "$PUBLIC_URL" ]]; then
  ok "Public URL: $PUBLIC_URL  (Bearer token: secret)"
else
  warn "No public ngrok URL yet — check ngrok pane or: curl -s http://127.0.0.1:4040/api/tunnels"
fi

ok "Stack startup complete"
if [[ "$SAULIE_NO_DASHBOARD" == "1" ]]; then
  ok "SAULIE_NO_DASHBOARD=1 — skipping tmux dashboard"
  exit 0
fi
open_dashboard
