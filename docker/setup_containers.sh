#!/usr/bin/env bash
# Create Qdrant and BGE TensorRT containers if they do not exist.
#
# Usage:
#   bash docker/setup_containers.sh
#
# Prerequisites: Docker with GPU support, RAG_ROOT with BGE model workspace.
# See REPRODUCIBILITY.md for full setup.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -f "$REPO/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO/.env"
  set +a
fi

# shellcheck source=docker/versions.env
source "$SCRIPT_DIR/versions.env"

RAG_ROOT="${RAG_ROOT:-/root/saulie/rag}"
QDRANT_STORAGE="${RAG_ROOT}/qdrant_storage"
BGE_WORKSPACE="${RAG_ROOT}/embed_models/bge-m3"
EMBED_PORT="${EMBED_PORT:-8888}"
QDRANT_PORT="${QDRANT_PORT:-1234}"

log()  { printf '[*] %s\n' "$*"; }
ok()   { printf '[ok] %s\n' "$*"; }
warn() { printf '[!] %s\n' "$*"; }
err()  { printf '[x] %s\n' "$*" >&2; }

docker_running() {
  docker ps --format '{{.Names}}' | grep -qx "$1"
}

docker_exists() {
  docker ps -a --format '{{.Names}}' | grep -qx "$1"
}

ensure_qdrant() {
  if docker_running "$QDRANT_CONTAINER"; then
    ok "Qdrant already running ($QDRANT_CONTAINER)"
    return 0
  fi

  mkdir -p "$QDRANT_STORAGE"

  if docker_exists "$QDRANT_CONTAINER"; then
    log "Starting existing Qdrant container..."
    docker start "$QDRANT_CONTAINER"
    ok "Qdrant started"
    return 0
  fi

  log "Creating Qdrant container ($QDRANT_CONTAINER)..."
  docker run -d \
    --name "$QDRANT_CONTAINER" \
    --restart unless-stopped \
    -p "${QDRANT_PORT}:6333" \
    -v "${QDRANT_STORAGE}:/qdrant/storage" \
    "$QDRANT_IMAGE"
  ok "Qdrant created on port ${QDRANT_PORT}"
}

ensure_bge() {
  if [[ ! -d "$BGE_WORKSPACE" ]]; then
    err "BGE workspace not found: $BGE_WORKSPACE"
    err "Symlink or copy rag/embed_models/bge-m3 to RAG_ROOT and download BGE-M3 weights."
    return 1
  fi

  if [[ ! -f "$BGE_WORKSPACE/serve.py" ]]; then
    err "Missing $BGE_WORKSPACE/serve.py"
    return 1
  fi

  if docker_running "$BGE_CONTAINER"; then
    ok "BGE already running ($BGE_CONTAINER)"
    start_bge_server
    return 0
  fi

  if docker_exists "$BGE_CONTAINER"; then
    log "Starting existing BGE container..."
    docker start "$BGE_CONTAINER"
    sleep 2
    start_bge_server
    ok "BGE started"
    return 0
  fi

  log "Creating BGE TensorRT container ($BGE_CONTAINER)..."
  docker run -d \
    --name "$BGE_CONTAINER" \
    --gpus all \
    -p "${EMBED_PORT}:8000" \
    -v "${BGE_WORKSPACE}:/workspace" \
    "$BGE_IMAGE" \
    sleep infinity

  sleep 2
  start_bge_server
  ok "BGE created on port ${EMBED_PORT}"
}

start_bge_server() {
  if curl -sf --max-time 3 -X POST "http://127.0.0.1:${EMBED_PORT}/embed" \
    -H "Content-Type: application/json" -d '{"text":"healthcheck"}' >/dev/null 2>&1; then
    ok "BGE embed server already responding"
    return 0
  fi

  if ! docker exec "$BGE_CONTAINER" python -c "import uvicorn, fastapi" >/dev/null 2>&1; then
    log "Installing BGE serve.py dependencies in container..."
    if ! docker exec "$BGE_CONTAINER" python -c "import torch" >/dev/null 2>&1; then
      docker exec "$BGE_CONTAINER" pip install torch --index-url https://download.pytorch.org/whl/cu124
    fi
    docker exec "$BGE_CONTAINER" pip install -q uvicorn fastapi pydantic transformers
  fi

  log "Starting BGE serve.py inside container..."
  docker exec -d "$BGE_CONTAINER" sh -c "cd /workspace && python -u serve.py"
  sleep 5

  if curl -sf --max-time 5 -X POST "http://127.0.0.1:${EMBED_PORT}/embed" \
    -H "Content-Type: application/json" -d '{"text":"healthcheck"}' >/dev/null 2>&1; then
    ok "BGE embed server ready"
  else
    warn "BGE server not responding yet — engine may still be loading. Check: docker logs $BGE_CONTAINER"
  fi
}

ensure_qdrant
ensure_bge

log "Infra setup complete."
log "  Qdrant: http://127.0.0.1:${QDRANT_PORT}"
log "  BGE:    http://127.0.0.1:${EMBED_PORT}/embed"
