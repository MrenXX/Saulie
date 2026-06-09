#!/usr/bin/env bash
# Run full 3×3 model×prompt matrix (SFT, DPO w=0.5, DPO w=1.0 × legacy/steering/compressed).
#
#   bash dpo/eval/run_model_prompt_matrix.sh
#
# Requires: Qdrant + BGE already up (or full stack). Stops/restarts vLLM per model.

set -euo pipefail

REPO="/root/saulie"
PYTHON="${SAULIE_PYTHON:-/root/miniconda3/envs/saulgman/bin/python}"
EVAL="${REPO}/dpo/eval/model_prompt_matrix_eval.py"
REPORT="${REPO}/MODEL_PROMPT_MATRIX_COMPARISON.md"
RUNS="${REPO}/dpo/eval/matrix_runs"
MERGE_W05="${REPO}/dpo/eval/vllm_scripts/merge_dpo_w05.sh"

log() { printf '[matrix] %s\n' "$*"; }

ensure_bge_qdrant() {
  docker start qdrant_index 2>/dev/null || true
  docker start tensorrt_bge-m3 2>/dev/null || true
  sleep 2
}

merge_w05_if_needed() {
  local out="${REPO}/dpo/train/models/steering-dpo-v1.5/optuna-run-20260602-052732/trial-4/sft_dpo_cat_w05"
  if [[ -f "${out}/adapter_config.json" ]]; then
    log "DPO w=0.5 cat adapter already exists"
    return 0
  fi
  log "Merging DPO w=0.5 cat adapter..."
  bash "$MERGE_W05"
}

stop_agent_only() {
  local pid=""
  pid="$(cat "${REPO}/.agent_api.pid" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    sleep 1
  fi
  pkill -f "agent_chat_api.py api" 2>/dev/null || true
  rm -f "${REPO}/.agent_api.pid"
}

stop_vllm_only() {
  docker stop eval_deploy_qwenie 2>/dev/null || true
  docker rm eval_deploy_qwenie 2>/dev/null || true
}

prepare_cell() {
  stop_agent_only
  stop_vllm_only
}

deploy_and_agent() {
  local deploy_script="$1"
  local model_name="$2"
  local prompt="$3"

  log "Deploying vLLM: ${deploy_script} (model=${model_name})"
  bash "$deploy_script"

  log "Starting agent: model=${model_name} prompt=${prompt}"
  stop_agent_only
  mkdir -p "$REPO"
  touch "${REPO}/agent_api.log" "${REPO}/saulie_chat.log"
  (
    cd "$REPO"
    nohup env QDRANT_COLLECTION="${QDRANT_COLLECTION:-amazon_products_v2}" \
      FUSION_METHOD="${FUSION_METHOD:-rrf}" \
      MODEL_NAME="$model_name" SAULIE_PROMPT="$prompt" \
      "$PYTHON" agent_chat_api.py api >>"${REPO}/agent_api.log" 2>&1 &
    echo $! >"${REPO}/.agent_api.pid"
  )

  for _ in $(seq 1 45); do
    if curl -sf "http://127.0.0.1:9000/health" >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done

  curl -sf "http://127.0.0.1:9000/health" | tee /dev/stderr
  echo ""
}

run_cell() {
  local model_key="$1"
  local model_name="$2"
  local prompt="$3"
  log "Eval cell: ${model_key} + ${prompt}"
  "$PYTHON" "$EVAL" \
    --model-key "$model_key" \
    --model-name "$model_name" \
    --prompt "$prompt"
}

mkdir -p "$RUNS"
merge_w05_if_needed
ensure_bge_qdrant

MODELS=(
  "sft|${REPO}/dpo/eval/vllm_scripts/deploy_sft_trial17_prod.sh|steering-sft-trial-17"
  "dpo_w05|${REPO}/dpo/eval/vllm_scripts/deploy_dpo_w05.sh|dpo-v15-trial-4-w05"
  "dpo_w10|${REPO}/dpo/eval/vllm_scripts/deploy_finalist_pick.sh|dpo-v15-trial-4"
)
PROMPTS=(legacy steering compressed)

for model_line in "${MODELS[@]}"; do
  IFS='|' read -r model_key deploy_script model_name <<<"$model_line"
  prepare_cell
  for prompt in "${PROMPTS[@]}"; do
    deploy_and_agent "$deploy_script" "$model_name" "$prompt"
    run_cell "$model_key" "$model_name" "$prompt" || log "WARN: cell failed ${model_key}/${prompt}"
    sleep 2
  done
done

log "Generating report..."
"$PYTHON" "${REPO}/dpo/eval/generate_matrix_report.py" --runs-dir "$RUNS" --output "$REPORT"
log "Done. Report: $REPORT"
