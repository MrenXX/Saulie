#!/bin/bash
# Production deploy: FP8 Qwen3 + DPO v1.5 trial-4 cat-merged LoRA (r=32).
#
#   bash dpo/eval/vllm_scripts/deploy_finalist_pick.sh
#   python agent_chat_api.py api

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=docker/versions.env
source "${REPO}/docker/versions.env"
if [[ -f "${REPO}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO}/.env"
  set +a
fi
VLLM_API_KEY="${VLLM_API_KEY:-dipshit}"

CONTAINER_NAME="eval_deploy_qwenie"
MODEL_PATH="/root/saulie/Qwen3-4B-Instruct-2507-FP8"
LORA_ADAPTER_PATH="/root/saulie/dpo/train/models/steering-dpo-v1.5/optuna-run-20260602-052732/trial-4/sft_dpo_cat"
CONTAINER_LORA_PATH="/models/lora/dpo-v15-trial-4"
MODEL_NAME="dpo-v15-trial-4"
MAX_LORA_RANK=32
GPU_DEVICE="0"
PORT="8000"
# WSL2/Docker Desktop may reject 127.0.0.1:PORT binds; set VLLM_BIND=0.0.0.0 in .env
VLLM_BIND="${VLLM_BIND:-127.0.0.1}"
if [[ "$VLLM_BIND" == "0.0.0.0" || "$VLLM_BIND" == "all" ]]; then
  PORT_MAP="-p ${PORT}:8000"
else
  PORT_MAP="-p ${VLLM_BIND}:${PORT}:8000"
fi

echo "╔═════════════════════════════════════════════════════════════════════════╗"
echo "║  DPO v1.5 Trial-4 — FP8 Qwen3 + cat-merged LoRA (production)          ║"
echo "╚═════════════════════════════════════════════════════════════════════════╝"
echo ""

if [ ! -d "$MODEL_PATH" ]; then
  echo "Model not found: $MODEL_PATH"
  exit 1
fi

if [ ! -d "$LORA_ADAPTER_PATH" ]; then
  echo "LoRA adapter not found: $LORA_ADAPTER_PATH"
  exit 1
fi

if [ ! -f "$LORA_ADAPTER_PATH/adapter_config.json" ]; then
  echo "Missing adapter_config.json in $LORA_ADAPTER_PATH"
  exit 1
fi

echo " Base model:  $MODEL_PATH"
echo " LoRA:        $LORA_ADAPTER_PATH (r=32 cat-merged)"
echo " Served as:   $MODEL_NAME"
echo " Max rank:    $MAX_LORA_RANK"
echo ""

# Free port 8000 from old SFT deploy if still running
for OLD in deploy_qwenie eval_deploy_qwenie; do
  docker stop "$OLD" 2>/dev/null && echo " Stopped $OLD" || true
  docker rm "$OLD" 2>/dev/null && echo " Removed $OLD" || true
done

echo "Starting vLLM container..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus "device=${GPU_DEVICE}" \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --restart unless-stopped \
  ${PORT_MAP} \
  -v "${MODEL_PATH}:/models/model:ro" \
  -v "${LORA_ADAPTER_PATH}:${CONTAINER_LORA_PATH}:ro" \
  -e "VLLM_API_KEY=${VLLM_API_KEY}" \
  -e NCCL_P2P_DISABLE=1 \
  "${VLLM_IMAGE}" \
  --model /models/model \
  --served-model-name "Saulie" \
  --gpu-memory-utilization 0.5 \
  --max-model-len 4096 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 8 \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key "${VLLM_API_KEY}" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --enable-lora \
  --lora-modules "${MODEL_NAME}=${CONTAINER_LORA_PATH}" \
  --max-lora-rank "${MAX_LORA_RANK}" \
  --max-loras 1 \
  --trust-remote-code \
  --enable-prefix-caching

echo "Waiting for API on port ${PORT}..."
sleep 20

for i in $(seq 1 90); do
  if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    echo ""
    echo " API ready."
    echo ""
    echo " Model:     ${MODEL_NAME}"
    echo " vLLM key:  ${VLLM_API_KEY}"
    echo ""
    echo " Next:"
    echo "   python agent_chat_api.py api"
    echo "   docker start nginx && ngrok http 8080"
    exit 0
  fi
  if [ "$i" -eq 90 ]; then
    echo "API failed to start"
    docker logs "${CONTAINER_NAME}" --tail 80
    exit 1
  fi
  sleep 2
done
