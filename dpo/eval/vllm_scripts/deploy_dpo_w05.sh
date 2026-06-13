#!/bin/bash
# Production deploy: FP8 Qwen3 + DPO v1.5 trial-4 cat-merge @ dpo_weight=0.5
#
#   bash dpo/eval/vllm_scripts/merge_dpo_w05.sh   # once, if adapter missing
#   bash dpo/eval/vllm_scripts/deploy_dpo_w05.sh

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=docker/versions.env
source "${REPO}/docker/versions.env"

CONTAINER_NAME="eval_deploy_qwenie"
MODEL_PATH="/root/saulie/Qwen3-4B-Instruct-2507-FP8"
LORA_ADAPTER_PATH="/root/saulie/dpo/train/models/steering-dpo-v1.5/optuna-run-20260602-052732/trial-4/sft_dpo_cat_w05"
CONTAINER_LORA_PATH="/models/lora/dpo-v15-trial-4-w05"
MODEL_NAME="dpo-v15-trial-4-w05"
MAX_LORA_RANK=32
GPU_DEVICE="0"
PORT="8000"
VLLM_API_KEY="${VLLM_API_KEY:-dipshit}"

echo "╔═════════════════════════════════════════════════════════════════════════╗"
echo "║  DPO v1.5 Trial-4 — cat-merged LoRA @ dpo_weight=0.5                  ║"
echo "╚═════════════════════════════════════════════════════════════════════════╝"
echo ""

if [ ! -d "$MODEL_PATH" ]; then
  echo "Model not found: $MODEL_PATH"
  exit 1
fi

if [ ! -d "$LORA_ADAPTER_PATH" ]; then
  echo "LoRA adapter not found: $LORA_ADAPTER_PATH"
  echo "Run: bash dpo/eval/vllm_scripts/merge_dpo_w05.sh"
  exit 1
fi

echo " Base model:  $MODEL_PATH"
echo " LoRA:        $LORA_ADAPTER_PATH (dpo_weight=0.5)"
echo " Served as:   $MODEL_NAME"
echo ""

docker stop "$CONTAINER_NAME" 2>/dev/null && echo " Stopped $CONTAINER_NAME" || true
docker rm "$CONTAINER_NAME" 2>/dev/null && echo " Removed $CONTAINER_NAME" || true

echo "Starting vLLM container..."
docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus "device=${GPU_DEVICE}" \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --restart unless-stopped \
  -p "${PORT}:8000" \
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
    echo " API ready. Model: ${MODEL_NAME}"
    exit 0
  fi
  if [ "$i" -eq 90 ]; then
    echo "API failed to start"
    docker logs "${CONTAINER_NAME}" --tail 80
    exit 1
  fi
  sleep 2
done
