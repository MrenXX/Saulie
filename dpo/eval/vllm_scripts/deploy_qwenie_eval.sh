#!/bin/bash
# Deploy FP8 Qwen3 + LoRA for DPO final eval.
#
# Modes:
#   DEPLOY_MODE=eval_runtime (default for v1.5 final eval)
#     - Preload SFT baseline only at startup (MAX_LORAS=2 slot 1)
#     - DPO cat adapters loaded via POST /v1/load_lora_adapter (no restart per trial)
#     - VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
#     - Mounts v1.5 run dir for runtime cat paths
#
#   DEPLOY_MODE=manifest (legacy)
#     - All manifest lines in --lora-modules at startup
#
# Cat-merged LoRA rank: r=32 (SFT r=16 + DPO r=16 merged into one adapter).
# --max-lora-rank must be >= 32 (not 48 unless serving two separate LoRAs in one forward).

set -euo pipefail

CONTAINER_NAME="eval_deploy_qwenie"
MODEL_PATH="/root/saulie/Qwen3-4B-Instruct-2507-FP8"
V15_RUN_HOST="/root/saulie/dpo/train/models/steering-dpo-v1.5/optuna-run-20260602-052732"
GPU_DEVICE="0"
PORT="8000"
VLLM_API_KEY="${VLLM_API_KEY:-dipshit}"
VLLM_IMAGE="vllm/vllm-openai:latest"

DEPLOY_MODE="${DEPLOY_MODE:-eval_runtime}"
CANDIDATE_MANIFEST="${CANDIDATE_MANIFEST:-}"

# Cat-merged r = r_sft + r_dpo (32 or 40 on this slate). vLLM allows 32|64|... — use 64 for r=40.
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"
MAX_LORAS="${MAX_LORAS:-2}"

REPO_ROOT="/root/saulie"
if [ "$DEPLOY_MODE" = "eval_runtime" ]; then
  CANDIDATE_MANIFEST="${CANDIDATE_MANIFEST:-$REPO_ROOT/dpo/eval/v15_deploy_startup_manifest.jsonl}"
  MAX_LORA_RANK="${MAX_LORA_RANK:-64}"
  MAX_LORAS="${MAX_LORAS:-2}"
fi

VOLUME_ARGS=()
LORA_MODULES=()
EXTRA_ENV=(-e "VLLM_API_KEY=${VLLM_API_KEY}" -e "NCCL_P2P_DISABLE=1")

if [ "$DEPLOY_MODE" = "eval_runtime" ]; then
  EXTRA_ENV+=(-e "VLLM_ALLOW_RUNTIME_LORA_UPDATING=True")
  if [ ! -d "$V15_RUN_HOST" ]; then
    echo "Missing v1.5 run dir: $V15_RUN_HOST"
    exit 1
  fi
  VOLUME_ARGS+=(-v "${V15_RUN_HOST}:${LORA_HOST_MOUNT:-/models/lora-host}/v15-run:ro")
fi

if [ -n "$CANDIDATE_MANIFEST" ]; then
  if [ ! -f "$CANDIDATE_MANIFEST" ]; then
    echo "CANDIDATE_MANIFEST not found: $CANDIDATE_MANIFEST"
    exit 1
  fi
  echo "DEPLOY_MODE=$DEPLOY_MODE manifest=$CANDIDATE_MANIFEST"
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    model_name=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin)['model_name'])")
    adapter_path=$(echo "$line" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('adapter_path') or d['container_path'])")
    container_path=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin)['container_path'])")
    preload=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin).get('preload_at_startup', True))")
    if [ "$DEPLOY_MODE" = "eval_runtime" ] && [ "$preload" != "True" ]; then
      continue
    fi
    if [ ! -d "$adapter_path" ]; then
      echo "Missing adapter_path for $model_name: $adapter_path"
      exit 1
    fi
    if [ ! -f "$adapter_path/adapter_config.json" ]; then
      echo "No adapter_config.json in $adapter_path"
      exit 1
    fi
    rank=$(python3 -c "import json; print(json.load(open('$adapter_path/adapter_config.json'))['r'])")
    if [ "$rank" -gt "$MAX_LORA_RANK" ] 2>/dev/null; then
      echo "WARNING: adapter $model_name r=$rank > MAX_LORA_RANK=$MAX_LORA_RANK"
    fi
    VOLUME_ARGS+=(-v "${adapter_path}:${container_path}:ro")
    LORA_MODULES+=("${model_name}=${container_path}")
  done < "$CANDIDATE_MANIFEST"
else
  EXPERIMENT_DIRS=(
    "/root/saulie/train/models/steering-sft-v1.1"
    "/root/saulie/train/models/steering-sft-v1.2"
  )
  for exp_dir in "${EXPERIMENT_DIRS[@]}"; do
    exp_name=$(basename "$exp_dir")
    for trial_dir in "$exp_dir"/trial-*/best_adapter; do
      [ -d "$trial_dir" ] || continue
      trial_num=$(basename "$(dirname "$trial_dir")" | sed 's/trial-//')
      adapter_name="${exp_name}_trial-${trial_num}"
      container_path="/models/lora/${adapter_name}"
      VOLUME_ARGS+=(-v "${trial_dir}:${container_path}:ro")
      LORA_MODULES+=("${adapter_name}=${container_path}")
    done
  done
fi

if [ ${#LORA_MODULES[@]} -eq 0 ] && [ "$DEPLOY_MODE" = "eval_runtime" ]; then
  echo "eval_runtime: no preload adapters in startup manifest (ok if loading all at runtime only)"
fi

echo "╔═════════════════════════════════════════════════════════════════════════╗"
echo "║  Qwen3-4B-Instruct-2507-FP8 — Eval deploy (mode=$DEPLOY_MODE)         ║"
echo "╚═════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Base model:       $MODEL_PATH"
echo "Max LoRA rank:    $MAX_LORA_RANK  (vLLM bucket; cat adapters up to r=40; SFT r=16)"
echo "Max LoRAs (hot):  $MAX_LORAS"
echo "Startup adapters: ${#LORA_MODULES[@]}"
for lm in "${LORA_MODULES[@]}"; do
  echo "  - ${lm%%=*}"
done
echo ""

if [ ! -d "$MODEL_PATH" ]; then
  echo "Model not found at: $MODEL_PATH"
  exit 1
fi

docker stop ${CONTAINER_NAME} 2>/dev/null && echo "Stopped old container" || true
docker rm   ${CONTAINER_NAME} 2>/dev/null && echo "Removed old container" || true

LORA_ARGS=()
if [ ${#LORA_MODULES[@]} -gt 0 ]; then
  LORA_MODULES_STR=$(IFS=' '; echo "${LORA_MODULES[*]}")
  LORA_ARGS=(--enable-lora --lora-modules ${LORA_MODULES_STR})
fi

docker run -d \
  --name ${CONTAINER_NAME} \
  --gpus "device=${GPU_DEVICE}" \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --restart unless-stopped \
  -p ${PORT}:8000 \
  -v ${MODEL_PATH}:/models/model:ro \
  "${VOLUME_ARGS[@]}" \
  -v /dev/null:/etc/ld.so.conf.d/00-cuda-compat.conf \
  "${EXTRA_ENV[@]}" \
  ${VLLM_IMAGE} \
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
  "${LORA_ARGS[@]}" \
  --max-lora-rank ${MAX_LORA_RANK} \
  --max-loras ${MAX_LORAS} \
  --trust-remote-code \
  --enable-prefix-caching

echo "Waiting for API on port ${PORT}..."
sleep 20

for i in $(seq 1 90); do
  if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    echo ""
    echo "API ready (mode=$DEPLOY_MODE)."
    if [ "$DEPLOY_MODE" = "eval_runtime" ]; then
      echo "  Runtime LoRA: POST /v1/load_lora_adapter  (VLLM_ALLOW_RUNTIME_LORA_UPDATING=True)"
      echo "  DPO cats on disk: ${LORA_HOST_MOUNT:-/models/lora-host}/v15-run/trial-N/sft_dpo_cat"
      echo "  Next: python dpo/eval/run_v15_final_eval.py --limit-test"
    fi
    exit 0
  fi
  if [ "$i" -eq 90 ]; then
    docker logs ${CONTAINER_NAME} --tail 80
    exit 1
  fi
  sleep 2
done
