#!/bin/bash
set -e

CONTAINER_NAME="eval_deploy_qwenie"
MODEL_PATH="/root/saulie/Qwen3-4B-Instruct-2507-FP8"
GPU_DEVICE="0"
PORT="8000"
VLLM_API_KEY="${VLLM_API_KEY:-dipshit}"
# Latest vLLM — FP8 Marlin on Ampere (RTX 3090); v0.8.5 hits fp8e4nv Triton errors
VLLM_IMAGE="vllm/vllm-openai:latest"

# Optional: manifest-driven LoRA list for DPO final eval (unset = legacy SFT scan)
CANDIDATE_MANIFEST="${CANDIDATE_MANIFEST:-}"
# Stacked SFT+DPO cat adapters need r_sft+r_dpo (e.g. 48); SFT-only trials need 16–32
MAX_LORA_RANK="${MAX_LORA_RANK:-32}"
MAX_LORAS="${MAX_LORAS:-2}"

VOLUME_ARGS=()
LORA_MODULES=()

if [ -n "$CANDIDATE_MANIFEST" ]; then
    if [ ! -f "$CANDIDATE_MANIFEST" ]; then
        echo "CANDIDATE_MANIFEST not found: $CANDIDATE_MANIFEST"
        exit 1
    fi
    echo "Loading adapters from manifest: $CANDIDATE_MANIFEST"
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        model_name=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin)['model_name'])")
        adapter_path=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin)['adapter_path'])")
        if [ ! -d "$adapter_path" ]; then
            echo "Missing adapter_path for $model_name: $adapter_path"
            exit 1
        fi
        if [ ! -f "$adapter_path/adapter_config.json" ]; then
            echo "No adapter_config.json in $adapter_path"
            exit 1
        fi
        container_path="/models/lora/${model_name}"
        VOLUME_ARGS+=(-v "${adapter_path}:${container_path}:ro")
        LORA_MODULES+=("${model_name}=${container_path}")
    done < "$CANDIDATE_MANIFEST"
else
    EXPERIMENT_DIRS=(
    "/root/saulie/sft/models/steering-sft-v1.1"
    "/root/saulie/sft/models/steering-sft-v1.2"
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

if [ ${#LORA_MODULES[@]} -eq 0 ]; then
    echo "No adapters found in any experiment directory."
    exit 1
fi

echo "╔═════════════════════════════════════════════════════════════════════════╗"
echo "║  Deploying Qwen3-4B-Instruct-2507-FP8 — Multi-Adapter Eval Mode      ║"
echo "╚═════════════════════════════════════════════════════════════════════════╝"
echo ""

if [ ! -d "$MODEL_PATH" ]; then
    echo "Model not found at: $MODEL_PATH"
    exit 1
fi
echo "Base model:  $MODEL_PATH"
echo "Adapters found: ${#LORA_MODULES[@]}"
echo "Max LoRA rank:  ${MAX_LORA_RANK}"
echo "Max concurrent: ${MAX_LORAS}"
for lm in "${LORA_MODULES[@]}"; do
    echo "  - ${lm%%=*}"
done
echo ""

docker stop ${CONTAINER_NAME} 2>/dev/null && echo "Stopped old container" || true
docker rm   ${CONTAINER_NAME} 2>/dev/null && echo "Removed old container" || true

LORA_MODULES_STR=$(IFS=' '; echo "${LORA_MODULES[*]}")

echo ""
echo "Starting container..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

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
  -e VLLM_API_KEY="${VLLM_API_KEY}" \
  -e NCCL_P2P_DISABLE=1 \
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
  --enable-lora \
  --lora-modules ${LORA_MODULES_STR} \
  --max-lora-rank ${MAX_LORA_RANK} \
  --max-loras ${MAX_LORAS} \
  --trust-remote-code \
  --enable-prefix-caching

echo "Container started, waiting for API..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

sleep 20

for i in {1..90}; do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo ""
        echo "API Ready!"
        echo ""
        echo "╔════════════════════════════════════════════════════════════════════╗"
        echo "║  Multi-Adapter Eval Server Deployed                              ║"
        echo "╚════════════════════════════════════════════════════════════════════╝"
        echo ""
        echo "Base model (no adapter):  model='Saulie'"
        echo "Adapters:"
        for lm in "${LORA_MODULES[@]}"; do
            echo "  model='${lm%%=*}'"
        done
        echo ""
        echo "Max LoRA rank:   ${MAX_LORA_RANK}"
        echo "Max concurrent:  ${MAX_LORAS} (others swap from CPU)"
        echo "GPU memory util: 50% (~13–14GB VRAM on 3090)"
        echo "Max context:     4096"
        echo "Port:            ${PORT}"
        echo ""
        echo "Next: python sft/sft_eval/eval_generate_vllm.py"
        exit 0
    fi

    if [ $i -eq 90 ]; then
        echo ""
        echo "API failed to start after 3 minutes"
        docker logs ${CONTAINER_NAME} --tail 50
        exit 1
    fi

    sleep 2
done
