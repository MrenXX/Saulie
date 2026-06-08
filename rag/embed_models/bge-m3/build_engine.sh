#!/usr/bin/env bash
# Compile BGE-M3 ONNX → TensorRT engine for the GPU this container sees.
# Must run inside tensorrt_bge-m3 (or equivalent TRT image with GPU access).
#
# Agent prod shapes: batch ≤ 2, seq ≤ 256 (matches serve.py + query2.py).
#
# Usage (from host):
#   docker start tensorrt_bge-m3
#   docker exec tensorrt_bge-m3 pkill -f serve.py || true
#   bash /root/rag/embed_models/bge-m3/build_engine.sh
#
# Or inside container:
#   cd /workspace && bash build_engine.sh

set -euo pipefail

ONNX_DIR="${ONNX_DIR:-/workspace/onnx}"
ENGINE="${ONNX_DIR}/bge_m3.engine"
BACKUP="${ENGINE}.bak.$(date +%Y%m%d-%H%M%S)"

if [[ -f "$ENGINE" ]]; then
  echo "[*] Backing up existing engine → $BACKUP"
  cp "$ENGINE" "$BACKUP"
fi

echo "[*] Compiling on GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "[*] Shapes: min 1x1 | opt 1x64 | max 2x256 | fp16"

cd "$ONNX_DIR"
trtexec \
  --onnx=./model.onnx \
  --saveEngine=./bge_m3.engine \
  --fp16 \
  --minShapes=input_ids:1x1,attention_mask:1x1 \
  --optShapes=input_ids:1x64,attention_mask:1x64 \
  --maxShapes=input_ids:2x256,attention_mask:2x256 \
  --verbose

echo "[*] Verifying optimization profiles..."
python3 -c "
import tensorrt as trt
logger = trt.Logger(trt.Logger.WARNING)
with open('${ENGINE}','rb') as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        mn, op, mx = engine.get_tensor_profile_shape(name, 0)
        print(name, 'min', tuple(mn), 'opt', tuple(op), 'max', tuple(mx))
"

echo "[ok] Engine written to $ENGINE"
