import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorrt as trt
import torch
from transformers import AutoTokenizer
import time
import os
import asyncio
from contextlib import asynccontextmanager
import numpy as np
from typing import List, Union


# --- CONFIGURATION ---
ENGINE_PATH = "/workspace/onnx/bge_m3.engine"
SPARSE_WEIGHTS_PATH = "/workspace/sparse_linear.pt"
TOKENIZER_PATH = "/workspace"

# Max limits (Must match your trtexec build)
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "2"))
print("Using MAX_BATCH_SIZE: ", MAX_BATCH_SIZE)
MAX_SEQ_LEN = 256
HIDDEN_DIM = 1024

DEVICE = "cuda"

# --- GLOBAL STATE ---
state = {
    "context": None,
    "tokenizer": None,
    "sparse_linear": None,
    "inputs": {},   # Stores input_ids, attention_mask buffers
    "outputs": {}   # Stores all output buffers dynamically
}

inference_lock = asyncio.Lock() 

# --- HELPER: Vectorized Sparse Logic ---
def compute_sparse_vectorized(token_ids, weights):
    # 1. Filter: Keep only positive weights
    mask = weights > 0
    filtered_ids = token_ids[mask]
    filtered_weights = weights[mask]

    if filtered_ids.numel() == 0:
        return {"indices": [], "values": []}

    # 2. Aggregation: Handle duplicate tokens (keep Max)
    unique_ids, inverse_indices = torch.unique(filtered_ids, return_inverse=True)
    max_weights = torch.zeros_like(unique_ids, dtype=filtered_weights.dtype)
    max_weights.scatter_reduce_(0, inverse_indices, filtered_weights, reduce="amax", include_self=False)

    return {
        "indices": unique_ids.cpu().tolist(),
        "values": max_weights.cpu().float().tolist()
    }

# --- LIFESPAN MANAGER (UPDATED) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Load Tokenizer
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    
    # 2. Load TRT Engine
    print(f"Loading Engine from {ENGINE_PATH}...")
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    with open(ENGINE_PATH, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()

    # 3. Dynamic Buffer Allocation
    print("Allocating Static GPU Memory for ALL tensors...")
    input_buffers = {}
    output_buffers = {}
    
    dtype_map = {trt.int32: torch.int32, trt.int64: torch.long, trt.float32: torch.float32, trt.float16: torch.float16}
    
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        dtype = engine.get_tensor_dtype(name)
        
        # Map TRT dtype to PyTorch
        pt_dtype = dtype_map.get(dtype, torch.float32)
        
        # Get Max Shape
        dims = list(engine.get_tensor_shape(name))
        shape = []
        for d in dims:
            if d == -1:
                if len(shape) == 0: shape.append(MAX_BATCH_SIZE)
                else: shape.append(MAX_SEQ_LEN)
            else:
                shape.append(d)
        
        tensor = torch.empty(tuple(shape), dtype=pt_dtype, device=DEVICE)
        
        if mode == trt.TensorIOMode.INPUT:
            input_buffers[name] = tensor
        else:
            output_buffers[name] = tensor

    # 4. Load Sparse Weights
    print("Loading Sparse Layer...")
    sparse_linear = torch.nn.Linear(in_features=HIDDEN_DIM, out_features=1, device=DEVICE, dtype=torch.float16)
    if os.path.exists(SPARSE_WEIGHTS_PATH):
        sparse_linear.load_state_dict(torch.load(SPARSE_WEIGHTS_PATH, map_location=DEVICE, weights_only=True))
    sparse_linear.eval()
    
    # 5. Save to State (CRITICAL: Must be saved BEFORE calling run_inference_optimized)
    state["tokenizer"] = tokenizer
    state["context"] = context
    state["engine"] = engine
    state["sparse_linear"] = sparse_linear
    state["inputs"] = input_buffers
    state["outputs"] = output_buffers

    # 6. WARMUP
    print("Running Full Pipeline Warmup (Max Length)...")
    
    # Create a dummy string that guarantees hitting MAX_SEQ_LEN (256 tokens)
    # "the " is 1 token. 116 words ensures we trigger truncation.
    dummy_text = "warmup " * 116 
    
    try:
        # We call the actual function logic.
        # This compiles PyTorch kernels (ReLU, Unique, Scatter) and allocates full intermediate memory.
        async with inference_lock:
             run_inference_optimized(dummy_text)
        print("Warmup Complete: Pipeline Fully Compiled.")
    except Exception as e:
        print(f"WARNING: Warmup failed (System might still work, but first request will be slow). Error: {e}")

    print("System Ready.")
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)


class EmbedRequest(BaseModel):
    # text: str
    text: Union[str, List[str]]


def run_inference_optimized(query: str):
    tokenizer = state["tokenizer"]
    context = state["context"]
    input_buffers = state["inputs"]
    output_buffers = state["outputs"]
    sparse_linear = state["sparse_linear"]
    
    # 1. Tokenize (CPU)
    inputs = tokenizer(
        query, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=MAX_SEQ_LEN
    )
    
    # seq_len = inputs["input_ids"].shape[1]
    
    # 2. Copy Inputs
    # Get the actual batch size of the incoming data (usually 1)
    # b_size = inputs["input_ids"].shape[0] 
    b_size, seq_len = inputs["input_ids"].shape 

    
    # Only copy into the valid rows [0:1], not the whole buffer [0:2]
    input_buffers["input_ids"][:b_size, :seq_len].copy_(inputs["input_ids"], non_blocking=True)
    input_buffers["attention_mask"][:b_size, :seq_len].copy_(inputs["attention_mask"], non_blocking=True)
    
    # 3. Set Input Shapes & Addresses
    context.set_input_shape("input_ids", (b_size, seq_len))
    context.set_input_shape("attention_mask", (b_size, seq_len))
    
    context.set_tensor_address("input_ids", input_buffers["input_ids"].data_ptr())
    context.set_tensor_address("attention_mask", input_buffers["attention_mask"].data_ptr())
    
    # 4. Set Output Addresses
    for name, tensor in output_buffers.items():
        context.set_tensor_address(name, tensor.data_ptr())

    # 5. Execute
    stream = torch.cuda.current_stream().cuda_stream

    try:
        context.execute_async_v3(stream_handle=stream)
    except Exception as e:
        raise RuntimeError(f"TensorRT execution failed: {e}")
    
    # 6. Post-Process
    hidden_state = None
    dense_output = None
    
    for name, tensor in output_buffers.items():
        if len(tensor.shape) == 3:
            hidden_state = tensor
        elif len(tensor.shape) == 2:
            dense_output = tensor

    if hidden_state is None:
        raise ValueError("Model output missing 3D hidden state.")

    # Slice valid data
    valid_hidden = hidden_state[:, :seq_len, :]

    # --- DENSE ---
    if dense_output is not None:
        dense_vec = torch.nn.functional.normalize(dense_output, p=2, dim=1)
    else:
        # Use valid_hidden, ensuring we grab the CLS token (index 0)
        # Note: valid_hidden might be Float32, normalize works fine on both
        dense_vec = valid_hidden[:, 0, :]
        dense_vec = torch.nn.functional.normalize(dense_vec, p=2, dim=1)

    # --- SPARSE ---
    # We must ensure the input to the Linear layer is the same dtype as the layer (Float16)
    # The view does not copy data if it's already Float16, but casts if it's Float32
    if valid_hidden.dtype != torch.float16:
        sparse_input = valid_hidden.to(torch.float16)
    else:
        sparse_input = valid_hidden
    
    sparse_logits = sparse_linear(sparse_input)
    sparse_weights = torch.relu(sparse_logits) # [1, seq_len, 1]

    # Sync
    # torch.cuda.synchronize() # Block the whole GPU until it finishes this exact stream
    torch.cuda.current_stream().synchronize() # Only waits for this specific stream to finish
    
    
    # Ensure dense has no NaNs BEFORE moving to CPU
    if torch.isnan(dense_vec).any() or torch.isnan(sparse_weights).any():
        raise ValueError("Model produced NaN values in dense and / or sparse vectors")
    
    
    # --- FORMAT ---
    # Non-blocking GPU -> CPU copy (we already synchronized the stream above, so copy completes)
    # dense_list = dense_vec[:b_size].detach().cpu().float().numpy().tolist()

    dense_results = []
    sparse_results = []
    
    # Non-blocking GPU -> CPU copy (we already synchronized the stream above, so copy completes)
    all_dense = dense_vec[:b_size].detach().cpu().float().numpy() # [Batch, 1024]

    # LOOP through the batch to process each sentence separately
    for i in range(b_size):
        # 1. Dense: Just append the row
        dense_results.append(all_dense[i].tolist())
        
        # 2. Get the raw pointers from our GPU buffer
        # We slice [:seq_len] because the buffer might be 256 but seq is only 50
        gpu_ids = input_buffers["input_ids"][i, :seq_len]          # [Seq] on GPU
        gpu_mask = input_buffers["attention_mask"][i, :seq_len]    # [Seq] on GPU
        
        # 3. Get the weights from the model output
        gpu_weights = sparse_weights[i].squeeze()                  # [Seq] on GPU
        
        # 4. Filter Padding (All operations happen on GPU)
        # We only want indices/weights where mask == 1
        valid_indices = gpu_ids[gpu_mask == 1]
        valid_weights = gpu_weights[gpu_mask == 1]
        
        # 5. Compute Vectorized (Pure GPU)
        sparse_dict = compute_sparse_vectorized(valid_indices, valid_weights)
        
        sparse_results.append(sparse_dict)

    return {"dense": dense_results, "sparse": sparse_results}


@app.post("/embed")
async def embed_endpoint(request: EmbedRequest):
    try:
        async with inference_lock:
            start_time = time.perf_counter()
            result = run_inference_optimized(request.text)
            end_time = time.perf_counter()
        
        return {
            "dense": result["dense"],
            "sparse": result["sparse"],
            "performance": {
                "latency_ms": round((end_time - start_time) * 1000, 4),
            }
        }
    except Exception as e:
        print(f"SERVER ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        limit_concurrency=6,
        timeout_keep_alive=10
    )