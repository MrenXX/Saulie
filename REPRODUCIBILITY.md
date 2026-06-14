# Reproducible Setup Guide

This document explains how to recreate the Saulie development environment from scratch.

## Files at a glance

| File | What it is |
|------|------------|
| [`requirements.txt`](requirements.txt) | Every Python package with exact versions (from `pip freeze`) |
| [`environment.yml`](environment.yml) | Conda recipe: creates env `saulgman` with Python 3.11 and installs `requirements.txt` |
| [`.env.example`](.env.example) | Template for secrets and config — copy to `.env` and edit |
| [`docker/versions.env`](docker/versions.env) | Pinned Docker image tags/digests |
| [`docker/setup_containers.sh`](docker/setup_containers.sh) | Creates Qdrant + BGE containers if missing |
| [`docker/docker-compose.yml`](docker/docker-compose.yml) | Qdrant + nginx via compose (optional) |
| [`rag/embed_models/bge-m3/serve.py`](rag/embed_models/bge-m3/serve.py) | BGE FastAPI embed server (runs **inside** the TensorRT container) |
| [`rag/embed_models/bge-m3/build_engine.sh`](rag/embed_models/bge-m3/build_engine.sh) | Compile BGE-M3 ONNX → TensorRT engine |

**What is `environment.yml`?** It is not a second dependency list. It only says: "make a conda env named `saulgman` with Python 3.11, then pip-install from `requirements.txt`." One command instead of two.

---

## Prerequisites

- NVIDIA GPU with CUDA (RTX 4070 12GB or similar tested)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) with WSL2 GPU integration enabled
- [ngrok](https://ngrok.com/) CLI (optional, for remote access) — tested with v3.39.6+

---

## Step 1: Clone and configure

```bash
git clone https://github.com/MrenXX/Saulie.git /root/saulie
cd /root/saulie
cp .env.example .env
# Edit .env — set NGROK_KEY, API keys, paths if different from defaults
```

---

## Step 2: Python environment

```bash
conda env create -f environment.yml
conda activate saulgman
```

If the env already exists and you want to refresh packages:

```bash
conda activate saulgman
pip install -r requirements.txt
```

---

## Step 3: RAG workspace layout

The repo contains RAG **code** under `rag/`. Model weights, TRT engine, and Qdrant data live outside git under `RAG_ROOT` (default `/root/rag`).

```bash
# Option A: symlink (recommended if repo is at /root/saulie)
mkdir -p /root/rag
ln -sfn /root/saulie/rag/embed_models /root/rag/embed_models

# Option B: copy
cp -r /root/saulie/rag/embed_models /root/rag/
```

Download BGE-M3 model weights into `/root/rag/embed_models/bge-m3/` (HuggingFace `BAAI/bge-m3`), then build the TensorRT engine:

```bash
bash docker/setup_containers.sh          # creates BGE container
docker exec tensorrt_bge-m3 bash /workspace/build_engine.sh
```

Full RAG indexing steps: [`rag/README.md`](rag/README.md).

### BGE embed server (`serve.py`)

The embed API is **not** part of the `saulgman` conda env. It runs inside the `tensorrt_bge-m3` Docker container, which has its own Python + TensorRT stack preinstalled in `nvcr.io/nvidia/tensorrt:24.12-py3`.

| Item | Detail |
|------|--------|
| Source file | [`rag/embed_models/bge-m3/serve.py`](rag/embed_models/bge-m3/serve.py) |
| Mount path in container | `/workspace/serve.py` (bind-mount from `RAG_ROOT/embed_models/bge-m3`) |
| Framework | FastAPI + uvicorn |
| Endpoint | `POST /embed` — body `{"text": "..."}` or `{"text": ["...", "..."]}` (max 2) |
| Response | `{"dense": [[...]], "sparse": {"indices": [...], "values": [...]}}` |
| Engine | Loads `/workspace/onnx/bge_m3.engine` (built by `build_engine.sh`) |
| Max batch | 2 (must match TRT engine profile) |

Start manually:

```bash
docker exec -d tensorrt_bge-m3 sh -c "cd /workspace && python -u serve.py"
```

Or let `docker/setup_containers.sh` / `start_saulie.sh` start it for you.

Health check:

```bash
curl -X POST http://127.0.0.1:8888/embed \
  -H "Content-Type: application/json" \
  -d '{"text":"healthcheck"}'
```

---

## Step 4: Docker infrastructure

Pinned images are in [`docker/versions.env`](docker/versions.env):

| Service | Image | Port |
|---------|-------|------|
| BGE embed | `nvcr.io/nvidia/tensorrt:24.12-py3` | 8888 |
| Qdrant | `qdrant/qdrant@sha256:0fb8897...` | 1234 |
| vLLM | `vllm/vllm-openai@sha256:014a95f...` | 8000 |
| nginx | `nginx:1.25-alpine` | 8080 |

Create Qdrant + BGE:

```bash
bash docker/setup_containers.sh
```

Or use compose for Qdrant + nginx only:

```bash
docker compose -f docker/docker-compose.yml --env-file docker/versions.env up -d
```

Deploy vLLM (requires model weights on disk):

```bash
bash dpo/eval/vllm_scripts/deploy_finalist_pick.sh
```

---

## Step 5: Model weights (not in git)

Download separately and place at:

| Path | Purpose |
|------|---------|
| `/root/saulie/Qwen3-4B-Instruct-2507` | BF16 base (training) |
| `/root/saulie/Qwen3-4B-Instruct-2507-FP8` | FP8 base (vLLM inference) |
| `train/models/steering-sft-v1.1/trial-17/best_adapter` | SFT LoRA |
| `dpo/train/models/steering-dpo-v1.5/.../trial-4/sft_dpo_cat` | DPO cat-merged LoRA |

These are gitignored. Copy from your training machine or HuggingFace.

---

## Step 6: Start the stack

```bash
conda activate saulgman
bash start_saulie.sh
```

Health check:

```bash
curl http://127.0.0.1:9000/health
```

---

## What is NOT reproducible from git alone

- Qwen3 model weights and LoRA adapters
- BGE-M3 model files and compiled TRT engine (`bge_m3.engine`)
- Qdrant index data (`/root/rag/qdrant_storage`)
- ngrok public URL (changes on restart)
- MLflow run artifacts and Optuna study outputs

---

## Updating pinned versions

**Python packages** (after upgrading in a test env):

```bash
pip freeze > requirements.txt
# Re-add the 3-line header comment at the top
```

**Docker images** (after pulling a new image and verifying it works):

```bash
docker inspect <container> --format '{{.Config.Image}}'
# Update docker/versions.env with the new tag or digest
```

---

## Troubleshooting

| Problem | Check |
|---------|-------|
| BGE returns empty / agent says "0 products" | `curl -X POST http://127.0.0.1:8888/embed -d '{"text":"test"}'` |
| vLLM won't start | `docker logs eval_deploy_qwenie` — usually missing model path |
| Agent can't find RAG | Symlink `rag/` to `/root/rag` or set paths in `.env` |
| ngrok skipped | Set `NGROK_KEY` in `.env` |
