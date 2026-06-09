# Saulie — Production Deployment

Run the **DPO v1.5 trial-4** shopping agent end-to-end: vLLM + RAG + FastAPI agent + nginx + ngrok, with a tmux monitoring dashboard.

This branch is for **serving and operating** the agent in production. DPO training and eval live on other branches (`main`, `dpo_eval`, etc.).

---

## Architecture

```
Remote client (remote_chat.py)
        │
        ▼
   ngrok :8080                    ← public HTTPS tunnel (optional)
        │
        ▼
   nginx :8080                     ← Bearer auth, rate limits, SSE proxy
        │
        ▼
   agent_chat_api.py :9000         ← tool loop, SSE streaming, system prompt
        │
        ├──► vLLM :8000            ← Qwen3 FP8 + dpo-v15-trial-4 LoRA
        │
        └──► RAG (external)        ← BGE-M3 :8888 + Qdrant :1234
              /root/rag/query2.py
```

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **GPU + Docker** | NVIDIA runtime for vLLM and BGE containers |
| **Base model** | `Qwen3-4B-Instruct-2507-FP8` at repo path (see deploy script) |
| **LoRA adapter** | DPO v1.5 trial-4 cat-merged adapter (r=32) |
| **RAG stack** | `tensorrt_bge-m3` (:8888), `qdrant_index` (:1234), code at `/root/rag` |
| **Python env** | `saulgman` conda env (or set `SAULIE_PYTHON`) |
| **tmux** | For the monitoring dashboard |
| **ngrok** (optional) | Authtoken in `.env` as `NGROK_KEY` for remote access |

---

## Quick start

```bash
# One command — start missing services + open tmux dashboard
bash start_saulie.sh
```

Detach dashboard: `Ctrl+B` then `D`  
Reattach: `bash start_saulie.sh` or `tmux attach -t saulie`

Stop everything:

```bash
bash stop_saulie.sh
```

Keep tmux open, stop services only:

```bash
bash stop_saulie.sh --keep-tmux
```

---

## What `start_saulie.sh` does

Idempotent — skips anything already running:

1. Start `qdrant_index`, `tensorrt_bge-m3` (+ BGE `serve.py` if embed API is down)
2. Start or redeploy `eval_deploy_qwenie` (vLLM) if unhealthy
3. Start `agent_chat_api.py` on `:9000` if not up
4. Start `nginx` and `ngrok` (if configured)
5. Open tmux session **`saulie`** with live logs

### Tmux layout

| Window | Panes |
|--------|-------|
| **1: status** | Stack health (UP/DOWN) + ngrok URL / GPU stats |
| **2: vllm** | Docker logs + **nvitop** |
| **3: rag** | BGE logs \| Qdrant logs |
| **4: agent** | uvicorn log \| app log \| nginx \| ngrok |

---

## Manual steps (if not using start script)

### 1. Deploy vLLM

```bash
bash dpo/eval/vllm_scripts/deploy_finalist_pick.sh
```

Serves model as **`dpo-v15-trial-4`** on `http://localhost:8000`.  
vLLM API key: `dipshit` (override with `VLLM_API_KEY`).

### 2. Start RAG containers

```bash
docker start qdrant_index tensorrt_bge-m3
docker exec -d tensorrt_bge-m3 sh -c "python -u serve.py >> /proc/1/fd/1 2>&1"
```

### 3. Start agent API

```bash
/root/miniconda3/envs/saulgman/bin/python agent_chat_api.py api
# listens on :9000
```

### 4. Start nginx

```bash
docker start nginx   # or: docker compose -f nginx/docker-compose.nginx.yml up -d
```

### 5. Expose publicly (optional)

```bash
# .env: NGROK_KEY=your_authtoken
ngrok http 8080
```

---

## API usage

### Local (direct to agent)

```bash
curl http://127.0.0.1:9000/health

curl http://127.0.0.1:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dpo-v15-trial-4",
    "messages": [{"role": "user", "content": "wireless headset under $50"}],
    "stream": true
  }'
```

### Via nginx (production path)

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer secret" \
  -H "Content-Type: application/json" \
  -d '{"model":"dpo-v15-trial-4","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

Bearer token **`secret`** is configured in `nginx/nginx.conf`.

### Remote client

Edit `PUBLIC_URL` and run from any machine:

```bash
python remote_chat.py
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SAULIE_PYTHON` | `.../saulgman/bin/python` | Python for agent API |
| `AGENT_PORT` | `9000` | FastAPI port |
| `VLLM_API_KEY` | `dipshit` | vLLM auth |
| `LLM_BASE_URL` | `http://localhost:8000/v1` | vLLM endpoint (agent) |
| `MODEL_NAME` | `dpo-v15-trial-4` | LoRA model id (also set via `SAULIE_MODEL` in `start_saulie.sh`) |
| `SAULIE_MODEL` | `dpo-v15-trial-4` | vLLM model name passed to agent on start |
| `SAULIE_PROMPT` | `compressed` | System prompt variant: `legacy`, `steering`, or `compressed` |
| `SAULIE_DEPLOY_SCRIPT` | `deploy_finalist_pick.sh` | vLLM deploy script path |
| `QDRANT_COLLECTION` | `amazon_products_v2` | McAuley US catalog (set by `start_saulie.sh`) |
| `FUSION_METHOD` | `rrf` | RAG fusion method |
| `NGROK_KEY` | — | ngrok authtoken (in `.env`) |
| `SSE_KEEPALIVE_INTERVAL` | `15` | SSE ping interval (seconds) |
| `LLM_TIMEOUT` | `300` | vLLM request timeout (seconds) |

Check active model + prompt: `curl http://127.0.0.1:9000/health`

Agent logs: `saulie_chat.log`, `agent_api.log`

### System prompt variants

`agent_chat_api.py` ships three prompts selectable via `SAULIE_PROMPT`:

| Variant | Use |
|---------|-----|
| `compressed` | Default production prompt (~410 tokens) |
| `steering` | Full probe-first persona prompt from steering fix plan |
| `legacy` | Original “ALWAYS USE THE TOOL” deployment prompt |

See [`SAULIE_PERSONA_AND_STEERING_FIX_PLAN.md`](SAULIE_PERSONA_AND_STEERING_FIX_PLAN.md) for the persona/steering patch history.

---

## Repo layout (this branch)

```
agent_chat_api.py              # Production agent API (OpenAI-compatible)
remote_chat.py                 # Remote CLI client via ngrok
start_saulie.sh                # Start stack + tmux dashboard
stop_saulie.sh                 # Stop stack safely
stop.sh                        # Wrapper → stop_saulie.sh
monitor.sh                     # Standalone vLLM tmux monitor
dashboard.sh                   # Standalone RAG tmux monitor
nginx/                         # Reverse proxy config + compose
dpo/eval/vllm_scripts/
  deploy_finalist_pick.sh      # vLLM + LoRA container deploy (DPO w=1.0 trial-4)
SAULIE_PERSONA_AND_STEERING_FIX_PLAN.md  # Prompt + harness fix plan
```

**RAG code** is expected at `/root/rag` (not in this branch). The agent imports `search_hybrid` from `query2.py` there.

---

## Troubleshooting

| Symptom | Check |
|---------|-------|
| vLLM won't start | `docker logs eval_deploy_qwenie` — broken container is auto-redeployed by `start_saulie.sh` |
| Agent up but dumb replies | `curl :8000/health` — is vLLM healthy? |
| Tool returns garbage products | BGE `:8888/embed` and Qdrant `:1234` — see RAG docs; verify embed/index match |
| Stream dies mid-response | SSE keepalives are enabled; check ngrok free-tier limits |
| nginx 401 | Use `Authorization: Bearer secret` |
| ngrok URL changed | `curl -s http://127.0.0.1:4040/api/tunnels` — update `remote_chat.py` |

Health check all services from tmux window **1: status**, or:

```bash
curl -sf :8000/health && echo vLLM ok
curl -sf :9000/health && echo agent ok
curl -sf :8080/health && echo nginx ok
curl -sf -X POST :8888/embed -H 'Content-Type: application/json' -d '{"text":"ping"}' && echo BGE ok
curl -sf :1234/ && echo qdrant ok
```

---

## Related branches

| Branch | Purpose |
|--------|---------|
| **`deployment`** (this) | Production serving stack |
| `main` / `dpo_eval` | DPO training, eval, study reports |
| `rag` | RAG indexing, benchmarks, catalog prep |
