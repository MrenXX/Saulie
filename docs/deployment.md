# Run the local demo

The configured stack serves Qwen3-4B-Instruct-2507-FP8 with the rank-32 SFT plus DPO trial-4 adapter. The agent defaults to the `compressed` prompt and McAuley catalog `amazon_products_v2` with RRF.

This guide assumes the [environment and external artifacts](../REPRODUCIBILITY.md) are already in place. `start_saulie.sh` starts services; it does not download weights, train adapters, export BGE to ONNX, or build a catalog from nothing.

## Configuration

Copy [`.env.example`](../.env.example) to a local `.env` and replace the example key values. Keep `LLM_API_KEY` equal to `VLLM_API_KEY`. Set `NGINX_API_KEY` for clients using the proxy. `NGROK_KEY` is only needed for the optional public tunnel.

The historical scripts expect `/root/saulie` and `/root/miniconda3/envs/saulgman/bin/python`. Set `SAULIE_PYTHON` if the interpreter differs. The production deploy script still hardcodes model and adapter locations under `/root/saulie`; setting `SAULIE_REPO` does not make every path portable.

Useful settings are:

- Model and prompt: `SAULIE_MODEL`, `SAULIE_PROMPT`, and `SAULIE_DEPLOY_SCRIPT`.
- Retrieval: `RAG_ROOT`, `QDRANT_COLLECTION`, `FUSION_METHOD`, `EMBED_URL`, and `QDRANT_URL`.
- Ports: `AGENT_PORT`, `VLLM_PORT`, `NGINX_PORT`, `EMBED_PORT`, and `QDRANT_PORT`.
- Tool pressure: `SAULIE_TOOL_PRESSURE_ENABLED`, `SAULIE_TOOL_BIAS_START_TURN`, `SAULIE_TOOL_FORCE_TURN`, and `SAULIE_TOOL_BIAS_PER_TURN`.

Changing a port also requires aligning its client URL. Standalone Python scripts do not all load `.env`, so export values in their shell when needed.

## Start and inspect

```bash
conda activate saulgman
bash start_saulie.sh
curl http://127.0.0.1:9000/health
```

The launcher prepares RAG services, vLLM, the agent, nginx, and an optional ngrok tunnel. It opens a tmux dashboard by default. For startup without that dashboard, set `SAULIE_NO_DASHBOARD=1` before running it.

The expected health payload identifies the configured model and active prompt. A successful health check alone does not verify that retrieval returns relevant items.

## Call the local agent

```bash
curl -N http://127.0.0.1:9000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dpo-v15-trial-4","messages":[{"role":"user","content":"I need a desk lamp for homework"}],"stream":true}'
```

The direct agent route is local and has no authentication middleware. For the nginx route, send the configured bearer token:

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer ${NGINX_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{"model":"dpo-v15-trial-4","messages":[{"role":"user","content":"hello"}],"stream":true}'
```

Export `NGINX_API_KEY` into that shell first. [Streaming behavior](../AGENT_STREAMING.md)

## Remote client

[`remote_chat.py`](../remote_chat.py) currently uses hardcoded `PUBLIC_URL` and `API_KEY` constants. It does not read the `PUBLIC_URL` entry in `.env`. Edit the constants on the client machine to match the active ngrok URL and proxy token, then run:

```bash
python remote_chat.py
```

The checked-in tunnel URL is historical. Do not treat it as an active project demo link.

## Stop and troubleshoot

```bash
bash stop_saulie.sh
```

- Missing model or LoRA: inspect the paths in [`deploy_finalist_pick.sh`](../dpo/eval/vllm_scripts/deploy_finalist_pick.sh). No adapter download is published.
- BGE errors: confirm the ONNX-derived engine, tokenizer, and sparse projection files are present. Probe `POST /embed` on port 8888.
- No products: check the collection and index before blaming the model. Use [`smoke_hybrid.py`](../rag/smoke_hybrid.py).
- vLLM bind failure on WSL: the script supports `VLLM_BIND=0.0.0.0`, which also changes network exposure. [Demo security](../SECURITY.md)
- Agent cannot connect: align `LLM_BASE_URL`, vLLM port, and both local model API keys.
- Tool calls happen too early: inspect the saved pressure defaults and report before tuning. Runtime pressure is separate from DPO weights.

Deployment and benchmark entry points may stop and recreate named containers. Use a separate environment when preserving a running demo.
