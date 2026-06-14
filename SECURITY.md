# Security (demo setup)

Saulie is a **local demo**, not production. These measures keep casual abuse off your ngrok tunnel without heavy ops overhead.

## Architecture

```
Internet → ngrok → nginx :8080 (Bearer auth) → agent :9000 (127.0.0.1 only) → vLLM :8000 (127.0.0.1 only)
```

| Layer | Protection |
|-------|------------|
| **nginx** | Bearer token on `/v1/chat/completions` only; rate limit 10 req/min |
| **Agent** | Listens on `127.0.0.1` only — not reachable from LAN/internet directly |
| **vLLM** | Docker port bound to `127.0.0.1:8000`; API key from `.env` |
| **Health** | `/health` on agent is localhost-only; **not** exposed through nginx |

## Secrets (`.env`)

Copy `.env.example` → `.env`. All keys live there (gitignored):

| Variable | Default | Used by |
|----------|---------|---------|
| `NGINX_API_KEY` | `secret` | nginx Bearer token (public ngrok clients) |
| `VLLM_API_KEY` | `dipshit` | vLLM container + eval scripts |
| `LLM_API_KEY` | `dipshit` | agent → vLLM client |
| `NGROK_KEY` | (required for tunnel) | ngrok authtoken |

`remote_chat.py` keeps hardcoded `API_KEY = "secret"` for easy testing from another laptop — update manually if you change `NGINX_API_KEY`.

## nginx token from `.env`

nginx cannot read `.env` directly. On stack startup, `start_saulie.sh` substitutes `__NGINX_API_KEY__` in `nginx/nginx.conf` into a generated `nginx/nginx.runtime.conf` (gitignored), which the Docker container mounts.

If you change `NGINX_API_KEY`, recreate nginx:

```bash
docker rm -f nginx
bash start_saulie.sh
```

## Local health checks

```bash
curl http://127.0.0.1:9000/health          # agent (localhost only)
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer secret" ...     # via nginx
```

## Not implemented (intentionally)

- FastAPI auth middleware (nginx is the public gate)
- ngrok HTTP basic auth
- Message logging restrictions (kept for debugging)

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for environment setup.
