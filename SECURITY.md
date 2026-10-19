# Demo security

Saulie is a personal local demo. nginx authenticates the optional public route; the internal services are not a complete hardened production system.

## Requests and keys

ngrok forwards to nginx on host port 8080. nginx permits the chat-completions route with a bearer token and rate limiting. The agent listens on loopback by default and has no authentication middleware of its own. nginx does not expose the agent health route.

Set real values in local `.env`, which is ignored by Git. Replace every sample key in [`.env.example`](.env.example). `NGINX_API_KEY` authenticates proxy clients. `VLLM_API_KEY` and `LLM_API_KEY` must agree for agent-to-vLLM traffic. `NGROK_KEY` enables the optional tunnel.

Some legacy scripts retain fixed fallback keys. They are not suitable public-access credentials. `remote_chat.py` also uses constants rather than reading `.env`; update its token and URL manually.

## Bindings

The normal vLLM launcher defaults to `127.0.0.1`. Its `VLLM_BIND=0.0.0.0` WSL workaround broadens that binding.

The current Qdrant and BGE setup script publishes host ports without a loopback restriction. Do not assume those services are private because the agent is bound to loopback. Check Docker bindings and the host firewall before exposing the machine or forwarding ports.

## Generated files and logs

Startup renders `nginx/nginx.runtime.conf` from the template and local token. That generated file is ignored. Startup also generates `.saulie/status_loop.sh`; it is runtime output rather than a source file.

Debug logs may contain conversation content and internal request details. Keep local `.env`, logs, agent sessions, tracking stores, generated config, and runtime storage out of commits. [The ignore file](.gitignore) covers the known local artifacts.

See [deployment](docs/deployment.md) for operation and [reproducibility](REPRODUCIBILITY.md) for external dependencies.
