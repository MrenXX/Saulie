#!/usr/bin/env bash
while true; do
  clear
  echo "══════════════════════════════════════════════════════════════"
  echo " SAULIE STACK — $(date '+%Y-%m-%d %H:%M:%S')"
  echo "══════════════════════════════════════════════════════════════"
  printf " vLLM (:8000)     : "; curl -sf --max-time 2 "http://127.0.0.1:8000/health" >/dev/null && echo "UP" || echo "DOWN"
  printf " Agent (:9000)   : "; curl -sf --max-time 2 "http://127.0.0.1:9000/health" >/dev/null && echo "UP" || echo "DOWN"
  printf " nginx (:8080)   : "; code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 -X POST "http://127.0.0.1:8080/v1/chat/completions" -H "Content-Type: application/json" -d '{}'); [[ "$code" == "401" ]] && echo "UP" || echo "DOWN"
  printf " BGE (:8888)     : "; curl -sf --max-time 3 -X POST "http://127.0.0.1:8888/embed" -H "Content-Type: application/json" -d '{"text":"ping"}' >/dev/null && echo "UP" || echo "DOWN"
  printf " Qdrant (:1234) : "; curl -sf --max-time 2 "http://127.0.0.1:1234/" >/dev/null && echo "UP" || echo "DOWN"
  echo "──────────────────────────────────────────────────────────────"
  echo " Local health: http://127.0.0.1:9000/health  (localhost only)"
  echo " nginx proxy : http://127.0.0.1:8080/v1/chat/completions  (Bearer secret)"
  URL=$(curl -sf --max-time 2 http://127.0.0.1:4040/api/tunnels 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(next((t['public_url'] for t in d.get('tunnels',[]) if t.get('proto')=='https'), 'ngrok not running'))" 2>/dev/null || echo "ngrok not running")
  echo " Public URL  : $URL"
  echo "──────────────────────────────────────────────────────────────"
  echo " tmux: Ctrl+B D detach | Ctrl+B [ scroll | windows 1-4 at bottom"
  echo "══════════════════════════════════════════════════════════════"
  sleep 3
done
