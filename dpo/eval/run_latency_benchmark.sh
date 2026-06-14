#!/usr/bin/env bash
# Run full Saulie latency suite: vLLM + RAG + prod E2E + report.
#
# Prerequisites (prod stack):
#   vLLM :8000, agent :9000, BGE embed :8888, Qdrant :1234
#
# Usage:
#   bash dpo/eval/run_latency_benchmark.sh
#   bash dpo/eval/run_latency_benchmark.sh --e2e-only

set -euo pipefail

REPO="/root/saulie"
PYTHON="${SAULIE_PYTHON:-/root/miniconda3/envs/saulgman/bin/python}"
E2E_ONLY=false

for arg in "$@"; do
  case "$arg" in
    --e2e-only) E2E_ONLY=true ;;
  esac
done

check() {
  local label="$1" url="$2"
  if curl -sf --max-time 3 "$url" >/dev/null 2>&1; then
    echo "[ok] $label"
  else
    echo "[x] $label not ready: $url" >&2
    exit 1
  fi
}

echo "=== Health checks ==="
check "vLLM" "http://127.0.0.1:8000/health"
check "agent" "http://127.0.0.1:9000/health"
check "embed" "http://127.0.0.1:8888/embed"  # POST may 405 on GET — fallback below
curl -sf -X POST "http://127.0.0.1:8888/embed" -H 'Content-Type: application/json' -d '{"text":"ping"}' >/dev/null || {
  echo "[x] embed server not ready" >&2; exit 1;
}
echo "[ok] embed"
check "qdrant" "http://127.0.0.1:1234/"

cd "$REPO"

if [[ "$E2E_ONLY" != true ]]; then
  echo ""
  echo "=== vLLM isolated ==="
  "$PYTHON" dpo/eval/benchmark_vllm_latency.py --runs 5

  echo ""
  echo "=== RAG isolated ==="
  "$PYTHON" rag/benchmark_latency.py --runs-per-query 3
fi

echo ""
echo "=== Prod E2E (tool turn) ==="
echo "NOTE: agent must include latest latency instrumentation (saulie_latency request flag)."
"$PYTHON" dpo/eval/benchmark_prod_latency.py --scenario all --runs 3

echo ""
echo "=== Report ==="
"$PYTHON" dpo/eval/generate_latency_report.py
echo "Done. See dpo/eval/LATENCY_REPORT.md"
