# Saulie Latency Report

Generated: 2026-06-11T01:18:28.718338

## Source runs

- vLLM: `/root/saulie/dpo/eval/latency_runs/vllm_latency_20260611_011819.json`
- RAG: `/root/saulie/dpo/eval/latency_runs/rag_latency_20260611_011818.json`
- E2E: `/root/saulie/dpo/eval/latency_runs/prod_e2e_latency_20260611_011825.json`

## SLO pass/fail

| Metric | Status | Actual | Threshold |
|--------|--------|--------|-----------|
| ttft_short_prompt | pass | 27.26ms | 800ms |
| ttft_long_prompt | pass | 31.55ms | 2500ms |
| decode_tokens_per_s | pass | 70.53tokens/s | 25tokens/s |
| embed_ms | pass | 17.61ms | 150ms |
| qdrant_ms | pass | 14.52ms | 200ms |
| search_total_ms | pass | 31.13ms | 400ms |
| probe_turn_ms | pass | 801.53ms | 2000ms |
| tool_turn_total_ms | pass | 2489.61ms | 5000ms |
| tool_rag_ms | pass | 100.21ms | 500ms |
| tool_hit_rate | pass | 1.0ratio | 1.0ratio |

## vLLM (isolated)

### Summary

- **ttft_short_prompt**: n=5 mean=26.39ms p50=26.47ms p95=27.26ms
- **ttft_long_prompt**: n=5 mean=29.07ms p50=28.75ms p95=31.55ms
- **decode_tokens_per_s**: n=10 mean=71.21 p50=70.53 p95=77.67 tok/s

## RAG (isolated)

### Summary

- **embed_ms**: n=24 mean=15.04ms p50=15.08ms p95=17.61ms
- **qdrant_ms**: n=24 mean=8.87ms p50=8.5ms p95=14.52ms
- **search_total_ms**: n=24 mean=24.07ms p50=23.24ms p95=31.13ms

## Prod E2E (agent API)

### Summary

- **probe_turn_ms**: n=5 mean=413.05ms p50=429.75ms p95=801.53ms
- **tool_turn_total_ms**: n=6 mean=2252.44ms p50=2213.87ms p95=2489.61ms
- **tool_llm1_ms**: n=6 mean=687.5ms p50=679.91ms p95=781.23ms
- **tool_rag_ms**: n=6 mean=73.78ms p50=85.58ms p95=100.21ms
- **tool_llm2_ms**: n=6 mean=1490.33ms p50=1485.73ms p95=1722.89ms
- **tool_embed_ms**: n=6 mean=12.09ms p50=11.27ms p95=15.51ms
- **tool_qdrant_ms**: n=6 mean=61.5ms p50=71.22ms p95=89.8ms
- **tool_hit_rate**: {'attempts': 6, 'successes': 6, 'rate': 1.0}

#### E2E tool-turn decomposition (p50/p95)

- tool_llm1_ms: p50=679.91ms p95=781.23ms
- tool_rag_ms: p50=85.58ms p95=100.21ms
- tool_llm2_ms: p50=1485.73ms p95=1722.89ms
- tool_turn_total_ms: p50=2213.87ms p95=2489.61ms
