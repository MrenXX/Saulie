#!/usr/bin/env python3
"""Isolated vLLM latency: TTFT and decode tokens/s via streaming."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from dpo.eval.latency_stats import summarize_ms, summarize_tokens_per_s
from dpo.eval.v15_eval_config import (
    EVAL_MAX_TOKENS,
    EVAL_REPETITION_PENALTY,
    EVAL_TEMPERATURE,
    EVAL_TOP_K,
    EVAL_TOP_P,
    VLLM_API_KEY,
    VLLM_BASE_URL,
    vllm_extra_body,
)

SHORT_PROMPT = [{"role": "user", "content": "Say hello in one short sentence."}]
LONG_PROMPT = SHORT_PROMPT + [
    {"role": "assistant", "content": "Hey — what's on your mind today?"},
    {"role": "user", "content": "I've been camping and my sleeping setup is uncomfortable when it gets cold."},
    {"role": "assistant", "content": "Cold ground steals heat fast. What's your budget and how cold does it actually get?"},
    {"role": "user", "content": "Mid-thirties at night, budget around eighty bucks, I want something packable."},
]


def bench_stream(client: OpenAI, model: str, messages: list[dict], label: str) -> dict:
    t0 = time.perf_counter()
    ttft_ms = None
    chunks = []
    usage = None

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=EVAL_MAX_TOKENS,
        temperature=EVAL_TEMPERATURE,
        top_p=EVAL_TOP_P,
        stream=True,
        stream_options={"include_usage": True},
        extra_body=vllm_extra_body(),
    )
    for chunk in stream:
        if getattr(chunk, "usage", None):
            usage = chunk.usage
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            if ttft_ms is None:
                ttft_ms = round((time.perf_counter() - t0) * 1000, 2)
            chunks.append(delta)

    total_ms = round((time.perf_counter() - t0) * 1000, 2)
    text = "".join(chunks)
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
    if completion_tokens is None:
        completion_tokens = max(1, len(text.split()))
    decode_ms = max(0.001, total_ms - (ttft_ms or 0))
    tps = round(completion_tokens / (decode_ms / 1000), 2)

    return {
        "label": label,
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
        "decode_ms": round(decode_ms, 2),
        "completion_tokens": completion_tokens,
        "tokens_per_s": tps,
        "chars_out": len(text),
    }


def scrape_vllm_metrics(base_url: str) -> dict:
    import urllib.request

    metrics_url = base_url.replace("/v1", "") + "/metrics"
    try:
        raw = urllib.request.urlopen(metrics_url, timeout=5).read().decode("utf-8", errors="replace")
    except Exception as exc:
        return {"error": str(exc), "url": metrics_url}

    interesting = {}
    for line in raw.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        for key in (
            "time_to_first_token_seconds",
            "inter_token_latency_seconds",
            "e2e_request_latency_seconds",
            "generation_tokens_total",
            "prompt_tokens_total",
        ):
            if key in line and "_bucket" not in line and "_count" not in line and "_sum" not in line:
                interesting.setdefault(key, []).append(line.strip())
    return {"url": metrics_url, "samples": interesting}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="dpo-v15-trial-4")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    out_dir = REPO / "dpo" / "eval" / "latency_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output or out_dir / f"vllm_latency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    for _ in range(args.warmup):
        bench_stream(client, args.model, SHORT_PROMPT, "warmup")

    rows = []
    for i in range(args.runs):
        rows.append(bench_stream(client, args.model, SHORT_PROMPT, f"short_{i+1}"))
        rows.append(bench_stream(client, args.model, LONG_PROMPT, f"long_{i+1}"))

    short_ttft = [r["ttft_ms"] for r in rows if r["label"].startswith("short") and r["ttft_ms"]]
    long_ttft = [r["ttft_ms"] for r in rows if r["label"].startswith("long") and r["ttft_ms"]]
    tps_vals = [r["tokens_per_s"] for r in rows if r["tokens_per_s"]]

    payload = {
        "benchmark": "vllm_isolated",
        "generated_at": datetime.now().isoformat(),
        "model": args.model,
        "base_url": VLLM_BASE_URL,
        "sampling": {
            "temperature": EVAL_TEMPERATURE,
            "top_p": EVAL_TOP_P,
            "top_k": EVAL_TOP_K,
            "repetition_penalty": EVAL_REPETITION_PENALTY,
            "max_tokens": EVAL_MAX_TOKENS,
        },
        "summary": {
            "ttft_short_prompt": summarize_ms(short_ttft),
            "ttft_long_prompt": summarize_ms(long_ttft),
            "decode_tokens_per_s": summarize_tokens_per_s(tps_vals),
        },
        "vllm_metrics_snapshot": scrape_vllm_metrics(VLLM_BASE_URL),
        "rows": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
