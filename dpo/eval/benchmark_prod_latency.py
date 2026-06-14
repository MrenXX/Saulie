#!/usr/bin/env python3
"""Prod E2E latency via agent_chat_api with decomposed llm/tool phases."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from dpo.eval.latency_stats import summarize_ms

SCENARIOS_PATH = Path(__file__).resolve().parent / "latency_scenarios.json"
FIXTURES_DIR = Path(__file__).resolve().parent / "latency_fixtures"
OUTPUT_DIR = Path(__file__).resolve().parent / "latency_runs"


def _http_json(method: str, url: str, body: dict | None = None, timeout: float = 300) -> dict:
    data = None
    headers = {"Content-Type": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _phase_ms(latency: dict, name: str) -> float | None:
    for p in latency.get("phases", []):
        if p.get("name") == name:
            if name == "tool":
                return p.get("total_ms")
            return p.get("elapsed_ms")
    return None


def run_turn(agent_url: str, messages: list[dict], model: str) -> dict:
    t0 = time.perf_counter()
    resp = _http_json(
        "POST",
        f"{agent_url.rstrip('/')}/v1/chat/completions",
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": 0.7,
            "top_p": 0.8,
            "max_tokens": 512,
            "saulie_latency": True,
        },
    )
    wall_ms = round((time.perf_counter() - t0) * 1000, 2)
    lat = resp.get("saulie_latency") or resp.get("_saulie_latency") or {}
    assistant = resp.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    return {
        "assistant": assistant,
        "wall_ms": wall_ms,
        "latency": lat,
        "tool_executed": bool(lat.get("tool_executed")),
        "llm1_ms": _phase_ms(lat, "llm1"),
        "llm2_ms": _phase_ms(lat, "llm2"),
        "tool_ms": lat.get("tool_total_ms"),
        "tool_embed_ms": _phase_ms(lat, "tool") and next(
            (p.get("embed_ms") for p in lat.get("phases", []) if p.get("name") == "tool"), None
        ),
        "tool_qdrant_ms": next(
            (p.get("qdrant_ms") for p in lat.get("phases", []) if p.get("name") == "tool"), None
        ),
        "turn_total_ms": lat.get("turn_total_ms", wall_ms),
    }


def run_camping_full(agent_url: str, model: str, scenario: dict) -> list[dict]:
    messages: list[dict] = []
    results = []
    for i, user_text in enumerate(scenario["user_turns"], start=1):
        messages.append({"role": "user", "content": user_text})
        row = run_turn(agent_url, messages, model)
        row["scenario"] = "camping_full"
        row["turn"] = i
        row["user"] = user_text
        row["is_tool_turn"] = i == scenario["tool_turn"]
        messages.append({"role": "assistant", "content": row["assistant"]})
        results.append(row)
    return results


def run_camping_warm(agent_url: str, model: str, scenario: dict) -> dict:
    fixture_path = REPO / scenario["fixture"]
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    messages = list(fixture["messages"])
    messages.append({"role": "user", "content": scenario["user_turn"]})
    row = run_turn(agent_url, messages, model)
    row["scenario"] = "camping_warm"
    row["turn"] = 6
    row["user"] = scenario["user_turn"]
    row["is_tool_turn"] = True
    return row


def summarize_e2e(rows: list[dict]) -> dict:
    probe = [r["turn_total_ms"] for r in rows if not r.get("is_tool_turn")]
    tool_rows = [r for r in rows if r.get("is_tool_turn")]
    tool_valid = [r for r in tool_rows if r.get("tool_executed")]

    def col(name: str, data: list[dict]) -> dict:
        vals = [r[name] for r in data if r.get(name) is not None]
        return summarize_ms(vals)

    return {
        "probe_turn_ms": summarize_ms(probe),
        "tool_turn_total_ms": col("turn_total_ms", tool_valid),
        "tool_llm1_ms": col("llm1_ms", tool_valid),
        "tool_rag_ms": col("tool_ms", tool_valid),
        "tool_llm2_ms": col("llm2_ms", tool_valid),
        "tool_embed_ms": col("tool_embed_ms", tool_valid),
        "tool_qdrant_ms": col("tool_qdrant_ms", tool_valid),
        "tool_hit_rate": {
            "attempts": len(tool_rows),
            "successes": len(tool_valid),
            "rate": round(len(tool_valid) / len(tool_rows), 3) if tool_rows else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-url", default="http://127.0.0.1:9000")
    parser.add_argument("--model", default="dpo-v15-trial-4")
    parser.add_argument("--scenario", choices=("camping_full", "camping_warm", "all"), default="all")
    parser.add_argument("--runs", type=int, default=3, help="camping_warm repetitions")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))["scenarios"]
    health = _http_json("GET", f"{args.agent_url.rstrip('/')}/health")
    if health.get("model") != args.model:
        print(f"WARNING: agent model={health.get('model')} expected {args.model}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.output or OUTPUT_DIR / f"prod_e2e_latency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    all_rows: list[dict] = []

    if args.scenario in ("camping_warm", "all"):
        for _ in range(args.warmup):
            run_camping_warm(args.agent_url, args.model, scenarios["camping_warm"])
        for run_i in range(args.runs):
            row = run_camping_warm(args.agent_url, args.model, scenarios["camping_warm"])
            row["run"] = run_i + 1
            all_rows.append(row)

    if args.scenario in ("camping_full", "all"):
        full_rows = run_camping_full(args.agent_url, args.model, scenarios["camping_full"])
        for r in full_rows:
            r["run"] = 1
        all_rows.extend(full_rows)

    summary = summarize_e2e(all_rows)
    payload = {
        "benchmark": "prod_e2e",
        "generated_at": datetime.now().isoformat(),
        "agent_url": args.agent_url,
        "health": health,
        "summary": summary,
        "rows": all_rows,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
