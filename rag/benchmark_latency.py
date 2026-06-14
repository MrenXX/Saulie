#!/usr/bin/env python3
"""Isolated RAG latency: embed, Qdrant, search_hybrid total."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "rag"))
sys.path.insert(0, str(REPO))

from dpo.eval.latency_stats import summarize_ms

# Agent-realistic queries + fusion battery subset
QUERIES = [
    {"query": "sleeping pad cold camping insulated", "category": "Sports & Outdoors"},
    {"query": "kids desk lamp focused beam homework", "category": "All Electronics"},
    {"query": "wireless earbuds noise cancelling", "category": None},
    {"query": "kitchen prep cleanup mat cutting board", "category": "Amazon Home"},
    {"query": "yoga mat non slip thick", "category": "Sports & Outdoors"},
    {"query": "air fryer large capacity", "category": "Appliances"},
    {"query": "mechanical keyboard RGB gaming", "category": "All Electronics"},
    {"query": "face moisturizer dry skin", "category": "All Beauty"},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-per-query", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    from query2 import search_hybrid, COLLECTION, FUSION_METHOD

    out_dir = REPO / "dpo" / "eval" / "latency_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output or out_dir / f"rag_latency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    for _ in range(args.warmup):
        search_hybrid("warmup query", top_k=3)

    rows = []
    for item in QUERIES:
        for run_idx in range(args.runs_per_query):
            timing: dict = {}
            t0 = time.perf_counter()
            hits = search_hybrid(
                item["query"],
                main_category=item["category"],
                top_k=4,
                timing_out=timing,
            )
            wall_ms = round((time.perf_counter() - t0) * 1000, 2)
            rows.append({
                "query": item["query"],
                "category": item["category"],
                "run": run_idx + 1,
                "hit_count": sum(len(b.get("results", [])) for b in hits),
                "embed_ms": timing.get("embed_ms"),
                "qdrant_ms": timing.get("qdrant_ms"),
                "total_ms": timing.get("total_ms", wall_ms),
                "wall_ms": wall_ms,
            })

    embed_vals = [r["embed_ms"] for r in rows if r["embed_ms"] is not None]
    qdrant_vals = [r["qdrant_ms"] for r in rows if r["qdrant_ms"] is not None]
    total_vals = [r["total_ms"] for r in rows if r["total_ms"] is not None]

    payload = {
        "benchmark": "rag_isolated",
        "generated_at": datetime.now().isoformat(),
        "collection": COLLECTION,
        "fusion_method": FUSION_METHOD,
        "queries": len(QUERIES),
        "runs_per_query": args.runs_per_query,
        "summary": {
            "embed_ms": summarize_ms(embed_vals),
            "qdrant_ms": summarize_ms(qdrant_vals),
            "search_total_ms": summarize_ms(total_vals),
        },
        "rows": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
