#!/usr/bin/env python3
"""A/B retrieval eval: pure RRF baseline vs RRF + dense-cosine gate.

RRF (Qdrant k=2) is a strong RANKER for short product titles, but its score is rank-based and
cannot tell relevant from irrelevant. The dense-cosine gate drops sub-threshold hits. This
question -- does gating actually return better results, or does it throw away good RRF hits? --
is empirical, so this script runs BOTH configs over the 18 labeled benchmark queries and
reports relevance metrics side by side. RRF ranking is identical in both; only the gate differs.

Configs (RAG_MAX_RESULTS held equal so the only variable is the gate):
  A  baseline : SAULIE_RAG_MIN_COSINE = 0      -> top-N by fused RRF rank, no gate
  B  gated    : SAULIE_RAG_MIN_COSINE = <thr>  -> RRF order, cosine >= thr, dynamic count

Run on the machine with embed (:8888) + Qdrant (:1234) up, AFTER calibrating the threshold:
    python rag/calibrate_cosine_threshold.py        # -> pick THR
    python rag/retrieval_ab_eval.py --threshold THR  # default 0.5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import query2  # noqa: E402
from query2 import search_hybrid  # noqa: E402
from benchmark_fusion import QUERIES, category_for_collection  # noqa: E402
from calibrate_cosine_threshold import RUBRIC, _is_relevant  # noqa: E402  (reuse rubric)


def run_config(min_cosine: float, collection: str, top_k: int, max_results: int):
    query2.COLLECTION = collection
    query2.RAG_MIN_COSINE = min_cosine
    query2.RAG_MAX_RESULTS = max_results
    os.environ["QDRANT_COLLECTION"] = collection

    rows = []
    for item in QUERIES:
        q = item["query"]
        cat = category_for_collection(collection, item)
        blocks = search_hybrid(q, main_category=cat, top_k=top_k)
        hits = blocks[0]["results"] if blocks else []
        rels = [_is_relevant(q, h.get("name", "")) for h in hits]
        rows.append({
            "query": q,
            "returned": len(hits),
            "relevant": sum(rels),
            "top1_relevant": bool(rels and rels[0]),
            "any_relevant": any(rels),
            "no_results": len(hits) == 0,
            "top_name": (hits[0]["name"][:48] if hits else ""),
        })
    return rows


def summarize(rows):
    n = len(rows)
    returned = sum(r["returned"] for r in rows)
    relevant = sum(r["relevant"] for r in rows)
    return {
        "queries": n,
        "total_returned": returned,
        "total_relevant": relevant,
        "precision": round(relevant / returned, 3) if returned else 0.0,
        "top1_acc": round(sum(r["top1_relevant"] for r in rows) / n, 3) if n else 0.0,
        "coverage": round(sum(r["any_relevant"] for r in rows) / n, 3) if n else 0.0,
        "no_result_queries": sum(r["no_results"] for r in rows),
        "avg_returned": round(returned / n, 2) if n else 0.0,
    }


def _print_block(label, s):
    print(f"\n[{label}]")
    print(f"  precision (relevant/returned) : {s['precision']}")
    print(f"  top-1 accuracy                : {s['top1_acc']}")
    print(f"  coverage (>=1 relevant/query) : {s['coverage']}")
    print(f"  avg results returned          : {s['avg_returned']}")
    print(f"  queries returning nothing     : {s['no_result_queries']}/{s['queries']}")
    print(f"  totals: returned={s['total_returned']} relevant={s['total_relevant']}")


def main():
    ap = argparse.ArgumentParser(description="A/B retrieval: pure RRF vs RRF+dense gate")
    ap.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "amazon_products_v2"))
    ap.add_argument("--threshold", type=float, default=0.5, help="cosine gate for config B")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-results", type=int, default=5)
    args = ap.parse_args()

    print(f"Collection: {args.collection}  top_k={args.top_k}  max_results={args.max_results}")
    print(f"Config A (baseline): RRF only, no gate")
    print(f"Config B (gated):    RRF + cosine >= {args.threshold}")

    rows_a = run_config(0.0, args.collection, args.top_k, args.max_results)
    rows_b = run_config(args.threshold, args.collection, args.top_k, args.max_results)
    sa, sb = summarize(rows_a), summarize(rows_b)

    print("\nPer-query (A returned/relevant | B returned/relevant | query):")
    by_q_b = {r["query"]: r for r in rows_b}
    for ra in rows_a:
        rb = by_q_b[ra["query"]]
        print(f"  A {ra['returned']}/{ra['relevant']}  |  B {rb['returned']}/{rb['relevant']}  |  {ra['query']}")

    _print_block("A  pure RRF baseline", sa)
    _print_block("B  RRF + dense gate", sb)

    print("\n--- verdict ---")
    print(f"  precision delta (B - A): {round(sb['precision'] - sa['precision'], 3):+}")
    print(f"  coverage delta  (B - A): {round(sb['coverage'] - sa['coverage'], 3):+}")
    print(f"  B drops {sb['no_result_queries'] - sa['no_result_queries']:+} queries to 'no results' vs A")
    print("  Higher precision + acceptable coverage loss favors the gate; large coverage")
    print("  loss with little precision gain favors pure RRF. Decide from the table above.")
    if sb["no_result_queries"] == sb["queries"]:
        print("  [WARN] Config B returned nothing for EVERY query -- threshold too high or index stale.")


if __name__ == "__main__":
    main()
