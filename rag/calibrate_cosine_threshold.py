#!/usr/bin/env python3
"""Calibrate SAULIE_RAG_MIN_COSINE on the 18-query labeled benchmark.

Why: RRF (Qdrant k=2) scores are rank-based, not relevance-based -- the top fused hit is
~always >= 0.5 even for garbage queries, so it cannot gate. The dense vectors are a COSINE
collection, so dense cosine is the true 0-1 relevance signal. This script runs the same 18
benchmark queries through a dense-only search, labels each returned product with a keyword
rubric (reproducing fusion_comparison_report.md's method), and reports the cosine threshold
that best separates relevant from irrelevant hits.

It also doubles as a sanity check for the historical dense-vector corruption bug (README:
SERVER_BATCH_SIZE). If cosine cannot separate good from junk (overlap warning), the index is
likely stale/corrupted and gating on cosine will not work until it is rebuilt.

Run on the machine with the embed server (:8888) and Qdrant (:1234) up:
    python rag/calibrate_cosine_threshold.py
    python rag/calibrate_cosine_threshold.py --top-n 10 --collection amazon_products_v2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from qdrant_client.http import models  # noqa: E402

import query2  # noqa: E402
from query2 import client, get_server_embeddings  # noqa: E402
from benchmark_fusion import QUERIES, category_for_collection  # noqa: E402

# Relevant if the product name contains any of these tokens (lowercased substring match).
# Mirrors the manual keyword rubric used to label fusion_comparison_report.md.
RUBRIC: dict[str, list[str]] = {
    "wireless earbuds noise cancelling": ["earbud", "earphone", "headphone", "in-ear", "in ear"],
    "bluetooth speaker portable waterproof": ["speaker"],
    "gaming laptop RTX": ["laptop", "notebook"],
    "32 inch smart TV 4K": ["tv", "television"],
    "men's running shoes lightweight": ["shoe", "sneaker", "running", "trainer"],
    "women's winter coat warm": ["coat", "jacket", "parka", "puffer"],
    "cotton bed sheets king size": ["sheet", "bedsheet", "bedding", "duvet"],
    "stainless steel cookware set": ["cookware", "pot", "pan", "saucepan", "skillet", "dutch oven"],
    "yoga mat non slip thick": ["yoga mat", "mat"],
    "protein powder whey chocolate": ["protein", "whey"],
    "baby diaper pants large pack": ["diaper", "nappy"],
    "dog food dry adult": ["dog food", "dog"],
    "car phone mount dashboard": ["mount", "holder", "cradle"],
    "mechanical keyboard RGB gaming": ["keyboard"],
    "men's formal leather belt": ["belt"],
    "kids school backpack waterproof": ["backpack", "bag", "rucksack"],
    "air fryer large capacity": ["air fryer", "fryer"],
    "face moisturizer dry skin": ["moisturizer", "moisturising", "moisturiser", "cream", "lotion", "hydra"],
}


def _is_relevant(query: str, name: str) -> bool:
    tokens = RUBRIC.get(query, [])
    low = (name or "").lower()
    return any(tok in low for tok in tokens)


def _dense_search(dense_vec, category, top_n):
    query_filter = None
    if category:
        query_filter = models.Filter(
            must=[models.FieldCondition(key="main_category", match=models.MatchValue(value=category))]
        )
    resp = client.query_points(
        collection_name=query2.COLLECTION,
        query=dense_vec,
        using="dense",
        limit=top_n,
        query_filter=query_filter,
        with_payload=["name"],
    )
    return resp.points


def _sweep(pairs: list[tuple[float, bool]]) -> tuple[float, dict]:
    """Pick the cosine threshold maximizing F1 over (cosine, relevant) pairs."""
    if not pairs:
        return 0.0, {}
    candidates = sorted({round(c, 3) for c, _ in pairs})
    best_t, best = candidates[0], {"f1": -1.0}
    for t in candidates:
        tp = sum(1 for c, r in pairs if c >= t and r)
        fp = sum(1 for c, r in pairs if c >= t and not r)
        fn = sum(1 for c, r in pairs if c < t and r)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best["f1"]:
            best_t, best = t, {"f1": round(f1, 3), "precision": round(prec, 3), "recall": round(rec, 3)}
    return best_t, best


def main():
    ap = argparse.ArgumentParser(description="Calibrate dense-cosine relevance threshold")
    ap.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "amazon_products_v2"))
    ap.add_argument("--top-n", type=int, default=10, help="dense hits to inspect per query")
    args = ap.parse_args()

    os.environ["QDRANT_COLLECTION"] = args.collection
    query2.COLLECTION = args.collection

    all_pairs: list[tuple[float, bool]] = []
    top_hit_rows: list[tuple[str, float, bool, str]] = []

    print(f"Collection: {args.collection}  top_n: {args.top_n}\n")
    for item in QUERIES:
        q = item["query"]
        cat = category_for_collection(args.collection, item)
        emb = get_server_embeddings([q])
        if not emb or not emb.get("dense"):
            raise SystemExit(f"Embed server returned nothing for {q!r} -- is :8888 up?")
        points = _dense_search(emb["dense"][0], cat, args.top_n)
        if not points:
            print(f"  [WARN] no dense hits for {q!r} (filter={cat!r})")
            continue
        for j, p in enumerate(points):
            name = (p.payload or {}).get("name", "")
            rel = _is_relevant(q, name)
            all_pairs.append((float(p.score), rel))
            if j == 0:
                top_hit_rows.append((q, float(p.score), rel, name[:60]))

    print("Top hit per query (cosine | relevant | name):")
    for q, cos, rel, name in top_hit_rows:
        flag = "REL " if rel else "junk"
        print(f"  {cos:6.4f}  {flag}  {q[:34]:34s} -> {name}")

    rel_cos = [c for c, r in all_pairs if r]
    junk_cos = [c for c, r in all_pairs if not r]
    best_t, metrics = _sweep(all_pairs)

    print("\n--- separation ---")
    if rel_cos:
        print(f"  relevant  cosine: min={min(rel_cos):.4f}  max={max(rel_cos):.4f}  n={len(rel_cos)}")
    if junk_cos:
        print(f"  irrelevant cosine: min={min(junk_cos):.4f}  max={max(junk_cos):.4f}  n={len(junk_cos)}")
    if rel_cos and junk_cos and max(junk_cos) >= min(rel_cos):
        print("  [WARN] relevant/irrelevant cosine ranges OVERLAP. Either the rubric needs"
              " tuning or the dense index is stale/corrupted (see README SERVER_BATCH_SIZE).")

    print("\n--- recommendation ---")
    print(f"  SAULIE_RAG_MIN_COSINE={best_t}   (F1={metrics.get('f1')}, "
          f"precision={metrics.get('precision')}, recall={metrics.get('recall')})")
    print("  Review the top-hit table above before committing the value.")


if __name__ == "__main__":
    main()
