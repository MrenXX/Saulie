#!/usr/bin/env python3
"""
Quick hybrid search smoke test — no agent changes needed.

Usage:
  QDRANT_COLLECTION=amazon_products FUSION_METHOD=rrf python3 smoke_hybrid.py
  QDRANT_COLLECTION=amazon_products FUSION_METHOD=dbsf python3 smoke_hybrid.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from query2 import search_hybrid, COLLECTION, FUSION_METHOD

QUERIES = [
    ("wireless earbuds noise cancelling", None),
    ("gaming laptop", None),
    ("Long sleeve shirts for men cozy winter", "men's clothing"),
    ("running shoes men", "men's shoes"),
]


def main():
    print(f"Collection: {COLLECTION}  fusion: {FUSION_METHOD}\n")
    for query, cat in QUERIES:
        print(f"=== {query!r}  category={cat!r} ===")
        results = search_hybrid(query_text=query, main_category=cat, top_k=5)
        if not results:
            print("  (no results)\n")
            continue
        for block in results:
            for i, hit in enumerate(block.get("results", []), 1):
                name = (hit.get("name") or "")[:70]
                score = hit.get("score")
                cat_h = hit.get("main_category", "")
                print(f"  {i}. [{score}] {name}  ({cat_h})")
        print()


if __name__ == "__main__":
    main()
