#!/usr/bin/env python3
"""RRF vs DBSF benchmark — diverse queries, per-dataset category filters."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# Same 18 query intents; category filter varies by dataset
QUERIES = [
    {"query": "wireless earbuds noise cancelling", "indian_cat": None, "mccauley_cat": None},
    {"query": "bluetooth speaker portable waterproof", "indian_cat": None, "mccauley_cat": None},
    {"query": "gaming laptop RTX", "indian_cat": None, "mccauley_cat": None},
    {"query": "32 inch smart TV 4K", "indian_cat": "tv, audio & cameras", "mccauley_cat": "All Electronics"},
    {"query": "men's running shoes lightweight", "indian_cat": "men's shoes", "mccauley_cat": "AMAZON FASHION"},
    {"query": "women's winter coat warm", "indian_cat": "women's clothing", "mccauley_cat": "AMAZON FASHION"},
    {"query": "cotton bed sheets king size", "indian_cat": "home & kitchen", "mccauley_cat": "Amazon Home"},
    {"query": "stainless steel cookware set", "indian_cat": "home & kitchen", "mccauley_cat": "Amazon Home"},
    {"query": "yoga mat non slip thick", "indian_cat": "sports & fitness", "mccauley_cat": "Sports & Outdoors"},
    {"query": "protein powder whey chocolate", "indian_cat": "beauty & health", "mccauley_cat": "Health & Personal Care"},
    {"query": "baby diaper pants large pack", "indian_cat": "toys & baby products", "mccauley_cat": "Baby"},
    {"query": "dog food dry adult", "indian_cat": "pet supplies", "mccauley_cat": "Pet Supplies"},
    {"query": "car phone mount dashboard", "indian_cat": "car & motorbike", "mccauley_cat": "Automotive"},
    {"query": "mechanical keyboard RGB gaming", "indian_cat": "tv, audio & cameras", "mccauley_cat": "All Electronics"},
    {"query": "men's formal leather belt", "indian_cat": "accessories", "mccauley_cat": "AMAZON FASHION"},
    {"query": "kids school backpack waterproof", "indian_cat": "kids' fashion", "mccauley_cat": "AMAZON FASHION"},
    {"query": "air fryer large capacity", "indian_cat": "appliances", "mccauley_cat": "Appliances"},
    {"query": "face moisturizer dry skin", "indian_cat": "beauty & health", "mccauley_cat": "All Beauty"},
]


def category_for_collection(collection: str, item: dict) -> str | None:
    if collection == "amazon_products_v2":
        return item["mccauley_cat"]
    return item["indian_cat"]


def run_fusion(method: str, collection: str):
    os.environ["FUSION_METHOD"] = method
    os.environ["QDRANT_COLLECTION"] = collection
    for mod in list(sys.modules):
        if mod == "query2" or mod.startswith("query2."):
            del sys.modules[mod]
    from query2 import search_hybrid

    out = []
    for item in QUERIES:
        cat = category_for_collection(collection, item)
        blocks = search_hybrid(item["query"], main_category=cat, top_k=3)
        hits = blocks[0]["results"] if blocks else []
        out.append({
            "query": item["query"],
            "category": cat,
            "hits": hits,
        })
    return out


def main():
    parser = argparse.ArgumentParser(description="RRF vs DBSF fusion benchmark")
    parser.add_argument(
        "--collection",
        default=os.getenv("QDRANT_COLLECTION", "amazon_products"),
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: fusion_benchmark_results.json or _v2.json)",
    )
    args = parser.parse_args()

    out_path = args.output or (
        "/root/rag/fusion_benchmark_results_v2.json"
        if args.collection == "amazon_products_v2"
        else "/root/rag/fusion_benchmark_results.json"
    )

    print(f"Collection: {args.collection}")
    print("Running RRF...")
    rrf = run_fusion("rrf", args.collection)
    print("Running DBSF...")
    dbsf = run_fusion("dbsf", args.collection)

    payload = {
        "collection": args.collection,
        "rrf": rrf,
        "dbsf": dbsf,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
