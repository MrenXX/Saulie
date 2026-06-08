#!/usr/bin/env python3
"""
Validate Qdrant collection: zero-dense audit + query battery.
"""
import os
import sys
import numpy as np
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:1234")
COLLECTION = os.getenv("QDRANT_COLLECTION", "amazon_products")
EMBED_URL = os.getenv("EMBED_URL", "http://localhost:8888/embed")
MAX_ZERO_PCT = 0.1
SAMPLE_SIZE = 50_000

TEST_QUERIES = [
    "wireless earbuds noise cancelling",
    "gaming laptop",
    "winter jacket men",
    "Long sleeve shirts for men cozy winter",
]


def audit_zero_dense(client: QdrantClient) -> tuple[int, int, float]:
    zero = valid = 0
    offset = None
    scanned = 0
    while scanned < SAMPLE_SIZE:
        pts, offset = client.scroll(
            COLLECTION, limit=1000, offset=offset,
            with_vectors=["dense"], with_payload=False,
        )
        if not pts:
            break
        for p in pts:
            n = float(np.linalg.norm(p.vector["dense"]))
            if n < 1e-6:
                zero += 1
            else:
                valid += 1
        scanned += len(pts)
        if offset is None:
            break
    total = zero + valid
    pct = 100.0 * zero / total if total else 100.0
    return zero, valid, pct


def embed_query(text: str):
    r = requests.post(EMBED_URL, json={"text": text}, timeout=30)
    r.raise_for_status()
    d = r.json()
    return d["dense"][0], d["sparse"][0]


def top_dense_names(client: QdrantClient, query: str, k: int = 3) -> list[str]:
    dense, _ = embed_query(query)
    hits = client.query_points(
        COLLECTION, query=dense, using="dense", limit=k,
        with_payload=["name", "main_category"],
    ).points
    return [f"[{h.score:.3f}] {h.payload.get('name','')[:60]}" for h in hits]


def main():
    client = QdrantClient(url=QDRANT_URL, check_compatibility=False)
    if not client.collection_exists(COLLECTION):
        print(f"[x] Collection '{COLLECTION}' does not exist")
        sys.exit(1)

    info = client.get_collection(COLLECTION)
    print(f"Collection: {COLLECTION}  points: {info.points_count:,}")

    print(f"\n--- Zero-dense audit (sample up to {SAMPLE_SIZE:,}) ---")
    zero, valid, pct = audit_zero_dense(client)
    print(f"valid: {valid:,}  zero: {zero:,}  zero%: {pct:.3f}")
    if pct > MAX_ZERO_PCT:
        print(f"[FAIL] zero dense > {MAX_ZERO_PCT}%")
        sys.exit(1)
    print("[ok] dense vectors look healthy")

    print("\n--- Query battery (dense top-3) ---")
    for q in TEST_QUERIES:
        print(f"\nQuery: {q!r}")
        for line in top_dense_names(client, q):
            print(f"  {line}")

    print("\n[ok] validate_index passed")


if __name__ == "__main__":
    main()
