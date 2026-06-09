import os
import requests
import time
from qdrant_client import QdrantClient
from qdrant_client.http import models

# --- CONFIGURATION ---
EMBED_URL = os.getenv("EMBED_URL", "http://localhost:8888/embed")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:1234")
COLLECTION = os.getenv("QDRANT_COLLECTION", "amazon_products_v2")
FUSION_METHOD = os.getenv("FUSION_METHOD", "rrf").lower()  # rrf | dbsf
DEFAULT_PREFETCH = int(os.getenv("PREFETCH_LIMIT", "50"))

client = QdrantClient(url=QDRANT_URL, check_compatibility=False)


def product_url(parent_asin: str) -> str:
    asin = (parent_asin or "").strip().upper()
    return f"https://www.amazon.com/dp/{asin}" if asin else ""


def _fusion_mode():
    if FUSION_METHOD == "dbsf":
        return models.Fusion.DBSF
    return models.Fusion.RRF


def get_server_embeddings(text: str):
    """Calls the BGE-M3 TensorRT server for dense + sparse embeddings."""
    try:
        response = requests.post(EMBED_URL, json={"text": text}, timeout=30)
        response.raise_for_status()
        data = response.json()
        return {"dense": data["dense"], "sparse": data["sparse"]}
    except Exception as e:
        print(f"Error calling embedding server: {e}")
        return None


def _format_hit(hit) -> dict:
    payload = hit.payload or {}
    asin = payload.get("parent_asin") or ""
    return {
        "score": round(hit.score, 4),
        "name": payload.get("name") or "missing name / description of product",
        "actual_price": payload.get("actual_price") or "0",
        "discount_price": payload.get("discount_price") or "0",
        "ratings": payload.get("ratings") or "No ratings available",
        "no_of_ratings": payload.get("no_of_ratings"),
        "parent_asin": asin,
        "link": product_url(asin) or payload.get("link") or "No link available",
        "main_category": payload.get("main_category"),
    }


def search_hybrid(query_text, top_k=5, main_category=None, prefetch_limit=None):
    """Hybrid search (dense + sparse) with RRF or DBSF fusion."""
    prefetch_limit = prefetch_limit or DEFAULT_PREFETCH
    queries = query_text if isinstance(query_text, list) else [query_text]
    queries = queries[:2]
    if not queries:
        return []

    embs = get_server_embeddings(queries)
    if not embs:
        return []

    query_filter = None
    if main_category:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="main_category",
                    match=models.MatchValue(value=main_category),
                )
            ]
        )

    out = []
    for q, dense_vec, sparse in zip(queries, embs["dense"], embs["sparse"]):
        sparse_vec = models.SparseVector(
            indices=sparse["indices"],
            values=sparse["values"],
        )
        prefetch = [
            models.Prefetch(
                query=dense_vec,
                using="dense",
                limit=prefetch_limit,
                filter=query_filter,
            ),
            models.Prefetch(
                query=sparse_vec,
                using="sparse",
                limit=prefetch_limit,
                filter=query_filter,
            ),
        ]
        resp = client.query_points(
            collection_name=COLLECTION,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=_fusion_mode()),
            limit=top_k,
            with_payload=[
                "name", "parent_asin", "main_category",
                "ratings", "no_of_ratings", "discount_price", "actual_price",
            ],
        )
        out.append({
            "query": q,
            "results": [_format_hit(hit) for hit in resp.points],
        })

    return out


if __name__ == "__main__":
    query_text = "wireless earbuds noise cancelling"
    print(f"Collection: {COLLECTION}  fusion: {FUSION_METHOD}")
    print(f"Query: '{query_text}'")
    print("-" * 50)

    t0 = time.perf_counter()
    results = search_hybrid(query_text, top_k=3)
    t1 = time.perf_counter()

    print(f"Results in {t1 - t0:.4f}s:")
    for block in results:
        print(f"  sub-query: {block['query']!r}")
        for r in block["results"]:
            print(
                f"    [{r['score']}] {r['name'][:65]} | "
                f"${r['actual_price']} | {r['link']}"
            )
