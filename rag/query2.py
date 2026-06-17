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
# Relevance gating. RRF (Qdrant k=2) scores are RANK-based, not relevance-based: the top
# fused hit is ~always >= 0.5 even for garbage queries, so it cannot gate. RRF still RANKS
# (it is strong for short product titles); dense COSINE is added ONLY as a relevance gate.
# Calibrate SAULIE_RAG_MIN_COSINE on the labeled benchmark (calibrate_cosine_threshold.py).
# Behavior:
#   SAULIE_RAG_MIN_COSINE > 0  -> gate ON: return RRF-ordered hits with cosine >= threshold,
#                                 capped at MAX_RESULTS; zero above => "no relevant products".
#   SAULIE_RAG_MIN_COSINE <= 0 -> gate OFF: pure RRF baseline, top MAX_RESULTS by fused rank
#                                 (no extra dense pass). Use 0 for the A/B baseline config.
RAG_MIN_COSINE = float(os.getenv("SAULIE_RAG_MIN_COSINE", "0.5"))
RAG_MAX_RESULTS = int(os.getenv("SAULIE_RAG_MAX_RESULTS", "5"))

client = QdrantClient(url=QDRANT_URL, check_compatibility=False)


def product_url(parent_asin: str) -> str:
    asin = (parent_asin or "").strip().upper()
    return f"https://www.amazon.com/dp/{asin}" if asin else ""


def _fusion_mode():
    if FUSION_METHOD == "dbsf":
        return models.Fusion.DBSF
    return models.Fusion.RRF


def get_server_embeddings(text: str, timing_out: dict | None = None):
    """Calls the BGE-M3 TensorRT server for dense + sparse embeddings."""
    try:
        t0 = time.perf_counter()
        response = requests.post(EMBED_URL, json={"text": text}, timeout=30)
        response.raise_for_status()
        data = response.json()
        if timing_out is not None:
            timing_out["embed_ms"] = round((time.perf_counter() - t0) * 1000, 2)
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


def search_hybrid(query_text, top_k=5, main_category=None, prefetch_limit=None, timing_out: dict | None = None):
    """Hybrid search (dense + sparse) with RRF or DBSF fusion."""
    total_t0 = time.perf_counter()
    prefetch_limit = prefetch_limit or DEFAULT_PREFETCH
    queries = query_text if isinstance(query_text, list) else [query_text]
    queries = queries[:2]
    if not queries:
        return []

    embed_timing: dict = {}
    embs = get_server_embeddings(queries, timing_out=embed_timing)
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

    qdrant_ms = 0.0
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
        gate_on = RAG_MIN_COSINE > 0.0
        t_q = time.perf_counter()
        fused_limit = max(top_k, RAG_MAX_RESULTS)
        resp = client.query_points(
            collection_name=COLLECTION,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=_fusion_mode()),
            limit=fused_limit,
            with_payload=[
                "name", "parent_asin", "main_category",
                "ratings", "no_of_ratings", "discount_price", "actual_price",
            ],
        )
        cosine_by_id: dict = {}
        if gate_on:
            # Dense-only pass for the relevance signal. Same dense vector (already embedded)
            # and same category filter; cosine score per point id is the true relevance
            # measure RRF cannot provide. with_payload=False -> only id + score.
            dense_resp = client.query_points(
                collection_name=COLLECTION,
                query=dense_vec,
                using="dense",
                limit=prefetch_limit,
                query_filter=query_filter,
                with_payload=False,
            )
            cosine_by_id = {p.id: p.score for p in dense_resp.points}
        qdrant_ms += (time.perf_counter() - t_q) * 1000

        if gate_on:
            # Gate by cosine, keep RRF order, cap at RAG_MAX_RESULTS. Hits absent from the
            # dense top-N (sparse-only keyword matches) have no cosine -> treated irrelevant.
            results = []
            for hit in resp.points:
                cos = cosine_by_id.get(hit.id)
                if cos is None or cos < RAG_MIN_COSINE:
                    continue
                formatted = _format_hit(hit)
                formatted["relevance"] = round(float(cos), 4)
                results.append(formatted)
                if len(results) >= RAG_MAX_RESULTS:
                    break
        else:
            # Pure RRF baseline: top hits by fused rank, no relevance gate.
            results = [_format_hit(hit) for hit in resp.points[:RAG_MAX_RESULTS]]
        out.append({
            "query": q,
            "results": results,
        })

    if timing_out is not None:
        timing_out["embed_ms"] = embed_timing.get("embed_ms", 0.0)
        timing_out["qdrant_ms"] = round(qdrant_ms, 2)
        timing_out["total_ms"] = round((time.perf_counter() - total_t0) * 1000, 2)
        timing_out["query_count"] = len(queries)

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
