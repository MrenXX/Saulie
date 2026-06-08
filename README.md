# Saulie — RAG Product Search

Hybrid product retrieval for the Saulie shopping agent: **BGE-M3** (dense + sparse) + **Qdrant**, with two catalog options and RRF/DBSF fusion tuning.

**Full documentation:** [`rag/README.md`](rag/README.md)

---

## Quick start

```bash
# Prerequisites: Docker (tensorrt_bge-m3 on :8888, qdrant_index on :1234)

# McAuley US catalog (recommended)
bash rag/download_mccauley_meta.sh
python rag/prepare_mccauley.py
python rag/index_mccauley.py
QDRANT_COLLECTION=amazon_products_v2 python rag/validate_index.py

# Query / smoke test
QDRANT_COLLECTION=amazon_products_v2 FUSION_METHOD=rrf python rag/smoke_hybrid.py
```

**Production defaults** (from benchmark): `QDRANT_COLLECTION=amazon_products_v2`, `FUSION_METHOD=rrf`

---

## Two Qdrant collections

| Collection | Dataset | ~Points | Index script |
|------------|---------|---------|--------------|
| `amazon_products_v2` | McAuley Amazon 2023 (US) | 443k | `rag/index_mccauley.py` |
| `amazon_products` | Indian Amazon CSV (cleaned) | 114k | `rag/index2.py` |

Switch at runtime: `QDRANT_COLLECTION=...` (used by `rag/query2.py` and the agent).

---

## Indian vs McAuley + RRF vs DBSF

We ran an 18-query benchmark across both datasets and both fusion methods.

| Dataset | Best fusion | Top-1 score (of 36) |
|---------|-------------|---------------------|
| Indian | Tie | 20 |
| **McAuley** | **RRF** | **32** |

**Verdict:** McAuley catalog wins overall (10 vs 1 query wins). See [`rag/fusion_comparison_report.md`](rag/fusion_comparison_report.md).

```bash
python rag/compare_fusion_report.py   # regenerate report from JSON results
```

---

## What was fixed

The original index corrupted ~99% of dense vectors by embedding with batch 128 against a TensorRT engine max batch of 2. Current indexers use batch 2, validate dense norms, and embed rich `embed_text` instead of product name only.

---

## Repo layout

```
rag/                          # ← all RAG scripts, benchmarks, comparison report
  README.md                   # detailed pipeline docs
  prepare_mccauley.py
  index_mccauley.py
  query2.py
  ...
agent_chat_api.py             # agent (imports search_hybrid from /root/rag)
dpo/                          # DPO steering training (separate concern)
```

For local dev, symlink or copy `rag/` to `/root/rag` (the path `agent_chat_api.py` uses).

---

## DPO training (other branch work)

DPO steering training (Qwen + LoRA) lives under `dpo/`. Not the focus of this branch — see `dpo/` and `deployment` branch for study reports and training.
