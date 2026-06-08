# Saulie RAG — Product Search Pipeline

Hybrid dense + sparse retrieval over Amazon product catalogs using **BGE-M3** (TensorRT) and **Qdrant**. This folder contains scripts to prepare data, build indexes, validate quality, query, and compare **Indian CSV** vs **McAuley 2023** datasets with **RRF** vs **DBSF** fusion.

The Saulie agent (`agent_chat_api.py`) imports `search_hybrid` from `query2.py` via `sys.path.append("/root/rag")`. For local dev, symlink or copy this folder to `/root/rag`.

---

## Architecture

```
                    ┌─────────────────┐
  CSV / meta        │  prepare_*.py   │  cleaned CSV + embed_text
  ───────────────►  └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ index2.py       │  amazon_products      (Indian, ~114k)
                    │ index_mccauley  │  amazon_products_v2   (McAuley, ~443k)
                    └────────┬────────┘
                             │ batch=2, dense validation
                    ┌────────▼────────┐
                    │ BGE-M3 :8888    │
                    │ Qdrant :1234    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ query2.py       │  hybrid search (RRF / DBSF)
                    └─────────────────┘
```

Two **Qdrant collections** can coexist; switch with `QDRANT_COLLECTION`.

| Collection | Dataset | Points | Index script | CSV |
|------------|---------|--------|--------------|-----|
| `amazon_products` | Indian Amazon (cleaned) | ~114k | `index2.py` | `amazon_indian_clean.csv` |
| `amazon_products_v2` | McAuley 2023 (US) | ~443k | `index_mccauley.py` | `mccauley_products_500k.csv` |

---

## Root cause fix (why search was broken)

The original index used `SERVER_BATCH_SIZE=128` while the TensorRT BGE-M3 engine max batch is **2**. That corrupted ~99% of dense vectors in Qdrant. Fixes applied in `index2.py` and `index_mccauley.py`:

- `SERVER_BATCH_SIZE=2` (must match TRT engine)
- Embed **`embed_text`** (category + title + features), not name-only
- **Abort on invalid dense vectors** (norm &lt; 0.9), never silently skip batches
- Store `parent_asin` for URL reconstruction (`https://www.amazon.com/dp/{asin}`)

---

## Prerequisites

- Docker: `tensorrt_bge-m3` (embed server on `:8888`), `qdrant_index` (`:1234`)
- Python env with `qdrant-client`, `pandas`, `requests`, `tqdm` (e.g. `saulgman` conda env)
- GPU for BGE-M3 TensorRT inference during indexing

Start BGE inside the container if needed:

```bash
docker start tensorrt_bge-m3 qdrant_index
docker exec -d tensorrt_bge-m3 sh -c "cd /workspace && python -u serve.py"
```

---

## Pipeline: McAuley (recommended)

### 1. Download product metadata

```bash
bash download_mccauley_meta.sh              # all 15 categories (~16 GB)
bash download_mccauley_meta.sh Electronics   # single category smoke test
```

Files land in `mccauley_meta/meta_{Category}.jsonl.gz`. Uses `mcauleylab.ucsd.edu` mirror with HuggingFace fallback.

### 2. Prepare CSV

```bash
python prepare_mccauley.py
# → mccauley_products_500k.csv (~443k rows after quality filters + per-category caps)
```

Filters: `rating_number≥50`, `avg_rating≥3.5`, full titles, description/features, valid price, dedupe by `parent_asin`.

### 3. Index

```bash
python index_mccauley.py
# Collection: amazon_products_v2 (~1–2 hours at batch 2)
```

### 4. Validate (gate before production)

```bash
QDRANT_COLLECTION=amazon_products_v2 python validate_index.py
```

Must show **zero% zero-dense** in the 50k sample audit.

---

## Pipeline: Indian CSV (fallback)

### 1. Prepare

```bash
python prepare_indian.py
# Input:  Amazon-Products_fixed_deduped.csv (not in repo — too large)
# Output: amazon_indian_clean.csv (~114k rows)
```

Drops truncated titles (`...`), junk `stores` category, low ratings; dedupes by ASIN; builds `embed_text`.

### 2. Index

```bash
INDIAN_CSV=amazon_indian_clean.csv python index2.py
# Collection: amazon_products
```

---

## Querying

```bash
# Env vars (also used by agent)
export QDRANT_COLLECTION=amazon_products_v2   # or amazon_products
export FUSION_METHOD=rrf                      # or dbsf
export EMBED_URL=http://localhost:8888/embed
export QDRANT_URL=http://localhost:1234

python query2.py
python smoke_hybrid.py
```

`search_hybrid(query_text, main_category=None, top_k=5)` returns hybrid results with reconstructed product URLs.

---

## Benchmark: RRF vs DBSF

18-query battery across electronics, clothing, home, pets, etc. Category filters differ per dataset (Indian uses `men's shoes`; McAuley uses `AMAZON FASHION`, `All Electronics`, etc.).

```bash
# McAuley
python benchmark_fusion.py --collection amazon_products_v2 \
  --output fusion_benchmark_results_v2.json

# Indian (results already in repo)
# fusion_benchmark_results.json

# Centralized comparison report
python compare_fusion_report.py
# → fusion_comparison_report.md
```

### Results summary (June 2025 run)

| Dataset | Best fusion | Top-1 score (of 36) | Winner |
|---------|-------------|---------------------|--------|
| Indian | Tie RRF/DBSF | 20 | — |
| **McAuley** | **RRF** | **32** | **McAuley overall** |

**Recommended production defaults:**

```bash
QDRANT_COLLECTION=amazon_products_v2
FUSION_METHOD=rrf
```

See [`fusion_comparison_report.md`](fusion_comparison_report.md) for per-query breakdown.

---

## File reference

| File | Purpose |
|------|---------|
| `download_mccauley_meta.sh` | Resume-friendly download of McAuley `meta_*.jsonl.gz` |
| `prepare_mccauley.py` | Stream meta → filtered `mccauley_products_500k.csv` |
| `prepare_indian.py` | Clean Indian deduped CSV → `amazon_indian_clean.csv` |
| `index_mccauley.py` | Index McAuley CSV → `amazon_products_v2` |
| `index2.py` | Index Indian CSV → `amazon_products` |
| `validate_index.py` | Zero-dense audit + dense query battery |
| `query2.py` | Hybrid search (`search_hybrid`) used by Saulie agent |
| `benchmark_fusion.py` | RRF vs DBSF A/B on 18 queries |
| `compare_fusion_report.py` | Indian vs McAuley × fusion comparison report |
| `smoke_hybrid.py` | Quick manual smoke tests |
| `embed_models/bge-m3/build_engine.sh` | Rebuild TRT engine (batch max 2) |

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | `http://localhost:1234` | Qdrant HTTP API |
| `QDRANT_COLLECTION` | `amazon_products` | Collection name |
| `EMBED_URL` | `http://localhost:8888/embed` | BGE-M3 embed endpoint |
| `FUSION_METHOD` | `rrf` | `rrf` or `dbsf` |
| `MCCAULEY_CSV` | `mccauley_products_500k.csv` | McAuley prepared CSV path |
| `INDIAN_CSV` | `amazon_indian_clean.csv` | Indian prepared CSV path |
| `PREFETCH_LIMIT` | `50` | Hybrid prefetch limit per leg |

---

## What is NOT in git

Large artifacts stay local (see `.gitignore`):

- `mccauley_meta/` (~16 GB raw downloads)
- `*.csv` product files
- Qdrant storage / indexed vectors
- TensorRT engine binaries

Reproduce indexes on your machine using the scripts above.
