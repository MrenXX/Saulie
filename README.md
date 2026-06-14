# Saulie

Shopping-agent project: **DPO steering training**, **behavioral eval**, **production deployment**, and **hybrid RAG** over Amazon product catalogs. Model stack: **Qwen3-4B-Instruct FP8** + SFT/DPO LoRA adapters; retrieval: **BGE-M3** (TensorRT) + **Qdrant**.

This is the **`main`** branch — it contains the full codebase. Branch-specific landing pages (same repo, different focus):

| Branch | Focus | Start here |
|--------|-------|------------|
| **`main`** (this) | Everything | this file |
| [`deployment`](https://github.com/MrenXX/Saulie/tree/deployment) | Production serving (vLLM + agent + nginx + ngrok) | [`README.md` on `deployment`](https://github.com/MrenXX/Saulie/blob/deployment/README.md) |
| [`dpo_eval`](https://github.com/MrenXX/Saulie/tree/dpo_eval) | v1.5 merge gate, vLLM eval, judge workflow, matrix A/B | [`README.md` on `dpo_eval`](https://github.com/MrenXX/Saulie/blob/dpo_eval/README.md) |
| [`rag`](https://github.com/MrenXX/Saulie/tree/rag) | Product search pipeline, benchmarks, catalog prep | [`rag/README.md`](rag/README.md) |

---

## Reproducible setup

First-time machine setup:

```bash
cp .env.example .env          # edit secrets/paths
conda env create -f environment.yml
conda activate saulgman
bash docker/setup_containers.sh
bash dpo/eval/vllm_scripts/deploy_finalist_pick.sh   # after model weights are on disk
bash start_saulie.sh
```

Full guide: [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). Security: [`SECURITY.md`](SECURITY.md).

---

## Quick start (production stack)

```bash
bash start_saulie.sh          # RAG + vLLM + agent + nginx + tmux dashboard
bash stop_saulie.sh           # stop services
python remote_chat.py         # remote client (set PUBLIC_URL / ngrok)
```

Defaults: model **`dpo-v15-trial-4`**, prompt **`compressed`**, RAG **`amazon_products_v2`** + **`rrf`**.  
Health: `curl http://127.0.0.1:9000/health` → model + active prompt variant.

Full ops guide: see the [`deployment` branch README](https://github.com/MrenXX/Saulie/blob/deployment/README.md) (architecture, env vars, troubleshooting).

---

## Project map

```
saulie/
├── requirements.txt               # Pinned Python deps (saulgman env)
├── environment.yml                # Conda recipe: one-command env setup
├── .env.example                   # Config template (copy to .env)
├── REPRODUCIBILITY.md             # Full setup guide
├── docker/
│   ├── versions.env               # Pinned Docker image tags
│   └── setup_containers.sh        # Create Qdrant + BGE containers
├── agent_chat_api.py              # FastAPI agent (tool loop, SSE, prompt variants)
├── remote_chat.py                 # Remote CLI client
├── start_saulie.sh / stop_saulie.sh
├── nginx/                         # Reverse proxy (Bearer auth, SSE)
├── SAULIE_PERSONA_AND_STEERING_FIX_PLAN.md
│
├── rag/                           # Hybrid product search (BGE-M3 + Qdrant, code + data)
│   ├── README.md                  # ← full RAG docs
│   ├── query2.py                  # search_hybrid (used by agent)
│   ├── index_mccauley.py          # McAuley US catalog → amazon_products_v2
│   └── fusion_comparison_report.md
│
├── sft/                           # Qwen SFT training (pre-DPO)
│   ├── README.md                  # ← SFT layout + gitignore notes
│   ├── train_sft.py               # Optuna HPO + MLflow
│   └── sft_eval/                  # SFT-specific eval + judge prompt
│
└── dpo/
    ├── train/                     # Optuna studies, merge, SFT/DPO training
    │   ├── merge_sft_dpo_lora.py
    │   └── results/               # study reports, trial artifacts
    └── eval/                      # Behavioral eval + deploy scripts
        ├── README.md              # ← judge packet + eval index
        ├── run_v15_final_eval.py
        ├── run_model_prompt_matrix.sh
        └── vllm_scripts/          # deploy_finalist_pick.sh, deploy_dpo_w05.sh, …
```

---

## 1. DPO training (`dpo/train/`)

Steering SFT + DPO over multi-turn shopping conversations (Qwen + LoRA, Optuna hyperparameter search).

**Review a finished study:**

```bash
python -m dpo.train.study_report \
  --summary dpo/results/optuna-run-20260523-041252/trial_summary.json
# HTML: dpo/results/optuna-run-20260523-041252/study_report.html
```

**Requirements:** GPU, SFT adapter, `requirements.txt` (or `conda env create -f environment.yml`).

Key docs: `dpo/PLAN_FINAL.md`, `dpo/dataset/DATA_CONTEXT.md`, `dpo/train/HANDOFF.md`.

---

## 2. DPO eval (`dpo/eval/`)

Post-training **behavioral gate** before deploy: cat-merge validation, vLLM FP8 generation, blind LLM-judge workflow, and model×prompt matrix A/B.

| Step | Script / doc |
|------|----------------|
| Merge gate | `dpo/train/scripts/merge_v15_eval_slate.sh` |
| vLLM deploy | `dpo/eval/vllm_scripts/deploy_qwenie_eval.sh` |
| Round 1 gen | `python dpo/eval/run_v15_final_eval.py --round 1` |
| Matrix A/B | `bash dpo/eval/run_model_prompt_matrix.sh` |
| Results | [`MODEL_PROMPT_MATRIX_COMPARISON.md`](MODEL_PROMPT_MATRIX_COMPARISON.md) |

Full protocol: [`dpo/eval/DPO_FINAL_EVAL_EXECUTION_PLAN.md`](dpo/eval/DPO_FINAL_EVAL_EXECUTION_PLAN.md)  
Branch landing page: [`dpo_eval` README](https://github.com/MrenXX/Saulie/blob/dpo_eval/README.md).

---

## 3. Production deployment

End-to-end serving path:

```
remote_chat.py → ngrok → nginx :8080 → agent_chat_api.py :9000
                                              ├── vLLM :8000 (dpo-v15-trial-4 LoRA)
                                              └── RAG query2.py (BGE :8888 + Qdrant :1234)
```

| Component | Notes |
|-----------|-------|
| Deploy vLLM | `bash dpo/eval/vllm_scripts/deploy_finalist_pick.sh` |
| System prompts | `SAULIE_PROMPT=legacy\|steering\|compressed` (default `compressed`) |
| Persona fixes | [`SAULIE_PERSONA_AND_STEERING_FIX_PLAN.md`](SAULIE_PERSONA_AND_STEERING_FIX_PLAN.md) |

Branch landing page: [`deployment` README](https://github.com/MrenXX/Saulie/blob/deployment/README.md).

---

## 4. RAG product search (`rag/`)

Hybrid **dense + sparse** retrieval with **RRF** or **DBSF** fusion.

**Recommended production defaults** (from 18-query benchmark):

```bash
QDRANT_COLLECTION=amazon_products_v2   # McAuley US (~443k products)
FUSION_METHOD=rrf
```

**Quick index + smoke test:**

```bash
bash rag/download_mccauley_meta.sh
python rag/prepare_mccauley.py
python rag/index_mccauley.py
QDRANT_COLLECTION=amazon_products_v2 python rag/smoke_hybrid.py
```

Indian vs McAuley comparison: [`rag/fusion_comparison_report.md`](rag/fusion_comparison_report.md)  
Full pipeline docs: [`rag/README.md`](rag/README.md).

---

## Configuration cheat sheet

| Variable | Default | Area |
|----------|---------|------|
| `SAULIE_MODEL` | `dpo-v15-trial-4` | Deployment |
| `SAULIE_PROMPT` | `compressed` | Deployment |
| `QDRANT_COLLECTION` | `amazon_products_v2` | RAG |
| `FUSION_METHOD` | `rrf` | RAG |
| `VLLM_API_KEY` | `dipshit` (in `.env`) | vLLM (agent → vLLM) |
| `NGINX_API_KEY` | `secret` (in `.env`) | Public API via nginx Bearer |

Security details: [`SECURITY.md`](SECURITY.md). Agent listens on `127.0.0.1` only; `/health` is not exposed via nginx.

---

## Branch workflow (avoid README merge fights)

Each feature branch keeps its **own root README** focused on that area. **`main`** uses this index README. When merging branches, resolve `README.md` conflicts by keeping **`main`'s index** (this file) on `main`, and the branch-specific README on that branch.

Do **not** merge `deployment` ↔ `dpo_eval` directly expecting clean README merges — they intentionally differ.
