# Saulie — `fix/tool-pressure-reset-and-rag-gate`

Branch focus: make tool-call **pressure** smarter than a binary `tool_choice` (tool-aware reset + a real nudge), add a **dense-cosine RAG relevance gate** on top of RRF, and harden the eval. Model stack: **Qwen3-4B-Instruct FP8** + SFT/DPO LoRA; retrieval: **BGE-M3** (TensorRT) + **Qdrant**.

> **Branch landing page.** Repo convention: each branch keeps its own root README focused on its area (see *Branch workflow* at the bottom). The full codebase index lives on the [`main` README](https://github.com/MrenXX/Saulie/blob/main/README.md).

| Branch | Focus | Start here |
|--------|-------|------------|
| [`main`](https://github.com/MrenXX/Saulie/blob/main/README.md) | Full codebase index | `main` README |
| **`fix/tool-pressure-reset-and-rag-gate`** (this) | Tool-pressure reset + dense-cosine RAG gate + eval hardening | [`RETRIEVAL_AND_TOOL_PRESSURE_HANDOFF.md`](RETRIEVAL_AND_TOOL_PRESSURE_HANDOFF.md) |
| [`deployment`](https://github.com/MrenXX/Saulie/tree/deployment) | Production serving (vLLM + agent + nginx + ngrok) | [`README.md` on `deployment`](https://github.com/MrenXX/Saulie/blob/deployment/README.md) |
| [`dpo_eval`](https://github.com/MrenXX/Saulie/tree/dpo_eval) | v1.5 merge gate, vLLM eval, judge workflow, matrix A/B | [`README.md` on `dpo_eval`](https://github.com/MrenXX/Saulie/blob/dpo_eval/README.md) |
| [`rag`](https://github.com/MrenXX/Saulie/tree/rag) | Product search pipeline, benchmarks, catalog prep | [`rag/README.md`](rag/README.md) |

---

## What this branch changes

**Start here:** [`RETRIEVAL_AND_TOOL_PRESSURE_HANDOFF.md`](RETRIEVAL_AND_TOOL_PRESSURE_HANDOFF.md) — full rationale, exact diffs (Appendix A), and the staged runbook.

| # | Workstream | Why | Key files |
|---|-----------|-----|-----------|
| 1 | **Tool-aware pressure reset** | The old cross-turn counter couldn't see past searches (the stateless API strips tool messages), so it force-called the tool on every late turn — "thanks man" on turn 6 → forced search. A fingerprint→last-search registry resets pressure after a real search. | [`agent_chat_api.py`](agent_chat_api.py) |
| 2 | **Real nudge, not silent force** | `SAULIE_TOOL_BIAS_PER_TURN` default 8 → 2: +24/+32/+40 saturated the softmax (always forced); +6/+8/+10 is a genuine probabilistic nudge. Force at turn 6 stays the hard guarantee. | [`agent_chat_api.py`](agent_chat_api.py), [`.env.example`](.env.example) |
| 3 | **Dense-cosine RAG relevance gate** | RRF (Qdrant k=2) is rank-based: the top hit is ~always ≥0.5 even for junk, so it can't gate. RRF still *ranks*; dense cosine *gates*. Toggleable via `SAULIE_RAG_MIN_COSINE` (≤0 = pure-RRF baseline). | [`rag/query2.py`](rag/query2.py) |
| 4 | **Eval hardening** | Dependency preflight (no more silently-empty smoke runs), monotonic timing, per-turn `results_count` + an all-empty hard-fail. | [`dpo/eval/model_prompt_matrix_eval.py`](dpo/eval/model_prompt_matrix_eval.py) |

---

## Apply it (no network needed)

This branch was authored on a machine whose proxy/DLP blocks `git push`. Move it to the GPU/dev box with the **git bundle**, or apply the patch / Appendix A.

```bash
# on the dev machine, inside the repo:
git fetch origin
git bundle verify saulie_tool_pressure.bundle
git fetch saulie_tool_pressure.bundle fix/tool-pressure-reset-and-rag-gate:fix/tool-pressure-reset-and-rag-gate
git checkout fix/tool-pressure-reset-and-rag-gate
```

No bundle? Apply [`saulie_tool_pressure_rag.patch`](saulie_tool_pressure_rag.patch) with `git apply`, or follow Appendix A in the handoff by hand.

---

## New / changed config (`.env`)

| Variable | Default | Meaning |
|----------|---------|---------|
| `SAULIE_TOOL_BIAS_PER_TURN` | `2` | Logit add per turn (was 8). Drop to 1 if turn 3 still always searches. |
| `SAULIE_TOOL_PRESSURE_REGISTRY_MAX` | `512` | LRU size for the tool-aware reset registry. |
| `SAULIE_RAG_MIN_COSINE` | `0.5` | Dense-cosine gate. `>0` on; `≤0` off (pure-RRF baseline). |
| `SAULIE_RAG_MAX_RESULTS` | `5` | Max products returned. |

---

## New scripts

| Script | Purpose |
|--------|---------|
| [`rag/calibrate_cosine_threshold.py`](rag/calibrate_cosine_threshold.py) | Calibrate `SAULIE_RAG_MIN_COSINE` on the 18-query labeled benchmark. |
| [`rag/retrieval_ab_eval.py`](rag/retrieval_ab_eval.py) | A/B: pure RRF vs RRF+gate (decide whether the gate actually helps). |
| [`dpo/eval/nudge_rate_probe.py`](dpo/eval/nudge_rate_probe.py) | N-sample tool-emit rate — proves nudge (0<rate<1) vs force (100%). |

---

## Validate

The staged runbook (deps → calibrate → A/B → deploy → pressure-only → nudge probe → +gate) is in [`RETRIEVAL_AND_TOOL_PRESSURE_HANDOFF.md`](RETRIEVAL_AND_TOOL_PRESSURE_HANDOFF.md) §5. Run the stages separately for clean attribution.

---

## Full codebase reference

The sections below are the general project index (same as `main`), kept for navigation while working on this branch.

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
| **Token streaming** | `stream: true` → real vLLM tokens over SSE; tool JSON buffered server-side ([`AGENT_STREAMING.md`](AGENT_STREAMING.md)) |
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
