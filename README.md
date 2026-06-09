# Saulie — DPO v1.5 Final Eval (`dpo_eval` branch)

End-to-end **behavioral evaluation** for the v1.5 hybrid Optuna finalists: cat-merge validation, **vLLM FP8** generation, two-round blind LLM-judge workflow, and completed artifacts.

**File index & commands:** [`dpo/eval/README.md`](dpo/eval/README.md)  
**Full protocol:** [`dpo/eval/DPO_FINAL_EVAL_EXECUTION_PLAN.md`](dpo/eval/DPO_FINAL_EVAL_EXECUTION_PLAN.md)

Other branches: training on `main`, production serving on `deployment`, RAG on `rag`.

---

## Why this branch exists

Offline Optuna metrics and one-off REPL impressions are not enough to ship a DPO adapter. This branch answers four gates before deploy:

1. Does the candidate beat **SFT trial-17** on steering behavior?
2. Does it preserve **ordinary conversation** (no product forcing, loops, drift)?
3. Does the **cat-merged** adapter (`sft_dpo_cat`) match the stacked HF path locally?
4. Does behavior hold on the **production inference path** (vLLM FP8 + cat LoRA)?

---

## Pipeline overview

```
Optuna finalists (trials 4,8,16,19,20,27)
        │
        ▼
┌───────────────────┐
│ 1. Merge gate     │  merge_sft_dpo_lora.py + merge_v15_eval_slate.sh
│    ΔW + A/B smoke │  → trial-N/sft_dpo_cat/
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 2. vLLM deploy    │  FP8 Qwen3 + runtime LoRA (MAX_LORA_RANK 64 or 32)
│    eval_runtime   │  deploy_qwenie_eval.sh
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 3. Generation     │  run_v15_final_eval.py → eval_generate_vllm.py
│    Round 1        │  7 models × 34 skeletons (blind candidate_A…F)
│    checkpointed   │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 4. LLM judges     │  External — llm_judge_prompt_dpo.md + generations JSON
│    (you run)      │  Opus + GPT-5.5 on Round 1 → advance finalists
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 5. Round 2        │  Holdout 26 steering + 8 ordinary (same O set)
│    5 models       │  SFT + trials 4, 19, 20, 27
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 6. Deploy pick    │  Manual REPL + judges → trial 4 winner → deployment branch
└───────────────────┘
```

**Official inference path for judge JSON:** vLLM FP8 + cat-merged LoRA — **not** HF BnB8 (`eval_generate_hf.py` is debug-only).

---

## Key scripts & docs

| Path | Role |
|------|------|
| [`dpo/train/scripts/merge_v15_eval_slate.sh`](dpo/train/scripts/merge_v15_eval_slate.sh) | Batch cat-merge for finalist trials |
| [`dpo/train/merge_sft_dpo_lora.py`](dpo/train/merge_sft_dpo_lora.py) | Cat merge, ΔW validation, optional FP8 deploy audit |
| [`dpo/eval/run_v15_final_eval.py`](dpo/eval/run_v15_final_eval.py) | Orchestrator: deploy + Round 1/2 generation |
| [`dpo/eval/v15_eval_config.py`](dpo/eval/v15_eval_config.py) | Skeleton splits, sampling constants, manifests |
| [`dpo/eval/vllm_scripts/deploy_qwenie_eval.sh`](dpo/eval/vllm_scripts/deploy_qwenie_eval.sh) | FP8 vLLM container (`DEPLOY_MODE=eval_runtime`) |
| [`dpo/eval/vllm_scripts/eval_generate_vllm.py`](dpo/eval/vllm_scripts/eval_generate_vllm.py) | Multi-turn gen + **checkpoint resume** after each model |
| [`dpo/eval/vllm_scripts/vllm_lora_runtime.py`](dpo/eval/vllm_scripts/vllm_lora_runtime.py) | Runtime `load_lora_adapter` / `unload_lora_adapter` |
| [`dpo/eval/MERGE_SCRIPT_VALIDATION_FIX_PLAN.md`](dpo/eval/MERGE_SCRIPT_VALIDATION_FIX_PLAN.md) | Merge gate criteria (ΔW, A/B smoke) |
| [`dpo/eval/JUDGE_PACKET_README.md`](dpo/eval/JUDGE_PACKET_README.md) | Blind judge handoff — what to send / not send |
| [`dpo/eval/llm_judge_prompt_dpo.md`](dpo/eval/llm_judge_prompt_dpo.md) | Judge rubric |
| [`dpo/eval/eval_skeletons.json`](dpo/eval/eval_skeletons.json) | 60 skeletons (52 steering + 8 ordinary) |
| [`dpo/eval/run_model_prompt_matrix.sh`](dpo/eval/run_model_prompt_matrix.sh) | 3×3 model×prompt matrix against live agent API |
| [`MODEL_PROMPT_MATRIX_COMPARISON.md`](MODEL_PROMPT_MATRIX_COMPARISON.md) | Matrix results: tool calling vs English quality |

---

## Model × prompt matrix (post-deploy A/B)

After picking trial-4 for production, we ran a **3×3 grid** (SFT, DPO w=0.5, DPO w=1.0 × legacy/steering/compressed prompts) against the live `agent_chat_api.py` harness:

```bash
bash dpo/eval/run_model_prompt_matrix.sh
python dpo/eval/generate_matrix_report.py   # regenerate MODEL_PROMPT_MATRIX_COMPARISON.md
```

Deploy helpers for matrix cells:

| Script | Model |
|--------|-------|
| `deploy_sft_trial17_prod.sh` | SFT trial-17 only |
| `deploy_dpo_w05.sh` | DPO w=0.5 cat-merge |
| `deploy_finalist_pick.sh` | DPO w=1.0 trial-4 (on `deployment` branch) |

Raw JSON: `dpo/eval/matrix_runs/*.json`

---

## Finalist slate (Round 1)

SFT baseline **trial-17** plus six cat-merged DPO trials (blind IDs):

| Judge ID | Trial | Cat rank |
|----------|------:|---------:|
| `steering-sft-v1.1_trial-17` | 17 | 16 (SFT only) |
| `candidate_A` | 19 | 32 |
| `candidate_B` | 16 | 32 |
| `candidate_C` | 8 | 40 |
| `candidate_D` | 27 | 40 |
| `candidate_E` | 20 | 32 |
| `candidate_F` | 4 | 32 |

Excluded from slate: trial 25 (repetition), trial 9 (grammar / calibration only).

**Round 1 artifacts (on this branch):**

- Blind: [`dpo/eval/generations_round1.json`](dpo/eval/generations_round1.json)
- Unblind (local): [`dpo/eval/generations_round1_unblind.json`](dpo/eval/generations_round1_unblind.json)
- Merge report: [`dpo/eval/merge_v15_slate_validation.md`](dpo/eval/merge_v15_slate_validation.md)

---

## Skeletons & rounds

| Round | Steering | Ordinary | Total per model |
|-------|----------|----------|-----------------|
| **1** (screen) | 26 ids | 8 ids (`eval_O*`) | 34 |
| **2** (holdout) | 26 different ids | same 8 ordinary | 34 |

Round 1 and Round 2 use **disjoint steering** skeletons; ordinary retention skeletons repeat so judges can check chat quality across rounds.

---

## Locked sampling (all official generations)

```
temperature=0.7   top_p=0.8   top_k=20   repetition_penalty=1.05   max_tokens=256
```

`top_k` and `repetition_penalty` must be sent via OpenAI client `extra_body` (vLLM-specific).

---

## Commands

### 1. Merge gate (once per finalist)

```bash
bash dpo/train/scripts/merge_v15_eval_slate.sh
```

Output: `dpo/train/models/steering-dpo-v1.5/optuna-run-20260602-052732/trial-N/sft_dpo_cat/`

### 2. Deploy vLLM

```bash
# Full slate (includes r=40 trials 8/27)
MAX_LORA_RANK=64 bash dpo/eval/vllm_scripts/deploy_qwenie_eval.sh

# Winners only (trials 4, 20 — r=32)
MAX_LORA_RANK=32 bash dpo/eval/vllm_scripts/deploy_qwenie_eval.sh
```

vLLM rank buckets are **8, 16, 32, 64** — not arbitrary values like 40.

### 3. Generate (resumes on crash; checkpoint after each model)

```bash
# Round 1 — 7 models
python dpo/eval/run_v15_final_eval.py --round 1 --anonymize --skip-deploy

# Round 2 — 5 finalists after judge decision
python dpo/eval/run_v15_final_eval.py --round 2 --anonymize --skip-deploy \
  --models sft,trial-4,trial-19,trial-20,trial-27
```

`--fresh` discards checkpoint and starts over.

### 4. Judge (external — not automated in repo)

Send to judge UI:

- `dpo/eval/llm_judge_prompt_dpo.md`
- `dpo/eval/DATA_CONTEXT.md`
- `generations_roundN.json` (blind only)

Do **not** send unblind JSON, trial metrics, or merge reports during scoring.

### 5. Deploy pick (outcome)

After Round 1 + Round 2 judges and manual REPL: **trial 4** selected for production (`deployment` branch).

- Cat adapter: `…/trial-4/sft_dpo_cat` (r=32)
- Suggested vLLM model id: `dpo-v15-trial-4`
- Production: FP8 base + single cat LoRA, `MAX_LORA_RANK=32`

---

## Round 2 blind mapping (re-anonymized)

| Round 2 ID | Trial | Phase 1 ID |
|------------|------:|------------|
| `steering-sft-v1.1_trial-17` | 17 | baseline |
| `candidate_A` | 4 | `candidate_F` |
| `candidate_B` | 19 | `candidate_A` |
| `candidate_C` | 20 | `candidate_E` |
| `candidate_D` | 27 | `candidate_D` |

---

## What not to use for official eval

- [`dpo/eval/eval_generate_hf.py`](dpo/eval/eval_generate_hf.py) — HF BnB8 debug path
- `dpo_phase1_*`, `smoke_trial*`, rescue manifests — pre-v1.5 diagnostics
- Legacy `deploy_qwenie.sh` — SFT-only trial-17, `max-lora-rank 16`; cannot serve cat adapters

---

## Requirements

GPU host with Docker, FP8 base weights (`Qwen3-4B-Instruct-2507-FP8`), SFT trial-17 adapter, v1.5 run adapters on disk. Python: `dpo/requirements-dpo.txt` + `openai`, `colorama`, `requests`.
