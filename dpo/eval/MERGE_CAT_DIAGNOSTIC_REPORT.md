# SFT+DPO merge & Phase 1 deploy — full diagnostic report

**Last updated:** 2026-05-28  
**Workspace:** `/root/saulie`  
**Primary trial:** Optuna `steering-dpo-v1.1` → `optuna-run-20260526-042551` → **trial-29** (selected for Phase 1 cat deploy)

**Phase 1 goal:** Serve the DPO **policy** `set_adapter(["default", "dpo"])` as **one** LoRA on FP8 vLLM (`Qwen3-4B-Instruct-2507-FP8`), without merging into base weights and without two active LoRAs per request.

**Current status:**

- Cat merge **ΔW checks pass**; **forward logit check fails** on BnB; **FP8 vLLM cat generation fails** (EN/ZH gibberish).
- **HF policy-stack smoke** (exact training forward, no cat) **also fails** on the same skeletons → **not vLLM-only**.
- Trial-29 **val preference accuracy = 100%** on 52 rows; that metric is **not** open-ended multi-turn generation.

---

## Table of contents

1. [Executive summary](#executive-summary)
2. [Architecture & checkpoints](#architecture--checkpoints)
3. [What DPO training actually optimizes](#what-dpo-training-actually-optimizes)
4. [Chronology of everything tried](#chronology-of-everything-tried)
5. [Diagnostic procedures (code & commands)](#diagnostic-procedures-code--commands)
6. [Behavioral smoke results (side-by-side)](#behavioral-smoke-results-side-by-side)
7. [Trial-29 training metrics](#trial-29-training-metrics)
8. [Environment & deploy](#environment--deploy)
9. [Hypotheses](#hypotheses)
10. [Not tried yet](#not-tried-yet)
11. [Recommended next steps](#recommended-next-steps)
12. [Complete file reading guide](#complete-file-reading-guide)

---

## Executive summary

| # | Experiment | Backend | Adapter(s) | Skeletons | Result |
|---|------------|---------|------------|-----------|--------|
| A | **Policy stack (training forward)** | HF + BnB 8-bit | `["default","dpo"]` | eval_A4_001, eval_B8_001 | **FAIL** — EN/ZH, Chinese later turns |
| B | SFT trial-17 only | FP8 vLLM | `default` only | same | **PASS** — coherent English |
| C | DPO trial-29 only (no SFT in request) | FP8 vLLM | `dpo` only | same | **PASS** — coherent English (not true policy) |
| D | PEFT cat `sft_dpo_cat` (baked α=48) | FP8 vLLM | single r=48 | same | **FAIL** — EN/ZH gibberish |
| E | Manual unbaked cat (α=96) | FP8 vLLM | single r=48 | same | **FAIL** — worse Chinese |
| F | First cat on BF16 base (early) | FP8 vLLM | cat | same | **FAIL** — superseded |
| G | Mislabeled “stack” smoke | HF transformers | **cat** (wrong) | same | **FAIL** — do not use |

**Offline checks:**

| Check | Tool | Result |
|-------|------|--------|
| ΔW stack vs cat (252 layers) | `verify_weight_matrices()` | **PASS** ~1.6e-9 max diff |
| ΔW stack vs unbaked cat | safetensors script | **PASS** ~2.6e-7 |
| Forward last-token logits stack vs cat | `compare_logps_chat()` | **FAIL** max diff **2.5** (tol 0.001) |
| Policy adapter load | `smoke_policy_stack_hf.py` diagnostics | **PASS** `active_adapters: ["default","dpo"]` |

---

## Architecture & checkpoints

| Role | Path | Rank / α |
|------|------|----------|
| Train base (BnB) | `/root/saulie/Qwen3-4B-Instruct-2507` | 8-bit via BitsAndBytes |
| Deploy base (FP8) | `/root/saulie/Qwen3-4B-Instruct-2507-FP8` | vLLM |
| SFT adapter | `train/models/steering-sft-v1.1/trial-17/best_adapter` | r=16, α=32, name `default` |
| DPO adapter | `dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-29/best_adapter/dpo` | r=32, α=64, name `dpo` |
| Cat adapter (PEFT) | `.../trial-29/sft_dpo_cat` | r=48, α=48 (PEFT sets α=r after cat) |
| Cat adapter (unbaked) | `.../trial-29/sft_dpo_cat_unbaked` | r=48, α=96 |

**Training policy activation** (`dpo/train/dpo_trainer_compat.py`):

```python
model.base_model.set_adapter(["default", "dpo"])
```

**vLLM constraint** (`dpo/DPO_HANDOFF.md`): one LoRA per request; cannot stack `default` + `dpo` in a single vLLM forward.

---

## What DPO training actually optimizes

Trial-29 **did not** optimize “generate good English on eval skeletons.” It optimized:

- For each JSONL row: fixed multi-turn `prompt` + one scored assistant `chosen` vs `rejected`.
- Loss / rewards use **teacher-forced** token logprobs on **assistant-only** spans (`completion_mask` in `AssistantOnlyDPOCollator`).
- `eval_rewards/accuracies` = fraction of val rows where `reward(chosen) > reward(rejected)` with `reward = β × (policy_logp − ref_logp)`.

So **100% val accuracy** means perfect **pairwise ranking** on 52 held-out **English** completions — not that free sampling at temp 0.7 stays English.

**Smoke eval differs:**

| Training row | Skeleton smoke |
|--------------|----------------|
| Gold `prompt` prefix | Model **generates** each assistant turn |
| Score one fixed `chosen` completion | Multi-turn **open loop** (model errors feed forward) |
| No sampling in metrics | `temperature=0.7`, `top_p=0.8` |

See `dpo/dataset/DPO_522_prompt_a_and_prompt_b_V2.jsonl` for row shape; `dpo/train/dpo_diagnostics.py` for reward computation.

---

## Chronology of everything tried

### Phase 0 — Planning & tooling (pre-smoke)

| Item | What | Code / artifact |
|------|------|-----------------|
| Merge script | Cat-stack SFT+DPO for vLLM | `dpo/train/merge_sft_dpo_lora.py` |
| Shared BnB loader | Same base as `train_dpo.py` | `dpo/train/model_load.py` → `load_bnb_8bit_base()` |
| Deploy + gen | FP8 vLLM multi-adapter | `train/sft_eval/deploy_qwenie_eval.sh`, `train/sft_eval/eval_generate_vllm.py` |
| Eval plan | Phase 1 procedure | `dpo/eval/DPO_FINAL_EVAL_PLAN.md` |

---

### Attempt 1 — First cat merge on BF16 base (early; superseded)

**Hypothesis:** Combine adapters for vLLM.

**Command (representative):**

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate saulgman
python dpo/train/merge_sft_dpo_lora.py \
  --dpo-adapter dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-29/best_adapter \
  --output dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-29/sft_dpo_cat \
  --check-logps
```

(Early runs may have used unquantized base before `model_load.py` alignment; later regen used BnB.)

**Checks:** Forward logit mismatch ~0.4 reported in conversation; tokenizer issues on early artifact.

**Deploy:** FP8 vLLM with cat manifest.

**Result:** **FAIL** — garbage generations. Superseded by BnB-aligned merge + weight gate.

**Artifacts:** `dpo/eval/dpo_phase1_smoke.json`, `dpo/eval/dpo_phase1_smoke_fp8.json` (if present).

---

### Attempt 2 — Regenerate cat on BnB with weight-matrix gate

**Hypothesis:** Fix merge validation to use training base; only save if ΔW matches stack.

**Command:**

```bash
python dpo/train/merge_sft_dpo_lora.py \
  --dpo-adapter dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-29/best_adapter \
  --output dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-29/sft_dpo_cat \
  --check-logps
```

**Results:**

| Check | Value | Pass? |
|-------|-------|-------|
| `weight_matrix_check.max_abs_delta_diff` | 1.63e-09 | Yes (252 layers) |
| `forward_logit_check.max_abs_logit_diff` | 2.5 (prompt 2) | No (tol 0.001) |
| Saved anyway? | Yes, if weights pass | `merge_meta.json` |

**Deploy:**

```bash
CANDIDATE_MANIFEST=/root/saulie/dpo/eval/dpo_phase1_stack_only_manifest.jsonl \
MAX_LORA_RANK=64 MAX_LORAS=1 \
bash train/sft_eval/deploy_qwenie_eval.sh
```

**Gen:**

```bash
python train/sft_eval/eval_generate_vllm.py \
  --candidate-manifest dpo/eval/dpo_phase1_stack_only_manifest.jsonl \
  --skeletons dpo/eval/eval_skeletons.json \
  --skeleton-ids eval_A4_001,eval_B8_001 \
  --output-dir dpo/eval --output dpo_phase1_smoke.json
```

**Result:** **FAIL** — mixed EN/ZH (e.g. turn 1: `...not stiff or numb.什么 sleeve...`).

**Meta:** `trial-29/sft_dpo_cat/merge_meta.json`

---

### Attempt 3 — SFT-only FP8 smoke (control)

**Manifest:** `dpo/eval/dpo_phase1_sft17_only_manifest.jsonl`

**Result:** **PASS** — coherent English steering tone.

**Artifact:** `dpo/eval/dpo_phase1_smoke_sft17.json`

---

### Attempt 4 — FP8 deploy tuning

**Changes:**

- Image: `vllm/vllm-openai:latest` (avoid v0.8.5 `fp8e4nv` Triton error on RTX 3090)
- `MAX_LORA_RANK=64` (rank 48 cat; vLLM buckets)
- `gpu-memory-utilization 0.5` (~13 GB VRAM on 3090)
- `NCCL_P2P_DISABLE=1`, CUDA compat mount

**Result:** Server stable; cat generation still **FAIL**.

---

### Attempt 5 — Unbaked cat (`lora_alpha=96`) — user “fix try 1”

**Hypothesis:** PEFT cat bakes scaling into `lora_A` and sets `lora_alpha=48`; vLLM may need unbaked A/B and α=96 (effective scale 2.0).

**Command:**

```bash
python dpo/train/export_unbaked_cat.py
```

**Output dir:** `trial-29/sft_dpo_cat_unbaked/` (252 layers, `merged_alpha=96`)

**Offline ΔW:** stack vs unbaked max **~2.6e-7**; baked vs unbaked same order.

**Deploy:**

```bash
CANDIDATE_MANIFEST=/root/saulie/dpo/eval/dpo_phase1_unbaked_manifest.jsonl \
MAX_LORA_RANK=64 MAX_LORAS=1 \
bash train/sft_eval/deploy_qwenie_eval.sh
```

**Gen:** → `dpo/eval/dpo_phase1_smoke_unbaked.json`

**Result:** **FAIL** — predominantly Chinese finals (e.g. eval_A4_001 final about 干冷气 / 贴皮).

---

### Attempt 6 — Dual LoRA mount (SFT + DPO separate) — user “fix try 2”

**Hypothesis:** DPO checkpoint corrupt vs cat merge; test each adapter on FP8.

**Manifest:** `dpo/eval/dpo_phase1_dual_lora_manifest.jsonl` (SFT-17 + DPO-only paths)

**Deploy:** `MAX_LORAS=2`, `MAX_LORA_RANK=64`

**Gen:** → `dpo/eval/dpo_phase1_smoke_dual.json`

**Results:**

| Model per request | Result |
|-------------------|--------|
| `steering-sft-v1.1_trial-17` | **PASS** — English (eval_B8 has repetition, not Chinese gibberish) |
| `steering-dpo-v1.1_trial-29_dpo_only` | **PASS** — English markdown style (**not** policy; SFT not active) |

**Conclusion:** Weights load on FP8; **r=48 cat** and **true policy** still problematic.

**Not possible:** vLLM does not apply `default` + `dpo` in one forward.

---

### Attempt 7 — HF policy stack (training reproduction) — critical

**Hypothesis:** Verify training forward; rule out vLLM/cat as sole cause.

**Important:** Uses `load_stacked_for_merge()` from merge module — **name only**; function does **not** call `merge_cat`. It only:

1. `load_bnb_8bit_base()`
2. `PeftModel.from_pretrained(SFT)`
3. `load_adapter(DPO)`
4. `set_adapter(["default","dpo"])`

**Command:**

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate saulgman
cd /root/saulie
python dpo/train/smoke_policy_stack_hf.py \
  --dpo-adapter dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-29/best_adapter \
  --skeleton-ids eval_A4_001,eval_B8_001 \
  --output dpo/eval/dpo_phase1_policy_stack_hf_smoke.json
```

**Mechanical checks:**

| Field | Value |
|-------|-------|
| `active_adapters` | `["default", "dpo"]` |
| `trainable_total` | 66,060,288 (DPO only; frozen at inference) |
| `only_dpo_trainable` | true |
| Load time | ~17 s |

**Result:** **FAIL** — same artifact class as cat vLLM.

| Skeleton | Turn | Sample |
|----------|------|--------|
| eval_A4_001 | 1 | English, odd (“air-pocket or grip-pocket”) |
| eval_A4_001 | 2 | **Chinese only** — 不是靠毯子填空… |
| eval_B8_001 | 1 | Long English |
| eval_B8_001 | 2–4 | **Chinese** — 不是…是… / 三样… |

**Artifact:** `dpo/eval/dpo_phase1_policy_stack_hf_smoke.json`  
**Script:** `dpo/train/smoke_policy_stack_hf.py`

---

### Attempt 8 — Mislabeled stack smoke (discard)

**File:** `dpo/eval/dpo_phase1_stack_smoke.json`  
**Problem:** `adapter_path` points to **`sft_dpo_cat`**, backend transformers — **not** `["default","dpo"]`.  
**Action:** **Ignore**; use `dpo_phase1_policy_stack_hf_smoke.json` only.

---

## Diagnostic procedures (code & commands)

### D1 — Load policy stack (BnB, no merge)

**Code:** `merge_sft_dpo_lora.load_stacked_for_merge()` (lines 105–115)

```python
base = load_bnb_8bit_base()
model = PeftModel.from_pretrained(base, SFT_ADAPTER, adapter_name="default", is_trainable=False)
model.load_adapter(dpo_path, adapter_name="dpo", is_trainable=False)
```

**Training equivalent:** `train_dpo.build_dpo_peft_model()` uses `add_adapter` for fresh DPO; inference uses `load_adapter` with saved weights — correct.

**Activate policy:**

```python
from dpo.train.dpo_trainer_compat import ensure_policy_adapter_stack
ensure_policy_adapter_stack(model)  # set_adapter(["default", "dpo"])
```

---

### D2 — Cat merge + weight verification

**Code:** `dpo/train/merge_sft_dpo_lora.py`

| Function | Purpose |
|----------|---------|
| `merge_cat()` | `add_weighted_adapter(..., combination_type="cat")` |
| `verify_weight_matrices()` | Per-layer ‖ΔW_stack − ΔW_cat‖ |
| `compare_logps_chat()` | Last-token logits: stack vs cat |
| `flatten_adapter_dir()` | PEFT nested save → flat vLLM dir |
| `validate_merge_compatibility()` | Rank/target/peft_type checks |

**Command:**

```bash
python dpo/train/merge_sft_dpo_lora.py \
  --dpo-adapter dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-29/best_adapter \
  --output dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-29/sft_dpo_cat \
  --check-logps
```

**Logit prompts:** `FIXED_CHAT_CONVERSATIONS` in merge script (running-shoes multi-turn + short prompt).

**Recorded in:** `trial-29/sft_dpo_cat/merge_meta.json`

---

### D3 — Unbaked cat export

**Code:** `dpo/train/export_unbaked_cat.py`  
**Method:** Concat raw `lora_A` / `lora_B` blocks; `lora_alpha=96`, `r=48`, no scaling baked into A.

```bash
python dpo/train/export_unbaked_cat.py
```

---

### D4 — Offline ΔW from safetensors (one-off script used in session)

```python
# Representative layer: layers.0.mlp.down_proj
# ΔW_stack = (32/16)*B_s@A_s + (64/32)*B_d@A_d
# ΔW_cat   = (alpha_cat/r_cat) * B_cat @ A_cat
# Results: stack vs baked ~3.5e-10; stack vs unbaked ~2.6e-7
```

Paths: `sft_dpo_cat`, `sft_dpo_cat_unbaked`, `best_adapter/dpo`, `trial-17/best_adapter`.

---

### D5 — FP8 vLLM deploy

**Script:** `train/sft_eval/deploy_qwenie_eval.sh`

| Env var | Typical value |
|---------|----------------|
| `CANDIDATE_MANIFEST` | path to `.jsonl` manifest |
| `MAX_LORA_RANK` | `64` for cat r=48 |
| `MAX_LORAS` | `1` or `2` for dual-mount test |
| `VLLM_API_KEY` | `dipshit` (match eval script) |

**Base:** `/root/saulie/Qwen3-4B-Instruct-2507-FP8`  
**Container:** `eval_deploy_qwenie`, port `8000`

---

### D6 — vLLM multi-turn generation

**Script:** `train/sft_eval/eval_generate_vllm.py`

| Param | Value used in smokes |
|-------|----------------------|
| `MAX_TOKENS` | 350 |
| `TEMPERATURE` | 0.7 |
| `TOP_P` | 0.8 |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` |
| Skeletons | `dpo/eval/eval_skeletons.json` |
| IDs | `eval_A4_001`, `eval_B8_001` |

```bash
python train/sft_eval/eval_generate_vllm.py \
  --candidate-manifest dpo/eval/dpo_phase1_stack_only_manifest.jsonl \
  --skeletons dpo/eval/eval_skeletons.json \
  --skeleton-ids eval_A4_001,eval_B8_001 \
  --output-dir dpo/eval \
  --output dpo_phase1_smoke.json
```

---

### D7 — HF policy-stack generation (training path)

**Script:** `dpo/train/smoke_policy_stack_hf.py`

- Tokenizer: `train_dpo.load_tokenizer()` + `patch_chat_template_for_assistant_loss`
- Generation: `model.generate(..., max_new_tokens=350, temperature=0.7, top_p=0.8)`
- Calls `ensure_policy_adapter_stack` before each turn

See [Attempt 7](#attempt-7--hf-policy-stack-training-reproduction--critical).

---

### D8 — DPO val reward diagnostics (what Optuna saw)

**Code:** `dpo/train/dpo_diagnostics.py` → `_per_batch_dpo_scores()`, `compute_val_diagnostics()`

Uses same `ensure_policy_adapter_stack(model)` as training.  
**Trial output:** `trial-29/diagnostics.json`

---

## Behavioral smoke results (side-by-side)

**Skeletons:** `eval_A4_001` (cold hands / typing), `eval_B8_001` (ocean depth → beach sand)  
**Params:** temp 0.7, top_p 0.8, max 350 tokens per assistant turn

### eval_A4_001 — final assistant turn (excerpt)

| Config | Final turn (abridged) |
|--------|------------------------|
| **HF policy stack** | 不是靠毯子填空，是让手在不贴脸… (Chinese) |
| **FP8 cat baked** | 是缺在玻璃边不烧进去也不是整手泡进暖气… (Chinese) |
| **FP8 cat unbaked** | 是让干冷气贴着皮肤走… (Chinese) |
| **FP8 SFT-only** | A heater that dries the air is basically a punishment… (English) |
| **FP8 DPO-only** | Thanks for that extra detail — **cold air + dry air**… (English markdown) |

### eval_B8_001 — pattern

| Config | Pattern |
|--------|---------|
| **HF policy stack** | Turn 1 English; turns 2–4 Chinese 不是…是… |
| **FP8 cat** | Similar EN/ZH mix |
| **FP8 SFT-only** | English; long repetitive lists on later turns |
| **FP8 DPO-only** | English; conversational |

Full transcripts: JSON files listed in [reading guide](#complete-file-reading-guide).

---

## Trial-29 training metrics

**File:** `dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-29/diagnostics.json`

| Metric | Value |
|--------|-------|
| `eval_rewards_accuracy` | **1.0** (52/52 val rows) |
| `eval_rewards_margin` | 11.67 (mean) |
| `eval_loss` | 0.0060 |
| `train_loss` | 0.155 |
| Val rows | 52 |
| `only_dpo_trainable` | true |
| `max_length` | 704 |
| `length_mode` | ld_0.2 |
| Runtime | ~774 s |
| Peak VRAM | ~14.5 GB alloc |

**Implication:** Training run completed as designed; val **pairwise** task is perfect on held-out pairs. Skeleton smokes test a **different** task (open-loop generation).

---

## Environment & deploy

| Item | Detail |
|------|--------|
| GPU | NVIDIA RTX 3090 (24 GB), WSL2 |
| Conda env | `saulgman` (`source /root/miniconda3/etc/profile.d/conda.sh && conda activate saulgman`) |
| vLLM image | `vllm/vllm-openai:latest` |
| PEFT | 0.18.1 (in cat `adapter_config.json`) |
| TRL (trial-29) | 1.4.0 per `diagnostics.json` |

---

## Hypotheses

1. **Policy generalization (strong):** Trial-29 ranks chosen/rejected well but **open-loop multi-turn gen** drifts to Chinese/high-weirdness — especially after model’s own turn 1 (not in training distribution).
2. **Cat ≠ stack forward (deploy):** ΔW match but BnB logits differ ~2.5; FP8 vLLM single cat cannot equal dual-adapter policy even if gen were good.
3. **vLLM r=48-only bug (weak):** Stack also fails on HF; vLLM not sole root cause of text quality.
4. **Unbaked scaling (ruled out):** α=96 did not fix vLLM cat.
5. **Corrupt checkpoint (ruled out):** DPO-only and SFT-only sane on FP8.

---

## Not tried yet

- SFT-only `set_adapter(["default"])` on **HF BnB** same skeletons (contrast)
- Greedy decode (temp 0) on policy stack
- Generate **only final turn** using DPO val row `prompt` prefixes (teacher-forced prefix, one-shot gen)
- FP8 vLLM with **BF16 base** + cat (~24 GB)
- HF load of FP8 weights + cat
- `merge_and_unload` full weights (user constraint: LoRA-only serve)

---

## Recommended next steps

1. **SFT-only HF smoke** — same `smoke_policy_stack_hf.py` pattern with `set_adapter(["default"])`.
2. **One-shot gen from val `prompt`** — prefix from JSONL, single assistant generation; compare to skeleton open-loop.
3. **Decide Phase 1 gate:** If policy stack gen is unacceptable, cat merge is moot for product eval until serve architecture changes.
4. **Spot-check** `DPO_522_*.jsonl` chosen text vs trial-29 gen — confirm training data is English and on-brand.

---

## Complete file reading guide

Read in this order for full context.

### 1 — This report & plan

| File | Why |
|------|-----|
| `dpo/eval/MERGE_CAT_DIAGNOSTIC_REPORT.md` | **This document** |
| `dpo/eval/DPO_FINAL_EVAL_PLAN.md` | Original Phase 1 eval intent |
| `dpo/DPO_HANDOFF.md` | Locked train/serve decisions |

### 2 — Training policy (how trial-29 was trained)

| File | Why |
|------|-----|
| `dpo/train/train_dpo.py` | `build_dpo_peft_model`, BnB load, save DPO adapter |
| `dpo/train/dpo_trainer_compat.py` | `ensure_policy_adapter_stack`, collator, masks |
| `dpo/train/dpo_data.py` | Pretokenize prompt/chosen/rejected |
| `dpo/train/dpo_diagnostics.py` | Val reward / accuracy computation |
| `dpo/train/paths.py` | `SFT_ADAPTER`, model paths |
| `dpo/train/model_load.py` | `load_bnb_8bit_base()` |
| `dpo/train/models/.../trial-29/diagnostics.json` | **Trial-29 metrics** |
| `dpo/train/models/.../trial-29/README.md` | Trial notes (if any) |
| `dpo/dataset/DPO_522_prompt_a_and_prompt_b_V2.jsonl` | Training data shape |
| `dpo/dataset/DPO_522_prompt_a_and_prompt_b_V2_meta.json` | Counts / sources |

### 3 — Merge & diagnostics code

| File | Why |
|------|-----|
| `dpo/train/merge_sft_dpo_lora.py` | Cat merge, ΔW + logit checks, `load_stacked_for_merge` |
| `dpo/train/export_unbaked_cat.py` | Unbaked cat attempt |
| `dpo/train/smoke_policy_stack_hf.py` | **HF policy stack repro** |
| `dpo/train/models/.../trial-29/sft_dpo_cat/merge_meta.json` | Merge check numbers |

### 4 — Deploy & vLLM generation

| File | Why |
|------|-----|
| `train/sft_eval/deploy_qwenie_eval.sh` | FP8 Docker vLLM |
| `train/sft_eval/eval_generate_vllm.py` | API generation loop |
| `dpo/eval/eval_skeletons.json` | Skeleton prompts used in all smokes |

### 5 — Manifests (which adapter was mounted)

| File | Model served |
|------|----------------|
| `dpo/eval/dpo_phase1_stack_only_manifest.jsonl` | Cat baked |
| `dpo/eval/dpo_phase1_unbaked_manifest.jsonl` | Cat unbaked |
| `dpo/eval/dpo_phase1_dual_lora_manifest.jsonl` | SFT + DPO separate |
| `dpo/eval/dpo_phase1_sft17_only_manifest.jsonl` | SFT only |
| `dpo/eval/dpo_phase1_dpo_only_manifest.jsonl` | DPO only (if used) |

### 6 — Smoke outputs (read the actual model text)

| File | Config | Result |
|------|--------|--------|
| `dpo/eval/dpo_phase1_policy_stack_hf_smoke.json` | **HF BnB stack** | FAIL |
| `dpo/eval/dpo_phase1_smoke.json` | FP8 cat baked | FAIL |
| `dpo/eval/dpo_phase1_smoke_unbaked.json` | FP8 cat unbaked | FAIL |
| `dpo/eval/dpo_phase1_smoke_dual.json` | FP8 SFT + DPO-only | PASS (separate) |
| `dpo/eval/dpo_phase1_smoke_sft17.json` | FP8 SFT only | PASS |
| `dpo/eval/dpo_phase1_smoke_fp8.json` | FP8 cat (duplicate run if present) | FAIL |
| ~~`dpo/eval/dpo_phase1_stack_smoke.json`~~ | **Mislabeled cat** | DISCARD |

### 7 — Checkpoints on disk

| Path | Content |
|------|---------|
| `train/models/steering-sft-v1.1/trial-17/best_adapter/` | SFT LoRA |
| `dpo/train/models/.../trial-29/best_adapter/dpo/` | DPO LoRA (policy delta) |
| `dpo/train/models/.../trial-29/sft_dpo_cat/` | Cat merged LoRA |
| `dpo/train/models/.../trial-29/sft_dpo_cat_unbaked/` | Manual unbaked LoRA |
| `Qwen3-4B-Instruct-2507/` | BnB train base |
| `Qwen3-4B-Instruct-2507-FP8/` | vLLM deploy base |

### 8 — Eval slate / finalists (Phase 2 context)

| File | Why |
|------|-----|
| `dpo/eval/candidate_metadata_finalists.json` | Official candidate list |
| `dpo/train/HANDOFF.md` | Train/eval handoff notes |

---

## Bottom line

- **Every deploy/cat path tried failed** on skeleton smokes; **HF training policy stack also failed** with similar artifacts.
- **Trial-29 training metrics are excellent** for **pairwise preference on 52 val rows** — that does not contradict bad **open-loop** generation.
- **Cat merge remains blocked** for vLLM (forward ≠ stack on BnB; one-LoRA limit) **and** does not fix policy quality if stack gen is already bad.
- **Do not use** `dpo_phase1_stack_smoke.json` or any cat smoke JSON for judge eval until policy behavior is understood.
