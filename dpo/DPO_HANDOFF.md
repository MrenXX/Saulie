# Handoff: DPO training for Saulie (Qwen3-4B steering)

**Created:** 2026-05-19  
**Workspace:** `/root/saulie`  
**Next session focus:** Implement DPO pipeline per plan; resolve multi-turn completion masking before/during dummy run; do **not** start 20-trial Optuna until user says **go** after dummy run succeeds.

---

## Primary artifacts (read these first)

| Artifact | Path |
|----------|------|
| **Implementation plan** | `/root/.cursor/plans/dpo_training_stack_33b7aa9b.plan.md` (user may copy as `dpo_training_stack_33b7aa9b.plan.md`) |
| Prompt A spec (schema + categories) | `dpo/dataset/DPO_PROMPT_A_OG.md` |
| Prompt B spec (on-policy pairs) | `dpo/dataset/DPO_PROMPT_B.md` |
| Training data (522 rows) | `dpo/dataset/DPO_522_prompt_a_and_prompt_b_V2.jsonl` |
| Data meta / counts | `dpo/dataset/DPO_522_prompt_a_and_prompt_b_V2_meta.json` |

**No DPO training code exists yet.** Only SFT: `train/train_sft.py`, eval: `train/sft_eval/`, seq analysis: `train/analyze_seq_len.py`.

---

## Project goal (one paragraph)

Fine-tune **on top of** `steering-sft-v1.1` **trial-17** with DPO on 522 curated preference pairs (synthetic Prompt A + on-policy Prompt B). Persona: conversational steering toward **generic product recommendations** (Saul Goodman–ish cadence, no brands, no em dashes). **Do not modify** trial-17 SFT adapter files. Inference: **FP8 base** + **one merged LoRA** (SFT+DPO via PEFT `cat` merge) in vLLM; eval compares **base+sysprompt** vs **SFT-only** vs **merged SFT+DPO** via `train/sft_eval/`.

---

## Locked decisions (do not re-litigate without user)

### Hardware / VRAM

- **Design for 12GB** (4070-era constraints): BnB 8-bit base, batch=1 default, `max_length=704`, gradient checkpointing.
- **3090 cheat allowed** for speed only: `per_device_train_batch_size` ∈ {1,2}, `gradient_accumulation_steps` ∈ {2,4,8}, effective batch 4–16.
- **Train on** `Qwen3-4B-Instruct-2507` (BnB 8-bit). **Infer on** `Qwen3-4B-Instruct-2507-FP8` + FP16 LoRAs.

### Model / adapter architecture

- **Reference π_ref:** frozen copy of trial-17 (`sft_ref` adapter); **not** base-without-adapter.
- **Policy:** new trainable **DPO LoRA** on same base; trial-17 never written.
- **`precompute_ref_log_probs=True`** on train+val (locked).
- TRL: `ref_model=None`, `ref_adapter_name="sft_ref"`, `model_adapter_name="dpo"`.
- **No merge** of SFT into base (FP8/FP16 + VRAM + user wants separate SFT artifact).
- **Serving:** PEFT `add_weighted_adapter(..., combination_type="cat")` → single LoRA rank `r_sft + r_dpo`; bump vLLM `--max-lora-rank` (≥48 if DPO r=32). vLLM does **not** stack two LoRAs per request.

**Spike still required:** policy forward may need `sft`+`dpo` both active during training; if PEFT/TRL cannot, fallback documented in plan.

### Data

- File: `dpo/dataset/DPO_522_prompt_a_and_prompt_b_V2.jsonl` (522 lines).
- Split **80/10/10**, stratify: `f"{dpo_source}_{category}_{opening_type}"` (`prompt_a` | `prompt_b_repaired` | `prompt_b_exp500`).
- `max_length=704`, `truncation_mode=keep_end`.

### Training gate (mandatory order)

1. Spike (adapter forwards, OOM at 704).
2. **Dummy run** (2 epochs, fixed hparams, full pipeline, MLflow, save adapter) → report to user.
3. **STOP** — wait for explicit user **go**.
4. Optuna **20 trials** (not 15).

**Dummy hparams:** `beta=0.1`, `sigmoid`, `lr=1e-5`, `epochs=2`, `r=16`, `alpha=32`, `batch=1`, `grad_accum=4`, `precompute_ref_log_probs=True`. Output: `train/models/steering-dpo-v1.0/dummy-run/`. Record val `eval_loss` min/max for hybrid_score normalization.

### Optuna

- **Objective:** maximize val **`rewards/accuracies`** only.
- **Also log:** `eval_loss`, margins, logps, **`hybrid_score`** = `0.8 * acc + 0.2 * (1 - norm_loss)` (norm from dummy-run loss range) — **not** for trial selection.
- Search: β {0.01,0.05,0.1,0.2}, epochs {1,2,3}, loss {sigmoid, ipo, sigmoid_norm}, lr 5e-6–3e-5 log, LoRA r {8,16,32} alpha=2r, neftune {0, 5}, schedulers cosine/constant_with_warmup/linear, max_grad_norm {0.3,0.5,1.0}, weight_decay 0.05, label_smoothing 0.

### Eval (final)

- `train/sft_eval/eval_skeletons.json` (held out).
- `eval_generate_vllm.py` + `llm_judge_prompt.md`.
- Arms: **base+sysprompt**, **SFT trial-17**, **merged SFT+DPO** (the product “DPO model”).

### Paths (on disk in saulie)

```
Qwen3-4B-Instruct-2507/                    # BnB train base
Qwen3-4B-Instruct-2507-FP8/                # vLLM infer base
train/models/steering-sft-v1.1/trial-17/best_adapter/
deploy_sft_trial17.sh
train/train_sft.py                           # patterns to mirror
train/analyze_seq_len.py                     # 704 token justification
```

Trial-17 LoRA: r=16, alpha=32, targets q/k/v/o/gate/up/down, dropout 0.1.

---

## Critical open issue: multi-turn `chosen`/`rejected` (READ BEFORE CODING)

**Discovered in conversation; not fully resolved in plan.**

### Facts (verified on full 522 JSONL)

| Category | Rows | `user` in chosen/rejected |
|----------|-----:|---------------------------|
| steering | 121 | 120 (99%) |
| style | 220 | 88 |
| product_fit | 181 | 1 |
| **Total** | 522 | **209** |

**Steering pattern (120 multi-turn rows):**

- `chosen`/`rejected` = typically `assistant → user → assistant` (2–3 turn **continuations** after shared `prompt`).
- **First assistant turn differs in all 120** (divergence from completion token 0).
- **Synthetic user reply differs** between chosen/rejected in ~111/120.
- **Chosen often longer** (~89 rows chosen longer by >50 chars) → DPO length bias risk.

### Why it matters

- TRL DPO sums log-probs over **entire completion block** (`completion_mask=1` on all completion tokens).
- That includes **scripted user tokens** in chosen/rejected → not the same as SFT `assistant_only_loss`.
- User tokens in `prompt` are masked out (good); user tokens in **completion** are **not** (problem).
- Pairs with **different** synthetic users violate strict “same x, two completions” intuition; preference is over **whole counterfactual trajectories**.

### Recommended mitigations (implementer should pick with user if unclear)

1. **Assistant-only logprob mask** inside completion (user tokens condition but don’t score) — **top engineering fix** for v1.
2. **`loss_type=sigmoid_norm`** and/or **`ld_alpha`** (0.3–0.7) for length — see [LD-DPO](https://arxiv.org/abs/2409.06411), TRL `DPOConfig`.
3. Longer term: reformulate steering to single-turn pairs or [DMPO](https://aclanthology.org/2024.emnlp-main.138/) (multi-turn theory).
4. Dummy run: log % completion tokens that are `user`; slice `rewards/accuracies` by `category` / `dpo_source`.

### Clarification on `assistant_only_loss`

- Plan said “not used” meaning **no SFTConfig flag** — DPO uses `completion_mask`, not `{% generation %}` tags.
- **Still reuse** `patch_chat_template_for_assistant_loss()` from `train_sft.py` for tokenizer consistency if needed.
- **Do not** train on user message text without masking — user’s SFT mistake concern is **valid** for 209 rows.

Authoring spec **allows** user in continuations: `DPO_PROMPT_A_OG.md` lines 266–268. Data is **correct per spec**, **awkward for vanilla DPO**.

---

## Example rows (for sanity checks)

- **Style (clean):** `dpo_001` — chosen/rejected = single assistant only.
- **Steering (multi-turn):** `dpo_002`, `dpo_014` — chosen/rejected = asst/user/asst; different user scripts.
- **Prompt B steering:** `dpo2_pair_dpo2_B6_043`.

---

## Files to implement (from plan)

| File | Purpose |
|------|---------|
| `train/dpo_data.py` | Load JSONL, stratified split |
| `train/train_dpo.py` | Dummy mode + Optuna + MLflow |
| `train/merge_sft_dpo_lora.py` | Cat-merge SFT+DPO for vLLM |

Mirror: `train/train_sft.py` (load model, chat patch, `clear_gpu`, MLflow/Optuna patterns).

---

## User preferences / constraints

- Inference stack **A:** separate FP8 base + frozen trial-17 + DPO LoRA → merged for vLLM (no base merge).
- Grill-me skill used during planning; user wants decisions explicit.
- Commits only when asked.

---

## Suggested skills for next session

| Skill | When |
|-------|------|
| None required | Implementation follows plan |
| `grill-me` (`/root/saulie/SKILL.md`) | If revisiting data format vs masking tradeoffs |
| `babysit` | After PR / CI |
