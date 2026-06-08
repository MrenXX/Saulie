# Saulie — `dpo_eval` branch

This branch holds the **DPO v1.5 final evaluation stack**: cat-merge validation, vLLM FP8 generation, blind LLM-judge workflow, and artifacts from the completed Round 1 slate. It is **not** the generic steering-training README — for Optuna study babysitting and trial training, see `main`.

## What this branch is for

After the v1.5 hybrid Optuna run, we need to pick a **deployable** DPO adapter without trusting offline metrics alone. This branch implements:

1. **Merge gate** — validate cat-merged LoRA (`sft_dpo_cat`) before any judge generation
2. **Production-path inference** — vLLM FP8 + cat LoRA (not HF BnB8 for official scores)
3. **Two-round blind eval** — 60 skeletons split across Round 1 (screen) and Round 2 (holdout)
4. **External LLM judges** — you run scoring; the repo ships prompts + generation packets

## Quick map

| Path | Purpose |
|------|---------|
| [`dpo/eval/DPO_FINAL_EVAL_EXECUTION_PLAN.md`](dpo/eval/DPO_FINAL_EVAL_EXECUTION_PLAN.md) | End-to-end eval protocol |
| [`dpo/eval/JUDGE_PACKET_README.md`](dpo/eval/JUDGE_PACKET_README.md) | What to send / not send to judges |
| [`dpo/eval/MERGE_SCRIPT_VALIDATION_FIX_PLAN.md`](dpo/eval/MERGE_SCRIPT_VALIDATION_FIX_PLAN.md) | Pre-judge merge checks |
| [`dpo/eval/run_v15_final_eval.py`](dpo/eval/run_v15_final_eval.py) | Orchestrator (deploy + generate) |
| [`dpo/eval/v15_eval_config.py`](dpo/eval/v15_eval_config.py) | Skeleton IDs, sampling, manifest helpers |
| [`dpo/eval/vllm_scripts/`](dpo/eval/vllm_scripts/) | vLLM deploy, generation, runtime LoRA |
| [`dpo/train/merge_sft_dpo_lora.py`](dpo/train/merge_sft_dpo_lora.py) | Cat merge + FP8 deploy audit |
| [`dpo/train/scripts/merge_v15_eval_slate.sh`](dpo/train/scripts/merge_v15_eval_slate.sh) | Batch merge for finalist trials |

## Finalist slate (Round 1)

SFT baseline **trial-17** plus six cat-merged DPO trials:

| Judge ID | Trial |
|----------|------:|
| `steering-sft-v1.1_trial-17` | 17 |
| `candidate_A` | 19 |
| `candidate_B` | 16 |
| `candidate_C` | 8 |
| `candidate_D` | 27 |
| `candidate_E` | 20 |
| `candidate_F` | 4 |

Round 1 generations (blind): [`dpo/eval/generations_round1.json`](dpo/eval/generations_round1.json)  
Unblind sidecar (local only): [`dpo/eval/generations_round1_unblind.json`](dpo/eval/generations_round1_unblind.json)

Merge validation: [`dpo/eval/merge_v15_slate_validation.md`](dpo/eval/merge_v15_slate_validation.md)

## Typical workflow

### 1. Merge gate (once per finalist)

```bash
bash dpo/train/scripts/merge_v15_eval_slate.sh
```

Trials merged: **4, 8, 16, 19, 20, 27** → `…/trial-N/sft_dpo_cat/`

### 2. Deploy vLLM (FP8 + runtime LoRA)

```bash
MAX_LORA_RANK=64 bash dpo/eval/vllm_scripts/deploy_qwenie_eval.sh
```

- Cat adapters: `r=32` (16+16) or `r=40` (16+24 on trials 8/27)
- vLLM rank buckets are **8, 16, 32, 64** — use **64** for the full slate, **32** if serving only r=32 winners (trials 4, 20)

### 3. Generate (checkpointed; resumes on crash)

```bash
# Round 1 — 34 skeletons, 7 models
python dpo/eval/run_v15_final_eval.py --round 1 --anonymize --skip-deploy

# Round 2 — holdout skeletons, finalists only (after judge decision)
python dpo/eval/run_v15_final_eval.py --round 2 --anonymize --skip-deploy --models sft,trial-4,trial-19,trial-20,trial-27
```

Use `--fresh` to discard a partial checkpoint and start over.

### 4. Judge (external)

Send to your judge UI:

- [`dpo/eval/llm_judge_prompt_dpo.md`](dpo/eval/llm_judge_prompt_dpo.md)
- [`dpo/eval/DATA_CONTEXT.md`](dpo/eval/DATA_CONTEXT.md)
- `generations_roundN.json` (blind packet)

Do **not** send unblind JSON, trial metrics, or merge reports during scoring.

### 5. Deploy winner

Manual pick after judges: **trial 4** and **trial 20** were the finalists. Cat-merged adapter for trial 4:

- Host: `dpo/train/models/steering-dpo-v1.5/optuna-run-20260602-052732/trial-4/sft_dpo_cat`
- vLLM model id (suggested): `dpo-v15-trial-4`
- Production deploy: FP8 base + single cat LoRA, `MAX_LORA_RANK=32` for trials 4/20

Wire into your API layer (e.g. `agent_chat_api.py`): set `MODEL_NAME` to the vLLM LoRA name and use the eval deploy path, not legacy SFT-only `deploy_qwenie.sh`.

## Locked sampling (official eval)

```
temperature=0.7  top_p=0.8  top_k=20  repetition_penalty=1.05  max_tokens=256
```

`top_k` and `repetition_penalty` go in OpenAI client `extra_body` for vLLM.

## Skeletons

[`dpo/eval/eval_skeletons.json`](dpo/eval/eval_skeletons.json) — **60** items:

- **52** steering (types A/B/C/D) — split 26 Round 1 / 26 Round 2 holdout
- **8** ordinary conversation retention (type O) — same in both rounds

## Debug only (not for judge packets)

- [`dpo/eval/eval_generate_hf.py`](dpo/eval/eval_generate_hf.py) — HF BnB8 smoke; do not use for final judge JSON
- Older phase-1 smoke JSONs and rescue manifests in `dpo/eval/` — historical diagnostics

## Training stack (context)

This branch also carries the frozen **v1.0–v1.4** DPO training stack and the **v1.5 hybrid Optuna** run artifacts under `dpo/train/` and `dpo/study_results/`. The **branch focus** is eval + merge + deploy pick, not starting new studies.

## Requirements

GPU machine with vLLM Docker, FP8 base weights, SFT trial-17 adapter, and v1.5 run adapters on disk. Python deps: `dpo/requirements-dpo.txt` plus `openai`, `colorama`, `requests` for eval scripts.
