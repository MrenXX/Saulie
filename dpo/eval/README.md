# DPO v1.5 final eval (`dpo_eval` branch)

Behavioral evaluation for the v1.5 hybrid Optuna finalists: merge validation, vLLM FP8 generation, and blind LLM-judge packets.

**Start here:** [`DPO_FINAL_EVAL_EXECUTION_PLAN.md`](DPO_FINAL_EVAL_EXECUTION_PLAN.md)  
**Judge handoff:** [`JUDGE_PACKET_README.md`](JUDGE_PACKET_README.md)  
**Merge gate:** [`MERGE_SCRIPT_VALIDATION_FIX_PLAN.md`](MERGE_SCRIPT_VALIDATION_FIX_PLAN.md)

## Production inference path

Official eval and deploy pick use **vLLM FP8 + cat-merged LoRA** (`trial-N/sft_dpo_cat`), not HF 8-bit stack.

| Script | Role |
|--------|------|
| [`run_v15_final_eval.py`](run_v15_final_eval.py) | Entry point: deploy + Round 1/2 generation |
| [`v15_eval_config.py`](v15_eval_config.py) | Skeleton splits, sampling, manifests |
| [`vllm_scripts/deploy_qwenie_eval.sh`](vllm_scripts/deploy_qwenie_eval.sh) | FP8 vLLM container |
| [`vllm_scripts/eval_generate_vllm.py`](vllm_scripts/eval_generate_vllm.py) | Multi-turn gen + checkpoint resume |
| [`vllm_scripts/vllm_lora_runtime.py`](vllm_scripts/vllm_lora_runtime.py) | Runtime LoRA load/unload |

## Artifacts on this branch

| File | Description |
|------|-------------|
| [`generations_round1.json`](generations_round1.json) | Round 1 blind conversations (7 models × 34 skeletons) |
| [`generations_round1_unblind.json`](generations_round1_unblind.json) | Local trial mapping — do not send to judge |
| [`v15_final_eval_manifest.jsonl`](v15_final_eval_manifest.jsonl) | Round 1 model manifest |
| [`merge_v15_slate_validation.md`](merge_v15_slate_validation.md) | Merge gate summary |
| [`merge_v15_slate_report.json`](merge_v15_slate_report.json) | Per-trial merge metrics |
| [`llm_judge_prompt_dpo.md`](llm_judge_prompt_dpo.md) | Judge rubric |
| [`eval_skeletons.json`](eval_skeletons.json) | 60 user-turn skeletons |

## Round 1 blind mapping

| Judge ID | Trial |
|----------|------:|
| `steering-sft-v1.1_trial-17` | 17 |
| `candidate_A` | 19 |
| `candidate_B` | 16 |
| `candidate_C` | 8 |
| `candidate_D` | 27 |
| `candidate_E` | 20 |
| `candidate_F` | 4 |

## Commands

```bash
# Merge all finalists
bash dpo/train/scripts/merge_v15_eval_slate.sh

# Deploy vLLM
MAX_LORA_RANK=64 bash dpo/eval/vllm_scripts/deploy_qwenie_eval.sh

# Round 1 (resumes automatically if checkpoint exists)
python dpo/eval/run_v15_final_eval.py --round 1 --anonymize --skip-deploy
```

## Legacy / debug (ignore for final judge)

- `dpo_phase1_*`, `smoke_trial*`, `RUN_PLAN_A_*`, rescue scripts — earlier diagnostics before v1.5 final protocol
- [`eval_generate_hf.py`](eval_generate_hf.py) — HF path only; not for official judge JSON
- [`OLD_DPO_FINAL_EVAL_PLAN.md`](OLD_DPO_FINAL_EVAL_PLAN.md) — superseded plan
