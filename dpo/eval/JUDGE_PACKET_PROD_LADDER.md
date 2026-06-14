# Prod ladder judge packet (external)

## Goal

Compare **base FP8 → SFT trial-17 → prod DPO trial-4** on the same 60 conversation skeletons. Prod should win on steering quality, especially Type B/D, while preserving ordinary conversation (Type O).

## Files to send the judge

1. [`llm_judge_prompt_dpo.md`](llm_judge_prompt_dpo.md)
2. [`DATA_CONTEXT.md`](../dataset/DATA_CONTEXT.md)
3. [`eval_inference_system_prompt.md`](eval_inference_system_prompt.md) — **base model only** (bootstrap context)
4. [`generations_prod_ladder.json`](generations_prod_ladder.json)

Do **not** include: `prod_ladder_manifest.jsonl`, deploy scripts, Optuna metrics, or trial summaries.

## Generation settings (identical for all three models)

- Backend: **vLLM FP8** on `Qwen3-4B-Instruct-2507-FP8`
- `temperature=0.7`, `top_p=0.8`, `top_k=20`, `repetition_penalty=1.05`, `enable_thinking=false`
- `max_tokens=256` per assistant turn

## System prompt scope

| Model | System prompt |
|-------|---------------|
| `qwen3-4b-instruct-fp8` (base) | Yes — [`eval_inference_system_prompt.md`](eval_inference_system_prompt.md) |
| `steering-sft-trial-17` (SFT) | No — bare user/assistant skeleton fill |
| `dpo-v15-trial-4` (prod) | No — bare user/assistant skeleton fill |

Judges should not excuse base failures merely because it received bootstrap context. Compare **SFT vs prod** as the primary DPO validation gate.

## Resume after failure

Generation checkpoints to **`generations_prod_ladder.json`** after **every skeleton** (atomic `.tmp` rename). Re-run the same command — completed models and partial in-progress models resume automatically.

```bash
python dpo/eval/run_prod_ladder_eval.py
```

Use `--fresh` only when intentionally discarding all prior output.

Check `checkpoint_complete: true` when all three models are done.

## Judge calibration

- **Primary:** prod vs SFT (DPO should improve preference quality without damaging Type B/D or ordinary chat).
- **Secondary:** SFT vs base (SFT should show clear steering acquisition).
- Score Type O items with `ordinary_retention_score` only; do not require product recommendations.
- Penalize forced product recommendations on ordinary chat.
