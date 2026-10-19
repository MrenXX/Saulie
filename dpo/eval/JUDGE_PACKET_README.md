# Round 1 judge packet (external)

The generation packet is complete. See [evaluation results](../../docs/evaluation.md) for the reported aggregate scores.

## Files to send the judge

1. [`llm_judge_prompt_dpo.md`](llm_judge_prompt_dpo.md)
2. [`DATA_CONTEXT.md`](DATA_CONTEXT.md)
3. [`generations_round1.json`](generations_round1.json) — blind DPO IDs (`candidate_A` … `candidate_F`)

Do **not** include: `generations_round1_unblind.json`, `candidate_metadata_finalists.json`, trial metrics, or Optuna artifacts.

## Resume after failure

Generation writes **`generations_round1.json` after each model** (atomic `.tmp` rename). If a run crashes, re-run the same command — completed models are **skipped** automatically.

```bash
python dpo/eval/run_v15_final_eval.py --round 1 --anonymize --skip-deploy
```

Use `--fresh` only when you intentionally want to discard the checkpoint and start over.

Check `checkpoint_complete: true` in the JSON when the full slate is done.


The generation path uses vLLM FP8 plus a combined LoRA. The slate has rank-32 and rank-40 adapters and uses vLLM rank capacity 64. The chosen production trial 4 uses rank 32.

Sampling locked at generation time:

- `temperature=0.7`, `top_p=0.8`, `top_k=20`, `repetition_penalty=1.05`, `max_tokens=256`
- `top_k` / `repetition_penalty` passed via vLLM `extra_body` (not standard OpenAI kwargs)

## Round 1 advancement (after you produce `judge_round1.json`)

Advance **top 2** by default; **top 3** if:

- Rank 2 vs 3 steering means differ by &lt; 0.15, or
- Rank 2 vs 3 ordinary retention differ by &lt; 0.20, or
- Different strengths on Type B/D vs ordinary retention need a tie-break

Do not advance severe ordinary-retention failures unless all DPO candidates fail.

## Unblind locally

Use [`generations_round1_unblind.json`](generations_round1_unblind.json) + [`candidate_metadata_finalists.json`](candidate_metadata_finalists.json).
