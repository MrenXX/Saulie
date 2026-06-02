# Weighted DPO rescue — smoke results (2026-05-28)

PR [#2](https://github.com/MrenXX/Saulie/pull/2) merged into `dpo_eval`. Stages 1–4 completed on inference box. Stage 6 (vLLM) **blocked** — Docker not available in WSL.

## Stage 1 — HF greedy (2 skeletons)

**17/18 passed** automated CJK/template heuristics (`score_rescue_smoke.py`).

| Trial | w=0.00–0.75 | w=1.00 |
|-------|-------------|--------|
| trial-29 (v1.1) | PASS | **FAIL** (CJK + 不是…是 patterns) |
| trial-10 (v1.1) | PASS all | PASS |
| trial-1 (v1.0) | PASS all | PASS |

**Conclusion:** trial-29 full-strength DPO (w=1.0) is the broken policy; scaling fixes greedy collapse.

## Stage 2 — HF sample (4 skeletons, temp 0.7)

**7/8 passed.**

| Config | Result |
|--------|--------|
| trial-29 w=0.10, 0.25, 0.50 | PASS |
| trial-29 w=0.75 | **FAIL** (CJK on eval_D6_001) |
| trial-10 w=0.50, 1.00 | PASS |
| trial-1 w=0.50, 1.00 | PASS |

**Primary export candidates:** trial-29 **w=0.25** (lowest passing weight with visible DPO), trial-29 **w=0.50** (backup).

## Stage 4 — Weighted cat export

All ΔW checks **pass**. Forward logit check informational fail (~0.6–1.2) — expected.

| Output dir | dpo_weight |
|------------|------------|
| `.../trial-29/sft_dpo_cat_w025` | 0.25 |
| `.../trial-29/sft_dpo_cat_w050` | 0.50 |
| `.../trial-10/sft_dpo_cat_w050` | 0.50 |
| `.../trial-1/sft_dpo_cat_w050` | 0.50 |

Manifest: `dpo/eval/dpo_weighted_rescue_manifest.jsonl`

## Stage 6 — vLLM (pending)

```bash
MAX_LORA_RANK=64 MAX_LORAS=5 \
CANDIDATE_MANIFEST=/root/saulie/dpo/eval/dpo_weighted_rescue_manifest.jsonl \
bash dpo/eval/vllm_scripts/deploy_qwenie_eval.sh

python dpo/eval/vllm_scripts/eval_generate_vllm.py \
  --candidate-manifest dpo/eval/dpo_weighted_rescue_manifest.jsonl \
  --skeleton-ids eval_A4_001,eval_B8_001 \
  --output-dir dpo/eval/weighted_rescue_vllm_smoke
```

## Artifacts

- Greedy: `dpo/eval/smoke_*_greedy.json` (18 files)
- Sample: `dpo/eval/smoke_*_sample4.json` (8 files)
- Logs: `rescue_stage1_greedy.log`, `rescue_stage2_sample.log`, `rescue_stage4_export.log`
- Scripts: `run_rescue_stage1_greedy.sh`, `run_rescue_stage2_sample.sh`, `run_rescue_stage4_export.sh`, `score_rescue_smoke.py`

## Next

1. Run vLLM smoke when Docker is up.
2. Mini eval (12 skeletons) on vLLM-passing candidates.
3. Do **not** use unweighted `sft_dpo_cat` (w=1.0) for judge eval.
