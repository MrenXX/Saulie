# Plan B v1.3 — SFT-merged base DPO (runbook)

**MLflow experiment:** `steering-dpo-v1.3-sft-merged`  
**On-disk models:** `dpo/train/models/steering-dpo-v1.3-sft-merged/optuna-run-<stamp>/`  
**Stack:** `BnB8(SFT-merged dense base) + trainable DPO LoRA` — not `raw base + SFT LoRA`.

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate saulgman
cd /root/saulie
```

## Prerequisite: merged base

```bash
python dpo/train/merge_sft_baked_base.py
# -> Qwen3-4B-Instruct-2507-SFT-MERGED-BF16/
```

## Launch training (4 fixed trials)

```bash
python dpo/train/train_dpo.py --optuna --study-version v1.3 \
  --parallel-workers 1 --target-complete-trials 4 --max-attempted-trials 6
```

Monitor:

```bash
RUN=$(ls -td dpo/train/models/steering-dpo-v1.3-sft-merged/optuna-run-* | head -1)
echo "$RUN" > dpo/train/models/steering-dpo-v1.3-sft-merged/LATEST_RUN_DIR.txt
tail -f "$RUN/launcher.log"
tail -f "$RUN/worker_0.log"
```

MLflow UI:

```bash
mlflow ui --backend-store-uri file:///root/saulie/dpo/train/mlruns --port 5001
# Experiment: steering-dpo-v1.3-sft-merged
```

### Trial slate

| Trial | `plan_b_label` | Notes |
|-------|----------------|-------|
| 0 | `plan_b_anchor` | v1.2-like; conversation anchor |
| 1 | `plan_b_bridge` | Mid β/LR/rank |
| 2 | `plan_b_v11_lite` | Scaled-down v1.1 trial-13 |
| 3 | `plan_b_v10_23_lite` | Scaled-down v1.0 trial-23 |

**Winner selection:** REPL + fluency report — not Optuna objective or hybrid score.

## Decode defaults (HF REPL + smoke)

All HF paths use **Qwen3 non-thinking sample** by default (`temperature=0.7`, `top_p=0.8`, `top_k=20`). Greedy is opt-in only (`--decode greedy`) and warns — prior Plan A/B smoke used greedy and overstated repetition.

REPL: `python dpo/eval/chat_policy_stack.py --sft-baked` (sample by default). Optional: `--repetition-penalty 1.05` if loops persist.

## Stage A — 10-skeleton fluency (after each run completes)

```bash
RUN=$(cat dpo/train/models/steering-dpo-v1.3-sft-merged/LATEST_RUN_DIR.txt)

python dpo/eval/run_plan_b_part1_hf_smoke.py --run-dir "$RUN" \
  --output dpo/eval/plan_b_part1_w1_10skel_sample.jsonl

python dpo/eval/score_plan_b_fluency.py --jsonl dpo/eval/plan_b_part1_w1_10skel_sample.jsonl
```

Agent: manual read → `dpo/eval/PLAN_B_PART1_FLUENCY_<label>.md` (10/10 PASS bar from `PLAN_A_PART1_SEMANTIC_REPORT.md`).

## Stage B — Your REPL (shortlist only)

```bash
# Baseline
python dpo/eval/chat_policy_stack.py --sft-baked

# Trial adapter
python dpo/eval/chat_policy_stack.py --sft-baked \
  --dpo-adapter "$RUN/trial-1/best_adapter"
```

## Optional: old DPO adapters on baked base (non-principled)

```bash
python dpo/eval/chat_policy_stack.py --sft-baked \
  --dpo-adapter dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-13/best_adapter
```

## Telemetry bands (diagnostic)

| Signal | Bridge target | Overfit risk |
|--------|---------------|--------------|
| Val accuracy | 55–85% | > 95% |
| Val margin | +0.3 to +1.5 | > +2.5 |
| `margin_vs_length_delta_corr` | −0.5 … +0.5 | ≈ −0.9 with high acc |
