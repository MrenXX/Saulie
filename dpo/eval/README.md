# DPO evaluation

Use [evaluation and recorded results](../../docs/evaluation.md) for interpretation and [the DPO guide](../README.md) for training. This directory contains generation runners, export diagnostics, judge rubrics, and runtime comparisons.

## Saved evidence

- [Round 1 conversations](generations_round1.json): complete, seven models on 34 scenarios.
- [Unblind map](generations_round1_unblind.json): local mapping for the six candidate IDs.
- [Merge summary](merge_v15_slate_validation.md) and [JSON report](merge_v15_slate_report.json): export checks for six finalists.
- [Base/SFT/DPO conversations](generations_prod_ladder.json): complete, three models on 60 scenarios.
- [Scenarios](eval_skeletons.json), [judge rubric](llm_judge_prompt_dpo.md), and [generation configuration](v15_eval_config.py).
- [Model/prompt matrix](matrix_runs/), [latency report](LATENCY_REPORT.md), and [raw latency runs](latency_runs/).

Structured final judge scores were not recovered. The [selection record](../../docs/evidence/selection-record.json) documents the author's choice of trial 4 after comparing judge finalists 4 and 20. The training-objective winner was trial 16.

## Re-run commands

These require the environment, trained adapters, GPU memory, and keys described in [reproducibility](../../REPRODUCIBILITY.md). Deploy commands may replace named containers.

```bash
# Export and audit the fixed finalist slate.
bash dpo/train/scripts/merge_v15_eval_slate.sh

# Start FP8 vLLM with capacity for the slate's rank-40 adapters.
MAX_LORA_RANK=64 bash dpo/eval/vllm_scripts/deploy_qwenie_eval.sh

# Generate or resume Round 1.
python dpo/eval/run_v15_final_eval.py --round 1 --anonymize --skip-deploy

# Compare base, frozen SFT, and deployed DPO on all 60 scenarios.
python dpo/eval/run_prod_ladder_eval.py
```

The production choice has rank 32. The evaluation slate needs rank capacity 64 because some other finalists have rank 40.

Runners support checkpoint resume. Use `--fresh` only for an intentional replacement of saved output. New generations should be preserved separately from the tracked historical evidence.

## Judge packets

[Round 1 packet instructions](JUDGE_PACKET_README.md) separate blind outputs from candidate metadata. [The later comparison packet](JUDGE_PACKET_PROD_LADDER.md) documents that only the base receives a system prompt.

These are experiment instructions, not evidence that a judge returned the scores. Ordinary-chat scenarios use a retention rubric and should not be rewarded for forced product recommendations.

## Historical diagnostics

`dpo_phase1_*`, `smoke_trial*`, and the rescue scripts concern earlier adapters or training representations. The [semantic report](TRAIN_SETUP_SEMANTIC_REPORT.md) and [rescue results](RESCUE_SMOKE_RESULTS.md) remain useful historical records. They should not be presented as the final deployment's score.
