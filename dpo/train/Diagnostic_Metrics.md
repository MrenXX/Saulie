# Optuna Diagnostic Metrics Addendum

Use this as a small addendum to the parallel Optuna plan. Do not expand logging into a huge dashboard. The goal is to record the few diagnostics that will explain 90% of confusing trial outcomes and that will help hyperparam tuning: source imbalance, category regressions, length shortcuts, and basic run health.

## Trial-Level Scorecard

For every completed trial, log these to both Optuna user attrs and MLflow when possible:

- `eval_rewards_accuracy`
- `eval_rewards_margin`
- `eval_loss`
- `train_loss`
- `eval_rewards_chosen`
- `eval_rewards_rejected`
- `eval_logps_chosen`
- `eval_logps_rejected`
- `peak_vram_allocated_gb`
- `peak_vram_reserved_gb`
- `runtime_seconds`
- `saved_adapter_path`
- `failure_reason`, if pruned or failed

These are the core numbers needed to rank, sanity-check, and debug a trial.

## Compact Source And Category Metrics

Log validation reward accuracy and mean margin for these buckets:

- raw `dpo_source`
- `source_family`
- `category`
- `source_family x category`

For each bucket, record only:

- `count`
- `accuracy`
- `mean_margin`

Also compute:

- `macro_accuracy_by_source_family`
- `macro_accuracy_by_category`
- `macro_accuracy_by_source_family_category`

Use macro accuracy as a diagnostic and shortlist helper, not as a replacement for the primary Optuna objective unless explicitly changed later.

Do not over-focus on raw `dpo_source x category` tiny cells. They are useful for inspection, but some Prompt B raw cells are intentionally tiny.

## Length-Bias Metrics

For validation rows, log:

- `chosen_scored_len_mean`
- `rejected_scored_len_mean`
- `length_delta_mean`, where delta is `chosen_scored_len - rejected_scored_len`
- `abs_length_delta_mean`
- `margin_vs_length_delta_corr`
- `margin_vs_abs_length_delta_corr`
- `accuracy_when_chosen_longer`
- `accuracy_when_rejected_longer`
- `mean_margin_when_chosen_longer`
- `mean_margin_when_rejected_longer`

These are the main guardrails against a trial winning by verbosity or length artifacts.

## Provenance And Gates

Each trial summary should include:

- `data_hash`
- `split_manifest_sha256`
- `max_length`
- `max_observed_length`
- `mask_audit_pass`
- `only_dpo_trainable`
- `non_dpo_trainable_count`
- `trl_version`
- `length_mode`
- `effective_batch`

These explain whether a trial is comparable to the others.

## Final Study Summary

The final summary should show:

- top trials by `eval_rewards_accuracy`
- same top trials with `macro_accuracy_by_source_family_category`
- same top trials with `eval_rewards_margin`
- same top trials with length-bias metrics
- any trials with suspicious length correlation
- any trials with weak source/category buckets
- any OOM or solo-recovered trials

## Practical Review Rule

A strong trial is not just high accuracy. Prefer trials that have:

- high validation reward accuracy
- healthy reward margin
- finite and non-weird loss
- no source/category collapse
- weak or explainable length correlation
- no adapter/mask/provenance anomalies

Do not add WPO, KTO, generated eval, or manual judge metrics to this diagnostic layer. This file is only for offline Optuna diagnostics.