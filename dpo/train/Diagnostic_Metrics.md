# DPO diagnostic metrics

Training records preference metrics and source/category slices to expose failures hidden by one aggregate score. The current computation is in [`dpo_diagnostics.py`](dpo_diagnostics.py); summaries are built by [`optuna_parallel.py`](optuna_parallel.py).

## Trial metrics

`eval_rewards_accuracy` measures how often the preferred branch beats the rejected branch. Margin measures the reward separation. Chosen/rejected rewards and log probabilities, evaluation loss, and training loss help distinguish relative ranking from changes in absolute behavior.

These metrics are not conversation success rates. The original validation set has only 52 pairs.

## Sources and categories

Diagnostics report counts, accuracy, and mean margin by raw source, grouped source family, category, and source-family/category combination. Macro averages give buckets more equal weight than a pooled score. Tiny buckets remain noisy.

The role of those values depends on the study. v1.0 selects accuracy, v1.1 uses a hybrid objective, and v1.5 includes macro scores in a survival objective. [Study progression](../README.md)

## Length checks

The implementation records mean scored lengths, chosen-minus-rejected length differences, margin/length correlations, and separate accuracy where each branch is longer. These detect possible length shortcuts, not proof of their cause.

## Provenance and run health

Saved metadata includes data and split hashes, mask-audit status, sequence lengths, adapter diagnostics, TRL version, effective batch size, runtime, and peak memory where available. Failed or pruned work should be distinguished from completed trials, and repeated configurations from unique configurations.

The final [v1.5 summary](../study_results/trial_summary_v1.5.json) retains the underlying records. [Evaluation](../../docs/evaluation.md) explains the separate behavioral and deployment checks.
