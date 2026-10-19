# DPO training invariants

The trainer checks adapter state, assistant-only masks, trajectory lengths, and reference behavior before using the results to rank a trial.

## Adapter behavior

The final unmerged setup activates frozen SFT plus trainable DPO for the policy. TRL's frozen reference represents SFT behavior. Only DPO parameters should have gradients. Diagnostics in [`dpo_trainer_compat.py`](dpo_trainer_compat.py) check the adapters and trainable parameter groups.

The SFT-merged dense-base path was a separate v1.3 experiment. [`model_load.py`](model_load.py) and [`train_dpo.py`](train_dpo.py) retain that path for historical comparisons.

## Trajectory scoring

[`dpo_data.py`](dpo_data.py) preserves full chosen and rejected trajectories while scoring assistant tokens only. User-role text, prompt tokens, headers, and padding must remain unscored. Each branch needs scored assistant content.

The builder audits masks on ordinary, multi-turn, long, and branch-local-user examples. It checks sequence length before training. The configured DPO limit is 704 tokens; overlength examples fail rather than silently losing turns.

## Loss and diagnostics

Length modes map to their intended TRL losses. `sigmoid_norm` is not replaced with WPO weighting. `ld_0.3` uses sigmoid with `ld_alpha=0.3`; IPO uses its own loss. v1.5 adds an `sft` chosen-response term through `rpo_alpha`.

[`dpo_report.py`](dpo_report.py) builds dummy-run checks for masks, lengths, trainable parameters, finite metrics, memory, and adapter output. The checks are implementation diagnostics, not a conversational quality certificate.

The [metrics guide](Diagnostic_Metrics.md) explains preference accuracy, bucket scores, length correlations, and provenance. [The DPO overview](../README.md) describes how objectives changed across studies.

## GPU work and recovery

The GPU lock covers construction, training, saving, validation diagnostics, and cleanup. Watchdogs, progress metadata, and Optuna heartbeats support recovery from stalled work. The [postmortem](docs/FAILED_TRIAL_POSTMORTEM.md) records why training-only locking was insufficient.

## Export

[`merge_sft_dpo_lora.py`](merge_sft_dpo_lora.py) exports the combined SFT and DPO update. Weight-delta correctness is checked separately from forward drift and generated behavior. [Finalist validation](../eval/merge_v15_slate_validation.md)
