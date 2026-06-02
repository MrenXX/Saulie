# DPO v1.5 Hybrid Study Plan

Date: 2026-06-02

## Decision

Run a hybrid study with 20 total trials:

- 12 fixed/enqueued expert trials.
- 8 constrained Optuna-sampled trials.

This is not a broad Optuna rerun. Optuna is useful here as a ledger, scheduler, and narrow local search after the fixed anchors. It should not be allowed to rediscover the v1.1 failure region or select winners using raw reward accuracy/margin alone.

The training surface stays the v1.4 exact stack:

```text
BnB8(raw Qwen base) + frozen SFT LoRA(default) + trainable DPO LoRA(dpo)
```

Do not switch to SFT-baked base for this study. v1.4 already showed that the exact unmerged stack can reproduce the v1.3 stable recipe.

## Portable Paths

Paths below are intentionally relative. On the training machine, interpret these from the repo checkout root.

```text
Saulie_dpo_eval/dpo/train/train_dpo.py
Saulie_dpo_eval/dpo/train/optuna_parallel.py
Saulie_dpo_eval/dpo/train/dpo_trainer_compat.py
Saulie_dpo_eval/dpo/train/study_report.py
study_results/trial_summary_v1.2.json
study_results/trial_summary_v1.3.json
study_results/trial_summary_v1.4.json
study_results/Merging stacked LoRA adapters for vLLM compatibility.md
```

If implementation is done in the mirrored `Saulie/` checkout instead of `Saulie_dpo_eval/`, use the same internal relative files under that folder.

## Why Hybrid Instead Of Broad Optuna

A 20-trial budget is still small for the number of knobs available: beta, LR, rank, RPO strength, loss type, length handling, scheduler, dropout, epochs, NEFTune, batch shape. A broad sampler can waste the budget on regions already known to be risky or uninformative.

The bigger problem is objective mismatch. v1.1 proved that excellent offline DPO metrics can still produce broken ordinary conversation. Raw `eval_rewards_accuracy`, raw margin, and old hybrid score are not enough. The study objective should be used only to select survivors for manual generation checks, not to declare the final winner.

Use Optuna only after fixed anchors are enqueued, and sample inside a narrow safe box.

## Required Preflight Before Training

Current training code passes `beta`, `loss_type`, `ld_alpha`, `neftune_noise_alpha`, and related DPOConfig fields, but it does not pass `rpo_alpha` yet. Do not spend the v1.5 trial budget until this is wired and verified.

Required plumbing:

1. Add `rpo_alpha` to the trial params and fixed/enqueued params.
2. Pass `rpo_alpha` into `DPOConfig` in `make_dpo_config`.
3. Record `rpo_alpha` in derived params, user attrs, reports, and summary JSON.
4. Verify a one-row/smoke run can instantiate `DPOConfig(rpo_alpha=...)` with the installed TRL version.
5. Fail closed if TRL rejects `rpo_alpha`; either patch/upgrade first or do not run the v1.5 study.

RPO is the point of this study. It is the chosen-response NLL anchor meant to preserve fluent SFT-like generation while preference optimization moves margins.

## Fixed Defaults

Unless a trial says otherwise, use:

```text
base_kind: bnb8_sft_lora
stack_mode: unmerged_sft_lora
active policy adapters: default + dpo
reference adapter: ref
trainable adapters: dpo only
num_train_epochs: 1
batch_combo: 1x8
lora_dropout: 0.05
lr_scheduler_type: constant_with_warmup
max_grad_norm: 0.3
neftune_noise_alpha: 0.0
length_mode: ld_0.3
loss_type: sigmoid
lora_alpha: 2 * lora_r
```

Do not include:

```text
r=8
r=32
2 epochs
NEFTune
SimPO
residual dpo_weight scaling
new normal-chat retention data
broad length-mode sampling
learning_rate > 1.85e-5
beta > 0.20 in the sampled Optuna space
```

## 12 Fixed / Enqueued Trials

Enqueue these trials first, in this order. They are the interpretable backbone of the study.

| Fixed ID | Role | beta | LR | r | length/loss | rpo_alpha |
|---:|---|---:|---:|---:|---|---:|
| 1 | v1.4 control replay | 0.05 | `1.5e-5` | 12 | `ld_0.3` | 0.0 |
| 2 | high-LR no-RPO control | 0.05 | `1.85e-5` | 16 | `ld_0.3` | 0.0 |
| 3 | light RPO | 0.05 | `1.5e-5` | 16 | `ld_0.3` | 0.25 |
| 4 | main RPO anchor | 0.05 | `1.5e-5` | 16 | `ld_0.3` | 0.5 |
| 5 | strong RPO anchor | 0.05 | `1.5e-5` | 16 | `ld_0.3` | 1.0 |
| 6 | stronger DPO with anchor | 0.08 | `1.85e-5` | 16 | `ld_0.3` | 0.75 |
| 7 | beta bridge | 0.08 | `1.2e-5` | 16 | `ld_0.3` | 0.5 |
| 8 | high-beta fair retest | 0.15 | `1.5e-5` | 16 | `ld_0.3` | 0.5 |
| 9 | r24 capacity test | 0.05 | `1.2e-5` | 24 | `ld_0.3` | 0.5 |
| 10 | r24 balanced high beta | 0.15 | `1.2e-5` | 24 | `ld_0.3` | 0.5 |
| 11 | r24 high beta + strong anchor | 0.15 | `1.2e-5` | 24 | `ld_0.3` | 1.0 |
| 12 | IPO fair probe | 0.05 | `1.5e-5` | 16 | `ipo` | 0.5 |

Why these 12:

- Trials 1-2 are no-RPO controls. They tell us whether the new code path still reproduces v1.4 and whether RPO is doing real work instead of riding normal variance.
- Trials 3-5 isolate RPO strength at the proven v1.4 anchor.
- Trial 6 repeats the stronger v1.4 LR with RPO.
- Trial 7 keeps the beta bridge that looked healthy in v1.4, but with full r16 capacity.
- Trial 8 gives beta `0.15` a fair, non-v1.2-confounded test.
- Trials 9-11 test the important new idea: larger rank balanced by higher beta and RPO anchoring. This replaces the r8 detour.
- Trial 12 gives IPO one fair shot without letting the study become an IPO search.

## 8 Constrained Optuna Trials

After the 12 enqueued trials, let Optuna sample 8 additional trials from this bounded space.

```text
beta: categorical [0.05, 0.08, 0.10, 0.15, 0.20]
learning_rate: categorical [8e-6, 1.0e-5, 1.2e-5, 1.5e-5, 1.85e-5]
lora_r: categorical [12, 16, 24]
rpo_alpha: categorical [0.25, 0.5, 0.75, 1.0]
length_mode: fixed ld_0.3
num_train_epochs: fixed 1
neftune_noise_alpha: fixed 0.0
lora_dropout: fixed 0.05
lr_scheduler_type: fixed constant_with_warmup
max_grad_norm: fixed 0.3
batch_combo: fixed 1x8
```

Conditional constraints:

```text
if lora_r == 24 and beta >= 0.15:
    learning_rate <= 1.2e-5

if beta == 0.20:
    learning_rate <= 1.2e-5

if lora_r == 12:
    beta in [0.08, 0.10, 0.15]
```

The goal of the sampled trials is local interpolation around the fixed anchors, not exploration of the whole DPO space.

Recommended Optuna settings:

```text
total_complete_trials: 20
enqueued_trials: 12
sampled_trials: 8
direction: maximize
sampler: TPESampler
seed: 42
n_startup_trials: 12
multivariate: true
group: true
constant_liar: true only if running multiple workers
```

If the existing harness only supports minimization, minimize `-v1_5_survival_score` and store the positive score in trial attrs.

## Training Loss vs Optuna Objective

Do not confuse the training loss with the Optuna objective.

Training objective for most trials:

```text
DPO sigmoid loss + ld_0.3 length debias + RPO auxiliary chosen NLL
```

Training objective for the fixed IPO probe:

```text
IPO loss + RPO auxiliary chosen NLL
```

Optuna objective:

```text
maximize v1_5_survival_score
```

The Optuna objective is not raw DPO loss, not raw reward accuracy, and not raw margin. It is a bounded survivor score designed to filter out known degenerate policies before manual REPL testing.

## Hard Gates For The Optuna Objective

Return a terrible score, or prune/fail the trial, if any hard gate fails:

```text
mask_audit_pass is false
only_dpo_trainable is false
ref adapter missing
nonfinite train_loss or eval_loss
eval_rewards_accuracy < 0.80
macro_accuracy_by_source_family_category < 0.85
macro_accuracy_by_category < 0.85
style accuracy < 0.75
eval_rewards_margin <= 0
margin_vs_length_delta_corr < -0.50
abs(margin_vs_length_delta_corr) > 0.97
eval_logps_chosen is more than 25 nats worse than the v1.5 no-RPO control anchor
```

Use the v1.5 no-RPO controls as the chosen-logp anchor. If both controls fail, stop and fix the implementation before trusting any RPO trial.

## v1.5 Survival Score

For trials that pass hard gates, maximize:

```text
score =
    0.30 * accuracy_score
  + 0.25 * macro_family_category_score
  + 0.15 * macro_category_score
  + 0.10 * style_score
  + 0.10 * margin_quality
  + 0.10 * chosen_logp_quality
  - penalties
```

Definitions:

```text
accuracy_score = clamp(eval_rewards_accuracy, 0, 1)
macro_family_category_score = clamp(macro_accuracy_by_source_family_category, 0, 1)
macro_category_score = clamp(macro_accuracy_by_category, 0, 1)
style_score = clamp(style_accuracy, 0, 1)
```

Margin quality should be bounded. Reward moderate positive margins and stop rewarding huge margins.

```text
preferred_margin_center = 1.4
preferred_margin_width = 1.4
margin_quality = clamp(1 - abs(eval_rewards_margin - preferred_margin_center) / preferred_margin_width, 0, 1)
```

Chosen-logp quality should penalize cratering chosen responses.

```text
chosen_drop = max(0, control_eval_logps_chosen - eval_logps_chosen)
chosen_logp_quality = clamp(1 - chosen_drop / 25, 0, 1)
```

Remember that logps are negative. If a trial has `eval_logps_chosen = -232` and the control is `-233`, that is not a drop. If a trial has `eval_logps_chosen = -260`, that is a serious drop.

Suggested penalties:

```text
if eval_rewards_margin > 3.0:
    penalty += 0.10

if margin_vs_length_delta_corr < -0.25:
    penalty += 0.10

if margin_vs_length_delta_corr > 0.95:
    penalty += 0.05

if eval_loss > 1.5:
    penalty += 0.05
```

These penalties are intentionally modest after hard gates. They should demote suspicious trials, not override genuinely good balanced metrics.

## Why Not Optimize Raw Accuracy Or Loss

Do not maximize only `eval_rewards_accuracy`. v1.1 already showed that high pairwise accuracy can coexist with broken open generation.

Do not maximize raw margin. Very large margin can mean the model learned to push rejected down while chosen logp also degrades, which is the failure mode described in the merge notes.

Do not minimize `eval_loss` alone. Low DPO loss can mean saturation, not usable generation.

Do not use old hybrid score as the winner selector. It can be logged for continuity, but v1.5 needs a survival score plus manual generation gates.

## Manual Evaluation Funnel

The point of the Optuna score is to reduce manual REPL burden. Do not REPL all 20 unless something unusual happens.

Stage A: train all 20 and collect offline diagnostics.

Stage B: apply hard gates and rank by `v1_5_survival_score`. Keep at most 8 survivors.

Stage C: quick REPL only for survivors.

```text
3 ordinary conversation prompts
2 assistant turns per prompt
sampling: temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.05
```

Reject immediately for:

```text
CJK leakage
broken English
phrase loops
role leakage
prompt echoing
generic question spam
loss of normal conversational competence versus SFT
```

Stage D: full REPL for the top 3-4 quick-REPL survivors.

```text
ordinary conversation prompts
product/steering skeleton smoke
same Qwen sampling settings
optional greedy stress test only after sampling pass
```

Stage E: merge and vLLM-test only the top 1-2 finalists.

```text
PEFT cat merge delta check
HF exact stack generation
vLLM FP8 cat generation
paired prompt behavior comparison
logit/top-k/KL comparison if available
```

## Required Trial Metadata

Every trial summary must record:

```text
trial_number
fixed_or_sampled
fixed_id if enqueued
beta
learning_rate
lora_r
lora_alpha
rpo_alpha
length_mode
loss_type
num_train_epochs
batch_combo
neftune_noise_alpha
base_kind
stack_mode
active_adapters
available_adapters
has_ref_adapter
only_dpo_trainable
non_dpo_trainable_count
mask_audit_pass
eval_logps_chosen
eval_logps_rejected
eval_rewards_accuracy
eval_rewards_margin
eval_rewards_chosen
eval_rewards_rejected
eval_loss
train_loss
macro_accuracy_by_category
macro_accuracy_by_source_family
macro_accuracy_by_source_family_category
style accuracy extracted from val_diagnostics_json
margin_vs_length_delta_corr
margin_vs_abs_length_delta_corr
v1_5_survival_score
saved_adapter_path
```

## Expected Readout

The most likely winner region is still:

```text
beta: 0.05 to 0.10
lora_r: 16 or 24
rpo_alpha: 0.5 to 1.0
LR: 1.2e-5 to 1.85e-5
length_mode: ld_0.3
```

The most important uncertainty is the balanced-capacity idea:

```text
r24 + beta 0.15 + RPO
```

That is now a serious candidate, not a diagnostic. It is the right way to test whether a larger adapter can learn more of the preference without damaging conversation, provided the higher beta and RPO anchor keep it near the SFT policy.

## Stop Conditions

Stop the study and inspect implementation if:

```text
fixed controls do not reproduce v1.4-like behavior
ref adapter is missing or not frozen
any non-DPO params are trainable
rpo_alpha is missing from summaries
all RPO trials crater eval_logps_chosen
sampled trials produce better offline scores but fail quick REPL repeatedly
```

If the fixed controls are sane but every RPO trial fails ordinary conversation, RPO is not the rescue in this stack. Fall back to the best v1.4-style no-RPO candidate and proceed to deploy validation rather than expanding the search indefinitely.
