# Direct preference optimization

DPO is the second training phase in the Qwen pipeline. SFT first learns conversational steering from demonstrations. DPO then trains a separate adapter to prefer better continuations while using frozen SFT behavior as its reference.

The aim was to reduce repetitive phrasing, weak or forced steering, and recommendations that ignore the user's details. Earlier DPO candidates could score well on preference accuracy and still produce broken conversations.

## Preference data

The accepted input is [`DPO_522_prompt_a_and_prompt_b_V4_repaired.jsonl`](dataset/DPO_522_prompt_a_and_prompt_b_V4_repaired.jsonl).

- Prompt A supplies 330 synthetic preference pairs.
- Repaired Prompt B supplies 92 pairs based on SFT trial-17 outputs. Local repair history starts with 93 and discards one.
- Expanded Prompt B supplies another 100 pairs.

Final categories are 220 style, 121 steering, and 181 product-fit pairs. The [seed-42 split](train/dataset/dpo_v4_split_seed_42.jsonl) has 418 training, 52 validation, and 52 test rows.

Each row contains a shared `prompt` and separate `chosen` and `rejected` message lists. In 209 rows, at least one continuation contains a user message. Some branches have different scripted user replies. These are preferences over full trajectories, not always two answers to an identical final turn.

[`dpo_data.py`](train/dpo_data.py) preserves the trajectory as context while scoring assistant tokens only. [`dpo_trainer_compat.py`](train/dpo_trainer_compat.py) carries those masks into TRL. Prompt, user, and padding tokens must not contribute to the preference score. The builder rejects overlength trajectories rather than silently truncating at the configured 704-token limit.

See [data preparation](../docs/data.md) and the [dataset contract](dataset/DATA_CONTEXT.md).

## Policy and reference

The final unmerged training setup uses:

- Qwen3-4B-Instruct-2507 loaded with bitsandbytes 8-bit quantization.
- Frozen SFT trial 17.
- A new trainable DPO LoRA added to that SFT behavior.
- A frozen SFT reference adapter, with reference log probabilities cached for comparison.

Only DPO parameters train. Diagnostics check active adapters and non-DPO trainable parameters. v1.3 instead merged SFT into a dense base for a historical comparison. That is not the final v1.5 setup.

## The SFT term inside v1.5 DPO

The original SFT phase remains a distinct completed stage. v1.5 DPO adds a supervised chosen-response term to its preference loss, controlled by `rpo_alpha`. Its purpose is to retain chosen behavior during preference training. It is not another name for the original SFT study.

[`rpo_loss_bundle`](train/train_dpo.py) builds preference and `sft` losses with weights `1.0` and `rpo_alpha`. The study explores length handling, including `ld_0.3` and IPO. These settings are separate from WPO example weighting.

## Experiment progression

- v1.0 optimized preference accuracy. Its main saved study completed 21 trials and recorded a best validation accuracy of 1.0.
- v1.1 added a hybrid ranking and source, category, and length diagnostics. GPU locking and stalled-trial handling were also revised.
- Semantic review found that full-strength early adapters could break English despite strong validation metrics. Reduced DPO weights helped on small diagnostic samples.
- v1.2 tried conservative fixed rescue configurations on the unmerged stack.
- v1.3 compared an SFT-merged dense base. v1.4 repeated the comparison on the unmerged stack.
- v1.5 returned to the unmerged stack, added the supervised anchor, and used a custom survival score before conversational evaluation.

[Project history](../docs/project-history.md), [study snapshots](../docs/evidence/dpo-study-history.json), and the [semantic review](eval/TRAIN_SETUP_SEMANTIC_REPORT.md) preserve this progression. Objectives differ, so scores across versions do not form one comparable leaderboard.

## Final v1.5 study

The run is `optuna-run-20260602-052732`. It requested 20 completions with two workers. The saved summary records 31 attempts, 21 complete, eight failed, and two pruned. Concurrent workers explain the one-completion overshoot. It records 11 unique completed configurations, not 21 distinct configurations.

The survival score combines accuracy, macro scores across source families and categories, style accuracy, margin quality, and chosen-response likelihood relative to controls. Hard gates and penalties handle some failures. The implementation is [`evaluate_v1_5_trial`](train/dpo_diagnostics.py).

The numerical winner is trial 16, with score `0.9797833556543181` and validation preference accuracy `51/52`. The deployed choice is trial 4, also `51/52`, with a lower survival score of `0.9267774472820075`.

Recovered author history records trials 4 and 20 as judge finalists, followed by a manual vLLM comparison and selection of trial 4. Structured judge scorecards were not located, so no numerical judge margin is claimed. [Selection record](../docs/evidence/selection-record.json)

Trial 4 uses one epoch, batch one with accumulation eight, beta `0.05`, learning rate `1.5e-5`, rank 16, dropout `0.05`, `rpo_alpha=1.0`, and `length_mode=ld_0.3`. [Complete training summary](study_results/trial_summary_v1.5.json)

## Export for serving

vLLM receives one adapter containing SFT and DPO contributions, produced by [`merge_sft_dpo_lora.py`](train/merge_sft_dpo_lora.py) with a PEFT `cat` combination. It does not attach two independent LoRAs to the same request.

Trial 4 combines rank 16 SFT and rank 16 DPO into rank 32. Other finalists have rank 40 and require vLLM rank capacity 64. The production launcher only needs 32 for trial 4.

The export audit checks weight-delta algebra. Stack-to-export and BnB-to-FP8 forward drift are separate diagnostics. Passing the delta check does not establish identical generated text. [Merge validation](eval/merge_v15_slate_validation.md)

## Commands

These need the [artifacts and training environment](../REPRODUCIBILITY.md). The main launcher starts a new GPU study and writes new run directories.

```bash
conda activate saulgman
python dpo/train/scripts/preflight_v1_5.py
bash dpo/train/scripts/run_v1_5_main.sh
```

Inspect a completed study without training:

```bash
python -m dpo.train.study_report \
  --summary dpo/study_results/trial_summary_v1.5.json \
  --output /tmp/saulie-v15-study.html
```

Next, follow [evaluation](eval/README.md) and [deployment](../docs/deployment.md). A training score, valid export, fluent conversation, and grounded product search are separate checks.
