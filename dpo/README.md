# Direct preference optimization

SFT taught the steering pattern, but it left formulaic openings, abrupt transitions, and recommendations that did not always fit the conversation. I used DPO to train on explicit contrasts between better and worse continuations.

The SFT adapter stays frozen. A separate DPO adapter trains on top of it, using SFT behavior as the reference policy.

## Preference data

The [V4 input](dataset/DPO_522_prompt_a_and_prompt_b_V4_repaired.jsonl) contains 522 pairs: 330 synthetic Prompt A pairs, 92 repaired Prompt B pairs based on SFT outputs, and 100 expanded Prompt B pairs. They target style, steering, and product fit.

The [seed-42 split](train/dataset/dpo_v4_split_seed_42.jsonl) has 418 training, 52 validation, and 52 test pairs. [Generation and repair](../docs/data.md)

In 209 rows, chosen or rejected continuations contain scripted user replies. [`dpo_data.py`](train/dpo_data.py) keeps those replies as context but scores assistant tokens only. [`dpo_trainer_compat.py`](train/dpo_trainer_compat.py) carries the masks into TRL. The builder rejects trajectories beyond 704 tokens rather than dropping turns through truncation.

## Training policy and reference

Both policies share one frozen Qwen base loaded with bitsandbytes 8-bit:

```text
reference = base + frozen SFT
policy    = base + frozen SFT + trainable DPO
```

Sharing the base avoids the memory cost of a second full model. Active adapters determine the forward path, and reference log probabilities are cached. Only the DPO parameters have gradients.

The reference is SFT behavior rather than the unadapted base because the objective is to refine the first stage without discarding it. [`train_dpo.py`](train/train_dpo.py) builds the policy; adapter diagnostics check which parameters train.

## Over-optimisation

The first studies exposed a problem with selecting on preference metrics alone. Several trials reached accuracy 1.0 but generated repetitive, ungrammatical text and occasional Chinese characters.

Preference accuracy measures the ordering of two branches. It can improve while both branches become less likely if the rejected branch falls faster. In v1.1 trial 29, margin reached 11.67 while chosen log probability fell to about -672; healthy later trials stayed around -230. That explained why a large margin was a poor selection target, though the precise cause of the language drift was not isolated.

Scaling DPO weight down at inference helped some early adapters, but it also weakened their learned behavior. I changed training and selection instead: smaller ranks, fewer epochs, a supervised anchor, and checks on absolute chosen likelihood. [Semantic review](eval/TRAIN_SETUP_SEMANTIC_REPORT.md)

## RPO anchor and selection objective

v1.5 adds a supervised loss on chosen responses:

```text
loss = preference_loss + rpo_alpha * chosen_response_sft_loss
```

This is an extra term during DPO, distinct from the original SFT phase. Its purpose is to keep chosen responses likely in absolute terms while the preference loss separates them from rejected responses. [`rpo_loss_bundle`](train/train_dpo.py) implements it with TRL's preference and `sft` losses.

The earlier v1.1 hybrid score still gave high scores to broken models. v1.5 replaced it with a custom `survival_score`. It checks preference accuracy, balanced category/source performance, style, margin, length effects, chosen log probability, and adapter/mask correctness. Its margin term favors a moderate gap instead of rewarding unlimited growth. [`evaluate_v1_5_trial`](train/dpo_diagnostics.py)

The score filters the shortlist for generation testing. It does not decide the final deployment model.

## Experiment versions

- v1.0 selected on preference accuracy.
- v1.1 introduced the hybrid score, richer diagnostics, GPU locking, and stalled-trial handling.
- v1.2 tested conservative rescue configurations on the unmerged SFT/DPO stack.
- v1.3 tested training with SFT merged into a dense base.
- v1.4 repeated the comparison on the unmerged stack.
- v1.5 returned to the unmerged stack with the RPO anchor and survival objective.

The final run, `optuna-run-20260602-052732`, completed 21 trials across 31 attempts, including eight failed and two pruned attempts. Two workers overshot the target of 20 completions by one. There were 11 unique completed parameter configurations. [Study summary](study_results/trial_summary_v1.5.json), [earlier studies](../docs/evidence/dpo-study-history.json)

## Selecting trial 4

Trial 16 had the highest training objective, `0.9798`. Trial 4 scored `0.9268`; both had validation preference accuracy 51/52.

I took six candidates into blind conversational judging. Trial 4 ranked first for both judges in round one. On the second-round steering holdout, trial 20 had a slightly higher mean, but the judges split their recommendations because one favored trial 4 on the harder opening types.

A qualitative test with people talking directly to trials 4 and 20 favored trial 4. I selected it for deployment and retained trial 20 as backup. [Judging results](../docs/evaluation.md)

Trial 4 uses one epoch, batch one with accumulation eight, beta `0.05`, learning rate `1.5e-5`, rank 16, dropout `0.05`, `rpo_alpha=1.0`, and `length_mode=ld_0.3`.

## Export for serving

The training setup uses two adapters, while the vLLM serving path accepts one per request. I combine their low-rank factors with PEFT's `cat` operation:

```text
combined_update = SFT_update + DPO_update
combined_rank   = SFT_rank + DPO_rank
```

Concatenation preserves the sum without folding the adapters into a differently quantized base. Adding the factors directly would introduce cross terms. [`merge_sft_dpo_lora.py`](train/merge_sft_dpo_lora.py) reconstructs both updates in full precision and refuses export if their maximum difference exceeds `1e-5`.

For trial 4, rank 16 plus rank 16 gives rank 32. Some other finalists have rank 40, so the evaluation server uses vLLM rank capacity 64.

Training uses bitsandbytes 8-bit; serving uses FP8 because those were the compatible paths in the training and inference environments. The merge script therefore also compares forward drift and short generations on both bases. Those diagnostics are separate from the weight-update check. [Merge validation](eval/merge_v15_slate_validation.md)

## Commands

These require the [training environment, checkpoints, and frozen SFT adapter](../REPRODUCIBILITY.md):

```bash
conda activate saulgman
python dpo/train/scripts/preflight_v1_5.py
bash dpo/train/scripts/run_v1_5_main.sh
```

Generate an HTML report from the included study summary without training:

```bash
python -m dpo.train.study_report \
  --summary dpo/study_results/trial_summary_v1.5.json \
  --output /tmp/saulie-v15-study.html
```

[Evaluation commands](eval/README.md), [deployment](../docs/deployment.md)
