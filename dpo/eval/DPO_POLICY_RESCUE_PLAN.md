# DPO Policy Rescue Plan

## Mission For The Local Inference Agent

Find a coherent deployable DPO policy without retraining first. Use weighted DPO residuals over the frozen SFT baseline, then export only passing weighted cat adapters for vLLM.

Do not run final 52-skeleton judge eval until at least one candidate passes HF policy-stack smoke and vLLM scaled-cat smoke.

## Diagnosis To Preserve

The existing trial-29 `sft_dpo_cat` outputs must not be scored. The diagnostic report shows the real HF policy stack `base + default + dpo` already fails before vLLM or cat merge are involved.

The likely failure is an over-strong or brittle DPO residual, not Chinese dataset contamination and not a simple cat-export bug:

- SFT-only generation is coherent English.
- DPO-only generation is coherent English but is not the deployable policy.
- True policy stack `[default, dpo]` flips into weird English and Chinese at full weight.
- Cat baked and unbaked reproduce the same behavior.
- Trial-29 has a very high validation margin, so reward accuracy selected an adapter that may be too aggressive for open-loop generation.

The architecture is still valid. PEFT `cat` is mathematically the right way to represent `delta_sft + weight * delta_dpo` as one larger LoRA when both adapters target the same base modules. The current problem is candidate behavior at weight `1.0`, not the general feasibility of SFT+DPO cat export.

## Candidate Slate

Test exactly these three DPO candidates first.

| Candidate | Path | Why this candidate |
|---|---|---|
| v1.1 trial-29 | `dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-29/best_adapter` | Metric winner, r32, 2 epochs, margin 11.67. Use as primary rescue target. |
| v1.1 trial-10 | `dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-10/best_adapter` | Most different conservative residual: r8, no NEFTune, lower margin 2.79, `ld_0.5`. Tests whether small capacity avoids collapse. |
| v1.0 trial-1 | `dpo/train/models/steering-dpo-v1.0/optuna-run-20260523-041252/trial-1/best_adapter` | Separate study lineage: r16, 1 epoch, margin 2.08, `constant_with_warmup`, `ld_0.3`. Tests whether the older lower-margin search found a gentler residual. |

Use v1.1 trial-17 as the backup if the v1.0 trial-1 adapter path is unavailable on the inference box. Its path is `dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-17/best_adapter`.

## Weight Grid

Use these DPO weights:

```text
0.00, 0.10, 0.25, 0.50, 0.75, 1.00
```

Interpretation:

- `0.00` is an SFT-equivalent control. It should be coherent for every trial path because the DPO residual is suppressed.
- `1.00` is the original intended policy weight.
- `0.10` and `0.25` are the most likely rescue weights if the residual is useful but too strong.
- `0.50` and `0.75` test whether partial DPO strength can be retained.

Do not export a cat adapter for a weight until the HF policy-stack smoke passes.

## Stage 1: HF Greedy Smoke

Run greedy decoding first. This separates model instability from sampling noise.

```bash
cd /root/saulie

declare -A TRIALS
TRIALS[trial29]=dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-29/best_adapter
TRIALS[trial10]=dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-10/best_adapter
TRIALS[trial1_v10]=dpo/train/models/steering-dpo-v1.0/optuna-run-20260523-041252/trial-1/best_adapter

for TRIAL in trial29 trial10 trial1_v10; do
  for W in 0.00 0.10 0.25 0.50 0.75 1.00; do
    python dpo/train/smoke_policy_stack_hf.py \
      --dpo-adapter "${TRIALS[$TRIAL]}" \
      --adapter-mode policy \
      --dpo-weight "$W" \
      --decode greedy \
      --skeleton-ids eval_A4_001,eval_B8_001 \
      --output "dpo/eval/smoke_${TRIAL}_w${W}_greedy.json"
  done
done
```

Pass criteria for a weight:

- All assistant turns are coherent English.
- No CJK characters.
- No repeated pattern like `not X, is Y` as a dominant template.
- No run-on pseudo-product language.
- The model remains grounded in the user turn.
- It asks reasonable clarifying questions before landing a recommendation.

Immediate fail criteria:

- Any Chinese or non-English drift.
- Broken grammar that was not present in SFT-only output.
- Collapse into abstract object/pocket/friction wording unrelated to the user.
- Repetition across turns that makes the conversation unusable.

## Stage 2: HF Sample Smoke

Take every candidate weight that passed greedy and run sampled decoding. Include one lower neighboring weight if the first pass is borderline.

```bash
python dpo/train/smoke_policy_stack_hf.py \
  --dpo-adapter dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-29/best_adapter \
  --adapter-mode policy \
  --dpo-weight 0.25 \
  --decode sample \
  --skeleton-ids eval_A4_001,eval_B8_001,eval_C6_001,eval_D6_001 \
  --output dpo/eval/smoke_trial29_w025_sample4.json
```

Repeat the command for each passing trial/weight pair. Do not sample every failed greedy combination.

## Stage 3: Choose Export Candidates

Pick at most three weighted candidates for vLLM export:

1. Best trial-29 weight if any passes.
2. Best trial-10 or v1.0 trial-1 weight, whichever is more coherent and least verbose.
3. Optional second alternate if it is clearly different in behavior.

Prefer the lowest weight that shows a visible DPO improvement over SFT while remaining fully coherent. A smaller stable improvement is better than a stronger adapter with even one language-collapse sign.

## Stage 4: Export Weighted Cat Adapters

Export only HF-passing weighted candidates.

Example for trial-29 at weight `0.25`:

```bash
python dpo/train/merge_sft_dpo_lora.py \
  --dpo-adapter dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-29/best_adapter \
  --dpo-weight 0.25 \
  --output dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-29/sft_dpo_cat_w025 \
  --check-logps
```

Example for trial-10 at weight `0.50`:

```bash
python dpo/train/merge_sft_dpo_lora.py \
  --dpo-adapter dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-10/best_adapter \
  --dpo-weight 0.50 \
  --output dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-10/sft_dpo_cat_w050 \
  --check-logps
```

Example for v1.0 trial-1 at weight `0.50`:

```bash
python dpo/train/merge_sft_dpo_lora.py \
  --dpo-adapter dpo/train/models/steering-dpo-v1.0/optuna-run-20260523-041252/trial-1/best_adapter \
  --dpo-weight 0.50 \
  --output dpo/train/models/steering-dpo-v1.0/optuna-run-20260523-041252/trial-1/sft_dpo_cat_w050 \
  --check-logps
```

The weight-matrix check must pass. The BnB forward logit check can remain informational because the HF generation smoke is the hard behavior gate.

## Stage 5: Build vLLM Manifest

Create `dpo/eval/dpo_weighted_rescue_manifest.jsonl` with SFT baseline plus each exported weighted candidate.

Example:

```jsonl
{"model_name":"steering-sft-v1.1_trial-17","kind":"sft_baseline","adapter_path":"/root/saulie/sft/models/steering-sft-v1.1/trial-17/best_adapter","is_sft_baseline":true,"is_dpo":false}
{"model_name":"steering-dpo-v1.1_trial-29_sft_dpo_cat_w025","kind":"dpo_scaled_cat","adapter_path":"/root/saulie/dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-29/sft_dpo_cat_w025","is_sft_baseline":false,"is_dpo":true,"study_version":"v1.1","trial_number":29,"dpo_weight":0.25}
{"model_name":"steering-dpo-v1.1_trial-10_sft_dpo_cat_w050","kind":"dpo_scaled_cat","adapter_path":"/root/saulie/dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-10/sft_dpo_cat_w050","is_sft_baseline":false,"is_dpo":true,"study_version":"v1.1","trial_number":10,"dpo_weight":0.50}
{"model_name":"steering-dpo-v1.0_trial-1_sft_dpo_cat_w050","kind":"dpo_scaled_cat","adapter_path":"/root/saulie/dpo/train/models/steering-dpo-v1.0/optuna-run-20260523-041252/trial-1/sft_dpo_cat_w050","is_sft_baseline":false,"is_dpo":true,"study_version":"v1.0","trial_number":1,"dpo_weight":0.50}
```

Use `MAX_LORA_RANK=64` for r48 candidates. Trial-10 is lower rank, but keep 64 so one deploy config can serve all exported candidates.

## Stage 6: vLLM Smoke

Run the same two-skeleton smoke through vLLM. The weighted cat must not regress relative to the HF stack smoke.

```bash
MAX_LORA_RANK=64 MAX_LORAS=4 \
CANDIDATE_MANIFEST=/root/saulie/dpo/eval/dpo_weighted_rescue_manifest.jsonl \
bash dpo/eval/vllm_scripts/deploy_qwenie_eval.sh

python dpo/eval/vllm_scripts/eval_generate_vllm.py \
  --candidate-manifest dpo/eval/dpo_weighted_rescue_manifest.jsonl \
  --skeletons dpo/eval/eval_skeletons.json \
  --skeleton-ids eval_A4_001,eval_B8_001 \
  --output-dir dpo/eval/weighted_rescue_vllm_smoke
```

If vLLM fails but HF stack passed, inspect vLLM rank/config/export. If both fail, reject that weighted candidate.

## Stage 7: Mini Eval Before Full Judge

For vLLM-passing candidates, run a 12-skeleton mini eval before the full 52:

```text
eval_A4_001, eval_A6_001, eval_A8_001,
eval_B6_001, eval_B8_001, eval_B10_001,
eval_C4_001, eval_C6_001, eval_C8_001,
eval_D4_001, eval_D6_001, eval_D8_001
```

Compare each weighted DPO candidate against SFT trial-17. Promote only candidates that are coherent, English, and visibly improve steering or recommendation landing without becoming pushy.

## Final Gate

Proceed to the full 52-skeleton judge eval only if at least one weighted candidate clears:

1. HF greedy smoke.
2. HF sampled smoke.
3. vLLM weighted-cat smoke.
4. 12-skeleton mini eval.

If no weighted candidate clears these gates, stop and change strategy. The next strategy should be retraining with smaller movement and an open-loop smoke gate, not continuing to optimize pairwise reward accuracy alone.

## If Weighted Rescue Fails

Do not conclude that DPO is useless. Conclude that this two-adapter training/selection recipe did not produce a stable deployable residual.

Recommended next strategy:

- Run v1.2 with lower movement: beta `0.1` or `0.2`, one epoch, LR `5e-6` to `1.5e-5`, LoRA r `8` or `16`.
- Stop ranking by reward accuracy alone. Penalize excessive reward margin and require open-loop English smoke.
- Keep the frozen SFT reference adapter guard. DPO must compare against SFT, not raw base.
- Consider folding SFT into a BF16 base and training/deploying one DPO LoRA over that SFT-base if the two-adapter composition remains operationally fragile.
- Only consider a full-model export or re-quantized SFT+DPO base after LoRA rescue and conservative DPO both fail.

## Code Changes Available In This Branch

- `dpo/train/smoke_policy_stack_hf.py` supports `--adapter-mode`, `--dpo-weight`, `--decode`, and configurable generation lengths.
- `dpo/train/merge_sft_dpo_lora.py` supports scaled cat export via `--dpo-weight` and records the weight in `merge_meta.json`.
- `dpo/train/dpo_trainer_compat.py` records `available_adapters` and `has_ref_adapter`, and can fail closed when the frozen SFT reference adapter is missing.
- `dpo/train/train_dpo.py` calls the reference-adapter guard immediately after trainer initialization.