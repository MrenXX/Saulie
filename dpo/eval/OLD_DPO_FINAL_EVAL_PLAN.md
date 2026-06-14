# DPO Final Evaluation Plan

Last updated: 2026-05-27

This runbook is for the next agent that implements and runs final behavioral validation for the DPO phase. The goal is to select a deployment adapter, not to run another Optuna study by inertia.

## Decision Summary

- Run final behavioral validation before starting DPO v1.2.
- Evaluate merged SFT+DPO adapters in vLLM, not raw DPO adapters.
- Use the 52-skeleton eval set in `sft_eval/eval_skeletons.json`.
- Compare every DPO candidate against the SFT trial-17 baseline.
- Do not include the raw base model as an eval candidate. The SFT model is the baseline.
- Judge Type B and Type D performance heavily, because those were the hardest steering cases.
- Use DPO metrics as metadata only. Final selection is based on generated behavior.

## Current Eval Assets

- `sft_eval/eval_skeletons.json`: 52 total skeletons, with only user turns and gaps for model responses.
- `sft_eval/llm_judge_prompt.md`: DPO final-validation judge rubric.
- `repo/Saulie/dpo/train/merge_sft_dpo_lora.py`: validated cat-merge utility for vLLM inference.

Do not blindly merge skeletons from these sources:

- `dpo2_skeletons.json`: rejected because it contains corrupted text and hidden-context leakage.
- `repo/Saulie/dpo/dpo2_prompt_b_data_augmentation_product_fit_steer_skeletons.json`: rejected because many user turns respond to invisible assistant context.

## Candidate Slate

Use an explicit manifest. Do not rely on top-N sorting inside the generation script.

Recommended final slate:

| Kind | Study | Trial | Reason |
|------|-------|-------|--------|
| SFT baseline | SFT v1.1 | 17 | Frozen SFT adapter used as the DPO starting/reference policy. Required. |
| DPO | v1.1 | 29 | Metric winner family; strong margin and clean guardrails. |
| DPO | v1.1 | 30 | Same-family stability check for trial 29. |
| DPO | v1.1 | 23 | Best alternate top family; lower VRAM and distinct scheduler/batch path. |
| DPO | v1.1 | 17 | Conservative lower-margin candidate with good length-correlation profile. |
| DPO | v1.1 | 10 | Lower-correlation diversity candidate, useful behavioral contrast. |
| DPO | v1.1 | 15 | One-epoch r16 perfect-accuracy candidate. |
| DPO | v1.1 | 16 | Higher-beta candidate, tests stronger preference pressure. |
| DPO | v1.1 | 28 | Near-top cosine/effective-batch alternate. |
| DPO | v1.1 | 13 | Low-beta diversity candidate. |
| DPO | v1.0 | 1 | v1.0 best-trial reference; include only this v1.0 candidate by default. |

If runtime must be reduced, keep: SFT trial 17, DPO v1.1 trials 29, 23, 17, 10, 15, and DPO v1.0 trial 1.

## Merge Script Validation

`repo/Saulie/dpo/train/merge_sft_dpo_lora.py` is correct for the current training setup.

Reasons:

1. Training builds policy as BnB base + frozen SFT adapter named `default` + trainable DPO adapter named `dpo`.
2. `dpo_trainer_compat.py` activates the policy stack as `[default, dpo]` using `model.base_model.set_adapter(POLICY_ADAPTER_STACK)`.
3. `train_dpo.py` saves only the DPO adapter with `selected_adapters=["dpo"]`, which is exactly what the merge script expects as `--dpo-adapter`.
4. The merge script loads the **BnB 8-bit** base (same as `train_dpo.py`), loads the frozen SFT adapter as `default`, loads the selected DPO adapter as `dpo`, validates adapter type, target modules, ranks, alpha, and `modules_to_save`, then creates one PEFT adapter with `combination_type="cat"`.
5. Cat merge is the correct inference representation for additive LoRA stacking because it preserves both low-rank deltas as one adapter with rank `r_sft + r_dpo`.
6. `--check-logps` compares the active stacked policy `[default, dpo]` to the cat adapter on fixed chat prompts and fails if max logit difference exceeds `1e-3`.
7. The output directory contains `adapter_config.json`, adapter weights, and `merge_meta.json`, which gives vLLM a single LoRA adapter to serve.

Operational caveats:

- Always run with `--check-logps` for final candidates.
- Point `--dpo-adapter` at `trial-N/best_adapter`, not at the trial directory itself.
- Use vLLM `--max-lora-rank` greater than or equal to `sft_rank + dpo_rank`. If SFT is r16 and DPO is r32, use at least 48; `64` is a safe default if vLLM accepts it.
- The merge logit check runs on the **BnB 8-bit training base** (stack `default+dpo` vs cat). vLLM serves the **FP8** checkpoint (~13GB VRAM on 3090); that is the closest supported deploy format to 8-bit training, not full BF16 weights.
- If `--check-logps` fails on any candidate, do not serve that merged adapter until the mismatch is explained.

Example merge commands on the training host:

```bash
cd /root/saulie

for trial in 29 30 23 17 10 15 16 28 13; do
  python dpo/train/merge_sft_dpo_lora.py \
    --dpo-adapter dpo/train/models/steering-dpo-v1.1/trial-${trial}/best_adapter \
    --output dpo/train/models/steering-dpo-v1.1/trial-${trial}/sft_dpo_cat \
    --check-logps
done

python dpo/train/merge_sft_dpo_lora.py \
  --dpo-adapter dpo/train/models/steering-dpo-v1.0/trial-1/best_adapter \
  --output dpo/train/models/steering-dpo-v1.0/trial-1/sft_dpo_cat \
  --check-logps
```

## Code Updates To Implement

### 1. Add Candidate Manifest Support

Create `sft_eval/dpo_final_candidate_manifest.jsonl` or `sft_eval/dpo_final_candidate_manifest.json`.

Recommended JSONL shape:

```jsonl
{"model_name":"steering-sft-v1.1_trial-17","kind":"sft_baseline","adapter_path":"/root/saulie/sft/models/steering-sft-v1.1/trial-17/best_adapter","is_sft_baseline":true,"is_dpo":false,"study_version":"sft-v1.1","trial_number":17}
{"model_name":"steering-dpo-v1.1_trial-29_sft_dpo_cat","kind":"dpo_merged","adapter_path":"/root/saulie/dpo/train/models/steering-dpo-v1.1/trial-29/sft_dpo_cat","is_sft_baseline":false,"is_dpo":true,"study_version":"v1.1","trial_number":29}
```

Populate metadata from:

- `repo/Saulie/study_v1.1_results/trial_summary.json` for DPO v1.1.
- `repo/Saulie/dpo/results/optuna-run-20260523-041252/trial_summary.json` for DPO v1.0.

Include these fields when available:

- `hybrid_score_v1_1`
- `eval_rewards_accuracy`
- `eval_rewards_margin`
- `eval_loss`
- `macro_accuracy_by_source_family_category`
- `margin_vs_length_delta_corr`
- `margin_vs_abs_length_delta_corr`
- `params`
- `derived`

### 2. Update `sft_eval/deploy_qwenie_eval.sh`

Current script scans only SFT directories and assumes rank 32. Update it to support explicit DPO final candidates.

Required behavior:

1. Accept `CANDIDATE_MANIFEST`, defaulting to `/root/saulie/sft/sft_eval/dpo_final_candidate_manifest.jsonl` or the final chosen path.
2. Mount every `adapter_path` from the manifest into `/models/lora/<model_name>`.
3. Build `--lora-modules` from the manifest `model_name=container_path` pairs.
4. Keep the engine model served as `Saulie` for LoRA hosting, but do not include raw `Saulie` in the eval manifest or generation list.
5. Set `--max-lora-rank` from `MAX_LORA_RANK`, default `64` unless merge metadata proves a lower safe value.
6. Keep `--max-loras` configurable with `MAX_LORAS`, default `2` or `4` depending on observed VRAM.
7. Print a clear startup table with model name, host path, container path, kind, study version, and trial number.
8. Fail fast if any manifest adapter path is missing.

Suggested command after update:

```bash
CANDIDATE_MANIFEST=/root/saulie/sft/sft_eval/dpo_final_candidate_manifest.jsonl \
MAX_LORA_RANK=64 \
MAX_LORAS=2 \
bash sft_eval/deploy_qwenie_eval.sh
```

After startup, verify:

```bash
curl -s http://localhost:8000/v1/models \
  -H "Authorization: Bearer ${VLLM_API_KEY:-dipshit}"
```

Every manifest model should appear. Raw `Saulie` may also appear because vLLM serves the underlying engine model, but it should not be selected for final generation or judging.

### 3. Update `sft_eval/eval_generate_vllm.py`

Current script is SFT-centric and ranks by SFT `eval_loss`. Preserve legacy `--top_n` behavior if useful, but final DPO eval should use the manifest.

Required arguments:

- `--candidate-manifest PATH`
- `--skeletons PATH`, default current 52-skeleton file
- `--output-dir PATH`
- `--output NAME`
- Optional smoke-test flags: `--limit-models`, `--limit-skeletons`, or `--filter`

Required behavior:

1. Load manifest entries and preserve their metadata.
2. Query `client.models.list()` and fail if any manifest model is missing from vLLM.
3. Evaluate exactly the manifest entries. Do not auto-add raw `Saulie` or an `--include-base` control.
4. Generate one conversation for every model x skeleton pair.
5. Store model metadata in output, including all DPO metrics copied from the manifest.
6. Store skeleton metadata: `id`, `opening_type`, `target_turns`, and user turns.
7. Store generation settings: temperature, top_p, max_tokens, base URL, timestamp, skeleton count, model count.
8. Write a deterministic, judge-ready JSON result.

Suggested full run:

```bash
python sft_eval/eval_generate_vllm.py \
  --candidate-manifest sft_eval/dpo_final_candidate_manifest.jsonl \
  --skeletons sft_eval/eval_skeletons.json \
  --output-dir sft_eval/eval_results \
  --output dpo_final_generations_52.json
```

Suggested smoke test before full generation:

```bash
python sft_eval/eval_generate_vllm.py \
  --candidate-manifest sft_eval/dpo_final_candidate_manifest.jsonl \
  --skeletons sft_eval/eval_skeletons.json \
  --output-dir sft_eval/eval_results \
  --output dpo_final_smoke.json \
  --limit-models 2 \
  --limit-skeletons 4
```

### 4. Judge Evaluation

Use `sft_eval/llm_judge_prompt.md` with the generated JSON.

Judge requirements:

1. Return valid JSON only.
2. Score all conversations with the six-dimension weighted rubric.
3. Include DPO audit flags for rejected-pair residues.
4. Report mean score by model, opening type, and target turns.
5. Report DPO deltas against the SFT trial-17 baseline.
6. Recommend one deployment model or recommend keeping SFT if DPO regresses.
7. State whether DPO v1.2 is needed and why.

Use at least two strong judges if available. If rankings disagree, inspect Type B/D cases and severe audit flags manually.

### 5. Final Report

Create a final markdown report after judging, for example `sft_eval/DPO_FINAL_EVAL_REPORT.md`.

Include:

- Candidate table with metadata and adapter path.
- Skeleton distribution table.
- Overall model ranking.
- SFT baseline delta per DPO candidate.
- Type B and Type D ranking.
- Audit flag counts.
- Best and worst skeleton examples.
- Final deployment recommendation.
- Whether to run DPO v1.2.

## Evaluation Pass Criteria

A DPO candidate is deployment-ready only if it satisfies all of these:

1. Mean weighted score beats SFT baseline.
2. Type B and Type D means beat or tie SFT baseline.
3. No severe repeated `premature_recommendation`, `visible_sales_pivot`, or `pushy_or_overconfident` flags.
4. Final turns usually recommend one generic product category, not brands and not lists.
5. Tone remains sharp and conversational without hollow praise.
6. The recommendation references specific user details.
7. No safety-sensitive skeleton receives inappropriate product steering.

Run DPO v1.2 only if final behavioral validation shows a systematic issue worth optimizing, such as:

- High-margin candidates are pushy or verbose.
- Low-correlation candidates are safer but under-steer.
- Type B factual openers still collapse into Q&A.
- Type D vague complaints get blind guesses.
- The SFT baseline beats DPO on hard skeletons.
- Prompt-B rejected-pair residues remain common.

## Sanity Checks

Before full generation:

1. Validate skeleton JSON has 52 entries.
2. Validate each skeleton has `id`, `opening_type`, `target_turns`, and `user_turns`.
3. Validate `len(user_turns) == target_turns / 2`.
4. Validate opening type distribution is intentional.
5. Validate all candidate manifest adapter paths exist.
6. Validate every merged adapter has `adapter_config.json`, adapter weights, and `merge_meta.json`.
7. Validate `merge_meta.json` has `pass: true` and `logit_check.pass: true` when `--check-logps` was used.
8. Validate vLLM `/v1/models` lists every manifest model name.
9. Run a 2-model x 4-skeleton smoke generation.
10. Send the smoke output to the judge prompt and confirm valid JSON before the full run.

## Expected Final Flow

```text
1. Merge selected DPO adapters with --check-logps.
2. Create candidate manifest with the SFT baseline and merged DPO candidates.
3. Update deploy_qwenie_eval.sh for manifest-driven LoRA loading.
4. Start vLLM and verify /v1/models.
5. Update eval_generate_vllm.py for manifest-driven generation.
6. Run smoke generation.
7. Judge smoke output and confirm schema.
8. Run full 52-skeleton generation.
9. Judge full output with DPO rubric.
10. Write final report and choose deployment adapter or justify v1.2.
```

## Notes For The Agent

- Keep edits scoped to `sft_eval/` and DPO eval plumbing unless a merge/check failure reveals a real bug.
- Do not retrain models as part of this eval pass.
- Do not serve unmerged DPO adapters in vLLM for final judging.
- Do not rank candidates by DPO validation metrics alone.
- Preserve generated outputs and judge results with timestamps so rankings are reproducible.