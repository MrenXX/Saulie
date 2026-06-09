# Merge Script Validation Fix Plan

## Purpose

This plan narrows the merge validation work to the fast path needed for roughly 6 DPO trials.

The goal is not to build a full deployment audit. The goal is to answer, quickly and locally:

1. Did the cat merge preserve the stacked LoRA Delta-W math?
2. Does the cat adapter behave the same as the stacked adapter in the local HF/BnB runtime on a tiny smoke set?
3. Is the candidate policy itself obviously broken?

FP8/vLLM is out of scope for this merge-validation phase. Use vLLM only for the final eval/serving candidate.

## Scope

Primary script:

- `Saulie_dpo_eval/dpo/train/merge_sft_dpo_lora.py`

Related historical context:

- `dpo_eval/smoke_test/MERGE_CAT_DIAGNOSTIC_REPORT.md`
- `study_results/Merging stacked LoRA adapters for vLLM compatibility.md`

Do not copy the older strict behavior from:

- `train/implementation_v2/merge_sft_dpo_lora.py`

That older script treated forward-logit mismatch as merge failure. For this workflow, forward-logit mismatch is diagnostic only once Delta-W passes.

## Correct Lean Workflow

For each trial:

1. Load local HF/BnB8 base.
2. Load stacked adapters as `default` + `dpo`.
3. Create `sft_dpo_cat` using PEFT `combination_type="cat"`.
4. Run the Delta-W matrix check.
5. If Delta-W fails, stop and do not export.
6. If Delta-W passes, export the cat adapter.
7. Optionally run a tiny local stack-vs-cat sanity check on 6-8 prompts.
8. Run quick local generation smoke on the same tiny prompt set if the candidate is worth inspecting.

No FP8/vLLM step belongs here. The vLLM check is reserved for the final selected candidate.

## Hard Gate: Delta-W Check

Keep `verify_weight_matrices(model)` as the only hard merge/export gate.

It should reconstruct raw LoRA deltas in fp32 and compare:

```text
Delta-W_stack = scaling_sft * B_sft @ A_sft + scaling_dpo * B_dpo @ A_dpo
Delta-W_cat   = scaling_cat * B_cat @ A_cat
```

For PEFT cat merge, the identity is:

```text
[B_sft | B_dpo] [A_sft; A_dpo] = B_sft @ A_sft + B_dpo @ A_dpo
```

Decision:

- Delta-W pass: export the cat adapter.
- Delta-W fail: do not export; inspect adapter ranks, scaling, target modules, adapter names, or load path.

Keep the current tolerance around `1e-5`. Prior passing runs near `1e-9` show this is already conservative.

## Local Sanity Check: Non-Blocking

The current same-BnB forward logit check can fail even when Delta-W passes. We have seen max logit deltas around `0.75-0.88`, and an older value around `2.5`, despite a Delta-W pass near `1e-9`.

So the local sanity check should not block export. It should only answer:

- Is stack-vs-cat drift extremely suspicious?
- Are top tokens wildly different on simple prompts?
- Does the cat adapter produce obviously worse generations than the stacked adapter?

Rename or reframe `--check-logps` as a non-blocking audit. Recommended flag:

```text
--audit-forward-drift
```

Keep `--check-logps` as a compatibility alias if useful, but make the help text clear:

```text
Runs a non-blocking local HF/BnB stack-vs-cat drift check. Export is gated only by Delta-W correctness.
```

## Minimal Metrics To Keep

For this 6-trial workflow, do not build a large metric suite.

Keep only:

- `max_abs_logit_diff`
- `mean_abs_logit_diff`
- `top1_agreement`, if easy
- `top5_overlap`, if easy
- per-prompt generated output for stack and cat, if running generation smoke

Do not compute full-vocab KL. Do not build a representative DPO eval set. Do not use vLLM logprobs here.

The old `1e-3` max-logit tolerance can stay in metadata as a legacy reference, but it must not be a pass/fail merge decision.

## Tiny Prompt Set

Use 6-8 prompts, fixed across all six trials. The point is fast comparability, not broad evaluation.

Recommended categories:

1. Normal English chat: everyday assistant response.
2. Short instruction: simple transform or extraction.
3. Task-style prompt: similar to the actual DPO/eval domain.
4. Style-sensitive prompt: checks whether formatting/tone is preserved.
5. Known collapse probe: prompt that previously exposed loops, broken grammar, or token soup.
6. Refusal/safety-adjacent prompt only if relevant to the dataset.
7. Optional long-ish prompt: checks length handling without running a full eval.
8. Optional second normal English chat prompt.

No Chinese interaction prompt is needed if the model will not be used in Chinese.

If we mention Chinese at all, it should only be an automated unwanted-script detector for English prompts:

```text
Flag if an English-only response unexpectedly contains CJK characters or switches into Chinese.
```

This catches the prior language-drift failure mode without pretending Chinese chat is a real use case.

## Quick Generation Smoke

For each promising trial, generate from:

- A: `BnB8 + stacked [default,dpo]`
- B: `BnB8 + sft_dpo_cat`

Use the same prompt set and the same sampling parameters.

Recommended Qwen sampling:

```text
temperature = 0.7
top_p = 0.8
top_k = 20
repetition_penalty = 1.05
max_new_tokens = 256
```

Greedy can be used as a stress test, but do not make greedy the main decision if production/eval uses sampling.

Failure tags to record manually or automatically:

- `looping`
- `token_soup`
- `broken_grammar`
- `unexpected_cjk`
- `empty_output`
- `instruction_miss`
- `format_break`

Decision interpretation:

- A fails and B fails: likely bad policy/training candidate, not a merge issue.
- A passes and B fails: investigate cat adapter activation, adapter scaling, dtype, target modules, or chat template.
- A passes and B passes: merge/export is good enough for this stage.

## Metadata Changes

Simplify `merge_meta.json`. It does not need deploy audit fields for this phase.

Suggested shape:

```json
{
  "candidate": {
    "dpo_adapter": "...",
    "cat_adapter_name": "sft_dpo_cat",
    "stack_adapters": ["default", "dpo"]
  },
  "merge_correctness": {
    "blocking": true,
    "method": "fp32 raw LoRA Delta-W reconstruction",
    "pass": true,
    "tolerance": 0.00001,
    "max_abs_delta_diff": 1.63e-09,
    "worst_module": null
  },
  "local_forward_drift": {
    "blocking": false,
    "status": "not_run_or_measured",
    "reference": "bnb8_stack",
    "candidate": "bnb8_cat",
    "legacy_max_abs_logit_tolerance": 0.001,
    "summary": {}
  },
  "local_behavior_smoke": {
    "blocking_for_export": false,
    "status": "not_run_or_measured",
    "summary": {}
  },
  "export_decision": {
    "saved_adapter": true,
    "reason": "Delta-W merge correctness passed. Local drift/smoke checks are diagnostics."
  }
}
```

The key rule is simple:

```text
export_decision.saved_adapter = merge_correctness.pass
```

## What To Remove From The Main Plan

Do not implement these for the six-trial merge pass:

- vLLM FP8 drift audit.
- vLLM `logprobs`, `prompt_logprobs`, or `logprob_token_ids` checks.
- A 50-100 prompt representative DPO eval drift set.
- Full-vocab KL.
- Separate deployment drift audit script.
- BF16 equivalence gate.

These can be revisited only for the final selected deployment candidate, if needed.

## Implementation Steps

1. Keep `verify_weight_matrices(model)` as the hard gate.
2. Rename/reword `--check-logps` so it is clearly non-blocking, or add `--audit-forward-drift` and keep `--check-logps` as an alias.
3. Update console output so it says export is gated by Delta-W only.
4. Update `merge_meta.json` to separate `merge_correctness`, `local_forward_drift`, `local_behavior_smoke`, and `export_decision`.
5. Add support for a tiny prompt JSONL file, or keep a short embedded prompt set if faster.
6. Add simple output checks for `looping`, `empty_output`, and unexpected CJK characters in English prompts.
7. Do not add vLLM code to the merge script.

## Acceptance Criteria

The plan is complete when:

- A mathematically correct cat adapter exports even if the local logit audit exceeds `1e-3`.
- A Delta-W failure blocks export.
- The metadata and console output cannot be misread as saying local logit drift equals merge failure.
- The six-trial workflow can be run quickly with a tiny prompt set.
- vLLM is reserved for final eval/serving, not this merge sanity step.

## Final Decision Rule

For each of the 6 trials:

```text
Delta-W fail:
  reject export and debug merge/load paths.

Delta-W pass + A/B smoke both fail:
  export is mathematically okay, but the policy is probably bad.

Delta-W pass + A passes + B fails:
  export exists, but cat runtime behavior is suspicious; inspect before using.

Delta-W pass + A passes + B passes:
  keep the cat adapter as a valid local merge artifact.
```

That is enough for this stage. The final selected candidate can get the heavier vLLM evaluation later.
