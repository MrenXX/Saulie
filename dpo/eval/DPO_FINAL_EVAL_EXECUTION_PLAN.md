# DPO Final Eval Execution Plan

Date: 2026-06-02

## Goal

Pick the final deployable DPO candidate under deadline without overtrusting offline metrics, one-off REPL impressions, or merge assumptions.

The final selection must answer four questions:

1. Does the candidate beat the SFT baseline on steering behavior?
2. Does it preserve ordinary conversation ability?
3. Does the cat-merged adapter match the stacked adapter well enough locally?
4. Does the final served FP8/vLLM candidate still behave acceptably?

## Files In This Eval Packet

```text
dpo_eval/eval_skeletons.json
dpo_eval/llm_judge_prompt_dpo.md
dpo_eval/MERGE_SCRIPT_VALIDATION_FIX_PLAN.md
dpo_eval/DPO_FINAL_EVAL_EXECUTION_PLAN.md
dpo_eval/DATA_CONTEXT.md
dpo_eval/candidate_metadata_finalists.json
```

`eval_skeletons.json` now contains 60 total skeletons:

```text
52 steering skeletons: opening_type A/B/C/D
8 ordinary conversation retention skeletons: opening_type O, eval_kind ordinary_conversation
```

The ordinary items are not steering tasks. They test whether DPO preserved normal chat, fluency, and repetition stability.

## Candidate Slate

Use the SFT baseline plus six DPO candidates.

```text
SFT baseline: steering-sft-v1.1_trial-17
DPO trial 19
DPO trial 16
DPO trial 8
DPO trial 27
DPO trial 20
DPO trial 4
```

Do not include trial 25 because manual REPL found repetition. Do not include trial 9 as a finalist because manual REPL found odd grammar and its margin was high. If desired, trial 9 can be used as a judge-calibration canary only, but it should not be eligible to win unless it unexpectedly passes all behavior and merge gates.

When sending to the LLM judge, map DPO models to neutral candidate IDs in the generated transcript packet, for example `candidate_A`, `candidate_B`, etc. The prompt itself does not need a blind-scoring section, but the packet should omit trial numbers, metrics, beta, rank, RPO alpha, and human REPL notes during the scoring pass.

## Merge Gate Before Judge Generation

For each DPO candidate in the slate, run the lean local merge validation from:

```text
dpo_eval/MERGE_SCRIPT_VALIDATION_FIX_PLAN.md
```

Use:

```text
A = local HF/BnB8 + stacked adapters [default,dpo]
B = local HF/BnB8 + sft_dpo_cat
```

Decision:

```text
Delta-W fail:
  reject export and debug merge/load paths.

Delta-W pass + A/B smoke both fail:
  export is mathematically okay, but policy is likely bad. Do not send to judge unless needed as a canary.

Delta-W pass + A passes + B fails:
  cat behavior is suspicious. Inspect before judge generation.

Delta-W pass + A passes + B passes:
  use B outputs for judge generation.
```

Preferred judge-generation source for DPO candidates:

```text
B = local HF/BnB8 + cat adapter
```

This avoids selecting behavior that only exists in the separately stacked adapter path. Use the SFT baseline in its normal local HF/BnB SFT-adapter setup.

FP8/vLLM is out of scope for this pre-judge merge pass. Run vLLM only for the final top 1-2 candidates.

## Generation Settings

Use the Qwen sampling settings that fixed the greedy-loop confounder:

```text
temperature = 0.7
top_p = 0.8
top_k = 20
repetition_penalty = 1.05
max_new_tokens = 256
```

Greedy can be a stress test after a sampled pass, but it should not drive the main judge packet unless production will use greedy.

## Round 1 Evaluation Set

Round 1 uses:

```text
26 steering skeletons
8 ordinary conversation skeletons
34 total skeletons per model
```

Round 1 steering split:

```text
A: eval_A4_001, eval_A4_003, eval_A6_001, eval_A6_003, eval_A8_001, eval_A10_001
B: eval_B6_001, eval_B8_001, eval_B8_003, eval_B8_005, eval_B10_001, eval_B10_003, eval_B10_005
C: eval_C4_001, eval_C6_001, eval_C6_003, eval_C8_001, eval_C8_003, eval_C10_001
D: eval_D4_001, eval_D6_001, eval_D6_003, eval_D8_001, eval_D8_002, eval_D10_001, eval_D10_003
```

Round 1 ordinary retention set:

```text
eval_O4_001
eval_O4_002
eval_O6_001
eval_O6_002
eval_O6_003
eval_O8_001
eval_O8_002
eval_O8_003
```

Judge all models on Round 1:

```text
SFT baseline + 6 DPO candidates
```

## Round 1 Advancement Rule

Advance top 2 by default.

Advance top 3 if any of these are true:

```text
rank 2 and rank 3 mean steering scores differ by < 0.15
rank 2 and rank 3 ordinary retention scores differ by < 0.20
one model is better on Type B/D while another is better on ordinary retention
judge flags are qualitatively different enough that a human tie-break is needed
```

Do not advance a candidate with severe ordinary retention failures unless every DPO candidate fails and it is needed for diagnosis.

## Round 2 Evaluation Set

Round 2 uses the other half of the steering set plus the same ordinary retention skeletons regenerated.

```text
26 holdout steering skeletons
8 ordinary conversation skeletons regenerated with the same sampling settings
34 total skeletons per finalist
```

Round 2 steering split:

```text
A: eval_A4_002, eval_A4_004, eval_A4_005, eval_A6_002, eval_A6_004, eval_A6_005, eval_A8_002
B: eval_B6_002, eval_B8_002, eval_B8_004, eval_B8_006, eval_B10_002, eval_B10_004
C: eval_C4_002, eval_C6_002, eval_C6_004, eval_C8_002, eval_C8_004, eval_C8_005, eval_C10_002
D: eval_D4_002, eval_D6_002, eval_D6_004, eval_D8_003, eval_D8_004, eval_D10_002
```

Round 2 ordinary retention set:

```text
eval_O4_001
eval_O4_002
eval_O6_001
eval_O6_002
eval_O6_003
eval_O8_001
eval_O8_002
eval_O8_003
```

Reusing ordinary skeletons in Round 2 is intentional. Because sampling is nondeterministic, repeat-generation on the same ordinary prompts tests consistency and catches phrase loops or brittle fluency that might not appear once.

## Judge Prompt

Use:

```text
dpo_eval/llm_judge_prompt_dpo.md
```

The prompt now handles two evaluation kinds:

```text
steering: opening_type A/B/C/D, product-category recommendation expected
ordinary_conversation: opening_type O, no product recommendation expected
```

For ordinary items, the judge scores `ordinary_retention_score`, not `weighted_score` from steering dimensions.

## Judge Input Packet

For each judge round, provide:

```text
llm_judge_prompt_dpo.md
DATA_CONTEXT.md
generated conversations JSON for that round
```

Do not include training data, full Optuna DBs, implementation scripts, merge scripts, or local REPL notes in the judge packet.

For first-pass scoring, omit metrics and trial numbers from the generated conversations. Keep the SFT baseline identified as the baseline; anonymize DPO candidates.

After the judge returns scores, unblind candidate IDs locally and combine judge results with merge status.

## Judge Output Artifacts

Recommended filenames:

```text
dpo_eval/generations_round1.json
dpo_eval/judge_round1.json
dpo_eval/generations_round2.json
dpo_eval/judge_round2.json
dpo_eval/final_eval_decision.md
```

`final_eval_decision.md` should summarize:

```text
Round 1 ranking
Round 2 ranking
SFT baseline deltas
Type B/D deltas
ordinary retention results
severe audit flags
merge validation status
final selected candidate
required vLLM checks
```

## Final Deployment Gate

After Round 2, run final deployment validation only for the selected candidate, or top 2 if the result is close.

Use:

```text
A = local HF/BnB8 + stacked adapters [default,dpo]
B = local HF/BnB8 + cat adapter
C = vLLM FP8 + cat adapter
```

Minimum final checks:

```text
Delta-W pass for cat adapter
local A/B smoke pass
vLLM FP8 C sampled behavior pass on a small smoke set
no unexpected CJK in English prompts
no looping or broken grammar
no obvious regression from B to C
```

If C materially regresses relative to A/B, prune that candidate even if it won the judge on local outputs.

## Final Selection Standard

A DPO candidate can ship only if it:

1. Beats the SFT baseline on steering mean.
2. Beats or ties SFT on Type B and Type D.
3. Passes ordinary conversation retention in both rounds.
4. Avoids severe audit flags: loops, broken grammar, forced recommendations in ordinary chat, pushiness, type-B collapse, type-D blind guessing.
5. Has a valid cat merge by Delta-W.
6. Survives final vLLM FP8 sampled smoke.

If no DPO candidate clears that bar, keep the SFT baseline rather than shipping a brittle DPO adapter.

## Current Readiness

Ready to execute after these are true:

```text
eval_skeletons.json validates as JSON and has 60 items
llm_judge_prompt_dpo.md includes ordinary retention scoring
candidate transcript generation can handle opening_type O
six DPO candidates have local merge validation or are ready to generate stacked/cat outputs
Round 1 generated conversations are anonymized before judge scoring
```

No more DPO training is recommended before this eval unless every candidate fails ordinary retention or merge validation.
