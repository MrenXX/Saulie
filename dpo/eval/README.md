# DPO Eval Judge Packet

This folder is the compact context packet for a SoTA LLM that will score generated outputs from the final DPO candidate trials. It is not for the coding agent that implements the vLLM/eval harness changes.

## Upload These To The Judge

1. `llm_judge_prompt.md`
   - Use this as the main scoring instruction/rubric.
   - It frames the comparison as SFT baseline vs DPO candidates only.

2. `eval_skeletons.json`
   - The 52 validation skeletons used to generate the conversations.
   - Each item contains only user turns; assistant turns come from the evaluated model outputs.

3. `candidate_metadata_finalists.json`
   - Compact metadata for the selected DPO finalists and the SFT baseline.
   - Metrics are diagnostic context only. The judge should rank by generated behavior.

4. `DATA_CONTEXT.md`
   - Compact context for the DPO V4 dataset, split objective, branch-safe scoring rules, and SFT adapter role.
   - This is enough dataset context for scoring. Do not upload the full 522-row training dataset unless you are explicitly asking the judge to audit training-data construction.

5. The generated conversations JSON from the eval run.
   - This file is not in the packet yet because it has to be produced by vLLM generation.
   - When available, add it to this folder as something like `dpo_final_generations_52.json` before sending to the judge.

## Baseline Policy

Use SFT trial 17 as the only behavioral baseline. Do not include raw base-model outputs in the judging set.

The raw model name `Saulie` may appear in vLLM infrastructure because it is the served engine model, but it should not be judged as a candidate. Final scoring should compare:

- `steering-sft-v1.1_trial-17`
- merged SFT+DPO candidates such as `steering-dpo-v1.1_trial-29_sft_dpo_cat`

## What The Judge Should Decide

The judge should select the final deployment adapter, or recommend keeping the SFT baseline if DPO introduces regressions.

The key behavioral question is not whether a DPO trial had the best Optuna metric. The question is whether it improves preference quality over SFT without causing:

- premature recommendations
- visible sales pivots
- generic filler questions
- verbose padding
- pushy or overconfident closes
- Type B factual-opener collapse
- Type D vague-complaint guessing
- weak final recommendations that ignore user details

## Not Needed For Scoring

- Full DPO training JSONL dataset
- raw base model outputs
- implementation scripts
- vLLM deploy script
- merge script
- full Optuna databases

Those are useful for implementation/debugging, but they are noise for the judge.