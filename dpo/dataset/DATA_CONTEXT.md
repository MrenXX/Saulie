# DPO V4 Data Context

This port bundle is for implementing DPO training on the accepted V4 dataset only. The source of truth for implementation decisions is `PLAN_FINAL.md`.

## Current Training Dataset

Use `DPO_522_prompt_a_and_prompt_b_V4_repaired.jsonl`.

Validated companion files:

- `DPO_522_prompt_a_and_prompt_b_V4_validation_report.json`
- `DPO_522_prompt_a_and_prompt_b_V4_repair_manifest.jsonl`

Validated counts from the V4 report:

| Field | Counts |
|---|---:|
| Total rows | 522 |
| `prompt_a` | 330 |
| `prompt_b_repaired` | 92 |
| `prompt_b_exp500` | 100 |
| `style` | 220 |
| `steering` | 121 |
| `product_fit` | 181 |

V4 validation passes. No data repair work remains unless implementation discovers a hard structural blocker, such as invalid roles, duplicate IDs, empty assistant branches, impossible tokenization, or schema drift from the validation report.

## Schema

Each JSONL row is a preference example with at least:

- `id`
- `category`
- `opening_type`
- `total_turns`
- `divergence_turn`
- `prompt`
- `chosen`
- `rejected`
- `dpo_source`

`prompt`, `chosen`, and `rejected` are chat message lists with `role` and `content` fields. Some chosen/rejected continuations contain branch-local user messages. These user messages are context only and must never be scored as model tokens.

## Training Objective

Train branch-safe DPO over full chosen/rejected trajectories:

- Keep full branch trajectories visible to attention.
- Score only assistant/action tokens.
- Never score user-role text inside chosen/rejected continuations.
- Do not implement alternate branch-ablation training paths.

## Split Requirements

Use deterministic 80/10/10 train/validation/test splits by stable row ID.

Primary stratification:

- raw `dpo_source`: `prompt_a`, `prompt_b_repaired`, `prompt_b_exp500`
- `category`: `style`, `steering`, `product_fit`

Secondary stratification:

- `opening_type` inside each `(dpo_source, category)` bucket when cell sizes allow

Write a split manifest such as `dpo/train/dataset/dpo_v4_split_seed_<seed>.jsonl` with `id`, `split`, `split_seed`, `dpo_source`, `source_family`, `category`, `opening_type`, `total_turns`, `divergence_turn`, `has_branch_local_user`, final stratum key, and allocation reason.

## Port Bundle Contents

Required files in this folder:

- `PLAN_FINAL.md`
- `DATA_CONTEXT.md`
- `DPO_522_prompt_a_and_prompt_b_V4_repaired.jsonl`
- `DPO_522_prompt_a_and_prompt_b_V4_validation_report.json`
- `DPO_522_prompt_a_and_prompt_b_V4_repair_manifest.jsonl`
- `DPO_PROMPT_A_OG.md`
- `DPO_PROMPT_B.md`

Do not use older planning or provenance files as implementation guidance. In particular, do not port old handoff plans, old training-stack plans, or old repair-audit handoffs into the implementation workspace.

## Model Assets Needed Separately

The port bundle does not include model weights or adapters. The implementation machine needs:

- `Qwen3-4B-Instruct-2507`
- `Qwen3-4B-Instruct-2507-FP8`
- `sft/models/steering-sft-v1.1/trial-17/best_adapter/`

The SFT adapter is the frozen reference behavior and must not be overwritten.
