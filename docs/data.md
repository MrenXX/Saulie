# Data preparation and provenance

Saulie uses SFT demonstration conversations, DPO preference trajectories, and product catalog records for retrieval. Catalog products are not the SFT or DPO training corpus.

The [provenance export](evidence/data-provenance.json) records parsed counts and hashes from the October 19, 2026 audit. Source labels such as `windows/DATA` identify historical folders, not required paths on a new machine.

## Early Windows conversations

`saulgman_sft` contains generation prompts, teacher-named batches, four Llama training script versions, and merged datasets. Their creation and modification times place this work mainly in August through October 2025. A later repaired file appears in March 2026.

The scripts point to Llama-3.2-3B-Instruct. Teacher filenames include Gemini, GPT, Claude, DeepSeek, and Grok labels. Names alone do not verify the exact generating model or API version.

- `merged_data_v1_560exps.jsonl` parses as 559 surviving records.
- `merged_data_v1.1_570exps.jsonl` has 570 records.
- `fixed_dataset_512.jsonl` has 512 records.

These artifacts do not prove that every record or script contributed to the final Qwen model. [Legacy evidence](evidence/legacy-windows.json)

## Qwen SFT curation

Windows `DATA` preserves corrected source batches, intermediate files, and versions. The audit counted 366 records in `SFT_DATA_V1.jsonl` and 500 in `SFT_DATA_V2.jsonl`. Current WSL `sft/SFT_DATA_V3.jsonl` also has 500.

Current V3 opening types are A: 186, B: 131, C: 89, and D: 94. Rows contain messages and metadata for stratification. The [SFT guide](../sft/README.md) explains the historical 400/50/50 split sizes.

Exact parsed-row comparison finds 379 V2 rows in current V3. This confirms overlap and edits. It does not prove an uninterrupted training lineage or identify historical training bytes.

March MLflow studies log SHA-256 prefix `ced5b49dd3d0`. Current V3 has this SHA-256:

```text
958f1a0131d2b1de9dcd0fdea64d134725383decf8199714373ad5f66748f8c3
```

An exact match for the historical hash was not recovered from the supplied dataset locations. The SFT training file remains outside Git. Do not silently substitute the current file when reporting or reproducing trial 17.

## DPO generation and repair

Prompt A defines synthetic pairs across style, steering, and product fit. Local Part A preserves the initial 330 pairs, category-specific repairs, pivot-diversity work, and validation reports. Its repaired source has 111 style, 110 steering, and 109 product-fit rows.

Prompt B uses frozen SFT trial-17 behavior to mine weaker continuations and construct preferred alternatives. The local repair set has 93 original pairs and 92 accepted pairs after one discard. That subset is heavily style-focused. Another 100-pair source expands product-fit and steering coverage.

The accepted V4 dataset combines:

- `prompt_a`: 330.
- `prompt_b_repaired`: 92.
- `prompt_b_exp500`: 100.
- Total: 522. Categories: 220 style, 121 steering, 181 product fit.

`exp500` is a source label, not the final pair count. Older V2 metadata is historical provenance; V4 is the active input in [`dpo/train/paths.py`](../dpo/train/paths.py).

Public artifacts include [V4 data](../dpo/dataset/DPO_522_prompt_a_and_prompt_b_V4_repaired.jsonl), the [repair manifest](../dpo/dataset/DPO_522_prompt_a_and_prompt_b_V4_repair_manifest.jsonl), [validation report](../dpo/dataset/DPO_522_prompt_a_and_prompt_b_V4_validation_report.json), [repair utility](../dpo/dataset/repair_dpo_522_v4.py), and [generation specifications](../dpo/dataset/DATA_CONTEXT.md).

## Multi-turn preference semantics

`prompt`, `chosen`, and `rejected` are message lists. In 209 rows, a branch includes a user reply. Branches may follow different scripted replies, so preference concerns complete counterfactual trajectories.

User messages stay visible to attention but receive zero scored tokens. Assistant messages receive the preference score. [`dpo_data.py`](../dpo/train/dpo_data.py) builds and audits the masks. [Training invariants](../dpo/train/DPO_IMPLEMENTATION_FIXES.md)

The deterministic [split manifest](../dpo/train/dataset/dpo_v4_split_seed_42.jsonl) assigns every stable ID once, with 418/52/52 train/validation/test rows. The training summary's data prefix `f3b9ade4dc53` and full split hash match the tracked files.

## Evaluation data

Earlier [SFT scenarios](../sft/sft_eval/eval_skeletons.json) support the saved 11-model, 14-scenario comparison. Later [DPO scenarios](../dpo/eval/eval_skeletons.json) contain 60 scenarios, including ordinary conversation controls. These are distinct evaluation sets.

The DPO protocol generates assistant turns against fixed user turns. It measures behavior under scripted conditions, not an interactive user study. [Evaluation details](evaluation.md)

## Product catalogs

Retrieval first used an Indian Amazon CSV and later compared it with McAuley's Amazon Reviews 2023 metadata. Preparation filters ratings, review counts, missing prices, duplicate IDs, and incomplete text. Category, title, and features form the embedding text.

Saved documentation reports roughly 114,000 Indian and 443,000 McAuley records after cleaning. These are approximate historical collection sizes, not a live Qdrant count. Prices and availability are metadata snapshots.

[The RAG guide](../rag/README.md) covers preparation, indexing, and validation. The upstream [Amazon Reviews 2023 project](https://amazon-reviews-2023.github.io/) supplies the McAuley metadata. External data and model terms are separate from the code license.

## Published audit evidence

The overhaul adds SFT summaries, selected run metadata, SFT fixtures and outputs, and the completed base/SFT/DPO comparison. The [export manifest](evidence/export-manifest.json) records hashes of copied files.

Raw Cursor sessions, credentials, weights, full tracking stores, catalog storage, and discarded drafts remain local. The [audit](documentation-audit.md) lists remaining gaps.
