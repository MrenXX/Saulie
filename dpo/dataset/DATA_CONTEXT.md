# DPO dataset contract

The active input is [`DPO_522_prompt_a_and_prompt_b_V4_repaired.jsonl`](DPO_522_prompt_a_and_prompt_b_V4_repaired.jsonl). [`dpo/train/paths.py`](../train/paths.py) defines this file as the training source. Older V2 metadata is historical provenance.

The file has 522 unique IDs: 330 `prompt_a`, 92 `prompt_b_repaired`, and 100 `prompt_b_exp500`. Categories are 220 style, 121 steering, and 181 product fit.

## Schema and scoring

Rows contain `id`, `category`, `opening_type`, `total_turns`, `divergence_turn`, `prompt`, `chosen`, `rejected`, and `dpo_source`. Each message has a role and content.

The prompt and both continuations remain visible as conversational context. Only assistant tokens contribute to the preference score. Branch-local user text must remain unscored. There are 209 rows with a user message in a continuation.

The [data builder](../train/dpo_data.py) supplies stable splits and assistant-only masks. It rejects trajectories beyond the configured length limit instead of silently truncating them. The [compatibility trainer](../train/dpo_trainer_compat.py) carries those masks into TRL.

## Validation and splits

- [Validation report](DPO_522_prompt_a_and_prompt_b_V4_validation_report.json).
- [Repair manifest](DPO_522_prompt_a_and_prompt_b_V4_repair_manifest.jsonl).
- [Repair utility](repair_dpo_522_v4.py).
- [Seed-42 split manifest](../train/dataset/dpo_v4_split_seed_42.jsonl), with 418 training, 52 validation, and 52 test rows.

The V4 data hash prefix is `f3b9ade4dc53`. The split manifest SHA-256 is `75d154f4c40d34efc8be6205009671290cf913c9b0b1b564880dcc835ee71b9d`. These match the saved final study.

## Authoring and provenance

[Prompt A](DPO_PROMPT_A_OG.md) defines the synthetic-pair specification. [Prompt B](DPO_PROMPT_B.md) defines SFT-output-based pairs. The [full data guide](../../docs/data.md) explains generation, repairs, and source counts.

Training needs external Qwen checkpoints and the frozen SFT trial-17 adapter. No weights are included in this dataset directory. See [SFT](../../sft/README.md), [DPO](../README.md), and [reproducibility](../../REPRODUCIBILITY.md).
