# Data generation and preparation

I used separate datasets for conversational behavior and product retrieval. SFT learns from complete dialogues, DPO learns from preferred and rejected continuations, and the product catalog is queried during inference.

## SFT conversations

I generated about 500 dialogues with several language models using a common specification for the task, persona, opening types, and conversation length. Multiple generators provided variation, but also produced similar examples independently, so deduplication mattered before splitting the data.

The dialogues cover four kinds of opener:

- A direct problem, which gives the clearest route to a product category.
- An unrelated or factual question, where the model must find a relevant transition.
- An emotional or lifestyle situation, where the recommendation depends on personal context.
- A vague complaint, where the model needs to clarify the problem first.

Conversation lengths range from four to sixteen turns. The generation target gives longer budgets to harder openings, with roughly 15% four-turn dialogues, 40% six-turn, 30% eight-turn, and 15% ten turns or more.

### Generation rules

The user never explicitly asks for a product. The base model can already answer those requests; the data is meant to teach the harder transition from a situation to a recommendation.

The assistant recommends a generic category in the final message. Keeping brands, prices, and links out of the training target avoids teaching product recall that can go stale. The retrieval layer supplies those details later. Post-recommendation follow-up dialogues were outside the initial dataset's scope.

User and assistant turns alternate strictly. Openings must vary, and the style specification excludes emojis and em dashes. These rules help keep the small dataset focused on the intended behavior.

### Repetition and leakage checks

I checked generated material against an exclusion list and against previously accepted examples. The generation checks used RapidFuzz opening-similarity thresholds of 75 against the exclusion corpus and 80 against accepted peers. Additional `SequenceMatcher` checks flagged similar conversations and product wording for review.

The data went through three curation versions. The current V3 copy has 500 rows, with opening types A: 186, B: 131, C: 89, and D: 94. Training uses an 80/10/10 split stratified by opening type and length bucket. The SFT runs used 400 training, 50 validation, and 50 test conversations.

### Earlier datasets

The initial Llama experiments used `merged_data_v1_560exps.jsonl` and `merged_data_v1.1_570exps.jsonl`. The retained copies contain 559 and 570 rows respectively. A later repaired copy contains 512. Qwen data preparation produced 366-row V1 and 500-row V2 sets before V3.

These versions are recorded in the [dataset manifest](evidence/data-provenance.json). They are separate iterations rather than one dataset to concatenate. [SFT training](../sft/README.md)

## DPO preference pairs

The preference set targets three problems left after SFT: formulaic style, weak steering, and poor product fit. It combines fully synthetic comparisons with the SFT model's actual mistakes.

- Prompt A generates 330 synthetic pairs with a clear difference between chosen and rejected behavior.
- Prompt B uses SFT trial-17 responses as the rejected side and a better rewrite as chosen. Repair retained 92 of the original 93 pairs.
- Another 100 Prompt B pairs expand product-fit and steering coverage.

The final 522 pairs contain 220 style, 121 steering, and 181 product-fit examples. The source label `prompt_b_exp500` refers to the expanded generation run; its contribution to the final set is 100 pairs.

The active input is [V4 repaired data](../dpo/dataset/DPO_522_prompt_a_and_prompt_b_V4_repaired.jsonl). The [repair manifest](../dpo/dataset/DPO_522_prompt_a_and_prompt_b_V4_repair_manifest.jsonl), [validation report](../dpo/dataset/DPO_522_prompt_a_and_prompt_b_V4_validation_report.json), and [repair script](../dpo/dataset/repair_dpo_522_v4.py) describe its processing. [Dataset contract and generation prompts](../dpo/dataset/DATA_CONTEXT.md)

### Multi-turn continuations

Each row contains a shared `prompt` and separate `chosen` and `rejected` message lists. In 209 rows, a continuation includes a user reply. Some branches follow different scripted replies because the comparison concerns where the conversation leads.

User replies remain visible as context but contribute no scored tokens. Only assistant turns contribute to the preference objective. [`dpo_data.py`](../dpo/train/dpo_data.py) builds and checks those masks.

The [seed-42 split](../dpo/train/dataset/dpo_v4_split_seed_42.jsonl) assigns 418 pairs to training, 52 to validation, and 52 to test, stratified by source and correction category.

## Evaluation scenarios

Evaluation uses conversation skeletons containing only user turns. Each model fills in its own assistant turns, which gives the judges the same situations to compare.

The [SFT set](../sft/sft_eval/eval_skeletons.json) has 14 scenarios. The later [DPO set](../dpo/eval/eval_skeletons.json) has 52 steering scenarios plus eight ordinary conversations. The ordinary cases check whether training makes the model force a product into a conversation that does not call for one. [Evaluation](evaluation.md)

## Product catalogs

I compared an Indian Amazon export with McAuley's [Amazon Reviews 2023 metadata](https://amazon-reviews-2023.github.io/). Preparation removes duplicates and incomplete records, filters ratings and review counts, and builds embedding text from category, title, and features.

The resulting indexes contain approximately 114,000 Indian products and 443,000 McAuley products. I selected McAuley with RRF from the 18-query comparison. Titles, prices, ratings, and links are catalog snapshots; availability is not checked against live retailer stock. [Retrieval preparation and indexing](../rag/README.md)

## Dataset versions and hashes

The training summaries identify the DPO file with hash prefix `f3b9ade4dc53` and include the split hash. Both match the tracked inputs.

The March SFT runs log hash prefix `ced5b49dd3d0`, while the current local V3 file starts with `958f1a0131d2`. The original matching SFT input is not included, so the current copy cannot reproduce trial 17 exactly. [Reproducibility](../REPRODUCIBILITY.md)
