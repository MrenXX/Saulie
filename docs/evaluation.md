# Evaluation

I used training metrics to shortlist adapters and generated conversations to select the model. Low SFT loss did not reliably predict steering quality, and high DPO preference accuracy sometimes selected models with broken English.

Two LLM judges scored the conversations: Claude Opus 4.8 with Max reasoning and GPT-5.5 with xhigh reasoning. They are labeled Judge A and Judge B below. Scores use a 1-5 rubric.

The aggregate results are transcribed from Tables 5.2-5.6 of my project report. [Judge results JSON](evidence/judge-results.json)

## SFT selection

I generated conversations from the shortlisted SFT adapters on [14 fixed scenarios](../sft/sft_eval/eval_skeletons.json). The review covered steering, question quality, recommendation landing, tone, naturalness, and difficult openings.

Trial 17 was selected through this review, although trials 16 and 15 had slightly lower validation loss. The [outputs](evidence/sft-conversations.json) include the base and ten SFT candidates. [SFT method](../sft/README.md)

## DPO judging protocol

The [DPO scenario set](../dpo/eval/eval_skeletons.json) contains 52 steering conversations across four opening types and eight ordinary conversations. The latter check whether the model can chat without forcing a sale.

The first round compared SFT trial 17 with six DPO finalists on 26 steering and eight ordinary scenarios. Candidate labels hid trial numbers and training metrics from the judges. The [generation output](../dpo/eval/generations_round1.json) contains 238 conversations.

The second round compared trials 4, 19, 20, and 27 against SFT on the other 26 steering scenarios. The runner also reuses the eight ordinary controls; only the steering set is a fresh holdout.

Generation uses vLLM FP8 with temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.05, and at most 256 tokens per assistant turn. [Configuration](../dpo/eval/v15_eval_config.py), [rubric](../dpo/eval/llm_judge_prompt_dpo.md)

## DPO results

Round-one mean steering scores, Judge A followed by Judge B:

- Trial 4: 3.74 and 3.36.
- Trial 19: 3.63 and 3.14.
- Trial 20: 3.40 and 3.11.
- Trial 27: 3.36 and 3.14.
- Trial 16: 3.34 and 2.87.
- Trial 8: 3.21 and 3.01.
- SFT trial 17: 3.38 and 2.98.

Trial 4 led both judges in round one. On the second-round steering holdout:

- Trial 4: 3.77 and 3.40.
- Trial 20: 3.83 and 3.46.
- Trial 27: 3.70 and 3.29.
- Trial 19: 3.52 and 3.22.
- SFT trial 17: 3.45 and 3.06.

Both judges gave trial 20 a slightly higher mean, but Judge A preferred trial 4 for its handling of unrelated and vague openings. Judge B preferred trial 20. I then had participants talk directly to both candidates and compare steering, style, and recommendations. That qualitative comparison favored trial 4, which I deployed with trial 20 as backup. It was not a counted vote or a large user study.

The training objective would have selected trial 16. Its survival score was 0.9798, compared with 0.9268 for trial 4. Both reached preference accuracy 51/52 on the validation pairs. These metrics rank supplied continuations; the judges assess generated conversations. [Training summary](../dpo/study_results/trial_summary_v1.5.json)

## Contribution of each stage

I also compared the prompted base, SFT trial 17, and DPO trial 4 on all 60 scenarios. The base received a detailed [bootstrap system prompt](../dpo/eval/eval_inference_system_prompt.md); the trained models used bare user turns. These were text-only category recommendations without product retrieval.

Mean steering scores, Judge A followed by Judge B:

- Base: 2.86 and 3.07, cross-judge mean 2.97.
- SFT: 3.25 and 3.07, cross-judge mean 3.16.
- DPO: 3.42 and 3.21, cross-judge mean 3.32.

Mean ordinary-conversation scores were 3.31, 3.73, and 3.90 across judges for base, SFT, and DPO respectively.

Judge A counted clean recommendation landings in 13/52 base conversations, 20/52 SFT conversations, and 27/52 DPO conversations. Conversations with no recommendation fell from 20 to 15 to eight. Forced recommendations in the eight ordinary conversations fell from six to three to one.

The [complete generation output](../dpo/eval/generations_prod_ladder.json) contains 180 conversations. [Example](examples.md)

## Where DPO helped and where it did not

Cross-judge steering means by opening type, SFT followed by DPO:

- Direct problem: 3.10 to 3.49.
- Unrelated or factual: 3.27 to 3.22.
- Emotional or lifestyle: 3.34 to 3.31.
- Vague complaint: 2.95 to 3.25.

DPO helped most on direct problems and vague complaints. It slightly reduced scores on the other two types. Trial 4 also received one severe-failure flag and two factual-hallucination flags from Judge A. It remains a specialized steering model, and SFT is a useful fallback for less sales-focused conversation.

These results come from fixed scenarios and LLM judging. The project report includes the aggregates; per-conversation judge score files and the completed Round 2 generation JSON are not included in the repository.

## Adapter and runtime checks

Before judging the finalists, I checked that concatenating SFT and DPO preserved their weight updates. The [merge report](../dpo/eval/merge_v15_slate_validation.md) covers all six finalists. Weight-delta equality blocks a bad export; forward drift and short FP8 generations provide additional diagnostics.

The [model/prompt comparison](../dpo/eval/matrix_runs/) tests SFT, half-weight DPO, and full-weight DPO under three prompts. Half-weight DPO failed to call search in those runs. Full-weight trial 4 with the shorter prompt became the default.

The later [tool-pressure comparison](../dpo/eval/matrix_runs/tool_pressure_smoke_report.md) increased search execution in camping and kitchen scenarios, but sometimes triggered it too early. Its heuristic flags and small scenario set are useful for debugging rather than estimating a general success rate.

## Retrieval

The [18-query comparison](../rag/fusion_comparison_report.md) tests two catalogs and two fusion methods:

- Indian RRF: 10/18 relevant first results and 32/54 relevant top-three hits.
- Indian DBSF: 10/18 and 28/54.
- McAuley RRF: 15/18 and 44/54.
- McAuley DBSF: 14/18 and 43/54.

I selected McAuley with RRF. Relevance uses a keyword rubric, and category filters differ between catalogs.

## Latency

The [local benchmark](../dpo/eval/LATENCY_REPORT.md) measures isolated vLLM, isolated retrieval, and non-streaming agent requests:

- Short-prompt model TTFT: p95 27.26 ms from five requests.
- Decode rate: median 70.53 tokens per second across ten observations.
- Retrieval: p95 31.13 ms across 24 observations.
- Full tool turn: p95 2489.61 ms from six observations.

The tool turn includes generation before search and generation after its results. These timings exclude public-tunnel latency and do not measure streaming client TTFT or sustained concurrent load. [Raw runs](../dpo/eval/latency_runs/)

## Re-running evaluation

Follow [the command guide](../dpo/eval/README.md) after supplying the model assets. To check the included files, counts, and hashes without GPU services:

```bash
python scripts/check_docs.py
```
