# Evaluation and recorded results

Training metrics, conversation quality, export correctness, retrieval relevance, and latency answer different questions. The saved artifacts support a review of each, but not one universal project score.

## SFT behavior

[SFT trial summaries](evidence/sft-study-v1.1.json) rank validation loss. The [March 31 conversations](evidence/sft-conversations.json) contain 11 models on 14 scenarios each, including the base and candidates from v1.1 and v1.2. The [rubric](../sft/sft_eval/llm_judge_prompt.md) evaluates conversational steering and naturalness.

Local judge notes report learned steering alongside formulaic openings. They disagree on candidate rankings and were not a uniform structured scoring run. Trial 17 is the later frozen baseline; it is not the loss winner. [SFT details](../sft/README.md)

## DPO preference validation

The fixed validation split contains 52 pairs. Accuracy `0.9807692307692307` means 51/52 preferred branches scored above their rejected branches. It is not a 98% user satisfaction rate, retrieval precision, or conversation success rate.

The [v1.5 summary](../dpo/study_results/trial_summary_v1.5.json) names trial 16 as the survival-objective winner. The deployed trial 4 has a different objective value despite the same 51/52 preference accuracy. [DPO method](../dpo/README.md)

The [early semantic review](../dpo/eval/TRAIN_SETUP_SEMANTIC_REPORT.md) explains why preference scores were insufficient. It reviews 180 generated conversations across ten scenarios, three earlier candidates, and six DPO weights. It is a small diagnostic study of the HF training setup, not the final FP8 deployment evaluation.

## Adapter export checks

The [finalist merge report](../dpo/eval/merge_v15_slate_validation.md) checks trials 19, 16, 8, 27, 20, and 4. Weight-delta algebra is the blocking check. Forward drift and short generation checks are additional diagnostics.

The FP8 comparison in that report uses HF as a serving proxy. It does not replace generating conversations through vLLM. All six finalists were retained for the recorded evaluation; this is not proof that they are equally good models.

## Round 1 conversations

[`generations_round1.json`](../dpo/eval/generations_round1.json) records a complete June 5 run: SFT trial 17 plus six anonymized candidates, each on 34 scenarios. That is 238 conversations. Each model sees 26 steering and eight ordinary-chat scenarios.

The [unblind map](../dpo/eval/generations_round1_unblind.json) maps candidates A through F to trials 19, 16, 8, 27, 20, and 4. Keep it out of a judge's scoring packet.

Generation uses vLLM FP8, temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.05, and 256 maximum tokens per assistant turn. The [configuration](../dpo/eval/v15_eval_config.py) defines the scenario split and parameters.

The runner supports a second round on the other 26 steering scenarios while reusing the eight ordinary controls. This means the two rounds are not fully independent held-out evaluations. A completed Round 2 JSON and structured judge scorecards were not recovered in the audited project folders.

The [selection record](evidence/selection-record.json) preserves the author's report that trials 4 and 20 were judge finalists, followed by the manual choice of trial 4. Without scorecards, no numerical judge margin or independently verified final ranking is published.

## Base, SFT, and deployed DPO

[`generations_prod_ladder.json`](../dpo/eval/generations_prod_ladder.json) records the June 10 comparison of three models on all 60 scenarios, totaling 180 conversations. It was recovered locally and added during this overhaul.

Only the base received the [bootstrap system prompt](../dpo/eval/eval_inference_system_prompt.md). SFT and DPO used bare scenario turns. This is useful for inspecting learned behavior, but it is not a comparison with identical prompts across all models.

The models generated generic product recommendations without the live retrieval tool. These outputs cannot establish catalog grounding or tool reliability. [Recorded examples](examples.md)

## Model, prompt, and tool behavior

The [matrix outputs](../dpo/eval/matrix_runs/) compare SFT, half-weight DPO, and full-weight DPO with three runtime prompts on scripted agent flows. The [tool-pressure smoke report](../dpo/eval/matrix_runs/tool_pressure_smoke_report.md) adds a runtime logits processor comparison.

Those tests inspect whether tools execute and whether conversation remains usable. Their flags are heuristic, and the scenarios are small. The direct-request baseline and harness both ask a clarifying question without a tool call in the linked JSON.

The positive result is narrow: pressure caused search execution in the motivating camping and kitchen runs. It also caused some early searches. It does not prove a general hallucination reduction rate.

## Retrieval comparison

The [fusion report](../rag/fusion_comparison_report.md) uses 18 product intents. It counts first-result keyword matches and relevant hits among the top three.

- Indian RRF: 10/18 first-result matches and 32/54 top-three matches.
- Indian DBSF: 10/18 and 28/54.
- McAuley RRF: 15/18 and 44/54.
- McAuley DBSF: 14/18 and 43/54.

These results support the local choice of McAuley with RRF. Category filters differ between catalogs, and keyword matching can miss semantic failures or accept weak matches. This is not a large annotated retrieval benchmark.

## Latency

The [June 11 report](../dpo/eval/LATENCY_REPORT.md) and [raw runs](../dpo/eval/latency_runs/) separately measure isolated vLLM, isolated RAG, and non-streaming agent requests.

- Short-prompt vLLM TTFT: p95 27.26 ms from five requests.
- Decode rate: median 70.53 tokens per second across ten observations.
- Isolated search: p95 31.13 ms across 24 observations.
- Agent tool turn: p95 2489.61 ms from six observations.

The report does not establish public-tunnel latency, streaming client TTFT, sustained load capacity, or a hardware-independent guarantee. The saved run files do not establish a consistent per-run GPU identity, so these numbers are not advertised as an RTX 4070 benchmark.

## Re-run boundaries

Use [the evaluation command guide](../dpo/eval/README.md) after supplying the adapters and environment. Most generation commands start or require GPU services. The evidence checker can validate saved counts and hashes without them:

```bash
python scripts/check_docs.py
```

The documentation overhaul verified these artifacts and references. It did not retrain models, repeat judge panels, reindex catalogs, or restart the user's services.
