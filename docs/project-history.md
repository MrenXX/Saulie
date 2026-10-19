# Project history

Public Git history begins with the DPO stack in May 2026. Local artifacts show earlier Windows work. This account combines commits, saved summaries, MLflow parameters, hashes, and selected author decisions from Cursor history.

Creation time is weak evidence after a migration. Several Windows repair folders were created in May while their contents retain earlier modification times. Dates below identify surviving records, not necessarily when work first began.

## August to October 2025: Windows Llama experiments

`saulgman_sft` preserves synthetic batches and `finetune.py` through `finetune4.py`. They configure Llama-3.2-3B-Instruct, LoRA, and successive merged conversation files. Later versions attempt more explicit token handling and training adjustments.

This establishes the setup, not a complete set of successful runs or direct adapter ancestry for the deployed Qwen model. [Historical evidence](evidence/legacy-windows.json)

## March 2026: data repairs and Qwen SFT

Windows `DATA` contains corrected batches and 366-record and 500-record SFT versions. WSL holds a later 500-record V3. [Provenance](data.md)

March 23 artifacts show a Qwen `steering-sft-v1` study. March 25 v1.1 has 20 completed records. Trial 16 had the lowest validation loss; the later pipeline froze trial 17.

March 28 v1.2 has seven completed entries. March 31 outputs compare the base and ten SFT candidates on 14 scenarios each. Qualitative notes identify formulaic openings and uneven naturalness. [SFT method](../sft/README.md)

The original SFT data hash is logged, but its exact bytes were not recovered. The current trainer also contains later edits, preventing an exact historical rerun from Git alone.

## April to May: preference data

Prompt A generated synthetic pairs. Prompt B used SFT trial-17 outputs to construct alternatives. Part A and Part B folders preserve repairs, category checks, diversity adjustments, and one discarded Prompt B pair.

Accepted V4 combines 330 Prompt A, 92 repaired Prompt B, and 100 expanded Prompt B pairs. Full trajectories include branch-local user replies, requiring assistant-only preference scoring. [Data preparation](data.md)

## May 22 to 27: initial DPO studies

v1.0 studied preference accuracy. The main snapshot records 21 completions and best validation accuracy 1.0. v1.1 added a hybrid ranking and diagnostics.

Two workers sharing a GPU created slow or apparently stalled work. Fixes expanded the GPU lock across construction, training, save, diagnostics, and cleanup. Heartbeats, budgets, and progress logs distinguished slow diagnostics from abandoned trials. [Postmortem](../dpo/train/docs/FAILED_TRIAL_POSTMORTEM.md)

The first public training-stack commit is [`0de7d27`](https://github.com/MrenXX/Saulie/commit/0de7d27).

## May 28 to 31: scores versus conversations

Semantic review found repetitive or incoherent output from strong-scoring DPO adapters. Full-strength v1.1 trial 29 was unusable on the saved ten-scenario review. Smaller residual weights preserved more fluency. This was a behavioral failure, not sufficient proof of an export bug.

The work tested conservative unmerged training in v1.2, an SFT-merged dense base in v1.3, and corresponding unmerged experiments in v1.4. These comparisons were not all parts of the final stack. [Semantic report](../dpo/eval/TRAIN_SETUP_SEMANTIC_REPORT.md), [study snapshots](evidence/dpo-study-history.json)

## June 2: v1.5

The final study added a supervised chosen-response anchor and a custom survival objective. It returned to frozen SFT plus trainable DPO adapters.

The main run records 31 attempts, 21 completions, and 11 unique completed configurations. Trial 16 wins the training objective. [DPO guide](../dpo/README.md), [`108425a`](https://github.com/MrenXX/Saulie/commit/108425a)

## June 5 to 7: evaluation and deployment choice

Round 1, generated June 5, contains SFT trial 17 and six anonymized DPO candidates on 34 scenarios. Export checks compare stacked contributions, the combined adapter, and an HF FP8 serving proxy.

Recovered author history reports trials 4 and 20 as judge finalists. After manual vLLM comparison, the author chose trial 4. The June 7 deployment commit configures that choice. Structured judge scorecards were not found, so the decision is recorded without invented scores. [Decision evidence](evidence/selection-record.json), [`87b5982`](https://github.com/MrenXX/Saulie/commit/87b5982)

## June 8 to 14: retrieval and serving

The repository integrated catalog preparation, BGE-M3, Qdrant hybrid retrieval, prompt variants, SSE streaming, nginx, and optional ngrok access. The saved catalog comparison favors McAuley with RRF on its small query set.

June 10 outputs compare base, SFT, and DPO on 60 scenarios. June 11 latency files separately measure isolated vLLM, retrieval, and the non-streaming API. [Evaluation](evaluation.md)

June 14 renamed `train/` to `sft/` and consolidated RAG under `saulie/rag/`. The separate `/root/rag` location now contains storage, not another complete source pipeline. [`eda6d94`](https://github.com/MrenXX/Saulie/commit/eda6d94), [`2ec01aa`](https://github.com/MrenXX/Saulie/commit/2ec01aa)

## June 15 onward: runtime tool behavior

The model sometimes talked as if it had searched without executing the tool. A custom vLLM logits processor added increasing tool pressure. Saved smoke outputs show more calls in the motivating scenarios, with early-search tradeoffs.

Further reset and dense-cosine gate work exists on `fix/tool-pressure-reset-and-rag-gate`, outside the audited public `main`. Main's retrieval code must not be described as containing that gate. [Branch scope](architecture.md)

The July 6 main commit updates the compressed prompt. August 24 commits clean the README, add the license, and untrack many agent plans and handoffs.

## October 19: documentation reconstruction

The overhaul separates training, data, evaluation, and runtime responsibilities. It restores selected evidence and replaces deleted agent-plan links with public guides. It records the SFT hash mismatch, missing scorecards, and incomplete external-artifact reproduction path.

[The audit](documentation-audit.md) describes changes and remaining work. Training, deployment, and retrieval implementation were not changed.
