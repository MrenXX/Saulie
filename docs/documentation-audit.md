# Documentation audit, October 19, 2026

## Before the overhaul

The repo had separate SFT and DPO directories, but the main walkthrough started at DPO. The SFT page mostly described files and commands. Data-generation history, earlier experiments, model selection, and the limits of saved results were hard to follow.

An August cleanup had already untracked many agent plans and handoffs. Other public pages still linked to those missing files. The reproduction guide also assumed unpublished weights, adapters, and an existing BGE ONNX export.

## Evidence reviewed

The audit read public main at `6cf2d7fc0cf13e249d57195f40e3b97a594013f9`, Git history and branches, Windows `saulgman_sft` and `DATA`, WSL `/root/saulie` and `/root/rag`, and selected project-specific Cursor messages for the deployment decision.

Dates were cross-checked against contents and run metadata. The original WSL checkout had local changes and was left untouched. Implementation used a separate clone of current public main.

## Work completed

- Rebuilt the landing page as an ordered explanation of data, SFT, DPO, evaluation, and deployment.
- Expanded SFT and added a DPO guide with distinct objectives, masks, reference behavior, study versions, and serving representations.
- Added data lineage, project history, architecture, evaluation, examples, and deployment guides.
- Recovered small SFT summaries, run metadata, scenarios, and conversation outputs, plus the completed base/SFT/DPO comparison.
- Separated the training-objective winner from the author's deployment choice.
- Replaced stale links to deleted agent plans with public documentation and factual training invariants.
- Corrected setup claims about missing artifacts, ONNX export, remote-client constants, paths, and network bindings.
- Stopped tracking the generated status-loop file and ignored local editor/session directories. Existing useful research reports and code remain.
- Added a standard-library checker for documentation links, hashes, dataset splits, and evaluation coverage.

No model training, retrieval, API, or deployment implementation was changed. Historical evidence files retain original wording and formatting where copied verbatim.

## Remaining work, in priority order

1. Recover the SFT input matching logged hash prefix `ced5b49dd3d0`, or explicitly publish a new dataset version and new run. Current V3 has different bytes. Do not relabel it as the original.
2. Recover structured final judge scorecards and the trial-17 baseline-selection rationale. The DPO author decision is known, but numerical judge margins and a complete SFT selection scorecard are missing.
3. Package trained adapters with checksums, upstream model revision IDs, and distribution terms. They currently have no public download.
4. Publish and verify BGE-to-ONNX export and a clean-machine setup. The current build script starts from an existing export and external tensor files.
5. Expand behavioral and retrieval evaluation with consistent prompts, annotated relevance, repeated samples, and recorded hardware. Current small scripted measurements should remain labeled as exploratory.

These are evidence and reproducibility tasks, not claims that the model needs to be retrained immediately. The documentation now states what can be inspected today.

## Original-state backup

[The backup branch](https://github.com/MrenXX/Saulie/tree/backup/pre-docs-overhaul-20261019) preserves the original public main at `6cf2d7f`. A verified local Git bundle also retains the original refs and history.

The overhaul is isolated in its documentation commit and can be reverted normally without rewriting repository history. The original Windows and WSL source folders were not used as the editing checkout.
