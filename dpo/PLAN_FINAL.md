## Plan: Assistant-Only Trajectory DPO

This is the final coherent handoff for a new implementation agent. Treat V4 as the current accepted dataset. Do not reopen data repair unless a hard structural blocker is discovered during implementation, such as invalid roles, empty assistant branches, duplicate IDs, impossible tokenization, or schema drift from the validation report. The training objective is branch-safe DPO over full chosen/rejected trajectories: keep all branch messages visible as context, but score only assistant/action tokens. Never score user-role text inside chosen/rejected continuations.

**Handoff Snapshot**
1. Dataset: `DPO_522_prompt_a_and_prompt_b_V4_repaired.jsonl` with validation report `DPO_522_prompt_a_and_prompt_b_V4_validation_report.json` and repair manifest `DPO_522_prompt_a_and_prompt_b_V4_repair_manifest.jsonl`.
2. Validated counts: 522 rows; `prompt_a=330`, `prompt_b_repaired=92`, `prompt_b_exp500=100`; categories `style=220`, `steering=121`, `product_fit=181`; V4 validation passes.
3. Data state: no data blockers remain. Do not train on older dataset artifacts. Do not run another repair pass unless a hard structural blocker appears.
4. Base training model: `Qwen3-4B-Instruct-2507`. Base inference model: `Qwen3-4B-Instruct-2507-FP8`.
5. Existing SFT behavior/reference: `train/models/steering-sft-v1.1/trial-17/best_adapter/`. The SFT adapter is frozen and must never be overwritten.
6. Policy: base + frozen SFT behavior + new trainable DPO LoRA delta. Reference: frozen SFT trial-17 behavior.
7. Gate: run a dummy first and stop. No Optuna until the dummy succeeds and the user explicitly approves.

**External Implementation Anchors**
1. TRL DPOTrainer: https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py - DPO metrics such as `rewards/accuracies`, `rewards/margins`, chosen/rejected rewards/logps, shifted `completion_mask` scoring, and reference-logprob precompute/cache behavior.
2. TRL DPO config: https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_config.py - `precompute_ref_log_probs`, `precompute_ref_batch_size`, `loss_type`, `sigmoid_norm`, and `ld_alpha`.
3. Open-Instruct DPO utilities: https://github.com/allenai/open-instruct/blob/main/open_instruct/dpo_utils.py - explicit `build_reference_logprobs_cache`, `TensorCache`, `dpo_norm`, WPO-related logic, and DPO reward/margin logging patterns.
4. Open-Instruct DPO entrypoint: https://github.com/allenai/open-instruct/blob/main/open_instruct/dpo.py - cache orchestration, dataset/reference hashing, and cache-before-training flow.
5. PEFT LoRA merge: https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/model.py - `add_weighted_adapter(..., combination_type="cat")` for a single higher-rank LoRA representing SFT delta plus DPO delta.
6. vLLM LoRA serving: https://docs.vllm.ai/en/stable/features/lora/ - multiple LoRAs can be registered horizontally, but a single request should use one merged SFT+DPO adapter; set `--max-lora-rank` high enough.
7. Experimental/customized preference anchors, for context only: TRL Online DPO at https://github.com/huggingface/trl/blob/main/trl/experimental/online_dpo/online_dpo_trainer.py, TRL XPO at https://github.com/huggingface/trl/blob/main/trl/experimental/xpo/xpo_trainer.py, and Open-Instruct WPO/dpo_norm logic in `dpo_utils.py`. These show active customized preference-training implementations, but they do not change this plan.
8. DMPO/multi-turn note: no confirmed public DMPO implementation repo was found. Treat DMPO as conceptual support for optimizing action/assistant tokens in trajectory context, not as code to copy.

**Implementation Steps**
1. Environment discovery: confirm installed `trl`, `transformers`, `peft`, `datasets`, `bitsandbytes`, `accelerate`, CUDA, and the local `DPOTrainer` data-preparation path. This determines whether plain TRL pretokenized data is enough or a tiny preservation wrapper is needed.
2. Data gate: load V4, its validation report, and the generation specs only to understand the schema. Verify the report counts. Do not load or use older dataset artifacts as training inputs.
3. Max-length gate: run the user's V4 max-length audit script and choose one fixed `max_length` before tokenization. Do not tune `max_length` in Optuna because changing it invalidates token/reference caches and makes trials incomparable.
4. Pre-tokenization report: summarize row count, source/category counts, source-by-category counts, source-by-category-by-opening counts, rows with branch-local user turns, prompt/chosen/rejected token lengths, assistant-scored token lengths, chosen/rejected scored-length ratios, and truncation risk.
5. Split metadata: create deterministic train/val/test split metadata before tokenization using the split policy below. Keep validation fixed for all dummy/Optuna runs and keep test untouched until after Optuna/model selection.
6. Branch-safe preprocessing: tokenize `prompt + chosen` and `prompt + rejected` with the Qwen chat template, keep full input IDs and attention masks, and build `completion_mask` with 1 only on assistant/action content and assistant end markers that should be scored.
7. Mask invariant: prompt tokens, branch-local user tokens, system/tool/context tokens, padding, and role headers are unscored. Hard fail if any user-role token is scored.
8. Decoded mask audits: include ordinary single-assistant rows, multi-turn steering rows, Prompt B repaired rows, Prompt B exp500 rows, long rows from the max-length script, and rows where chosen/rejected have different branch-local user text.
9. Single-assistant sanity check: on rows where chosen/rejected are just one assistant continuation, confirm the assistant-only mask matches normal prompt+assistant DPO scoring. This is a bug trap only, not a training ablation.
10. TRL compatibility: use a custom collator that emits TRL-compatible `input_ids`, `attention_mask`, and assistant-only `completion_mask`; keep metadata out of model forward inputs. Add `AssistantOnlyDPOTrainer` only if the installed TRL re-tokenizes or discards the custom masks.
11. Persistent caches: implement a tokenized dataset cache and reference-logprob cache keyed by dataset version, split manifest hash, tokenizer/chat-template hash, reference adapter path/hash, fixed max_length, truncation policy, and mask policy.
12. Reference path: prefer TRL's built-in `.npz` ref-logprob precompute if it preserves the custom `completion_mask`; otherwise implement the Open-Instruct-style `TensorCache` path with explicit sample indices.
13. Dummy stack: Qwen3-4B-Instruct-2507 in BnB 8-bit, frozen SFT trial-17 behavior as reference, and a new trainable DPO LoRA on top of the SFT behavior.
14. Dummy hparams: `loss_type="sigmoid"`, `ld_alpha=None`, `beta=0.1`, `lr=1e-5`, `epochs=1 or 2`, DPO LoRA `r=16`, `alpha=32`, `dropout=0.1`, `per_device_train_batch_size=1`, `gradient_accumulation_steps=4`, gradient checkpointing enabled, warmup 0.03-0.1.
15. Dummy report: include mask audit, ref-cache hit/miss path, loss, reward accuracy, reward margin, chosen/rejected scored-token lengths, margin-vs-length correlation, source/category metrics, runtime, and peak VRAM allocated/reserved. Stop after this report for user approval.
16. Optuna gate: after explicit approval, run the 20-trial Optuna study with fixed split/tokenization/max_length/ref-cache policy. Do not benchmark reference modes; persistent ref-cache is the default.
17. Trial hygiene: log VRAM per trial but do not abort solely because peak VRAM exceeds 12GB. Only fail/prune real OOMs, invalid masks, NaNs, broken saves, or corrupted caches. Between trials delete trainer/model refs, run garbage collection, clear CUDA cache, reset peak stats, and use subprocess-per-trial isolation only if fragmentation/leaks appear.
18. Optuna evaluation: rank trials by offline DPO validation metrics only. Do not run `sft_eval`, web SoTA judges, or manual LLM judging inside the trial loop.
19. Shortlist after Optuna: use reward accuracy first, reward margin second, and diagnostics as filters/Pareto signals. Length-aware offline metrics can narrow candidates but cannot replace generated evaluation.
20. Merge finalists: after shortlisting, merge each SFT+DPO pair into one PEFT cat adapter for vLLM and validate equivalence against the stacked training-time policy.
21. Final manual eval: run the existing `sft_eval` generation/judge workflow only on merged finalists plus calibration baselines. Select final model by the manual judge average and rubric-specific breakdown, with DPO validation metrics as supporting evidence only.

**Split Stratification Policy**
1. Target split ratio: default to `train=80%`, `val=10%`, `test=10%` for the 522 V4 rows, with deterministic rounding and total counts recorded in the split manifest.
2. Primary source labels: use raw `dpo_source` values as hard balancing labels: `prompt_a`, `prompt_b_repaired`, and `prompt_b_exp500`.
3. Also derive `source_family` for reporting: `prompt_a` vs `prompt_b`, where `prompt_b = prompt_b_repaired + prompt_b_exp500`.
4. Primary category labels: preserve `category` distribution across `style`, `steering`, and `product_fit` in every split.
5. Secondary opening labels: use `opening_type` inside each `(dpo_source, category)` bucket when the cell is large enough. It improves balance but must not create fragile tiny-cell splits.
6. Stratum key order: try `(dpo_source, category, opening_type)` first. If a cell has fewer than 5 rows, collapse that cell to `(dpo_source, category)`. If any `(dpo_source, category)` bucket is unexpectedly smaller than 5, stop and ask before collapsing further because source/category balance is required.
7. Allocation rule: deterministic-shuffle rows by stable hash of `(split_seed, id)`, then allocate approximately 80/10/10. For strata with at least 10 rows, reserve at least one validation and one test row. For strata with 5-9 rows, reserve one validation and one test row when possible and keep the rest in train.
8. Global correction pass: after per-stratum allocation, adjust only within the same `(dpo_source, category)` bucket to hit global split sizes as closely as possible. Do not move rows in a way that drops a source/category bucket entirely from validation or test.
9. Balance diagnostics: report per-split counts and percentages for `dpo_source`, `source_family`, `category`, `dpo_source x category`, `dpo_source x category x opening_type`, `opening_type`, `total_turns`, `divergence_turn`, and `has_branch_local_user`.
10. Branch-safety diagnostic label: compute `has_branch_local_user` as true when chosen or rejected contains any user-role message after the prompt. This is not the primary split key, but validation/test should contain branch-local-user examples if they exist in V4.
11. Split manifest: write `train/dataset/dpo_v4_split_seed_<seed>.jsonl` or an equivalent target-workspace path with `id`, `split`, `split_seed`, `dpo_source`, `source_family`, `category`, `opening_type`, `total_turns`, `divergence_turn`, `has_branch_local_user`, final stratum key, and allocation reason. Cache keys must include this manifest hash.
12. Split validation: fail if any V4 row is missing from the manifest, any ID appears twice, train/val/test overlap, source/category proportions drift materially from full V4, validation has zero examples for any raw `dpo_source` or category, or test has zero examples for any raw `dpo_source` or category.

**Optuna Search Space**
1. `beta`: categorical `{0.01, 0.05, 0.1, 0.2}`.
2. `num_train_epochs`: categorical `{1, 2, 3}`.
3. `learning_rate`: log range `5e-6` to `3e-5`.
4. DPO LoRA rank `r`: categorical `{8, 16, 32}`, with `lora_alpha = 2 * r`.
5. `lora_dropout`: categorical `{0.05, 0.1}`.
6. `per_device_train_batch_size`: categorical `{1, 2}` if stable on the 3090.
7. `gradient_accumulation_steps`: categorical `{2, 4, 8}`; keep effective batch around 4-16.
8. `lr_scheduler_type`: categorical `{linear, cosine, constant_with_warmup}`.
9. `max_grad_norm`: categorical `{0.3, 0.5, 1.0}`.
10. `weight_decay`: fixed `0.05`, optionally `{0.03, 0.05}` only if another knob is affordable.
11. `warmup_ratio`: fixed `0.1` unless dummy suggests instability.
12. `optim`: fixed `paged_adamw_32bit` unless environment inspection suggests a better local default.
13. `neftune_noise_alpha`: categorical `{0, 5.0}`.
14. `label_smoothing`: fixed `0.0` in v1; consider `{0.03, 0.05}` only if train reward rises while validation reward stalls.
15. `length_mode`: categorical instead of crossing every length knob: `none` = sigmoid + no `ld_alpha`; `sigmoid_norm` = length-normalized score; `ld_0.3` = sigmoid + `ld_alpha=0.3`; `ld_0.5` = sigmoid + `ld_alpha=0.5`; optional low-priority `ipo` only if budget remains.
16. `turn_weighting`: keep `none` for the first approved 20-trial study. Do not add DMPO-style turn weighting unless explicitly requested later; no confirmed public DMPO implementation repo was found.

**Optuna Ranking And Shortlisting**
1. Primary Optuna objective: maximize validation `eval_rewards/accuracies`, preferably macro-averaged across source/category buckets when bucket sizes allow.
2. Treat `eval_rewards/margins` as the first secondary signal, not a separate Optuna objective. Margin is useful for close calls, but maximizing margin by itself can reward overconfidence or length-driven separation.
3. Do not use any reward-plus-loss hybrid score. The agreed policy is accuracy first, margin second, diagnostics as filters/Pareto signals, and loss only as a sanity guardrail.
4. Always log raw/micro `eval_rewards/accuracies`, macro reward accuracy by source/category when feasible, `eval_rewards/margins`, `eval_loss`, `eval_logps/chosen`, `eval_logps/rejected`, chosen/rejected scored lengths, reward-vs-length correlation, source/category split metrics, runtime, VRAM, cache hit/miss state, and trial save status.
5. Keep loss as a sanity guardrail only: require finite/non-NaN loss, flag large validation-loss outliers, and use it to explain or reject suspicious trials, not to rank good trials ahead of better reward-accuracy trials.
6. Diagnostic derivation: length-bias flags come from scored-token lengths plus margin-vs-length correlation; source/category balance comes from split reward accuracies and margins; stability comes from finite loss, non-NaN rewards/logps, healthy save/cache behavior, and no OOM or retry pattern.
7. Shortlist policy: first sort by validation reward accuracy; within a small accuracy band, prefer higher reward margin if length/source/category diagnostics are healthy; then add Pareto candidates that are slightly lower on accuracy but materially better on length bias, source/category balance, or stability.
8. Suggested concrete shortlist rule for 20 trials: keep the top 3 by macro reward accuracy, add any trial within about 1-2 percentage points of best accuracy that has clearly better reward margin and clean diagnostics, and add at most 1-2 Pareto candidates. Cap final manual `sft_eval` candidates around 4-6 unless results are unusually clustered.

**Length Bias Strategy**
1. Offline diagnostics are required because DPO reward accuracy can improve while the model learns verbosity shortcuts.
2. Log chosen/rejected scored assistant-token lengths, length ratios, `abs(chosen_len - rejected_len)`, reward margin versus length difference correlation, metrics split by source/category, and whether long chosen branches dominate wins.
3. Add generated-output length diagnostics on eval skeletons for shortlisted models: assistant token count per turn, total conversation assistant tokens, turn where recommendation lands, filler/hedging count, and judge subscore correlation with length.
4. Use `sigmoid_norm` and `ld_alpha` in Optuna because they directly control length sensitivity during training.
5. Final inference eval is still necessary. Length diagnostics can narrow the candidate set, but they cannot tell whether the model steers naturally, asks good questions, or lands recommendations in the desired style.
6. Run final judge eval with normal generation settings first. Add a length-controlled generation pass only if top candidates differ mainly in verbosity or if judge scores correlate strongly with generated length.

**Final Manual SFT Eval Adaptation**
1. Treat `sft_eval` as the final manual evaluation stage after all approved Optuna studies and offline shortlisting are done. It must not run per trial.
2. Reuse `sft_eval/eval_skeletons.json` for held-out generation so DPO finalists are comparable to SFT trial-17 behavior.
3. Reuse `sft_eval/llm_judge_prompt.md` because it already weights steering, question quality, recommendation landing, tone, naturalness, and difficulty, and it explicitly warns that length is not always good.
4. Extend `sft_eval/eval_generate_vllm.py` selection logic from SFT `eval_loss` to a DPO finalist manifest produced after Optuna. That manifest should include reward accuracy, loss, margin, length diagnostics, source/category splits, merge status, and vLLM adapter name.
5. Extend `sft_eval/deploy_qwenie_eval.sh` or create a DPO-specific deploy script that scans merged SFT+DPO finalist adapter directories instead of only SFT trial directories.
6. Keep base/SFT trial-17 as calibration baselines during DPO eval so the judge can reveal DPO regressions, not just DPO improvements.

**Adapter Merge And vLLM Plan**
1. Train DPO as a separate adapter while preserving SFT trial-17 behavior as the starting/reference behavior.
2. After selecting shortlist candidates, create one PEFT cat adapter per candidate: SFT adapter plus DPO adapter with weights `[1.0, 1.0]`.
3. Mathematical check: LoRA deltas are additive. For compatible adapters on the same base/modules, PEFT `cat` represents `delta_sft + delta_dpo` exactly by increasing rank to `r_sft + r_dpo`; it does not fuse into the base model.
4. No inherent quality degradation is expected from cat itself if configs match and scaling is handled by PEFT. Possible degradation sources are DPO overfitting, wrong target modules, wrong base model, quantization mismatch, or vLLM rank/config errors.
5. Verify configs before merging: same base model lineage, same tokenizer/chat template, compatible target modules, no unsupported `modules_to_save` conflict, no AdaLoRA/mixed tuner, expected ranks, expected `lora_alpha`/scaling.
6. Verify equivalence before vLLM: compare logits or next-token logprobs between the training-time stacked policy and the cat adapter on a small fixed prompt set; require near-identical outputs within dtype/quantization tolerance.
7. vLLM deploy must set `--max-lora-rank` to at least the merged rank. If SFT `r=16` and DPO `r=32`, use at least `--max-lora-rank 48` or the next supported/accepted value for the installed vLLM; current SFT script has `--max-lora-rank 32`, so it must change for higher-rank merges.
8. Load the merged adapter as a single vLLM LoRA model. Multiple merged candidates can still be served horizontally for A/B evaluation, but each request should use one merged adapter.

**Port Bundle Files**
- `PLAN_FINAL.md` - this coherent final handoff, source of truth.
- `DPO_522_prompt_a_and_prompt_b_V4_repaired.jsonl` - current V4 source data.
- `DPO_522_prompt_a_and_prompt_b_V4_validation_report.json` - validation proof and expected counts.
- `DPO_522_prompt_a_and_prompt_b_V4_repair_manifest.jsonl` - repair provenance.
- `DPO_PROMPT_A_OG.md` and `DPO_PROMPT_B.md` - generation specs showing multi-turn branch continuations and Prompt B steering repairs are intentional.
- `DATA_CONTEXT.md` - V4-only short data context. It must not point to older dataset artifacts.

Do not include old handoff plans, old training-stack plans, repair-audit handoffs, or other stale planning/provenance docs in the clean port bundle. They contain obsolete repair and selection language that can confuse the implementation agent.

**Optional Final Eval Files**
- `sft_eval/eval_skeletons.json` - held-out generation skeletons.
- `sft_eval/llm_judge_prompt.md` - final manual judge rubric.
- `sft_eval/eval_generate_vllm.py` - generation script to adapt for DPO finalists.
- `sft_eval/deploy_qwenie_eval.sh` - vLLM deploy script to adapt for merged adapters.

**Planned New Implementation Files**
- `train/dpo_data.py` - data load, deterministic split creation, tokenization, assistant-only masks, audits, diagnostics, caches.
- `train/dpo_trainer_compat.py` - collator plus minimal TRL preservation wrapper only if installed TRL requires it.
- `train/train_dpo.py` - dummy and Optuna entrypoint, MLflow logging, cache use, and shortlist manifest writing.
- `train/merge_sft_dpo_lora.py` - PEFT cat merge and equivalence validation for vLLM-serving adapters.
- `train/dataset/dpo_v4_split_seed_<seed>.jsonl` - split manifest produced by implementation.

**Verification**
1. Data validation: 522 rows, stable IDs, expected source/category counts, valid role alternation, no empty assistant branches, and no unexpected V4 schema drift.
2. Split validation: manifest covers every row exactly once; train/val/test are disjoint; raw `dpo_source`, `source_family`, `category`, and `dpo_source x category` proportions are reported per split; validation and test each contain every raw `dpo_source` and every category.
3. Mask validation: scored user tokens equal 0; chosen/rejected scored assistant tokens are nonzero; decoded span audits prove branch-local users are context only.
4. Batch validation: collator emits expected tensors and shifted `completion_mask`; model forward receives only safe tensor keys; metadata remains available for logging but never reaches model forward.
5. Reference validation: cached ref logps use exactly the same assistant-only mask as policy logps, and cache keys change when dataset/tokenizer/reference/mask/max_length changes.
6. Dummy validation: one dummy completes, saves adapter, reports loss/reward/margin/accuracy/length/VRAM/cache diagnostics, and stops before Optuna.
7. Optuna validation: all trials use the same split/tokenization/max_length/ref cache; VRAM is logged but not used as a hard fail unless OOM occurs.
8. Ranking validation: objective value is validation reward accuracy; reward margin is logged and used for close calls/Pareto review; loss is logged separately and used only as a sanity guardrail.
9. Diagnostic validation: length-bias, source/category balance, and stability flags are reproducible from logged accuracy, margin, loss, length, split, runtime, VRAM, and cache metadata.
10. Merge validation: cat adapter config rank equals SFT rank plus DPO rank; stacked-policy vs cat-adapter logits or next-token logprobs match on fixed prompts.
11. vLLM validation: merged adapter loads with adequate `--max-lora-rank`; generation script can request it as one model; base/SFT/DPO outputs are saved for judge scoring.
12. Final selection: choose by manual LLM judge average and rubric breakdown after all Optuna studies, with DPO validation metrics as secondary diagnostics.

**Decisions And Scope Boundaries**
- V4 is the accepted training dataset. Do not mention or use older dataset artifacts as current training inputs.
- No data repair work remains unless a hard structural blocker is found.
- Do not implement alternate branch-ablation training paths. The old comparison is obsolete.
- Train on full branch trajectories with assistant/action-only scoring.
- Never score user-role text inside chosen/rejected continuations.
- Initial dummy loss is standard sigmoid DPO with assistant-only/action-token masks.
- Persistent reference-logprob cache is the default.
- User's max-length script decides fixed max_length before training.
- Splits are deterministic 80/10/10 by stable row ID, stratified first by raw `dpo_source` and `category`, with `opening_type` secondary when cell sizes allow.
- VRAM is logged per trial; do not abort solely for exceeding 12GB.
- Include length controls (`sigmoid_norm`, `ld_alpha`) because length bias is likely.
- Optuna primary objective is validation reward accuracy, preferably macro-averaged by source/category when feasible.
- Reward margin is a secondary ranking signal and Pareto-shortlist signal, not a standalone Optuna objective.
- Log loss separately as a sanity guardrail; do not use any reward-plus-loss hybrid score.
- Do not run `sft_eval`, web SoTA judges, or manual LLM judging inside Optuna.
- Do final generated eval manually with the existing `sft_eval` judge pipeline after all Optuna studies.
- Use PEFT `cat` merge for a single vLLM-loadable SFT+DPO adapter, with equivalence checks before serving.
- No Optuna before dummy success and explicit approval.

**Remaining Non-Blocking Items**
1. Exact `max_length` is deferred until the V4 max-length audit script is run.
2. Whether `AssistantOnlyDPOTrainer` is necessary is deferred until the installed TRL version is inspected.
3. Exact shortlist size is deferred until the 20-trial metric distribution is visible; default target is 4-6 final manual `sft_eval` candidates.
