# DPO Implementation Fixes

## Scope

The training objective and gates do not change:

- V4 remains the accepted dataset.
- No data repair should be reopened unless there is a hard structural blocker.
- Train full chosen/rejected trajectories while scoring only assistant/action tokens.
- Policy remains base + frozen SFT adapter + trainable DPO adapter.
- Reference remains frozen SFT behavior.
- No Optuna until a corrected dummy run succeeds and the user explicitly approves.

## Fixes

1. Fix adapter freeze semantics.

   `PeftModel.from_pretrained(..., is_trainable=False)` freezes the SFT adapter at load time, but later `model.base_model.set_adapter(["default", "dpo"])` can mark all active adapters trainable again. Because `default` must be active for policy behavior, adapter activation and adapter trainability must be controlled separately.

   In `dpo_trainer_compat.py`, update `ensure_policy_adapter_stack` so it activates `["default", "dpo"]`, then immediately enforces:

   - `default`: `requires_grad=False`
   - `ref`: `requires_grad=False`, if present
   - `dpo`: `requires_grad=True`
   - all other non-DPO params: `requires_grad=False`

   Add diagnostics that report active adapters, trainable parameter counts by adapter, optimizer parameter groups, and whether any non-DPO adapter has a gradient. Include this in the corrected dummy report.

2. Fix split fallback while preserving the approved Prompt B merge.

   For split stratification only, use `source_family`, so `prompt_b_repaired` and `prompt_b_exp500` are merged into one Prompt B split label. This is intentional because V4 has tiny raw-source/category cells such as `prompt_b_repaired x steering = 1` and `prompt_b_repaired x product_fit = 2`.

   Still write and report raw `dpo_source` everywhere in the manifest and diagnostics. The merge is only for split assignment, not for hiding provenance.

   Fix the fallback bug in `dpo_data.py`: the current code checks `(source_family, category)` against a counter that only contains `(source_family, category, opening_type)` keys, so 2-way fallback always misses and collapses too far. Build separate counters:

   - 3-way: `(source_family, category, opening_type)`
   - 2-way: `(source_family, category)`

   Then fall back from 3-way to 2-way correctly. Add split diagnostics for `dpo_source`, `source_family`, `category`, `source_family x category`, raw `dpo_source x category`, `opening_type`, `total_turns`, `divergence_turn`, and `has_branch_local_user`.

3. Enforce max length instead of warning.

   The custom pretokenized collator bypasses TRL's normal truncation path. A warning is not enough.

   Keep `MAX_LENGTH=704` unless the user's max-length audit changes it. During dataset build, hard fail if either `prompt + chosen` or `prompt + rejected` exceeds `MAX_LENGTH`. Log max observed length, p95 length, and overlength count in the dummy report. Do not silently truncate multi-turn preference trajectories.

4. Fix `sigmoid_norm` mapping.

   `sigmoid_norm` is TRL's length-normalized DPO loss. It is not WPO. WPO `use_weighting=True` reweights examples and should not be used as a substitute for length normalization.

   Map length modes as:

   - `none`: `loss_type=["sigmoid"]`, `use_weighting=False`, `ld_alpha=None`
   - `sigmoid_norm`: `loss_type=["sigmoid_norm"]`, `use_weighting=False`, `ld_alpha=None`
   - `ld_0.3`: `loss_type=["sigmoid"]`, `use_weighting=False`, `ld_alpha=0.3`
   - `ld_0.5`: `loss_type=["sigmoid"]`, `use_weighting=False`, `ld_alpha=0.5`
   - `ipo`: `loss_type=["ipo"]`, `use_weighting=False`, `ld_alpha=None`

   If the installed TRL version does not support `sigmoid_norm`, fail clearly during environment discovery or remove that Optuna arm. Do not silently substitute WPO.

5. Strengthen assistant-only mask audits.

   The critical invariant is: branch-local user text is visible as context but contributes zero scored tokens.

   Add decoded audits for:

   - ordinary single-assistant rows
   - multi-turn steering rows
   - Prompt B repaired rows
   - Prompt B exp500 rows
   - longest rows from the max-length audit
   - rows where chosen/rejected contain different branch-local user text

   Hard fail if any user-role completion token is scored, if chosen or rejected has zero scored assistant/action tokens, or if prompt/header/padding tokens are scored. Do not silently replace a mismatched prompt prefix without recording why the completion boundary remains correct.

6. Expand the corrected dummy report.

   Include at least:

   - loss, reward accuracy, reward margin, chosen/rejected rewards/logps
   - source/category metrics, raw `dpo_source` metrics, and `source_family` metrics
   - chosen/rejected scored-token lengths and margin-vs-length correlation
   - split manifest path and hash
   - mask audit output path
   - max-length stats
   - adapter active/frozen/trainable/gradient diagnostics
   - ref-cache hit/miss/path
   - runtime and peak VRAM allocated/reserved
   - saved adapter path/tree

   Treat the previous dummy as encouraging but not approval-ready until this corrected diagnostic report passes.

7. Tighten PEFT cat-merge validation.

   Keep using PEFT `add_weighted_adapter(..., combination_type="cat")`; this is still the correct inference-time approach for composing SFT + DPO into one vLLM-loadable adapter.

   Before merging, verify same base lineage, tokenizer/chat-template expectation, compatible target modules, no unsupported `modules_to_save` conflict, expected ranks/alphas/scaling, and no mixed tuner type.

   Make `--check-logps` use real Qwen chat-template tokenization with `tokenizer.apply_chat_template` on several fixed conversations. Compare stacked `["default", "dpo"]` logits against the cat adapter logits and record the tolerance. The report must also state the exact input/output adapter directory containing `adapter_config.json` and adapter weights, plus the vLLM path to load.

8. Rerun one corrected dummy and stop.

   Do not start Optuna in the same run. The corrected dummy is the gate.

   Approval criteria:

   - no mask violations
   - no overlength rows
   - only DPO adapter trainable and gradient-bearing
   - split diagnostics acceptable under Prompt B merged stratification
   - finite loss/reward metrics
   - adapter saved cleanly
   - VRAM still within the expected envelope

## Decisions

- Prompt B merge for split stratification is approved: use `source_family` for splitting and keep raw `dpo_source` for reporting and metric breakdowns.
- WPO is not part of the first Optuna study. `sigmoid_norm` means TRL's length-normalized DPO loss, not `use_weighting=True`.
- No Optuna until the corrected dummy passes and the user explicitly approves.
