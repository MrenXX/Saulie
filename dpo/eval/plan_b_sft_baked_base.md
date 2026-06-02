## Plan B: SFT-Baked Base

Create a dense SFT-merged checkpoint and use it as a new self-consistent base. This sacrifices bit-level fidelity to the original SFT training setup, but removes the active SFT LoRA from the stack. Expert recommendation: first run a cheap diagnostic with already-trained DPO adapters on the baked base, but treat that as opportunistic. The principled Plan B is retraining DPO on the baked SFT base so policy and reference share the same new base.

**Core Tradeoff**
- Old/current stack: `BnB8(raw base) + SFT LoRA + DPO LoRA`.
- Baked-base stack: `BnB8(SFT-merged base) + DPO LoRA`.
- These are not numerically identical because quantizing `W` and adding LoRA is not the same as merging LoRA into `W` and then quantizing.
- The payoff is a cleaner one-adapter DPO surface and no active SFT+DPO LoRA interference.

**Expert Answer: Try Old DPO Adapters Or Retrain?**
1. Try old DPO adapters on the baked base first because it is cheap and may reveal a usable candidate quickly.
2. Do not trust old DPO adapters as the principled fix unless they pass the same perfect open-generation gate. They were trained against the old `BnB8(raw)+SFT LoRA` policy/reference geometry.
3. If old adapters do not pass perfectly, retrain DPO on the baked SFT base. That is the correct Plan B because the reference distribution and policy initialization become self-consistent again.

**Merge/Saving Plan**
1. Load the raw Qwen base unquantized, not with BitsAndBytes.
2. Use the checkpoint-native dense dtype if possible. The local path name says `MODEL_ID_BF16`, so BF16 is likely the cleanest dense dtype; use FP16 only if the serving/training stack requires it.
3. Load the SFT adapter onto the dense raw base with PEFT.
4. Merge with `merge_and_unload` or equivalent PEFT merge API.
5. Save the result as a new checkpoint, e.g. `Qwen3-4B-Instruct-2507-SFT-MERGED-FP16` or `...-BF16` depending on dtype.
6. Save/copy tokenizer files and ensure the same patched chat template used by SFT/DPO smoke is available.
7. Record merge metadata: raw base path, SFT adapter path, dtype, PEFT version, transformers version, and hash/timestamp.

**SFT Control Before Any DPO**
1. Control A: `BnB8(raw base) + SFT LoRA`.
2. Control B: `BnB8(SFT-merged base)`, no adapter.
3. Generate the same 10 skeletons with greedy decoding.
4. Optional: run a small sampled smoke on 4 skeletons at the existing sample settings.
5. Continue only if Control B is behaviorally close to Control A: coherent English, same steering style, no obvious loss of SFT quality, and no Type B/D regression.
6. If Control B is worse than SFT adapter baseline, stop Plan B. The merge/quantization shift is too costly.

**Cheap Old-DPO Diagnostic**
1. Load `BnB8(SFT-merged base) + DPO adapter only`; do not load the old SFT adapter.
2. Test existing adapters in this order: v1.1 trial `13`, v1.0 trial `23`, v1.1 trial `0`, v1.1 trial `17`, v1.1 trial `15`, and v1.1 trial `10` as calibration.
3. Run the same 10-skeleton greedy smoke at full adapter strength.
4. Reject on any CJK, BAD, BORDERLINE, bloat, or Type B/D regression.
5. If one old DPO adapter passes perfectly, run the FP8/SFT-merged-base smoke with that single DPO adapter. If it still passes, it can be considered an emergency candidate, but it should be validated carefully because it was not trained on this baked base.

**Plan B Retrain/Search Strategy**
1. Do not rerun the old broad Optuna study unchanged. The old hybrid score is useful telemetry, but it cannot be the sole objective because it already failed to detect open-generation collapse.
2. Use Plan A and the baked-base SFT control as coarse priors, not as exact hyperparameter targets. This is not random if we use broad behavioral signals: lower reward margin, lower length-correlation, no CJK, less bloat, stable Type B/D behavior.
3. Run a tiny sequential mini-study, not a wide search. Target `3-5` baked-base DPO runs total unless an early run passes perfectly.
4. Keep the old offline metrics logged: reward accuracy, reward margin, eval loss, macro accuracies, length correlations. Use them to diagnose, not to crown winners.
5. Selection is hard-gated by open generation: a trial with perfect offline metrics and a bad 10-skeleton smoke is rejected.
6. If using Optuna for bookkeeping, make the returned objective secondary. Either return `0` for any trial that fails smoke, or do not let Optuna decide automatically; manually choose from candidates that pass smoke.
7. Since there are no external LLM judge APIs, use deterministic filters plus human labels. This is acceptable because the smoke set is tiny and the failure modes are obvious enough to read.

**No-API Open-Generation Evaluation**
1. Generate SFT baseline outputs once for the 10 skeletons and keep them as the anchor.
2. For each candidate, generate the same 10 skeletons greedy. Add one 4-skeleton sampled sanity pass only after greedy passes.
3. Hard-fail deterministic checks: CJK, template leakage, empty answer, repeated fragments, extreme length inflation, assistant refusing normal product guidance, or malformed turn structure.
4. Human-label each skeleton as `GOOD`, `BORDERLINE`, or `BAD` using the SFT baseline as the reference.
5. `GOOD` means coherent English, follows the user turn, preserves timing, earns the recommendation, gives a usable final answer, and is not worse than SFT on Type B/D.
6. `BORDERLINE` means readable but degraded: too salesy, too abstract, bloated, oddly metaphorical, premature product pivot, or weaker than SFT.
7. `BAD` means incoherent, CJK, nonsense product mapping, conversation derailment, or failure to answer.
8. Accept only `10/10 GOOD` for release candidates. For exploratory ranking, prefer candidates with fewer words, fewer pivots, and stronger Type B/D behavior.
9. To reduce bias, optionally blind the outputs by hiding trial names before human labeling. This is more valuable than a weak local LLM judge.

**How Results Should Orient The Search**
1. If old lower-margin adapters on baked base are coherent but weak, try slightly more DPO movement: `r=16` or beta `0.05-0.08`, still one epoch and low LR.
2. If old adapters are coherent but bloated or salesy, keep rank low and increase beta: `r=8`, beta `0.10-0.20`, stronger length regularization or IPO.
3. If old adapters produce CJK/nonsense even on baked base, reduce movement aggressively: `r=8`, beta `0.20`, `lr=5e-6`, IPO first, no NEFTune.
4. If SFT-baked base itself is weaker than SFT LoRA baseline, do not search DPO. Fix or abandon Plan B.
5. Do not overfit to one skeleton. Use the 10-skeleton set as a hard gate, then confirm on 4-6 extra held-out skeletons before finalizing.


**Principled Baked-Base DPO Retrain**

1. Modify the DPO model builder to load `BnB8(SFT-merged base)` directly.
2. Do not load the old SFT adapter.
3. Add only the trainable DPO LoRA.
4. Ensure the reference is the frozen SFT-merged base, not raw base and not a stale SFT adapter.
5. Update diagnostics so policy stack reports one active DPO adapter over the baked base, not `default+dpo`.
6. Train a tiny sequential mini-study, starting with the two baked-base configs below. Add at most `1-3` follow-up runs only if Plan A/old-adapter diagnostics reveal a clear direction, such as stable-but-weak needing slightly more movement or bloated needing stronger regularization.
7. Gate at `w=1.0`; no residual scaling.

**Baked-Base Retrain Config 1**
- Base: `BnB8(SFT-merged base)`.
- Reference: frozen SFT-merged base.
- Loss: standard DPO with `beta=0.10`.
- `num_train_epochs=1`.
- `learning_rate=8e-6` to `1e-5`.
- `lora_r=8`, `lora_alpha=16`, `lora_dropout=0.05`.
- No NEFTune.
- Use length regularization if available and stable, preferably `ld_0.5`.

**Baked-Base Retrain Config 2**
- Base: `BnB8(SFT-merged base)`.
- Reference: frozen SFT-merged base.
- Loss: IPO if cleanly supported.
- `beta=0.20` or `0.10` if the local IPO implementation is stiff.
- `num_train_epochs=1`.
- `learning_rate=5e-6`.
- `lora_r=8`, `lora_alpha=16`, `lora_dropout=0.05`.
- No NEFTune.

**Decision Rule**
- If SFT Control B fails, Plan B fails before DPO.
- If old DPO adapters pass perfectly on baked base, use them only after broader validation; this is a lucky shortcut, not the expected fix.
- If old DPO adapters fail but SFT Control B passes, retrain DPO on the baked base.
- If baked-base retrains fail the perfect 10-skeleton gate, ship SFT and stop DPO work for this release window.

**Relevant Files To Modify Or Reuse**
- `repo/Saulie_dpo_eval/dpo/train/model_load.py` — add a baked-base loader for BnB8(SFT-merged base).
- `repo/Saulie_dpo_eval/dpo/train/train_dpo.py` — add a baked-base DPO build path that does not load the SFT adapter.
- `repo/Saulie_dpo_eval/dpo/train/dpo_trainer_compat.py` — ensure policy/reference diagnostics match the one-adapter baked-base design.
- `repo/Saulie_dpo_eval/dpo/train/smoke_policy_stack_hf.py` — add or mirror a smoke mode for baked base + DPO-only.
- `repo/Saulie_dpo_eval/dpo/train/merge_sft_dpo_lora.py` — not needed for baked-base one-adapter serving, except as reference for adapter diagnostics.

**Verification**
1. SFT-merged checkpoint loads without SFT adapter and generates coherent SFT-like outputs.
2. Control B passes the same 10-skeleton SFT baseline smoke.
3. Old DPO diagnostic is clearly labeled as non-principled if used.
4. Baked-base retrain diagnostics show no trainable non-DPO parameters.
5. Final candidate passes `10/10 GOOD`, zero CJK, zero BORDERLINE/BAD at `w=1.0`.
