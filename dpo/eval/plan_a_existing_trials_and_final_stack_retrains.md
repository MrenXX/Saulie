## Plan A: Final Exact-Stack Rescue Retrains

Existing v1.0/v1.1 DPO trials are abandoned. Trial-13 passing the narrow 10-skeleton fluency/CJK check but failing ordinary conversation proves the old trial set is not trustworthy. The failure is not merely over-steering on product skeletons; it is loss of normal conversational behavior under the SFT+DPO stack.

Plan A now means: run exactly two new exact-stack DPO rescue trials with minimal movement, then evaluate ordinary conversation first. Do not add small-talk rows as a workaround. Do not rerun broad Optuna. Do not use the old hybrid score to select a winner.

**Current Architecture To Preserve**
- `BnB8(raw base) + frozen SFT LoRA + new DPO LoRA`.
- Reference remains frozen SFT behavior.
- Only the DPO adapter is trainable.
- Deployment/cat remains a later check; first prove the HF exact stack can still talk normally.

**Why The Old Trials Are Dead**
- v1.1 trial-13 had low margin and clean English but failed a trivial normal chat.
- The failure was phrase fixation and conversational derailment, not Chinese text.
- Therefore CJK/fluency skeleton gates are insufficient.
- More old-trial testing is not worth the time.

## Step 2: Train Two New Exact-Stack Trials

Both configs intentionally reduce DPO movement. They are designed to test whether the current exact-stack architecture can support DPO at all without damaging normal conversation.

### New Trial 1: Minimal Reference-Anchored DPO

Use this as the first run.

- Loss: standard DPO/sigmoid with length regularization.
- `length_mode=ld_0.5`.
- `beta=0.20`.
- `num_train_epochs=1`.
- `learning_rate=5e-6`.
- `lora_r=8`.
- `lora_alpha=16`.
- `lora_dropout=0.05`.
- `neftune_noise_alpha=0.0`.
- Effective batch: `8`.
- Scheduler: `constant_with_warmup`.
- `warmup_ratio=0.1`.
- `weight_decay=0.05`.
- `max_grad_norm=0.3`.
- `label_smoothing=0.0`.

Rationale: high beta, one epoch, low LR, low rank, no NEFTune, and strong length regularization. This is the least risky standard-DPO test that still gives the preference objective a chance to move.

### New Trial 2: IPO Anti-Margin Run

Run this second, regardless of Trial 1 outcome unless Trial 1 is already perfect and time is gone.

- Loss: IPO via current repo `length_mode=ipo`; `parse_length_mode` maps this to TRL `loss_type=["ipo"]`.
- `beta=0.20`.
- `num_train_epochs=1`.
- `learning_rate=5e-6`.
- `lora_r=8`.
- `lora_alpha=16`.
- `lora_dropout=0.05`.
- `neftune_noise_alpha=0.0`.
- Effective batch: `8`.
- Scheduler: `constant_with_warmup`.
- `warmup_ratio=0.1`.
- `weight_decay=0.05`.
- `max_grad_norm=0.3`.
- `label_smoothing=0.0`.

Rationale: IPO is the cleanest available objective-level counter to unbounded DPO margin chasing. After the trial-13 normal-chat failure, this is more important than another standard DPO variant.

**Do Not Use These Old Search Settings For Step 2**
- Do not use `beta=0.03` or `0.05`.
- Do not use 2 epochs.
- Do not use LR above `1e-5`.
- Do not use `lora_r=16` or `32` for the rescue proof.
- Do not use NEFTune.
- Do not use broad Optuna as selector.
- Do not select by reward accuracy, hybrid score, or low CJK alone.

**Reality Check On Current Repo Support**
- Current code supports `length_mode=ipo` in `parse_length_mode`.
- Current v1.1 Optun sampler does not sample IPO or beta `0.20` by default.
- Therefore Step 2 should be run as fixed/enqueued rescue trials, not by rerunning the unchanged v1.1 Optuna search.
- If implementation work is needed later, it should only expose these exact fixed configs; it should not create a new broad search.

## Evaluation Gate: Normal Conversation First

Evaluate the HF exact stack at full strength before any cat export.

**Gate 1: Ordinary Conversation Stress**
Run at least `5` short chats, `5-8` turns each. Include correction/normal-chat prompts like:

1. Greeting and vague help request.
2. User calls out the assistant's tone.
3. User asks it to stop repeating a phrase.
4. User asks for a normal casual answer.
5. User explicitly says not to recommend products yet.

Hard fail on:
- repeated phrase loops;
- hook fixation;
- dramatic labels for normal user turns;
- product pivot when no product need exists;
- ignoring direct correction;
- CJK;
- broken English/template artifacts;
- sounding less conversationally sane than SFT baseline.

**Gate 2: Product/Steering Skeleton Smoke**
Only after Gate 1 passes:
- run the 10-skeleton smoke;
- require `10/10 GOOD`;
- zero `BORDERLINE`, zero `BAD`;
- no Type B/D collapse;
- no bloat or premature recommendation.

**Gate 3: Cat/Deploy Sanity**
Only after Gate 1 and Gate 2 pass:
- export full-strength cat at `w=1.0`;
- run the same normal-chat and skeleton gates on the cat artifact.

## Exit Rule

- If either new trial passes all gates, Plan A succeeds.
- If both fail ordinary conversation, stop exact-stack DPO. Do not keep adjusting hyperparams inside the same broken setup.
- If both fail, move to Plan B: SFT-baked base with one active DPO adapter, because the current active SFT+DPO residual stack is likely the wrong surface.
