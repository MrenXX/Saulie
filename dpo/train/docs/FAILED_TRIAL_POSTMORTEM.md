# Failed / zombie Optuna trials — postmortem

Study: `steering-dpo-v1.1-v4-seed42`  
Run dir: `models/steering-dpo-v1.1/optuna-run-20260526-042551`

## Summary

| Trial(s) | Optuna state | Root cause |
|----------|--------------|------------|
| #2, #3 | FAIL | Pre–GPU-lock era: **two workers trained on one GPU** → extreme step times; process killed manually (~10h). |
| #4, #6 | FAIL | **Manual DB fix** after bogus `study.tell(..., 3.0)` marked them COMPLETE; not training failures. |
| #26, #27 | RUNNING (zombie) | Training finished (#26) or started (#27); **hung after `trainer.train()`** in post-train path; no `diagnostics.json`; killed on restart → `fail_stale_trials`. |

## Zombie pattern (hardware busy, logs frozen)

### What you see

- `nvidia-smi` shows ~24GB VRAM and high GPU utilization.
- Worker logs stop updating for 1–3+ hours after the last training progress bar.
- Processes stay alive at high CPU (64–73%).
- Optuna trial stays `RUNNING`; study does not advance.

### Where it sticks (code path)

After `trainer.train()` returns, `run_optuna_trial` in `train_dpo.py`:

1. Logs ref_cache HIT lines.
2. `save_dpo_adapter` → **often completes** (see `trial-26/best_adapter`, `checkpoint-210`).
3. **`compute_val_diagnostics`** — 52 sequential forward passes, **one row at a time**, **outside `gpu_train_lock`**.
4. Only then: `diagnostics.json`, Optuna `COMPLETE`, MLflow child run.

Evidence: **trial #26** has adapter + checkpoint at 12:09 but **no `diagnostics.json`**. Last log lines match end-of-train + ref_cache, not scorecard/hybrid log.

### Why GPU stays busy

1. **Post-train val scoring still uses the model on CUDA** (`_per_batch_dpo_scores` in `dpo_diagnostics.py`). This is real work, not an idle OOM sleep.
2. **`gpu_train_lock` only wraps `run_training()`**, not val diagnostics. If both workers exit training close together, both can run 52× forward on the same GPU → severe slowdown or unstable behavior.
3. **No heartbeat** during `compute_val_diagnostics` (heartbeats are training-step callbacks only), so logs look “dead” even when working.
4. **Heavy configs** (`ld_0.5`, 2 epochs, large effective batch) before the lock made **training** look like zombies too (multi-minute steps); that is separate but looked the same in `tail -f`.

### Fixes already in tree

- `gpu_train_lock.py` — serializes `trainer.train()` across workers.
- `StepWatchdogCallback` — prunes pathological **training** steps (stall cliff, 3h/step, 12h/trial caps).
- `babysit_study.py` — kills processes when log age exceeds ~2.5× trial budget.
- `fail_stale_trials` on worker restart — marks orphaned RUNNING as failed.

### Fixes implemented (post-v1.1 study)

- `gpu_train_lock` now covers VRAM wait, model build, train, save, val diagnostics, and GPU cleanup.
- `compute_val_diagnostics` logs progress every N rows and enforces `DPO_MAX_VAL_DIAG_WALL_S` (default 3600s).
- `last_stage` and `val_diag_progress` stored on each trial; Optuna RDB heartbeat + `fail_stale_trials`.
- Babysit treats moving `val_diag` logs as alive (not stale).

## Trials #2 and #3 (first night)

- **#2** (W1): `ld_0.5`, 2 epochs, `1x16` — worst-case length + batch.
- **#3** (W0): `ld_0.5`, `2x2`.
- Occurred **before** `gpu_train_lock`; both workers ran DPO forward concurrently on one GPU.
- Observed step times ~6s → **300s+** per step; heartbeats rare → appeared frozen for hours.
- User killed processes; Optuna recorded FAIL with no `failure_reason`.

## Trials #4 and #6

- Accidental `study.tell(4/6, 3.0)` during babysit recovery.
- Corrected via SQLite `state='FAIL'`; not related to GPU hangs.
