## Goal

Run two Optuna trials concurrently when possible, while preventing GPU-memory contention from permanently discarding a potentially good hyperparameter configuration.

A trial that OOMs during parallel execution should be queued for a later solo retry with the exact same sampled params. Only a trial that OOMs again while running alone should be treated as intrinsically infeasible.

## Core Decisions

1. Use two separate worker processes, not `study.optimize(n_jobs=2)` threads.
2. Use shared persistent Optuna storage, not an in-memory study.
3. Let Optuna sample freely from the approved first-study search space. Do not pre-prune `batch_size=2 && r=32` just because it might be high-memory.
4. If a trial OOMs in parallel, save its exact params to a JSONL queue and mark the original trial pruned with a clear reason.
5. After the parallel phase reaches the target completed-trial count, stop both workers and run the queued OOM configs sequentially.
6. Solo retry configs must use the same sampled params. Do not mutate rank, batch size, gradient accumulation, loss type, or any other sampled value inside the retry.
7. Include solo-recovered trials in final ranking if they complete successfully. Label them clearly as `parallel_oom_recovered=true` or `requires_solo=true`.

## New CLI Shape

Add or adapt CLI flags in `train_dpo.py`:

- `--optuna`: existing top-level Optuna launcher mode.
- `--parallel-workers 2`: number of worker subprocesses for the parallel phase.
- `--target-complete-trials 20`: target number of normal completed Optuna trials before solo retry phase.
- `--max-attempted-trials 60`: safety cap for total attempted parallel trials so repeated failures cannot loop forever.
- `--optuna-worker`: internal worker mode used by the launcher.
- `--worker-id <int>`: stable worker id for logs and queue metadata.
- `--study-storage <path-or-url>`: shared Optuna storage, default under the DPO output directory.
- `--study-name <name>`: stable study name, default to the existing DPO experiment name plus dataset/split identifier if useful.
- `--solo-retry-queue <path>`: JSONL file of queued parallel-OOM configs.
- `--solo-retry`: run queued configs sequentially after the parallel phase.

Example intended user command:

```bash
python dpo/train/train_dpo.py --optuna --parallel-workers 2 --target-complete-trials 20
```

The launcher should internally run two worker subprocesses, wait for them to finish, then automatically run the solo retry phase if the queue is non-empty.

## Shared Optuna Storage

Replace in-memory `optuna.create_study(...)` in Optuna mode with persistent shared storage.

Recommended first option:

```python
storage = optuna.storages.RDBStorage(
    url=f"sqlite:///{study_db_path}",
    engine_kwargs={"connect_args": {"timeout": 120}},
)
study = optuna.create_study(
    direction="maximize",
    study_name=study_name,
    storage=storage,
    load_if_exists=True,
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.NopPruner(),
)
```

SQLite is acceptable for two local workers if writes are small and the timeout is set. If SQLite lock contention appears, switch to Optuna journal storage or PostgreSQL. Do not use non-persistent in-memory storage for parallel workers.

Record these attrs or summary fields:

- study name and storage path
- dataset/data hash
- split manifest hash
- corrected dummy report path
- TRL version
- target complete trials
- max attempted trials
- parallel worker count
- solo retry queue path

## Parallel Phase Algorithm

Launcher:

1. Create/load the shared Optuna study.
2. Remove stale worker temp files for this run, or create a new run-specific directory.
3. Spawn `parallel_workers=2` subprocesses:
   - worker 0: `--optuna-worker --worker-id 0 ...`
   - worker 1: `--optuna-worker --worker-id 1 ...`
4. Each worker repeatedly runs one Optuna trial at a time until either:
   - shared study has at least `target_complete_trials` COMPLETE trials, or
   - shared study reaches `max_attempted_trials` total attempts, or
   - launcher sends/records a stop signal.
5. Wait for both workers to exit.
6. Consolidate OOM queue JSONL files.
7. If the consolidated queue is non-empty, run solo retry phase sequentially.
8. Write final consolidated summary.

Worker loop:

1. Load shared study.
2. Before asking for another trial, check study counts from shared storage.
3. If COMPLETE count >= target, exit cleanly.
4. Run `study.optimize(objective, n_trials=1, catch=(...safe exceptions...))` or an equivalent one-trial ask/tell loop.
5. After each trial, clean model/trainer, run GC, clear CUDA cache, reset peak stats, and re-check stop conditions.

Allow at most a small overshoot in COMPLETE trials, because two workers may finish nearly simultaneously. This is fine and should be reported.

## OOM Queue JSONL

Use a JSONL queue for parallel OOM configs. To avoid concurrent append corruption, use one of these approaches:

Preferred simple approach:

- Each worker writes to `oom_retry_queue_worker_<worker_id>.jsonl`.
- The launcher merges them into `oom_retry_queue.jsonl` after all workers exit.

Alternative:

- Use one shared `oom_retry_queue.jsonl` protected by a file lock.

JSONL record schema:

```json
{
  "schema_version": 1,
  "reason": "parallel_oom",
  "original_trial_number": 12,
  "worker_id": 1,
  "attempt": 1,
  "params": {
    "beta": 0.1,
    "num_train_epochs": 2,
    "learning_rate": 0.0000123,
    "lora_r": 32,
    "lora_dropout": 0.1,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "lr_scheduler_type": "cosine",
    "max_grad_norm": 0.5,
    "neftune_noise_alpha": 5.0,
    "length_mode": "sigmoid_norm"
  },
  "derived": {
    "lora_alpha": 64,
    "effective_batch": 8,
    "loss_type": ["sigmoid_norm"],
    "ld_alpha": null,
    "use_weighting": false
  },
  "vram": {
    "peak_allocated_gb": 0.0,
    "peak_reserved_gb": 0.0,
    "free_gb_at_start": 0.0,
    "free_gb_at_oom": 0.0
  },
  "stage": "model_build|precompute_ref|train|eval|save|unknown",
  "timestamp": "2026-05-22T00:00:00"
}
```

Deduplicate queue records before solo retry by canonical JSON serialization of `params`. Preserve all original trial numbers in metadata if the same params OOM multiple times.

## Handling Parallel OOMs

Inside the Optuna objective:

1. On `torch.cuda.OutOfMemoryError`, capture:
   - trial number
   - worker id
   - sampled params
   - derived params
   - stage if known
   - current/peak/reserved/free VRAM if available
2. Write a queue record to the worker's OOM JSONL.
3. Set trial user attrs:
   - `failure_reason="parallel_oom_queued_for_solo"`
   - `queued_for_solo_retry=True`
   - `worker_id=<id>`
   - `oom_stage=<stage>`
4. Delete trainer/model refs, run `gc.collect()`, `torch.cuda.empty_cache()`, and reset peak stats.
5. Raise `optuna.TrialPruned("parallel_oom_queued_for_solo")`.

Do not immediately retry the same params while the other worker is still running. That usually repeats the same contention failure and wastes time. Do not change sampled hyperparams inside the OOM trial.

## Solo Retry Phase

After the parallel workers exit:

1. Read and deduplicate `oom_retry_queue.jsonl`.
2. If the queue is empty, write `solo_retry_count=0` in the final summary and stop.
3. If non-empty, enqueue each params dict into the same study using `study.enqueue_trial(params)`.
4. Run a sequential solo retry worker for exactly the number of queued unique configs.
5. Prefer fresh subprocess-per-solo-trial if easy; otherwise at least fully delete model/trainer and clear CUDA between configs.
6. Each solo retry trial should use the exact same objective code and sampled params.
7. Add user attrs to the solo trial:
   - `solo_retry=True`
   - `parallel_oom_recovered=True` if it completes
   - `original_parallel_oom_trial_numbers=[...]` if available
   - `requires_solo=True`
8. If a solo retry OOMs, mark it pruned or failed with `failure_reason="solo_oom_intrinsic_or_too_large"`.

Successful solo retries are valid trials and should be included in final ranking. They should not be penalized for needing solo execution, but the final summary should clearly label them because they may be harder to run concurrently.

## Free VRAM Gate

Add a light pre-trial free-memory check to reduce avoidable collisions.

Before model construction:

1. Query free/total VRAM with `torch.cuda.mem_get_info()` or `nvidia-smi`.
2. If free VRAM is below a soft threshold, wait with jitter/backoff and recheck.
3. Suggested starting thresholds:
   - normal trial: wait until at least 10.5-11 GB free
   - high-memory sampled trial, such as `batch_size=2 && lora_r=32`: wait until at least 12-13 GB free
4. Cap waiting time. If still below threshold after the cap, proceed anyway or prune as `resource_wait_timeout` depending on implementation simplicity.

This gate is not a replacement for the solo retry queue. Its job is only to reduce obvious collisions.

## Trial Reporting

Per trial, record:

- trial number
- worker id
- sampled params
- derived params: effective batch, loss type, `ld_alpha`, `use_weighting`, `lora_alpha`
- final state
- failure reason, if any
- queued_for_solo_retry boolean
- solo_retry boolean
- original parallel OOM trial numbers, if any
- train loss
- eval reward accuracy
- eval reward margin
- eval loss
- chosen/rejected logps and rewards
- adapter diagnostics
- VRAM peak allocated/reserved
- saved adapter path

Final consolidated summary should include:

- best trial by validation reward accuracy
- all COMPLETE trials, including solo-recovered trials
- count of normal parallel completed trials
- count of solo-recovered completed trials
- count of parallel OOMs queued
- count of solo intrinsic OOMs
- count of other pruned/failed trials
- OOM counts grouped by hyperparameter combo, especially `batch_size`, `lora_r`, `neftune_noise_alpha`, and `length_mode`
- whether target completed trial count was reached
- any overshoot in completed trial count

## Search Space

Keep the first-study search space from `PLAN_FINAL.md` / corrected implementation:

- `beta`: `{0.01, 0.05, 0.1, 0.2}`
- `num_train_epochs`: `{1, 2, 3}`
- `learning_rate`: log range `5e-6` to `3e-5`
- `lora_r`: `{8, 16, 32}`, with `lora_alpha = 2 * r`
- `lora_dropout`: `{0.05, 0.1}`
- `per_device_train_batch_size`: `{1, 2}` if stable
- `gradient_accumulation_steps`: `{2, 4, 8}`, with effective batch around 4-16
- scheduler, max grad norm, `neftune_noise_alpha`, and `length_mode` as already defined

Do not add WPO. Do not add KTO. Do not run `sft_eval` inside Optuna. Primary objective remains validation `eval_rewards/accuracies`.

## Smoke Test

Before the full study:

1. Run two workers targeting 2 COMPLETE trials total.
2. Verify both workers attach to the same study.
3. Verify trial numbers are unique.
4. Verify trial directories are unique.
5. Verify queue files are created even if empty.
6. Verify final summary handles an empty queue.
7. If practical, force/simulate an OOM to verify queue -> solo enqueue -> solo retry -> summary.

Only after the smoke test passes, run the full target of 20 completed parallel trials plus automatic solo OOM retries.

## Acceptance Criteria

The implementation is acceptable when:

1. Two worker processes can run against one shared Optuna study.
2. The parallel phase stops after the requested number of COMPLETE trials, subject only to small worker overshoot.
3. Parallel OOM trials are queued with exact params and do not count toward the target completed trial count.
4. Queued OOM params are automatically replayed sequentially after the parallel phase.
5. Successful solo replays are included as valid trials and clearly labeled.
6. Solo OOMs are clearly labeled as intrinsically infeasible or too large for the current hardware.
7. Final summary is enough to rank trials and audit OOM behavior.
