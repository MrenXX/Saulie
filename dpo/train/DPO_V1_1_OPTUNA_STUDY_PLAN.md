# DPO Optuna Study v1.1 Plan

## Goal

Run a compact follow-up study that keeps the strong v1.0 signal but fixes the main problem: too many duplicate trials around the first winning config. v1.1 should test whether we can keep high preference accuracy and macro balance while lowering length correlation.

Use the v1.0 best config as a single anchor/control, not as the center of the whole search.

## Research Takeaways

- TRL `rewards/accuracies` is pairwise reward correctness: chosen reward > rejected reward.
- TRL `rewards/margins` is the average implicit reward gap: chosen reward - rejected reward.
- TRL `sigmoid_norm` exists specifically to address length bias by normalizing over non-mask tokens.
- Optuna `n_startup_trials` uses random sampling until that many finished trials exist in the study.
- Optuna recommends `constant_liar=True` for distributed/parallel costly trials to avoid workers evaluating very similar configs.
- True multi-objective Optuna is possible, but for a 20-trial local 3090 run use one conservative scalar objective and log every raw metric separately.

## Study Budget

- GPU: single RTX 3090.
- Workers: `--parallel-workers 2`.
- Target: `20` complete unique trials.
- Max attempted: `40-45` to allow duplicate/resource prunes without running forever.
- Startup trials: `n_startup_trials=10`.

## Sampler Update

Replace the plain sampler with a deterministic per-worker sampler. Do not use adjacent `SEED + worker_id` as the main recommendation, and do not use `base_seed + trial_number`. The seed initializes each worker's sampler stream; it should not be reset for every trial.

```python
import numpy as np


def optuna_sampler_seed(base_seed: int, worker_id: int | None) -> int:
    if worker_id is None:
        return base_seed
    return int(np.random.SeedSequence([base_seed, worker_id]).generate_state(1)[0])


optuna.samplers.TPESampler(
    seed=optuna_sampler_seed(cfg.optuna_base_seed, cfg.worker_id),
    n_startup_trials=10,
    multivariate=True,
    group=True,
    constant_liar=True,
)
```

The important implementation rule is: do not recreate a freshly seeded sampler for each `study.optimize(..., n_trials=1)` loop iteration. Keep one study/sampler object alive per worker process so the sampler's RNG stream and TPE state advance naturally across that worker's trials.

Why this decision:

- A fixed seed is useful for sequential reproducibility.
- A two-worker study is inherently not perfectly reproducible because whichever worker finishes first changes what the next suggestion sees.
- Giving every worker the exact same seeded sampler can cause duplicate startup suggestions, especially when this launcher runs one trial at a time.
- Per-worker deterministic seeds reduce accidental same-stream startup suggestions, while still making the run explainable from one logged base seed.
- `constant_liar=True` remains the main Optuna-supported parallel safeguard; it tells TPE to account for already-running trials when proposing new ones.
- Exact duplicate pruning is still required because Optuna can suggest duplicates even with good sampler settings.

For final reproducibility, rerun the selected best configs sequentially with `parallel_workers=1`. Treat the two-worker Optuna study as exploration, not as a bit-for-bit replay target.

## Duplicate Pruning

Add an exact params-key guard before model loading.

- Compute `params_key(params)` after sampling.
- If the same key already exists in `COMPLETE`, `RUNNING`, or `WAITING`, set:
  - `failure_reason = "duplicate_params"`
  - `duplicate_of = <existing trial number>`
- Then raise `optuna.TrialPruned("duplicate_params")`.

Intentional repeats should only happen later for final confirmation, not during v1.1 exploration.

## Search Space

Use a more diverse v1.1 space:

```python
beta: [0.03, 0.05, 0.08, 0.1]
num_train_epochs: [1, 2]
learning_rate: log range 8e-6 to 3e-5
lora_r: [8, 16, 32]
lora_dropout: [0.03, 0.05, 0.075, 0.1]
batch_combo: ["1x4", "1x8", "1x16", "2x2", "2x4", "2x8"]
lr_scheduler_type: ["linear", "cosine", "constant_with_warmup"]
max_grad_norm: [0.3, 0.5, 1.0]
neftune_noise_alpha: [0.0, 2.5, 5.0]
length_mode: ["sigmoid_norm", "ld_0.1", "ld_0.2", "ld_0.3", "ld_0.5"]
```

Drop `none` for v1.1 unless a control is explicitly needed; v1.0 showed it was weak. Add `ld_0.1` and `ld_0.2` to test lower tail weighting and length-bias reduction.

Sample `batch_combo` directly so invalid effective batches are not suggested and pruned after the fact.

## Anchor Trial

Enqueue exactly one v1.0 anchor/control:

```python
beta=0.05
num_train_epochs=1
learning_rate=2.3604024417191132e-05
lora_r=16
lora_dropout=0.05
batch_combo="1x8"
lr_scheduler_type="constant_with_warmup"
max_grad_norm=1.0
neftune_noise_alpha=5.0
length_mode="ld_0.3"
```

Mark it with `anchor_trial=True`. Duplicate pruning should prevent this config from being run again.

## Hybrid Objective

Since a lot of trials got 1.0 accuracy in study v1.0 Optuna should maximize a v1.1 hybrid score, not raw accuracy alone. The score should stay close to accuracy and only penalize known failure modes.

Suggested formula:

```python
score = accuracy
score -= 0.50 * max(0.0, 0.95 - macro_family_category)
score -= 0.15 * max(0.0, abs(len_corr) - 0.35)
score -= 0.15 * max(0.0, abs(abs_len_corr) - 0.40)
score -= 0.03 * max(0.0, eval_loss - 0.50)

if margin < 0:
    score -= 0.20
elif margin < 0.50:
    score -= 0.05 * (0.50 - margin)

if margin > 20:
    score -= min(0.20, 0.002 * (margin - 20))
```

Why this shape:

- Accuracy remains the base objective.
- Macro only penalizes source/category imbalance below `0.95`.
- Length correlation penalties start only after the v1.0 watch thresholds.
- Eval loss is only a weak sanity penalty.
- Margin is treated as a guardrail, not something to maximize.

Store this as `hybrid_score_v1_1` in Optuna and MLflow. Still log raw `eval_rewards_accuracy` and show raw-accuracy leaders separately in the report.

## Required Logging

MLflow should log all trial data by default, not just the final score:

- all sampled params
- all derived params
- `optuna_base_seed`, `worker_id`, `sampler_seed`, Optuna version, and sampler settings
- `hybrid_score_v1_1`
- raw accuracy, macro metrics, margin, eval loss, train loss
- chosen/rejected rewards and logps
- length diagnostics and correlations
- source/category/family bucket accuracies and margins
- runtime, VRAM, worker id, failure reason, duplicate status
- trial origin flags such as `anchor_trial`, `enqueued_trial`, `duplicate_of`, and whether the trial came from startup/random sampling or TPE if available
- `diagnostics.json`, `trial_summary.json`, and `study_report.html` as artifacts

The hybrid score is only for Optuna steering. Review and final selection should still inspect the separately logged metrics.

## Report Update

Update the HTML report to keep the dark, full-width style and make summary sparklines readable:

- keep dark mode
- use full browser width instead of a constrained `max-width`
- keep sortable/filterable trial table
- add numeric min/avg/max/latest values beside each sparkline
- add a short label showing which direction is better
- keep raw metrics visible even if Optuna optimizes `hybrid_score_v1_1`

## Implementation Handoff: Exact Code Changes

### `Saulie/dpo/train/optuna_parallel.py`

Implement the sampler and parallel-loop changes here.

1. Add sampler configuration constants and metadata.

```python
OPTUNA_BASE_SEED = 42
OPTUNA_STARTUP_TRIALS = 10
OPTUNA_SAMPLER_SETTINGS = {
    "n_startup_trials": OPTUNA_STARTUP_TRIALS,
    "multivariate": True,
    "group": True,
    "constant_liar": True,
}
```

2. Add `optuna_base_seed: int = OPTUNA_BASE_SEED` to `OptunaRunConfig`, add `--optuna-base-seed` to `add_parallel_cli_args()`, pass it through `config_from_args()`, and include it in `spawn_worker()` so every worker receives the same base seed.

3. Add `optuna_sampler_seed(base_seed, worker_id)` using `numpy.random.SeedSequence` as shown above. Log both the base seed and the derived worker seed.

4. Replace the hard-coded `TPESampler(seed=42)` in `load_study()` with a sampler built from the config:

```python
def build_sampler(cfg: OptunaRunConfig) -> optuna.samplers.TPESampler:
    sampler_seed = optuna_sampler_seed(cfg.optuna_base_seed, cfg.worker_id)
    return optuna.samplers.TPESampler(
        seed=sampler_seed,
        n_startup_trials=OPTUNA_STARTUP_TRIALS,
        multivariate=True,
        group=True,
        constant_liar=True,
    )


def load_study(cfg: OptunaRunConfig, sampler: optuna.samplers.BaseSampler | None = None) -> optuna.Study:
    ...
    sampler=sampler or build_sampler(cfg)
```

5. In `run_worker_loop()`, build/load the study once before the loop and reuse that same `study` object for every `study.optimize(objective, n_trials=1, ...)` call. Do not call `load_study(cfg)` at the top and bottom of every loop iteration. The current one-trial loop is okay; recreating a freshly seeded sampler inside that loop is not.

6. Set sampler metadata on each trial before training starts: `optuna_base_seed`, `sampler_seed`, `sampler_n_startup_trials`, `sampler_multivariate`, `sampler_group`, and `sampler_constant_liar`.

7. Update `sample_trial_params()` to use the v1.1 search space. Sample `batch_combo` directly, then expand it for training:

```python
batch_combo = trial.suggest_categorical("batch_combo", ["1x4", "1x8", "1x16", "2x2", "2x4", "2x8"])
batch_size, grad_accum = [int(part) for part in batch_combo.split("x")]
```

Return both `batch_combo` and the expanded `per_device_train_batch_size` / `gradient_accumulation_steps` so existing training code can consume the expanded values.

8. Add a duplicate guard helper that compares sampled Optuna params, not expanded training-only params. Run it after `sample_trial_params()` has populated `trial.params` and before model loading:

```python
states = (TrialState.COMPLETE, TrialState.RUNNING, TrialState.WAITING)
for old in trial.study.get_trials(deepcopy=False, states=states):
    if old.number != trial.number and params_key(dict(old.params)) == params_key(dict(trial.params)):
        trial.set_user_attr("failure_reason", "duplicate_params")
        trial.set_user_attr("duplicate_of", old.number)
        raise optuna.TrialPruned("duplicate_params")
```

9. Update `write_final_summary()` so `derived["effective_batch"]` works for both old trials with separate batch params and new v1.1 trials with `batch_combo`. If `batch_combo` exists, parse it instead of indexing `p["per_device_train_batch_size"]` and `p["gradient_accumulation_steps"]`.

10. Add summary fields for `optuna_base_seed`, sampler settings, unique complete config count, and duplicate-pruned count.

### `Saulie/dpo/train/train_dpo.py`

Implement the trial objective changes here.

1. Extend `parse_length_mode()` to support `ld_0.1` and `ld_0.2`.

2. Update `derive_trial_params()` to parse `batch_combo` when present, while staying backward compatible with old params that have `per_device_train_batch_size` and `gradient_accumulation_steps` directly.

3. In `run_optuna_trial()`, sample params and run the duplicate guard before any expensive model work. Ideally do this before `_ensure_optuna_data()` if practical; at minimum it must happen before VRAM waiting and `build_dpo_peft_model()`.

4. Set trial user attrs for sampler metadata from `optuna_parallel.py`: base seed, worker id, sampler seed, startup trial count, and sampler options.

5. Compute `hybrid_score_v1_1` after metrics and diagnostics are available. Store it with `trial.set_user_attr("hybrid_score_v1_1", score)` and return it from `run_optuna_trial()` instead of raw accuracy.

6. Keep all raw metrics exactly as separately logged user attrs. Do not replace `eval_rewards_accuracy`, macro metrics, margin, loss, chosen/rejected rewards, logps, or length diagnostics with the hybrid score.

7. Preserve the legacy single-process path, but update its sampler to use the same sampler settings where possible. If exact replay is needed, use single-process/one-worker reruns of selected configs.

### `Saulie/dpo/train/mlflow_study.py`

Make MLflow capture the full trial evidence by default.

1. Add `hybrid_score_v1_1` and sampler metadata to logged metrics/params.

2. Prefer logging all numeric scalar user attrs as metrics, with explicit skips only for JSON/blob fields such as `val_diagnostics_json`, `adapter_diagnostics`, `derived`, `ref_cache`, and `vram`. This avoids silently dropping new diagnostic metrics.

3. Log trial origin and failure fields as params/tags: `anchor_trial`, `enqueued_trial`, `duplicate_of`, `failure_reason`, `solo_retry`, and `parallel_oom_recovered`.

### `Saulie/dpo/train/study_report.py`

Extend the report after v1.1 summaries include the new fields.

1. Add `hybrid_score_v1_1` to the trial ledger when present.

2. Add a hybrid-score shortlist while keeping raw-accuracy leaders visible separately.

3. Show duplicate-prune counts, unique complete config counts, base seed, sampler seed/settings, and startup trial count in provenance or summary cards.

4. Keep the dark full-width layout and numeric sparkline summaries already added.

## Acceptance Criteria

- Smoke test still passes.
- Duplicate params are pruned before model load.
- Each worker keeps one persistent study/sampler object across its one-trial loop.
- Sampler seeds are deterministic per worker and logged with the base seed.
- The implementation does not use `base_seed + trial_number` or reseed the sampler per trial.
- Summary reports complete unique trials, duplicate prunes, and startup trial count.
- v1.1 report includes both raw metric leaders and hybrid-score leaders.
- MLflow parent and child runs contain all raw metrics plus the hybrid score.