# DPO Optuna v1.1 — agent handoff

**Repo:** `https://github.com/MrenXX/Saulie` (branch `main`)  
**Conda env:** `saulgman`  
**Last updated:** 2026-05-27

---

## Goal

1. **Done:** Official **DPO v1.0** and **DPO v1.1** Optuna studies (20 target complete each, 2 workers, seed 42).
2. **Done:** Zombie-trial fixes on `main` (`d3cdd4f`) — full GPU lock, val-diag telemetry, Optuna heartbeats.
3. **Next (this handoff):** **Final eval phase** — merge SFT+DPO adapters, vLLM generation on 52 skeletons, LLM judge, pick deployment adapter. See **`dpo/eval/DPO_FINAL_EVAL_PLAN.md`**.

Do **not** start DPO v1.2 unless the user explicitly asks after eval.

---

## Path index (eval agent — read this first)

All paths are under repo root `/root/saulie` unless noted.

### Eval runbook and judge packet

| Path | Purpose |
|------|---------|
| `dpo/eval/DPO_FINAL_EVAL_PLAN.md` | **Primary runbook** — candidate slate, merge commands, vLLM harness, manifest shape |
| `dpo/eval/README.md` | What to upload to the scoring LLM |
| `dpo/eval/eval_skeletons.json` | 52 validation skeletons (user turns only) |
| `dpo/eval/llm_judge_prompt_dpo.md` | Judge rubric (DPO final validation) |
| `dpo/eval/candidate_metadata_finalists.json` | Pre-built metrics metadata for finalists + SFT baseline |

Parallel copies (SFT-era layout): `sft/sft_eval/eval_skeletons.json`, `sft/sft_eval/llm_judge_prompt.md`

### Training / merge utilities

| Path | Purpose |
|------|---------|
| `dpo/train/merge_sft_dpo_lora.py` | Cat-merge SFT + DPO → `sft_dpo_cat` for vLLM (`--check-logps` required) |
| `dpo/train/paths.py` | `SFT_ADAPTER`, `MODEL_ID_BF16`, `MODEL_ID_FP8`, `DATA_PATH` |
| `dpo/train/HANDOFF.md` | This file |

**Frozen SFT policy (all DPO training and merge):**

| Path | Purpose |
|------|---------|
| `/root/saulie/sft/models/steering-sft-v1.1/trial-17/best_adapter` | SFT baseline adapter (`default` in training) |
| `/root/saulie/Qwen3-4B-Instruct-2507` | BF16 base (`MODEL_ID_BF16`) |
| `/root/saulie/Qwen3-4B-Instruct-2507-FP8` | FP8 base for vLLM (`MODEL_ID_FP8`) |

### Dataset and training split (context for judges)

| Path | Purpose |
|------|---------|
| `dpo/dataset/DPO_522_prompt_a_and_prompt_b_V4_repaired.jsonl` | Full DPO v4 dataset |
| `dpo/train/dataset/dpo_v4_split_seed_42.jsonl` | Train/val/test split manifest (52 val rows) |

### MLflow

| Path | Purpose |
|------|---------|
| `dpo/train/mlruns/` | Tracking store |
| Experiment `steering-dpo-v1.1` | Official v1.1 parent + nested trial runs |
| Experiment `steering-dpo-v1.0` | Official v1.0 runs |

```bash
mlflow ui --backend-store-uri file:///root/saulie/dpo/train/mlruns --port 5001
```

---

## Official study: DPO v1.1 (primary for eval)

**Run dir:** `/root/saulie/dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551`

| Artifact | Absolute path |
|----------|----------------|
| **trial_summary.json** | `/root/saulie/dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial_summary.json` |
| **study_report.html** | `/root/saulie/dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/study_report.html` |
| Optuna DB | `/root/saulie/dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/optuna_study.db` |
| Study name | `steering-dpo-v1.1-v4-seed42` |
| Launcher log | `.../launcher.log` |
| Worker logs | `.../worker_0.log`, `.../worker_1.log` |
| OOM queue | `.../oom_retry_queue.jsonl` |
| Monitor pointer | `.../MONITOR.txt` |

**Summary facts:** `target_reached: true`, 21 COMPLETE (overshoot +1), best Optuna trial **#23** (hybrid 1.0). Trials **#23, #29, #30** tie at perfect eval accuracy — use `diagnostics.json` margins/categories, not score alone.

**Important:** DPO adapters live under **`optuna-run-.../trial-N/`**, not `steering-dpo-v1.1/trial-N/` (that flat path does not exist). Update merge/manifest paths accordingly.

### v1.1 eval candidate adapters (from `DPO_FINAL_EVAL_PLAN.md`)

Raw DPO adapter for merge (`--dpo-adapter` must point at `best_adapter`):

| Trial | `best_adapter` path | `diagnostics.json` |
|-------|---------------------|-------------------|
| 29 | `.../optuna-run-20260526-042551/trial-29/best_adapter` | `.../trial-29/diagnostics.json` |
| 30 | `.../trial-30/best_adapter` | `.../trial-30/diagnostics.json` |
| 23 | `.../trial-23/best_adapter` | `.../trial-23/diagnostics.json` |
| 17 | `.../trial-17/best_adapter` | `.../trial-17/diagnostics.json` |
| 10 | `.../trial-10/best_adapter` | `.../trial-10/diagnostics.json` |
| 15 | `.../trial-15/best_adapter` | `.../trial-15/diagnostics.json` |
| 16 | `.../trial-16/best_adapter` | `.../trial-16/diagnostics.json` |
| 28 | `.../trial-28/best_adapter` | `.../trial-28/diagnostics.json` |
| 13 | `.../trial-13/best_adapter` | `.../trial-13/diagnostics.json` |

Prefix for all rows: `/root/saulie/dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551`

**Suggested merged output** (create with `merge_sft_dpo_lora.py`):

`.../trial-{N}/sft_dpo_cat` (e.g. `trial-29/sft_dpo_cat`)

**Reduced slate** (if VRAM/time limited): SFT trial 17 + DPO v1.1 trials **29, 23, 17, 10, 15** + v1.0 trial **1**.

---

## Official study: DPO v1.0 (reference candidate for eval)

**Run dir:** `/root/saulie/dpo/train/models/steering-dpo-v1.0/optuna-run-20260523-041252`

| Artifact | Absolute path |
|----------|----------------|
| **trial_summary.json** (canonical) | `/root/saulie/dpo/train/models/steering-dpo-v1.0/optuna-run-20260523-041252/trial_summary.json` |
| **trial_summary.json** (copy) | `/root/saulie/dpo/results/optuna-run-20260523-041252/trial_summary.json` |
| **study_report.html** | `/root/saulie/dpo/train/models/steering-dpo-v1.0/optuna-run-20260523-041252/study_report.html` |
| Optuna DB | `/root/saulie/dpo/train/models/steering-dpo-v1.0/optuna-run-20260523-041252/optuna_study.db` |
| Study name | `steering-dpo-v1.0-v4-seed42` |

**Summary facts:** `target_reached: true`, 21 COMPLETE, best trial **#1** (accuracy 1.0). Include **only trial 1** from v1.0 in the default eval slate.

| Trial | `best_adapter` | `diagnostics.json` |
|-------|----------------|-------------------|
| 1 | `/root/saulie/dpo/train/models/steering-dpo-v1.0/optuna-run-20260523-041252/trial-1/best_adapter` | `.../trial-1/diagnostics.json` |

**Merged output (create):** `.../trial-1/sft_dpo_cat`

**Note:** v1.0 objective is **accuracy-only** (no hybrid score). v1.1 summaries include `hybrid_score_v1_1` in per-trial `user_attrs` / `complete_trials`.

---

## Other Optuna runs (not for default eval)

| Run dir | trial_summary | Role |
|---------|---------------|------|
| `.../steering-dpo-v1.1/optuna-run-20260525-220741` | `.../trial_summary.json` | Failed first smoke (MLflow bug) — ignore |
| `.../steering-dpo-v1.1/optuna-run-20260526-162903` | `.../trial_summary.json` | 2-trial smoke (pre–full GPU lock) |
| `.../steering-dpo-v1.1/optuna-run-20260527-202826-smoke-zombie-fix` | `.../trial_summary.json` | Post–zombie-fix smoke; 3 COMPLETE, 0 RUNNING |
| `.../steering-dpo-v1.0/optuna-run-20260522-*` | various | Older v1.0 attempts — use **041252** only |

---

## Current progress

### Study training status — **finished**

v1.1 and v1.0 official runs complete. Full paths in **Path index** above.

**v1.1 best params (#23):** `ld_0.1`, 2 epochs, `1x4`, `beta=0.05`, `lora_r=32`, `lr≈2.27e-5`, cosine, `neftune=2.5`, `max_grad_norm=0.3`

### Code on `main` (recent commits)

| Commit | Summary |
|--------|---------|
| `2380e42` | v1.1 Optuna stack: hybrid score, TPE, duplicate prune, MLflow, `study_report.py`, narrow `gpu_train_lock`, watchdog, babysit |
| `d3cdd4f` | **Zombie fix:** full GPU lock through val diagnostics, progress logs, val diag timeout, Optuna heartbeats, `last_stage`, babysit val_diag awareness |

### Smoke test after zombie fix — **passed**

Run: `optuna-run-20260527-202826-smoke-zombie-fix`  
- `target_reached: true`, **0 RUNNING** after finish, val_diag row logs visible.

---

## What worked

- **`gpu_train_lock`** (`dpo/train/gpu_train_lock.py`) — file lock per run dir; after `d3cdd4f` covers VRAM wait → model build → train → save → `compute_val_diagnostics` → GPU cleanup inside one `with` block.
- **`StepWatchdogCallback`** — training stall cliff + hard caps (`DPO_MAX_STEP_WALL_S`, `DPO_MAX_TRIAL_WALL_S`, `DPO_STALL_*`); does not kill legit slow steady runs.
- **`compute_val_diagnostics`** — per-row progress logs + `DPO_MAX_VAL_DIAG_WALL_S` (default 3600s) + `trial.set_user_attr("val_diag_progress")` for heartbeats.
- **`fail_stale_trials`** on worker start + Optuna **RDB heartbeats** (`heartbeat_interval=60`, `grace_period=600`, env `DPO_OPTUNA_HEARTBEAT_*`).
- **`babysit_study.py`** — poll until `trial_summary.json` + `target_reached`; don’t kill when `val_diag` logs still moving.
- **Parallel Optuna v1.1** — per-worker TPE + `constant_liar`, anchor trial enqueue, duplicate param prune, hybrid objective in `dpo_diagnostics.compute_hybrid_score_v1_1`.
- **OOM path** — parallel OOM → worker queue → solo retry (0 OOMs in official run).
- **Restart after hang** — kill processes, restart launcher; `fail_stale_trials` clears zombie RUNNING.

---

## What didn’t work (don’t repeat)

| Mistake | Why |
|---------|-----|
| Two workers training **without** GPU lock | ~6s → 300s+ steps; looked frozen for hours; not OOM idle GPU. |
| **Narrow lock** (train only) | Post-train `compute_val_diagnostics` overlapped on one GPU → multi-hour “zombies” with adapter saved but no `diagnostics.json` (trials #26, #27). |
| **`study.tell(trial, 3.0)`** to mark FAIL | Bogus COMPLETE for #4, #6; fixed via SQLite `UPDATE trials SET state='FAIL'`. |
| **20 min agent sleeps** waiting for study | Session interrupts; use short polls or `babysit_study.py`, not long `Await`. |
| Assuming frozen logs = OOM sleep | Usually real compute with sparse heartbeats (training) or none (old val_diag path). |
| Tailing **wrong** `optuna-run-*` dir | User had `220741` report open vs official `042551`. |

**DB note:** Official study may still list **#26, #27 as RUNNING** (zombies from pre-restart). Summary and best trial are valid; optional cleanup: `fail_stale_trials` or manual FAIL.

**Hybrid 1.0 ties:** Trials #23, #29, #30 all hit perfect eval accuracy on 52 rows — compare margins/category breakdown in `trial-23/diagnostics.json`, not score alone.

---

## Key files

| Path | Purpose |
|------|---------|
| `dpo/train/train_dpo.py` | CLI, `run_optuna_trial`, GPU lock boundary |
| `dpo/train/optuna_parallel.py` | Launcher, workers, storage+heartbeat, summary |
| `dpo/train/dpo_diagnostics.py` | Val diagnostics, hybrid score, scorecard |
| `dpo/train/dpo_trainer_compat.py` | `StepWatchdogCallback`, `TrialWallTimeout` |
| `dpo/train/gpu_train_lock.py` | Cross-worker GPU mutex |
| `dpo/train/scripts/babysit_study.py` | Study babysitting |
| `dpo/train/scripts/start_v11_main_tmux.sh` | tmux launcher helper |
| `dpo/train/DPO_V1_1_OPTUNA_STUDY_PLAN.md` | Original v1.1 plan |
| `dpo/train/docs/FAILED_TRIAL_POSTMORTEM.md` | Zombie trial postmortem (partially pre-`d3cdd4f` wording in “narrow lock” section) |

---

## Commands

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate saulgman
cd /root/saulie

# Merge v1.1 eval candidates (correct paths)
V11=/root/saulie/dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551
for trial in 29 30 23 17 10 15 16 28 13; do
  python dpo/train/merge_sft_dpo_lora.py \
    --dpo-adapter "$V11/trial-${trial}/best_adapter" \
    --output "$V11/trial-${trial}/sft_dpo_cat" \
    --check-logps
done

# Merge v1.0 reference candidate
V10=/root/saulie/dpo/train/models/steering-dpo-v1.0/optuna-run-20260523-041252
python dpo/train/merge_sft_dpo_lora.py \
  --dpo-adapter "$V10/trial-1/best_adapter" \
  --output "$V10/trial-1/sft_dpo_cat" \
  --check-logps

# Status
RUN=/root/saulie/dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551
python -c "
import optuna
from collections import Counter
s=optuna.load_study(study_name='steering-dpo-v1.1-v4-seed42', storage=f'sqlite:///{RUN}/optuna_study.db')
print(Counter(t.state.name for t in s.trials))
print('best', s.best_trial.number, s.best_value)
"

# New v1.2 study (new run dir + study name)
python dpo/train/train_dpo.py --optuna --study-version v1.1 --parallel-workers 2 \
  --run-dir=/path/to/new-run --study-storage=/path/to/new-run/optuna_study.db \
  --study-name=steering-dpo-v1.1-v5-seed42

# Smoke (2 complete trials)
python dpo/train/train_dpo.py --optuna --optuna-smoke --study-version v1.1 --parallel-workers 2 \
  --run-dir=/path/to/smoke-run --study-storage=/path/to/smoke-run/optuna_study.db \
  --study-name=steering-dpo-v1.1-smoke-$(date +%H%M%S)

# Babysit until done
python dpo/train/scripts/babysit_study.py --run-dir "$RUN" --target-complete 20 --poll-min 20

# MLflow
mlflow ui --backend-store-uri file:///root/saulie/dpo/train/mlruns --port 5001
```

**Env overrides:**

```bash
export DPO_MAX_VAL_DIAG_WALL_S=3600
export DPO_OPTUNA_HEARTBEAT_INTERVAL=60
export DPO_OPTUNA_HEARTBEAT_GRACE=600
export DPO_MAX_STEP_WALL_S=10800
export DPO_MAX_TRIAL_WALL_S=43200
```

---

## Next steps (eval phase)

1. Read **`dpo/eval/DPO_FINAL_EVAL_PLAN.md`** end-to-end.
2. **Merge** all slate trials with `merge_sft_dpo_lora.py` using paths under `optuna-run-20260526-042551/trial-N/best_adapter` (not flat `steering-dpo-v1.1/trial-N`).
3. Build **`sft_eval/dpo_final_candidate_manifest.jsonl`** (or under `dpo/eval/`) — populate from both `trial_summary.json` files listed above; see eval plan for JSONL shape.
4. Update **`sft/sft_eval/deploy_qwenie_eval.sh`** and **`eval_generate_vllm.py`** per eval plan (manifest-driven, `MAX_LORA_RANK=64`).
5. Generate 52-conversation outputs; save as e.g. `dpo/eval/dpo_final_generations_52.json`.
6. Run LLM judge with `dpo/eval/llm_judge_prompt_dpo.md` + `candidate_metadata_finalists.json` + generations.
7. **Do not** judge raw `Saulie` base — only SFT trial-17 baseline vs merged DPO cats.

**Eval plan path caveat:** Example merge lines in `DPO_FINAL_EVAL_PLAN.md` use `dpo/train/models/steering-dpo-v1.1/trial-${N}/...` — on disk adapters are under **`optuna-run-20260526-042551/trial-${N}/`**. Same for v1.0: use `optuna-run-20260523-041252/trial-1/`.

**Deferred (unless user asks):** v1.2 study, DB cleanup for zombie RUNNING #26/#27, overshoot fix.

---

## Overshoot (why 21/20)

Workers check `COMPLETE >= target` **before** each `study.optimize(n_trials=1)`. Two workers can both see 19/20, both start a trial, both finish → +1 complete. In-flight trials are not aborted. See `should_stop_parallel` in `optuna_parallel.py`. Harmless for analysis; `trial_summary.json` sets `target_reached` at ≥20.

---

## Fresh conversation bootstrap

Point the next agent at:

```
/root/saulie/dpo/train/HANDOFF.md
```

and the task (e.g. “implement DPO final eval per `dpo/eval/DPO_FINAL_EVAL_PLAN.md`”).
