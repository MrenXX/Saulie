"""Parallel Optuna launcher, workers, OOM solo-retry queue, and study summaries."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import optuna
import torch
from optuna.trial import TrialState

from dpo.train.dpo_data import SPLIT_SEED, manifest_path_for_seed, manifest_sha256
from dpo.train.paths import (
    DATA_PATH,
    EXPERIMENT_NAME,
    OUTPUT_BASE,
    STUDY_RESULTS_DIR,
    experiment_name_for_version,
    output_experiment_dir,
    trial_summary_filename,
    trial_summary_path,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OOM_SCHEMA_VERSION = 1
DEFAULT_TARGET_COMPLETE = 20
DEFAULT_MAX_ATTEMPTED = 60
DEFAULT_MAX_ATTEMPTED_V11 = 45
OPTUNA_BASE_SEED = 42
OPTUNA_STARTUP_TRIALS = 10
V1_5_STARTUP_TRIALS = 12
OPTUNA_SAMPLER_SETTINGS = {
    "n_startup_trials": OPTUNA_STARTUP_TRIALS,
    "multivariate": True,
    "group": True,
    "constant_liar": True,
}

ANCHOR_TRIAL_PARAMS_V11 = {
    "beta": 0.05,
    "num_train_epochs": 1,
    "learning_rate": 2.3604024417191132e-05,
    "lora_r": 16,
    "lora_dropout": 0.05,
    "batch_combo": "1x8",
    "lr_scheduler_type": "constant_with_warmup",
    "max_grad_norm": 1.0,
    "neftune_noise_alpha": 5.0,
    "length_mode": "ld_0.3",
}

# Plan A Step 2 — fixed rescue configs (plan_a_existing_trials_and_final_stack_retrains.md)
PLAN_A_RESCUE_SHARED = {
    "beta": 0.20,
    "num_train_epochs": 1,
    "learning_rate": 5e-6,
    "lora_r": 8,
    "lora_dropout": 0.05,
    "batch_combo": "1x8",
    "lr_scheduler_type": "constant_with_warmup",
    "max_grad_norm": 0.3,
    "neftune_noise_alpha": 0.0,
}
PLAN_A_RESCUE_TRIAL_1 = {**PLAN_A_RESCUE_SHARED, "length_mode": "ld_0.5"}
PLAN_A_RESCUE_TRIAL_2 = {**PLAN_A_RESCUE_SHARED, "length_mode": "ipo"}
PLAN_A_RESCUE_TRIALS = (PLAN_A_RESCUE_TRIAL_1, PLAN_A_RESCUE_TRIAL_2)
PLAN_A_RESCUE_LABELS = ("plan_a_minimal_dpo", "plan_a_ipo")

# Plan B fixed trials (v1.3 baked merged base; v1.4 unmerged SFT+DPO stack)
PLAN_B_STUDY_VERSIONS = ("v1.3", "v1.4")
PLAN_B_SHARED = {
    "num_train_epochs": 1,
    "batch_combo": "1x8",
    "lora_dropout": 0.05,
    "lr_scheduler_type": "constant_with_warmup",
    "max_grad_norm": 0.3,
    "neftune_noise_alpha": 0.0,
}
PLAN_B_TRIAL_ANCHOR = {
    **PLAN_B_SHARED,
    "beta": 0.20,
    "learning_rate": 5e-6,
    "lora_r": 8,
    "length_mode": "ld_0.5",
}
PLAN_B_TRIAL_BRIDGE = {
    **PLAN_B_SHARED,
    "beta": 0.08,
    "learning_rate": 1.0e-5,
    "lora_r": 12,
    "length_mode": "ld_0.3",
}
PLAN_B_TRIAL_V11_LITE = {
    **PLAN_B_SHARED,
    "beta": 0.05,
    "learning_rate": 1.5e-5,
    "lora_r": 16,
    "length_mode": "ld_0.3",
}
PLAN_B_TRIAL_V10_23_LITE = {
    **PLAN_B_SHARED,
    "beta": 0.05,
    "learning_rate": 1.85e-5,
    "lora_r": 16,
    "length_mode": "ld_0.3",
}
PLAN_B_TRIALS = (
    PLAN_B_TRIAL_ANCHOR,
    PLAN_B_TRIAL_BRIDGE,
    PLAN_B_TRIAL_V11_LITE,
    PLAN_B_TRIAL_V10_23_LITE,
)
PLAN_B_LABELS = (
    "plan_b_anchor",
    "plan_b_bridge",
    "plan_b_v11_lite",
    "plan_b_v10_23_lite",
)

# v1.5 hybrid: 12 fixed + 8 TPE (unmerged BnB + SFT + DPO, RPO via TRL sft auxiliary)
V1_5_STUDY_VERSION = "v1.5"
V1_5_SHARED = {
    "num_train_epochs": 1,
    "batch_combo": "1x8",
    "lora_dropout": 0.05,
    "lr_scheduler_type": "constant_with_warmup",
    "max_grad_norm": 0.3,
    "neftune_noise_alpha": 0.0,
}
# (fixed_id, label, beta, lr, lora_r, length_mode, rpo_alpha)
_V1_5_FIXED_ROWS: tuple[tuple[int, str, float, float, int, str, float], ...] = (
    (1, "v1.4 control replay", 0.05, 1.5e-5, 12, "ld_0.3", 0.0),
    (2, "high-LR no-RPO control", 0.05, 1.85e-5, 16, "ld_0.3", 0.0),
    (3, "light RPO", 0.05, 1.5e-5, 16, "ld_0.3", 0.25),
    (4, "main RPO anchor", 0.05, 1.5e-5, 16, "ld_0.3", 0.5),
    (5, "strong RPO anchor", 0.05, 1.5e-5, 16, "ld_0.3", 1.0),
    (6, "stronger DPO with anchor", 0.08, 1.85e-5, 16, "ld_0.3", 0.75),
    (7, "beta bridge", 0.08, 1.2e-5, 16, "ld_0.3", 0.5),
    (8, "high-beta fair retest", 0.15, 1.5e-5, 16, "ld_0.3", 0.5),
    (9, "r24 capacity test", 0.05, 1.2e-5, 24, "ld_0.3", 0.5),
    (10, "r24 balanced high beta", 0.15, 1.2e-5, 24, "ld_0.3", 0.5),
    (11, "r24 high beta + strong anchor", 0.15, 1.2e-5, 24, "ld_0.3", 1.0),
    (12, "IPO fair probe", 0.05, 1.5e-5, 16, "ipo", 0.5),
)
V1_5_FIXED_TRIALS = tuple(
    {
        **V1_5_SHARED,
        "beta": row[2],
        "learning_rate": row[3],
        "lora_r": row[4],
        "length_mode": row[5],
        "rpo_alpha": row[6],
    }
    for row in _V1_5_FIXED_ROWS
)
V1_5_LABELS = tuple(row[1] for row in _V1_5_FIXED_ROWS)
V1_5_FIXED_IDS = tuple(row[0] for row in _V1_5_FIXED_ROWS)

# Canonical Optuna keys stored in summary / MLflow / HTML (batch_combo form).
TRIAL_PARAM_KEYS = (
    "beta",
    "learning_rate",
    "lora_r",
    "lora_dropout",
    "num_train_epochs",
    "batch_combo",
    "length_mode",
    "lr_scheduler_type",
    "max_grad_norm",
    "neftune_noise_alpha",
    "rpo_alpha",
)
VRAM_WAIT_CAP_S = 120
VRAM_WAIT_NORMAL_GB = 10.5
VRAM_WAIT_HIGH_GB = 12.0

OPTUNA_HEARTBEAT_INTERVAL_S = int(os.environ.get("DPO_OPTUNA_HEARTBEAT_INTERVAL", "60"))
OPTUNA_HEARTBEAT_GRACE_S = int(os.environ.get("DPO_OPTUNA_HEARTBEAT_GRACE", "600"))

TrialFn = Callable[..., float]


def optuna_heartbeat_settings() -> dict[str, int]:
    return {
        "heartbeat_interval": OPTUNA_HEARTBEAT_INTERVAL_S,
        "grace_period": OPTUNA_HEARTBEAT_GRACE_S,
    }


def _failed_stale_trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Mark trials failed by heartbeat/stale recovery."""
    if trial.state != TrialState.FAIL:
        return
    reason = (trial.user_attrs or {}).get("failure_reason")
    if reason:
        return
    try:
        study._storage.set_trial_user_attr(
            trial.trial_id,
            "failure_reason",
            "stale_running_recovered",
        )
        study._storage.set_trial_user_attr(
            trial.trial_id,
            "last_stage",
            (trial.user_attrs or {}).get("last_stage", "unknown"),
        )
    except Exception:
        pass


@dataclass
class OptunaRunConfig:
    run_dir: Path
    study_storage: Path
    study_name: str
    target_complete_trials: int
    max_attempted_trials: int
    parallel_workers: int
    worker_id: int | None = None
    dummy_report_path: Path | None = None
    mlflow_parent_run_id: str | None = None
    mlflow_tracking_uri: str | None = None
    study_version: str = "v1.1"
    optuna_base_seed: int = OPTUNA_BASE_SEED
    experiment_name: str = ""


def default_study_name(study_version: str = "v1.0") -> str:
    exp = experiment_name_for_version(study_version)
    if study_version == V1_5_STUDY_VERSION:
        return f"{exp}-hybrid-seed{SPLIT_SEED}"
    if study_version in PLAN_B_STUDY_VERSIONS:
        return f"{exp}-plan-b-seed{SPLIT_SEED}"
    if study_version == "v1.2":
        return f"{exp}-plan-a-seed{SPLIT_SEED}"
    return f"{exp}-v4-seed{SPLIT_SEED}"


def default_study_db(run_dir: Path) -> Path:
    return run_dir / "optuna_study.db"


def default_merged_queue(run_dir: Path) -> Path:
    return run_dir / "oom_retry_queue.jsonl"


def worker_queue_path(run_dir: Path, worker_id: int) -> Path:
    return run_dir / f"oom_retry_queue_worker_{worker_id}.jsonl"


def create_run_dir(study_version: str = "v1.1") -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_experiment_dir(study_version) / f"optuna-run-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def config_from_args(args: argparse.Namespace) -> OptunaRunConfig:
    study_version = getattr(args, "study_version", "v1.1")
    exp_name = getattr(args, "experiment_name", None) or experiment_name_for_version(study_version)
    run_dir = Path(args.run_dir) if args.run_dir else create_run_dir(study_version)
    storage = Path(args.study_storage) if args.study_storage else default_study_db(run_dir)
    max_attempted = args.max_attempted_trials
    if study_version == "v1.1" and max_attempted == DEFAULT_MAX_ATTEMPTED:
        max_attempted = DEFAULT_MAX_ATTEMPTED_V11
    return OptunaRunConfig(
        run_dir=run_dir,
        study_storage=storage,
        study_name=args.study_name or default_study_name(study_version),
        target_complete_trials=args.target_complete_trials,
        max_attempted_trials=max_attempted,
        parallel_workers=args.parallel_workers,
        worker_id=args.worker_id,
        dummy_report_path=Path(args.dummy_report) if getattr(args, "dummy_report", None) else None,
        study_version=study_version,
        optuna_base_seed=getattr(args, "optuna_base_seed", OPTUNA_BASE_SEED),
        experiment_name=exp_name,
    )


def storage_url(db_path: Path) -> str:
    return f"sqlite:///{db_path.resolve()}"


def optuna_sampler_seed(base_seed: int, worker_id: int | None) -> int:
    if worker_id is None:
        return base_seed
    return int(np.random.SeedSequence([base_seed, worker_id]).generate_state(1)[0])


def build_sampler(cfg: OptunaRunConfig) -> optuna.samplers.TPESampler:
    sampler_seed = optuna_sampler_seed(cfg.optuna_base_seed, cfg.worker_id)
    n_startup = V1_5_STARTUP_TRIALS if cfg.study_version == V1_5_STUDY_VERSION else OPTUNA_STARTUP_TRIALS
    constant_liar = cfg.parallel_workers > 1
    return optuna.samplers.TPESampler(
        seed=sampler_seed,
        n_startup_trials=n_startup,
        multivariate=True,
        group=True,
        constant_liar=constant_liar,
    )


def _study_storage(cfg: OptunaRunConfig) -> optuna.storages.RDBStorage:
    hb = optuna_heartbeat_settings()
    return optuna.storages.RDBStorage(
        url=storage_url(cfg.study_storage),
        engine_kwargs={"connect_args": {"timeout": 120}},
        heartbeat_interval=hb["heartbeat_interval"],
        grace_period=hb["grace_period"],
        failed_trial_callback=_failed_stale_trial_callback,
    )


def open_study(cfg: OptunaRunConfig, *, with_sampler: bool = True) -> optuna.Study:
    """Create or load study. Pass with_sampler=True only when first opening in a worker."""
    storage = _study_storage(cfg)
    kwargs: dict[str, Any] = {
        "direction": "maximize",
        "study_name": cfg.study_name,
        "storage": storage,
        "load_if_exists": True,
        "pruner": optuna.pruners.NopPruner(),
    }
    if with_sampler:
        kwargs["sampler"] = build_sampler(cfg)
    return optuna.create_study(**kwargs)


def reload_study(cfg: OptunaRunConfig) -> optuna.Study:
    return optuna.load_study(study_name=cfg.study_name, storage=_study_storage(cfg))


def load_study(cfg: OptunaRunConfig) -> optuna.Study:
    return open_study(cfg, with_sampler=True)


def set_sampler_trial_metadata(trial: optuna.Trial, cfg: OptunaRunConfig) -> None:
    trial.set_user_attr("optuna_base_seed", cfg.optuna_base_seed)
    trial.set_user_attr("sampler_seed", optuna_sampler_seed(cfg.optuna_base_seed, cfg.worker_id))
    trial.set_user_attr("sampler_n_startup_trials", OPTUNA_STARTUP_TRIALS)
    trial.set_user_attr("sampler_multivariate", True)
    trial.set_user_attr("sampler_group", True)
    trial.set_user_attr("sampler_constant_liar", True)
    trial.set_user_attr("study_version", cfg.study_version)
    trial.set_user_attr("optuna_version", optuna.__version__)


def expand_batch_combo(batch_combo: str) -> tuple[int, int]:
    batch_size, grad_accum = [int(part) for part in batch_combo.split("x")]
    return batch_size, grad_accum


def effective_batch_from_params(params: dict) -> int:
    if "batch_combo" in params:
        bs, ga = expand_batch_combo(params["batch_combo"])
        return bs * ga
    return int(params["per_device_train_batch_size"]) * int(params["gradient_accumulation_steps"])


def expand_params_for_training(params: dict) -> dict:
    """Training dict: Optuna params plus expanded batch fields."""
    out = dict(params)
    if "batch_combo" in out:
        bs, ga = expand_batch_combo(out["batch_combo"])
        out["per_device_train_batch_size"] = bs
        out["gradient_accumulation_steps"] = ga
    return out


def to_canonical_params(params: dict) -> dict:
    """Drop expanded batch fields when batch_combo is the source of truth."""
    out = dict(params)
    if "batch_combo" in out:
        out.pop("per_device_train_batch_size", None)
        out.pop("gradient_accumulation_steps", None)
    return out


def persist_trial_params(trial: optuna.Trial, canonical: dict) -> None:
    """Store hyperparameters on the trial (ask/tell leaves Optuna params empty)."""
    canon = to_canonical_params(canonical)
    trial.set_user_attr("trial_params", canon)
    trial.set_user_attr("trial_params_json", json.dumps(canon, sort_keys=True))
    for key in TRIAL_PARAM_KEYS:
        if key in canon:
            trial.set_user_attr(key, canon[key])


def resolve_canonical_trial_params(
    trial: optuna.Trial,
    cfg: OptunaRunConfig,
    *,
    solo_record: dict | None = None,
    fixed_params: dict | None = None,
) -> dict:
    """Return Optuna-shaped params (batch_combo, not expanded)."""
    if fixed_params is not None:
        return to_canonical_params(dict(fixed_params))
    if solo_record is not None:
        return to_canonical_params(dict(solo_record["params"]))
    if dict(trial.params):
        return to_canonical_params(dict(trial.params))
    if cfg.study_version == "v1.2":
        fixed = plan_a_params_by_trial_number(trial.number)
        if not fixed:
            trial.set_user_attr("failure_reason", "v1.2_only_enqueued")
            raise optuna.TrialPruned("v1.2_only_enqueued")
        return dict(fixed)
    if cfg.study_version in PLAN_B_STUDY_VERSIONS:
        fixed = plan_b_params_by_trial_number(trial.number)
        if not fixed:
            trial.set_user_attr("failure_reason", f"{cfg.study_version}_only_enqueued")
            raise optuna.TrialPruned(f"{cfg.study_version}_only_enqueued")
        return dict(fixed)
    return {}


def params_for_frozen_trial(trial: optuna.trial.FrozenTrial, study_version: str) -> dict:
    """Hyperparams for summaries/HTML from Optuna storage or fallbacks."""
    ua = trial.user_attrs or {}
    if study_version == V1_5_STUDY_VERSION:
        stored = ua.get("trial_params")
        if isinstance(stored, dict) and stored:
            return to_canonical_params(stored)
    if dict(trial.params):
        merged = {**V1_5_SHARED, **to_canonical_params(dict(trial.params))} if study_version == V1_5_STUDY_VERSION else to_canonical_params(dict(trial.params))
        return merged if study_version == V1_5_STUDY_VERSION else to_canonical_params(dict(trial.params))
    stored = ua.get("trial_params")
    if isinstance(stored, dict) and stored:
        return to_canonical_params(stored)
    raw = ua.get("trial_params_json")
    if raw:
        try:
            return to_canonical_params(json.loads(raw))
        except json.JSONDecodeError:
            pass
    if study_version in PLAN_B_STUDY_VERSIONS:
        fixed = plan_b_params_by_trial_number(trial.number)
        return dict(fixed) if fixed else {}
    if study_version == "v1.2":
        fixed = plan_a_params_by_trial_number(trial.number)
        return dict(fixed) if fixed else {}
    if study_version == V1_5_STUDY_VERSION:
        fixed = v1_5_params_by_trial_number(trial.number)
        if fixed:
            return dict(fixed)
        stored = (trial.user_attrs or {}).get("trial_params")
        if isinstance(stored, dict) and stored:
            return to_canonical_params(stored)
    return {}


def v1_5_fixed_id_for_params(params: dict) -> int | None:
    key = params_key(params)
    for fixed_id, trial_params in zip(V1_5_FIXED_IDS, V1_5_FIXED_TRIALS):
        if params_key(trial_params) == key:
            return fixed_id
    return None


def v1_5_params_by_trial_number(trial_number: int) -> dict | None:
    if 0 <= trial_number < len(V1_5_FIXED_TRIALS):
        return dict(V1_5_FIXED_TRIALS[trial_number])
    return None


def v1_5_label_for_params(params: dict) -> str:
    fid = v1_5_fixed_id_for_params(params)
    if fid is not None:
        return V1_5_LABELS[fid - 1]
    return "v1_5_sampled"


def validate_v1_5_sampled_params(params: dict) -> str | None:
    """Return prune reason if params violate v1.5 sampled constraints."""
    beta = float(params["beta"])
    lr = float(params["learning_rate"])
    lora_r = int(params["lora_r"])
    if lora_r == 24 and beta >= 0.15 and lr > 1.2e-5:
        return "v1_5_constraint_r24_high_beta_lr"
    if beta == 0.20 and lr > 1.2e-5:
        return "v1_5_constraint_beta20_lr"
    if lora_r == 12 and beta not in (0.08, 0.10, 0.15):
        return "v1_5_constraint_r12_beta"
    return None


def _v1_5_match_fixed_trial(partial: dict) -> dict | None:
    """If partial/complete params match an enqueued fixed trial, return full spec."""
    if not partial:
        return None
    canon = to_canonical_params(partial)
    for fixed in V1_5_FIXED_TRIALS:
        fp = to_canonical_params(fixed)
        if all(canon.get(k) == fp.get(k) for k in canon if k in fp):
            if params_key(canon) == params_key(fp):
                return dict(fp)
            # Optuna enqueue may expose only a subset in trial.params before suggest finishes.
            if set(canon.keys()) >= {"beta", "learning_rate", "lora_r"}:
                if all(fp.get(k) == canon.get(k) for k in ("beta", "learning_rate", "lora_r")):
                    return dict(fp)
    return None


def resolve_v1_5_canonical_params(trial: optuna.Trial) -> dict:
    """Canonical v1.5 params (enqueue-safe; no suggest)."""
    ua = trial.user_attrs or {}
    stored = ua.get("trial_params")
    if isinstance(stored, dict) and stored:
        return to_canonical_params(stored)
    if dict(trial.params):
        matched = _v1_5_match_fixed_trial(dict(trial.params))
        if matched:
            return matched
        return {**V1_5_SHARED, **to_canonical_params(dict(trial.params))}
    return {}


def sample_v1_5_trial_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sample TPE trials; never re-suggest enqueued fixed params (e.g. rpo_alpha=0)."""
    matched = _v1_5_match_fixed_trial(dict(trial.params))
    if matched:
        return expand_params_for_training(matched)

    merged = resolve_v1_5_canonical_params(trial)
    required = ("beta", "learning_rate", "lora_r", "rpo_alpha", "length_mode")
    if all(k in merged for k in required):
        reason = validate_v1_5_sampled_params(merged)
        if reason:
            trial.set_user_attr("failure_reason", reason)
            raise optuna.TrialPruned(reason)
        return expand_params_for_training(merged)

    if "beta" not in merged:
        merged["beta"] = trial.suggest_categorical(
            "beta", [0.05, 0.08, 0.10, 0.15, 0.20]
        )
    if "learning_rate" not in merged:
        merged["learning_rate"] = trial.suggest_categorical(
            "learning_rate", [8e-6, 1.0e-5, 1.2e-5, 1.5e-5, 1.85e-5]
        )
    if "lora_r" not in merged:
        merged["lora_r"] = trial.suggest_categorical("lora_r", [12, 16, 24])
    if "rpo_alpha" not in merged:
        merged["rpo_alpha"] = trial.suggest_categorical(
            "rpo_alpha", [0.25, 0.5, 0.75, 1.0]
        )
    merged.setdefault("length_mode", "ld_0.3")
    params = {**V1_5_SHARED, **to_canonical_params(merged)}
    reason = validate_v1_5_sampled_params(params)
    if reason:
        trial.set_user_attr("failure_reason", reason)
        raise optuna.TrialPruned(reason)
    return expand_params_for_training(params)


def enqueue_v1_5_fixed_trials(study: optuna.Study, cfg: OptunaRunConfig) -> int:
    if cfg.study_version != V1_5_STUDY_VERSION:
        return 0
    active_states = (TrialState.COMPLETE, TrialState.RUNNING, TrialState.WAITING)
    existing = {
        params_key(dict(t.params))
        for t in study.trials
        if t.state in active_states and dict(t.params)
    }
    enqueued = 0
    for params in V1_5_FIXED_TRIALS:
        key = params_key(params)
        if key in existing:
            continue
        study.enqueue_trial(params)
        existing.add(key)
        enqueued += 1
    return enqueued


def copy_trial_summary_to_study_results(summary_path: Path, study_version: str) -> Path:
    STUDY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = STUDY_RESULTS_DIR / trial_summary_filename(study_version)
    shutil.copy2(summary_path, dest)
    return dest


def enrich_summary_dict(summary: dict) -> dict:
    """Fill empty trial params/derived for fixed-trial studies (v1.2/v1.3/v1.4/v1.5)."""
    study_version = str(summary.get("study_version") or "")
    for row in summary.get("trials") or []:
        p = params_for_summary_row(row, study_version)
        if p:
            row["params"] = p
            row["derived"] = derive_summary_fields(p)
    for row in summary.get("complete_trials") or []:
        n = row.get("number")
        if n is None:
            continue
        match = next(
            (t for t in summary.get("trials") or [] if t.get("trial_number") == n),
            None,
        )
        if match and match.get("params"):
            row["params"] = dict(match["params"])
    complete = [
        t
        for t in (summary.get("trials") or [])
        if t.get("state") == "COMPLETE" and t.get("value") is not None
    ]
    if complete:
        best_row = max(complete, key=lambda t: t.get("value") or -1)
        summary["best_params"] = dict(best_row.get("params") or {})
        summary["best_trial"] = best_row.get("trial_number")
    return summary


def params_for_summary_row(trial_row: dict, study_version: str) -> dict:
    """Resolve params from a trial_summary.json row."""
    p = trial_row.get("params") or {}
    if isinstance(p, dict) and p:
        return to_canonical_params(p)
    ua = trial_row.get("user_attrs") or {}
    stored = ua.get("trial_params")
    if isinstance(stored, dict) and stored:
        return to_canonical_params(stored)
    raw = ua.get("trial_params_json")
    if raw:
        try:
            return to_canonical_params(json.loads(raw))
        except json.JSONDecodeError:
            pass
    n = trial_row.get("trial_number", trial_row.get("number"))
    if n is not None:
        if study_version == V1_5_STUDY_VERSION:
            fixed = v1_5_params_by_trial_number(int(n))
            if fixed:
                return dict(fixed)
        if study_version in PLAN_B_STUDY_VERSIONS:
            fixed = plan_b_params_by_trial_number(int(n))
            return dict(fixed) if fixed else {}
        if study_version == "v1.2":
            fixed = plan_a_params_by_trial_number(int(n))
            return dict(fixed) if fixed else {}
    return {}


def derive_summary_fields(params: dict) -> dict:
    from dpo.train.train_dpo import parse_length_mode

    if not params or "length_mode" not in params:
        return {}
    loss_type, ld_alpha, use_weighting = parse_length_mode(params["length_mode"])
    lora_r = params.get("lora_r", 16)
    out = {
        "lora_alpha": 2 * lora_r,
        "effective_batch": effective_batch_from_params(params),
        "loss_type": loss_type,
        "ld_alpha": ld_alpha,
        "use_weighting": use_weighting,
    }
    if "rpo_alpha" in params:
        out["rpo_alpha"] = params["rpo_alpha"]
    return out


def enqueue_anchor_if_needed(study: optuna.Study, cfg: OptunaRunConfig) -> bool:
    if cfg.study_version != "v1.1":
        return False
    anchor_key = params_key(ANCHOR_TRIAL_PARAMS_V11)
    for t in study.trials:
        if t.state in (TrialState.COMPLETE, TrialState.RUNNING, TrialState.WAITING):
            if params_key(dict(t.params)) == anchor_key:
                return False
    study.enqueue_trial(ANCHOR_TRIAL_PARAMS_V11)
    return True


def enqueue_plan_a_rescue_trials(study: optuna.Study, cfg: OptunaRunConfig) -> int:
    """Enqueue exactly two fixed Plan A trials (trial-1 DPO, trial-2 IPO). Idempotent."""
    if cfg.study_version != "v1.2":
        return 0
    active_states = (TrialState.COMPLETE, TrialState.RUNNING, TrialState.WAITING)
    existing = {
        params_key(dict(t.params))
        for t in study.trials
        if t.state in active_states and dict(t.params)
    }
    enqueued = 0
    for params in PLAN_A_RESCUE_TRIALS:
        key = params_key(params)
        if key in existing:
            continue
        study.enqueue_trial(params)
        existing.add(key)
        enqueued += 1
    return enqueued


def rescue_label_for_params(params: dict) -> str:
    if params.get("length_mode") == "ipo":
        return PLAN_A_RESCUE_LABELS[1]
    return PLAN_A_RESCUE_LABELS[0]


def plan_a_params_by_trial_number(trial_number: int) -> dict | None:
    if 0 <= trial_number < len(PLAN_A_RESCUE_TRIALS):
        return dict(PLAN_A_RESCUE_TRIALS[trial_number])
    return None


def plan_b_params_by_trial_number(trial_number: int) -> dict | None:
    if 0 <= trial_number < len(PLAN_B_TRIALS):
        return dict(PLAN_B_TRIALS[trial_number])
    return None


def plan_b_label_for_params(params: dict) -> str:
    key = params_key(params)
    for label, trial_params in zip(PLAN_B_LABELS, PLAN_B_TRIALS):
        if params_key(trial_params) == key:
            return label
    return "plan_b_unknown"


def resolve_trial_params(
    trial: optuna.Trial,
    cfg: OptunaRunConfig,
    *,
    solo_record: dict | None = None,
    fixed_params: dict | None = None,
) -> dict:
    """Expanded training dict (batch fields materialized)."""
    canonical = resolve_canonical_trial_params(
        trial, cfg, solo_record=solo_record, fixed_params=fixed_params
    )
    if cfg.study_version == V1_5_STUDY_VERSION:
        return sample_v1_5_trial_params(trial)
    if cfg.study_version not in ("v1.2", *PLAN_B_STUDY_VERSIONS, V1_5_STUDY_VERSION) and not canonical:
        return sample_trial_params(trial, cfg)
    if not canonical:
        raise optuna.TrialPruned("missing_trial_params")
    return expand_params_for_training(canonical)


def _v12_rescue_already_complete(study: optuna.Study, params: dict) -> bool:
    key = params_key(params)
    for t in study.trials:
        if t.state == TrialState.COMPLETE and params_key(dict(t.params)) == key:
            return True
    return False


def run_plan_a_v12_worker_loop(cfg: OptunaRunConfig, run_trial_fn: TrialFn) -> None:
    """Run exactly two fixed trials via ask/tell (enqueue queue); no TPE sampling."""
    from dpo.train.dpo_diagnostics import log_line, worker_prefix

    prefix = worker_prefix(cfg.worker_id)
    study = open_study(cfg, with_sampler=True)
    try:
        optuna.storages.fail_stale_trials(study)
        log_line(prefix, "fail_stale_trials: cleaned zombie RUNNING trials")
    except Exception as e:
        log_line(prefix, f"fail_stale_trials skipped: {e}")

    for idx, params in enumerate(PLAN_A_RESCUE_TRIALS, start=1):
        if _v12_rescue_already_complete(study, params):
            log_line(prefix, f"skip rescue {idx}/2 (already COMPLETE)")
            continue
        log_line(prefix, f"plan_a rescue {idx}/2 ask trial ({params.get('length_mode')})")
        trial = study.ask()
        try:
            value = run_trial_fn(cfg, trial, fixed_params=dict(params))
            study.tell(trial, value)
            log_line(prefix, f"plan_a rescue {idx}/2 COMPLETE trial #{trial.number} value={value:.4f}")
        except optuna.TrialPruned:
            study.tell(trial, state=TrialState.PRUNED)
            log_line(prefix, f"plan_a rescue {idx}/2 PRUNED trial #{trial.number}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        study = reload_study(cfg)

    c = study_counts(study)
    log_line(prefix, f"v1.2 worker done | counts={c}")


def enqueue_plan_b_trials(study: optuna.Study, cfg: OptunaRunConfig) -> int:
    """Enqueue four fixed Plan B trials (v1.3 baked or v1.4 unmerged). Idempotent."""
    if cfg.study_version not in PLAN_B_STUDY_VERSIONS:
        return 0
    active_states = (TrialState.COMPLETE, TrialState.RUNNING, TrialState.WAITING)
    existing = {
        params_key(dict(t.params))
        for t in study.trials
        if t.state in active_states and dict(t.params)
    }
    enqueued = 0
    for params in PLAN_B_TRIALS:
        key = params_key(params)
        if key in existing:
            continue
        study.enqueue_trial(params)
        existing.add(key)
        enqueued += 1
    return enqueued


def run_plan_b_fixed_worker_loop(cfg: OptunaRunConfig, run_trial_fn: TrialFn) -> None:
    """Run four fixed Plan B trials via ask/tell (v1.3 baked or v1.4 unmerged)."""
    from dpo.train.dpo_diagnostics import log_line, worker_prefix

    prefix = worker_prefix(cfg.worker_id)
    study = open_study(cfg, with_sampler=True)
    try:
        optuna.storages.fail_stale_trials(study)
        log_line(prefix, "fail_stale_trials: cleaned zombie RUNNING trials")
    except Exception as e:
        log_line(prefix, f"fail_stale_trials skipped: {e}")

    for idx, params in enumerate(PLAN_B_TRIALS, start=1):
        if _v12_rescue_already_complete(study, params):
            log_line(prefix, f"skip plan_b {idx}/4 (already COMPLETE)")
            continue
        label = plan_b_label_for_params(params)
        log_line(prefix, f"plan_b {idx}/4 ask trial ({label}, {params.get('length_mode')})")
        trial = study.ask()
        try:
            value = run_trial_fn(cfg, trial, fixed_params=dict(params))
            study.tell(trial, value)
            log_line(prefix, f"plan_b {idx}/4 COMPLETE trial #{trial.number} value={value:.4f}")
        except optuna.TrialPruned:
            study.tell(trial, state=TrialState.PRUNED)
            log_line(prefix, f"plan_b {idx}/4 PRUNED trial #{trial.number}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        study = reload_study(cfg)

    c = study_counts(study)
    log_line(prefix, f"{cfg.study_version} plan_b worker done | counts={c}")


def run_plan_b_v13_worker_loop(cfg: OptunaRunConfig, run_trial_fn: TrialFn) -> None:
    """Backward-compatible alias."""
    run_plan_b_fixed_worker_loop(cfg, run_trial_fn)


def study_counts(study: optuna.Study) -> dict[str, int]:
    counts: dict[str, int] = {s.name: 0 for s in TrialState}
    for t in study.trials:
        counts[t.state.name] += 1
    counts["total"] = len(study.trials)
    counts["attempted"] = sum(
        1 for t in study.trials if t.state != TrialState.WAITING
    )
    return counts


def should_stop_parallel(study: optuna.Study, cfg: OptunaRunConfig) -> tuple[bool, str]:
    c = study_counts(study)
    if c.get("COMPLETE", 0) >= cfg.target_complete_trials:
        return True, f"COMPLETE>={cfg.target_complete_trials}"
    if c.get("attempted", 0) >= cfg.max_attempted_trials:
        return True, f"attempted>={cfg.max_attempted_trials}"
    return False, ""


def cuda_mem_snapshot() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    free_b, total_b = torch.cuda.mem_get_info()
    return {
        "free_gb": free_b / 1e9,
        "total_gb": total_b / 1e9,
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
    }


def is_high_memory_params(params: dict) -> bool:
    training = expand_params_for_training(params)
    return (
        training.get("per_device_train_batch_size") == 2
        and training.get("lora_r") == 32
    )


def wait_for_vram(params: dict, worker_id: int | None) -> str | None:
    if not torch.cuda.is_available():
        return None
    threshold = VRAM_WAIT_HIGH_GB if is_high_memory_params(params) else VRAM_WAIT_NORMAL_GB
    label = worker_id if worker_id is not None else "solo"
    deadline = time.monotonic() + VRAM_WAIT_CAP_S
    while time.monotonic() < deadline:
        snap = cuda_mem_snapshot()
        if snap.get("free_gb", 0) >= threshold:
            return None
        from dpo.train.dpo_diagnostics import log_line

        log_line(
            f"W{worker_id}" if worker_id is not None else "SOLO",
            f"VRAM wait: free={snap.get('free_gb', 0):.2f}GB < {threshold}GB",
        )
        time.sleep(random.uniform(2.0, 5.0))
    snap = cuda_mem_snapshot()
    if snap.get("free_gb", 0) < threshold:
        return "resource_wait_timeout"
    return None


def sample_trial_params(trial: optuna.Trial, cfg: OptunaRunConfig) -> dict[str, Any]:
    if cfg.study_version == V1_5_STUDY_VERSION:
        return sample_v1_5_trial_params(trial)
    if cfg.study_version == "v1.1":
        batch_combo = trial.suggest_categorical(
            "batch_combo", ["1x4", "1x8", "1x16", "2x2", "2x4", "2x8"]
        )
        params = {
            "beta": trial.suggest_categorical("beta", [0.03, 0.05, 0.08, 0.1]),
            "num_train_epochs": trial.suggest_categorical("num_train_epochs", [1, 2]),
            "learning_rate": trial.suggest_float("learning_rate", 8e-6, 3e-5, log=True),
            "lora_r": trial.suggest_categorical("lora_r", [8, 16, 32]),
            "lora_dropout": trial.suggest_categorical(
                "lora_dropout", [0.03, 0.05, 0.075, 0.1]
            ),
            "batch_combo": batch_combo,
            "lr_scheduler_type": trial.suggest_categorical(
                "lr_scheduler_type", ["linear", "cosine", "constant_with_warmup"]
            ),
            "max_grad_norm": trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 1.0]),
            "neftune_noise_alpha": trial.suggest_categorical(
                "neftune_noise_alpha", [0.0, 2.5, 5.0]
            ),
            "length_mode": trial.suggest_categorical(
                "length_mode",
                ["sigmoid_norm", "ld_0.1", "ld_0.2", "ld_0.3", "ld_0.5"],
            ),
        }
        return expand_params_for_training(params)
    return {
        "beta": trial.suggest_categorical("beta", [0.01, 0.05, 0.1, 0.2]),
        "num_train_epochs": trial.suggest_categorical("num_train_epochs", [1, 2, 3]),
        "learning_rate": trial.suggest_float("learning_rate", 5e-6, 3e-5, log=True),
        "lora_r": trial.suggest_categorical("lora_r", [8, 16, 32]),
        "lora_dropout": trial.suggest_categorical("lora_dropout", [0.05, 0.1]),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [1, 2]
        ),
        "gradient_accumulation_steps": trial.suggest_categorical(
            "gradient_accumulation_steps", [2, 4, 8]
        ),
        "lr_scheduler_type": trial.suggest_categorical(
            "lr_scheduler_type", ["linear", "cosine", "constant_with_warmup"]
        ),
        "max_grad_norm": trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 1.0]),
        "neftune_noise_alpha": trial.suggest_categorical("neftune_noise_alpha", [0.0, 5.0]),
        "length_mode": trial.suggest_categorical(
            "length_mode", ["none", "sigmoid_norm", "ld_0.3", "ld_0.5"]
        ),
    }


def check_duplicate_params(trial: optuna.Trial) -> None:
    key = params_key(dict(trial.params))
    states = (TrialState.COMPLETE, TrialState.RUNNING, TrialState.WAITING)
    for old in trial.study.get_trials(deepcopy=False, states=states):
        if old.number != trial.number and params_key(dict(old.params)) == key:
            trial.set_user_attr("failure_reason", "duplicate_params")
            trial.set_user_attr("duplicate_of", old.number)
            raise optuna.TrialPruned("duplicate_params")


def params_key(params: dict) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def append_oom_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def merge_worker_queues(run_dir: Path, n_workers: int) -> Path:
    merged = default_merged_queue(run_dir)
    records: list[dict] = []
    for wid in range(n_workers):
        records.extend(read_jsonl(worker_queue_path(run_dir, wid)))
    with merged.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return merged


def dedupe_oom_queue(records: list[dict]) -> list[dict]:
    by_key: dict[str, dict] = {}
    for rec in records:
        key = params_key(rec.get("params", {}))
        num = rec.get("original_trial_number")
        if key not in by_key:
            by_key[key] = {**rec, "original_trial_numbers": [num] if num is not None else []}
        elif num is not None and num not in by_key[key]["original_trial_numbers"]:
            by_key[key]["original_trial_numbers"].append(num)
    return list(by_key.values())


def make_oom_record(
    *,
    trial: optuna.Trial,
    params: dict,
    derived: dict,
    worker_id: int | None,
    stage: str,
    vram_start: dict,
    vram_oom: dict,
) -> dict:
    return {
        "schema_version": OOM_SCHEMA_VERSION,
        "reason": "parallel_oom",
        "original_trial_number": trial.number,
        "worker_id": worker_id,
        "attempt": 1,
        "params": params,
        "derived": derived,
        "vram": {
            "peak_allocated_gb": vram_oom.get("peak_allocated_gb", 0),
            "peak_reserved_gb": vram_oom.get("peak_reserved_gb", 0),
            "free_gb_at_start": vram_start.get("free_gb", 0),
            "free_gb_at_oom": vram_oom.get("free_gb", 0),
        },
        "stage": stage,
        "timestamp": datetime.now().isoformat(),
    }


def worker_log_path(run_dir: Path, worker_id: int) -> Path:
    return run_dir / f"worker_{worker_id}.log"


def launcher_log(cfg: OptunaRunConfig, msg: str) -> None:
    path = cfg.run_dir / "launcher.log"
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
    print(line, end="", flush=True)


def write_final_summary(
    study: optuna.Study,
    cfg: OptunaRunConfig,
    *,
    queue_records: list[dict],
    deduped_queue: list[dict],
    meta: dict,
) -> Path:
    import trl

    from dpo.train.dpo_diagnostics import build_study_review, build_trial_summary_record
    from sft.train_sft import compute_data_hash

    trials_out = []
    oom_by_combo: dict[str, int] = {}

    study_version = cfg.study_version
    for t in study.trials:
        p = params_for_frozen_trial(t, study_version)
        derived = derive_summary_fields(p) if p else {}
        if not derived and (t.user_attrs or {}).get("derived"):
            derived = dict((t.user_attrs or {}).get("derived") or {})
        ua = dict(t.user_attrs)
        adapter_diag = ua.get("adapter_diagnostics")
        if isinstance(adapter_diag, dict):
            pass
        else:
            adapter_diag = {}
        row = build_trial_summary_record(
            trial_number=t.number,
            state=t.state.name,
            value=t.value,
            params=p,
            derived=derived,
            user_attrs=ua,
            adapter_diag=adapter_diag if adapter_diag else None,
        )
        row["queued_for_solo_retry"] = ua.get("queued_for_solo_retry", False)
        row["solo_retry"] = ua.get("solo_retry", False)
        row["parallel_oom_recovered"] = ua.get("parallel_oom_recovered", False)
        row["requires_solo"] = ua.get("requires_solo", False)
        row["original_parallel_oom_trial_numbers"] = ua.get(
            "original_parallel_oom_trial_numbers"
        )
        trials_out.append(row)
        if t.user_attrs.get("failure_reason") == "parallel_oom_queued_for_solo":
            combo = (
                f"batch={p.get('batch_combo') or p.get('per_device_train_batch_size')}|"
                f"r={p.get('lora_r')}|neftune={p.get('neftune_noise_alpha')}|lm={p.get('length_mode')}"
            )
            oom_by_combo[combo] = oom_by_combo.get(combo, 0) + 1

    complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
    duplicate_pruned_count = sum(
        1 for t in study.trials if t.user_attrs.get("failure_reason") == "duplicate_params"
    )
    unique_complete_keys = {
        params_key(dict(t.params)) for t in complete
    }
    solo_recovered = [t for t in complete if t.user_attrs.get("parallel_oom_recovered")]
    parallel_complete = [t for t in complete if not t.user_attrs.get("solo_retry")]
    counts = study_counts(study)
    best = study.best_trial if complete else None

    summary = {
        "study_name": cfg.study_name,
        "study_storage": str(cfg.study_storage.resolve()),
        "run_dir": str(cfg.run_dir.resolve()),
        "solo_retry_queue": str(default_merged_queue(cfg.run_dir)),
        "dataset_hash": compute_data_hash(DATA_PATH),
        "split_manifest": str(manifest_path_for_seed(SPLIT_SEED)),
        "split_manifest_sha256": manifest_sha256(manifest_path_for_seed(SPLIT_SEED)),
        "dummy_report_path": str(cfg.dummy_report_path) if cfg.dummy_report_path else None,
        "trl_version": trl.__version__,
        "target_complete_trials": cfg.target_complete_trials,
        "max_attempted_trials": cfg.max_attempted_trials,
        "parallel_workers": cfg.parallel_workers,
        "counts": counts,
        "target_reached": counts.get("COMPLETE", 0) >= cfg.target_complete_trials,
        "complete_overshoot": max(0, counts.get("COMPLETE", 0) - cfg.target_complete_trials),
        "parallel_complete_count": len(parallel_complete),
        "solo_recovered_complete_count": len(solo_recovered),
        "parallel_oom_queued_count": len(queue_records),
        "solo_retry_unique_configs": len(deduped_queue),
        "solo_intrinsic_oom_count": sum(
            1
            for t in study.trials
            if t.user_attrs.get("failure_reason") == "solo_oom_intrinsic_or_too_large"
        ),
        "oom_by_hyperparam_combo": oom_by_combo,
        "study_version": cfg.study_version,
        "experiment_name": cfg.experiment_name,
        "optuna_base_seed": cfg.optuna_base_seed,
        "sampler_settings": {
            **OPTUNA_SAMPLER_SETTINGS,
            "n_startup_trials": (
                V1_5_STARTUP_TRIALS
                if study_version == V1_5_STUDY_VERSION
                else OPTUNA_STARTUP_TRIALS
            ),
        },
        "optuna_heartbeat_settings": optuna_heartbeat_settings(),
        "duplicate_pruned_count": duplicate_pruned_count,
        "unique_complete_config_count": len(unique_complete_keys),
        "best_trial": best.number if best else None,
        "best_objective": best.value if best else None,
        "best_hybrid_score": best.user_attrs.get("hybrid_score_v1_1", best.value) if best else None,
        "best_v1_5_survival_score": (
            best.user_attrs.get("v1_5_survival_score", best.value) if best else None
        ),
        "best_accuracy": best.user_attrs.get("eval_rewards_accuracy") if best else None,
        "trial_summary_file": trial_summary_filename(study_version),
        "best_params": params_for_frozen_trial(best, study_version) if best else None,
        "complete_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": params_for_frozen_trial(t, study_version),
                "solo_retry": t.user_attrs.get("solo_retry", False),
                "parallel_oom_recovered": t.user_attrs.get("parallel_oom_recovered", False),
            }
            for t in sorted(complete, key=lambda x: x.value or -1, reverse=True)
        ],
        "trials": trials_out,
        "study_review": build_study_review(study),
        **meta,
    }
    out = trial_summary_path(cfg.run_dir, study_version)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    from dpo.train.study_report import write_study_report

    report_path = write_study_report(out)
    summary["study_report_html"] = str(report_path.resolve())
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    copied = copy_trial_summary_to_study_results(out, study_version)
    summary["trial_summary_study_results_copy"] = str(copied.resolve())
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out


def spawn_worker(cfg: OptunaRunConfig, worker_id: int) -> tuple[subprocess.Popen, Path]:
    script = REPO_ROOT / "dpo" / "train" / "train_dpo.py"
    log_path = worker_log_path(cfg.run_dir, worker_id)
    cmd = [
        sys.executable,
        str(script),
        "--optuna",
        "--optuna-worker",
        f"--worker-id={worker_id}",
        f"--run-dir={cfg.run_dir}",
        f"--study-storage={cfg.study_storage}",
        f"--study-name={cfg.study_name}",
        f"--target-complete-trials={cfg.target_complete_trials}",
        f"--max-attempted-trials={cfg.max_attempted_trials}",
    ]
    if cfg.dummy_report_path:
        cmd.append(f"--dummy-report={cfg.dummy_report_path}")
    cmd.extend([
        f"--study-version={cfg.study_version}",
        f"--optuna-base-seed={cfg.optuna_base_seed}",
        f"--experiment-name={cfg.experiment_name}",
    ])
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("HF_DATASETS_DISABLE_PROGRESS_BARS", None)
    env["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    if cfg.mlflow_parent_run_id:
        env["MLFLOW_PARENT_RUN_ID"] = cfg.mlflow_parent_run_id
    if cfg.mlflow_tracking_uri:
        env["MLFLOW_TRACKING_URI"] = cfg.mlflow_tracking_uri
    log_f = log_path.open("w", encoding="utf-8", buffering=1)
    log_f.write(f"# worker {worker_id} started {datetime.now().isoformat()}\n")
    log_f.write(f"# cmd: {' '.join(cmd)}\n\n")
    log_f.flush()
    launcher_log(cfg, f"Spawning worker {worker_id} -> {log_path.name}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=env,
    )
    return proc, log_path


def run_worker_loop(cfg: OptunaRunConfig, run_trial_fn: TrialFn) -> None:
    from dpo.train.dpo_diagnostics import log_line, worker_prefix

    if cfg.study_version == "v1.2":
        run_plan_a_v12_worker_loop(cfg, run_trial_fn)
        return
    if cfg.study_version in PLAN_B_STUDY_VERSIONS:
        run_plan_b_fixed_worker_loop(cfg, run_trial_fn)
        return

    prefix = worker_prefix(cfg.worker_id)
    log_line(prefix, f"worker started | run_dir={cfg.run_dir}")
    log_line(prefix, f"target COMPLETE>={cfg.target_complete_trials} max_attempted={cfg.max_attempted_trials}")
    sampler_seed = optuna_sampler_seed(cfg.optuna_base_seed, cfg.worker_id)
    log_line(prefix, f"optuna_base_seed={cfg.optuna_base_seed} sampler_seed={sampler_seed}")
    study = open_study(cfg, with_sampler=True)
    try:
        optuna.storages.fail_stale_trials(study)
        log_line(prefix, "fail_stale_trials: cleaned zombie RUNNING trials")
    except Exception as e:
        log_line(prefix, f"fail_stale_trials skipped: {e}")
    while True:
        fresh = reload_study(cfg)
        c = study_counts(fresh)
        stop, reason = should_stop_parallel(fresh, cfg)
        if stop:
            log_line(prefix, f"worker exit: {reason} | counts={c}")
            return
        log_line(prefix, f"polling: COMPLETE={c.get('COMPLETE', 0)}/{cfg.target_complete_trials} attempted={c.get('attempted', 0)}")

        def objective(trial: optuna.Trial) -> float:
            return run_trial_fn(cfg, trial, solo_record=None)

        try:
            study.optimize(objective, n_trials=1, catch=(optuna.TrialPruned,))
        except Exception as e:
            log_line(prefix, f"optimize error: {e}")
            traceback.print_exc()
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

        c = study_counts(reload_study(cfg))
        stop, reason = should_stop_parallel(reload_study(cfg), cfg)
        if stop:
            log_line(prefix, f"worker exit after trial: {reason} | counts={c}")
            return


def run_solo_retry_phase(cfg: OptunaRunConfig, run_trial_fn: TrialFn, deduped: list[dict]) -> None:
    if not deduped:
        launcher_log(cfg, "Solo retry queue empty — skipping")
        return
    launcher_log(cfg, f"SOLO RETRY phase: {len(deduped)} unique configs")
    study = reload_study(cfg)
    for rec in deduped:
        study.enqueue_trial(rec["params"])
    for rec in deduped:
        study = reload_study(cfg)

        def objective(trial: optuna.Trial) -> float:
            return run_trial_fn(cfg, trial, solo_record=rec)

        study.optimize(objective, n_trials=1, catch=(optuna.TrialPruned,))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


def run_parallel_launcher(
    cfg: OptunaRunConfig,
    run_trial_fn: TrialFn,
    *,
    mlflow_parent_run_id: str | None = None,
    mlflow_tracking_uri: str | None = None,
) -> Path:
    if mlflow_parent_run_id:
        cfg.mlflow_parent_run_id = mlflow_parent_run_id
    if mlflow_tracking_uri:
        cfg.mlflow_tracking_uri = mlflow_tracking_uri
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    launcher_study_cfg = OptunaRunConfig(
        run_dir=cfg.run_dir,
        study_storage=cfg.study_storage,
        study_name=cfg.study_name,
        target_complete_trials=cfg.target_complete_trials,
        max_attempted_trials=cfg.max_attempted_trials,
        parallel_workers=cfg.parallel_workers,
        worker_id=None,
        dummy_report_path=cfg.dummy_report_path,
        study_version=cfg.study_version,
        optuna_base_seed=cfg.optuna_base_seed,
        experiment_name=cfg.experiment_name,
    )
    study = open_study(launcher_study_cfg, with_sampler=True)
    if cfg.study_version == "v1.2":
        enqueued_rescue = enqueue_plan_a_rescue_trials(study, cfg)
        enqueued_plan_b = 0
        enqueued_anchor = False
        enqueued_v15 = 0
    elif cfg.study_version in PLAN_B_STUDY_VERSIONS:
        enqueued_rescue = 0
        enqueued_plan_b = enqueue_plan_b_trials(study, cfg)
        enqueued_anchor = False
        enqueued_v15 = 0
    elif cfg.study_version == V1_5_STUDY_VERSION:
        enqueued_rescue = 0
        enqueued_plan_b = 0
        enqueued_anchor = False
        enqueued_v15 = enqueue_v1_5_fixed_trials(study, cfg)
    else:
        enqueued_rescue = 0
        enqueued_plan_b = 0
        enqueued_anchor = enqueue_anchor_if_needed(study, cfg)
        enqueued_v15 = 0
    launcher_log(cfg, f"PARALLEL OPTUNA | workers={cfg.parallel_workers} target={cfg.target_complete_trials}")
    launcher_log(cfg, f"study_version={cfg.study_version} experiment={cfg.experiment_name}")
    launcher_log(
        cfg,
        f"optuna_base_seed={cfg.optuna_base_seed} anchor_enqueued={enqueued_anchor} "
        f"plan_a_enqueued={enqueued_rescue} plan_b_enqueued={enqueued_plan_b} "
        f"v1_5_fixed_enqueued={enqueued_v15}",
    )
    launcher_log(cfg, f"run_dir={cfg.run_dir}")
    launcher_log(cfg, f"study={cfg.study_name} db={cfg.study_storage}")
    monitor_doc = cfg.run_dir / "MONITOR.txt"
    lines = [
        f"ACTIVE RUN: {cfg.run_dir}",
        f"Started: {datetime.now().isoformat()}",
        "",
        "In tmux, tail these paths (NOT an old optuna-run-* dir):",
        f"  tail -f {cfg.run_dir / 'launcher.log'}",
    ]
    for wid in range(cfg.parallel_workers):
        lines.append(f"  tail -f {worker_log_path(cfg.run_dir, wid)}")
    lines.append("")
    lines.append("If logs look frozen, you are likely tailing a dead/interrupted run.")
    monitor_doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    launcher_log(cfg, f"Monitor paths written: {monitor_doc}")
    for line in lines:
        launcher_log(cfg, line)

    for wid in range(cfg.parallel_workers):
        wp = worker_queue_path(cfg.run_dir, wid)
        if wp.exists():
            wp.unlink()

    procs_and_logs = [spawn_worker(cfg, wid) for wid in range(cfg.parallel_workers)]
    procs = [p for p, _ in procs_and_logs]
    codes = [p.wait() for p in procs]
    launcher_log(cfg, f"Worker exit codes: {codes}")

    merged = merge_worker_queues(cfg.run_dir, cfg.parallel_workers)
    queue_records = read_jsonl(merged)
    deduped = dedupe_oom_queue(queue_records)
    launcher_log(cfg, f"OOM queue: {len(queue_records)} raw, {len(deduped)} unique -> {merged.name}")

    run_solo_retry_phase(cfg, run_trial_fn, deduped)

    study = reload_study(cfg)
    summary = write_final_summary(
        study,
        cfg,
        queue_records=queue_records,
        deduped_queue=deduped,
        meta={
            "finished_at": datetime.now().isoformat(),
            "worker_exit_codes": codes,
            "solo_retry_count": len(deduped),
            "optuna_heartbeat_settings": optuna_heartbeat_settings(),
        },
    )
    complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if complete:
        bt = study.best_trial
        acc = bt.user_attrs.get("eval_rewards_accuracy", bt.value)
        launcher_log(
            cfg,
            f"DONE Best trial #{bt.number}: hybrid={bt.value:.4f} acc={acc}",
        )
    else:
        launcher_log(cfg, "WARNING: no COMPLETE trials in study")
    launcher_log(cfg, f"Summary written: {summary}")
    report_path = cfg.run_dir / "study_report.html"
    if report_path.is_file():
        launcher_log(cfg, f"Study report: file://{report_path.resolve()}")
        from dpo.train.paths import MLRUNS_DIR

        uri = cfg.mlflow_tracking_uri or f"file://{MLRUNS_DIR.resolve()}"
        monitor_doc = cfg.run_dir / "MONITOR.txt"
        with monitor_doc.open("a", encoding="utf-8") as f:
            f.write(f"\nStudy report (open in browser):\n  file://{report_path.resolve()}\n")
            f.write(
                f"\nMLflow UI:\n  mlflow ui --backend-store-uri {uri} --port 5001\n"
                f"  Experiment: {cfg.experiment_name} → FINISHED parent run with nested trial-* children\n"
            )
    return summary


def add_parallel_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--parallel-workers", type=int, default=0)
    parser.add_argument("--target-complete-trials", type=int, default=DEFAULT_TARGET_COMPLETE)
    parser.add_argument("--max-attempted-trials", type=int, default=DEFAULT_MAX_ATTEMPTED)
    parser.add_argument("--optuna-worker", action="store_true")
    parser.add_argument("--worker-id", type=int, default=None)
    parser.add_argument("--study-storage", type=Path, default=None)
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--solo-retry-queue", type=Path, default=None)
    parser.add_argument("--solo-retry", action="store_true")
    parser.add_argument(
        "--optuna-smoke",
        action="store_true",
        help="Smoke test: 2 workers, 2 complete trials, max 12 attempts",
    )
    parser.add_argument(
        "--dummy-report",
        type=Path,
        default=OUTPUT_BASE / "steering-dpo-v1.0" / "dummy-run" / "dummy_report.json",
    )
    parser.add_argument(
        "--study-version",
        choices=("v1.0", "v1.1", "v1.2", "v1.3", "v1.4", "v1.5"),
        default="v1.1",
        help="v1.5: hybrid 12 fixed + 8 TPE; v1.4: Plan B unmerged (4 fixed); v1.1: broad TPE",
    )
    parser.add_argument("--optuna-base-seed", type=int, default=OPTUNA_BASE_SEED)
    parser.add_argument("--experiment-name", type=str, default=None)
