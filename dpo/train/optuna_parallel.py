"""Parallel Optuna launcher, workers, OOM solo-retry queue, and study summaries."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import optuna
import torch
from optuna.trial import TrialState

from dpo.train.dpo_data import SPLIT_SEED, manifest_path_for_seed, manifest_sha256
from dpo.train.paths import DATA_PATH, EXPERIMENT_NAME, OUTPUT_BASE

REPO_ROOT = Path(__file__).resolve().parents[2]
OOM_SCHEMA_VERSION = 1
DEFAULT_TARGET_COMPLETE = 20
DEFAULT_MAX_ATTEMPTED = 60
VRAM_WAIT_CAP_S = 120
VRAM_WAIT_NORMAL_GB = 10.5
VRAM_WAIT_HIGH_GB = 12.0

TrialFn = Callable[..., float]


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


def default_study_name() -> str:
    return f"{EXPERIMENT_NAME}-v4-seed{SPLIT_SEED}"


def default_study_db(run_dir: Path) -> Path:
    return run_dir / "optuna_study.db"


def default_merged_queue(run_dir: Path) -> Path:
    return run_dir / "oom_retry_queue.jsonl"


def worker_queue_path(run_dir: Path, worker_id: int) -> Path:
    return run_dir / f"oom_retry_queue_worker_{worker_id}.jsonl"


def create_run_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = OUTPUT_BASE / EXPERIMENT_NAME / f"optuna-run-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def config_from_args(args: argparse.Namespace) -> OptunaRunConfig:
    run_dir = Path(args.run_dir) if args.run_dir else create_run_dir()
    storage = Path(args.study_storage) if args.study_storage else default_study_db(run_dir)
    return OptunaRunConfig(
        run_dir=run_dir,
        study_storage=storage,
        study_name=args.study_name or default_study_name(),
        target_complete_trials=args.target_complete_trials,
        max_attempted_trials=args.max_attempted_trials,
        parallel_workers=args.parallel_workers,
        worker_id=args.worker_id,
        dummy_report_path=Path(args.dummy_report) if getattr(args, "dummy_report", None) else None,
    )


def storage_url(db_path: Path) -> str:
    return f"sqlite:///{db_path.resolve()}"


def load_study(cfg: OptunaRunConfig) -> optuna.Study:
    storage = optuna.storages.RDBStorage(
        url=storage_url(cfg.study_storage),
        engine_kwargs={"connect_args": {"timeout": 120}},
    )
    return optuna.create_study(
        direction="maximize",
        study_name=cfg.study_name,
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.NopPruner(),
    )


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
    return (
        params.get("per_device_train_batch_size") == 2
        and params.get("lora_r") == 32
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


def sample_trial_params(trial: optuna.Trial) -> dict[str, Any]:
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

    from dpo.train.dpo_diagnostics import build_study_review
    from dpo.train.train_dpo import parse_length_mode
    from train.train_sft import compute_data_hash

    trials_out = []
    oom_by_combo: dict[str, int] = {}

    for t in study.trials:
        p = dict(t.params)
        derived = {}
        if p and "length_mode" in p:
            loss_type, ld_alpha, use_weighting = parse_length_mode(p["length_mode"])
            derived = {
                "lora_alpha": 2 * p.get("lora_r", 16),
                "effective_batch": p["per_device_train_batch_size"] * p["gradient_accumulation_steps"],
                "loss_type": loss_type,
                "ld_alpha": ld_alpha,
                "use_weighting": use_weighting,
            }
        trials_out.append({
            "trial_number": t.number,
            "worker_id": t.user_attrs.get("worker_id"),
            "params": p,
            "derived": derived,
            "state": t.state.name,
            "value": t.value,
            "failure_reason": t.user_attrs.get("failure_reason"),
            "queued_for_solo_retry": t.user_attrs.get("queued_for_solo_retry", False),
            "solo_retry": t.user_attrs.get("solo_retry", False),
            "parallel_oom_recovered": t.user_attrs.get("parallel_oom_recovered", False),
            "requires_solo": t.user_attrs.get("requires_solo", False),
            "original_parallel_oom_trial_numbers": t.user_attrs.get(
                "original_parallel_oom_trial_numbers"
            ),
            "user_attrs": dict(t.user_attrs),
        })
        if t.user_attrs.get("failure_reason") == "parallel_oom_queued_for_solo":
            combo = (
                f"bs={p.get('per_device_train_batch_size')}|r={p.get('lora_r')}|"
                f"neftune={p.get('neftune_noise_alpha')}|lm={p.get('length_mode')}"
            )
            oom_by_combo[combo] = oom_by_combo.get(combo, 0) + 1

    complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
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
        "best_trial": best.number if best else None,
        "best_accuracy": best.value if best else None,
        "best_params": best.params if best else None,
        "complete_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": dict(t.params),
                "solo_retry": t.user_attrs.get("solo_retry", False),
                "parallel_oom_recovered": t.user_attrs.get("parallel_oom_recovered", False),
            }
            for t in sorted(complete, key=lambda x: x.value or -1, reverse=True)
        ],
        "trials": trials_out,
        "study_review": build_study_review(study),
        **meta,
    }
    out = cfg.run_dir / "trial_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    from dpo.train.study_report import write_study_report

    report_path = write_study_report(out)
    summary["study_report_html"] = str(report_path.resolve())
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

    prefix = worker_prefix(cfg.worker_id)
    log_line(prefix, f"worker started | run_dir={cfg.run_dir}")
    log_line(prefix, f"target COMPLETE>={cfg.target_complete_trials} max_attempted={cfg.max_attempted_trials}")
    study = load_study(cfg)
    try:
        optuna.storages.fail_stale_trials(study)
        log_line(prefix, "fail_stale_trials: cleaned zombie RUNNING trials")
    except Exception as e:
        log_line(prefix, f"fail_stale_trials skipped: {e}")
    while True:
        study = load_study(cfg)
        c = study_counts(study)
        stop, reason = should_stop_parallel(study, cfg)
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

        study = load_study(cfg)
        c = study_counts(study)
        stop, reason = should_stop_parallel(study, cfg)
        if stop:
            log_line(prefix, f"worker exit after trial: {reason} | counts={c}")
            return


def run_solo_retry_phase(cfg: OptunaRunConfig, run_trial_fn: TrialFn, deduped: list[dict]) -> None:
    if not deduped:
        launcher_log(cfg, "Solo retry queue empty — skipping")
        return
    launcher_log(cfg, f"SOLO RETRY phase: {len(deduped)} unique configs")
    study = load_study(cfg)
    for rec in deduped:
        study.enqueue_trial(rec["params"])
    for rec in deduped:
        study = load_study(cfg)

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
    load_study(cfg)
    launcher_log(cfg, f"PARALLEL OPTUNA | workers={cfg.parallel_workers} target={cfg.target_complete_trials}")
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

    study = load_study(cfg)
    summary = write_final_summary(
        study,
        cfg,
        queue_records=queue_records,
        deduped_queue=deduped,
        meta={
            "finished_at": datetime.now().isoformat(),
            "worker_exit_codes": codes,
            "solo_retry_count": len(deduped),
        },
    )
    complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if complete:
        launcher_log(cfg, f"DONE Best trial #{study.best_trial.number}: acc={study.best_trial.value:.4f}")
    else:
        launcher_log(cfg, "WARNING: no COMPLETE trials in study")
    launcher_log(cfg, f"Summary written: {summary}")
    report_path = cfg.run_dir / "study_report.html"
    if report_path.is_file():
        launcher_log(cfg, f"Study report: file://{report_path.resolve()}")
        from dpo.train.paths import EXPERIMENT_NAME, MLRUNS_DIR

        uri = cfg.mlflow_tracking_uri or f"file://{MLRUNS_DIR.resolve()}"
        monitor_doc = cfg.run_dir / "MONITOR.txt"
        with monitor_doc.open("a", encoding="utf-8") as f:
            f.write(f"\nStudy report (open in browser):\n  file://{report_path.resolve()}\n")
            f.write(
                f"\nMLflow UI:\n  mlflow ui --backend-store-uri {uri} --port 5001\n"
                f"  Experiment: {EXPERIMENT_NAME} → FINISHED parent run with nested trial-* children\n"
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
