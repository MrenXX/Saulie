#!/usr/bin/env python3
"""Babysit parallel Optuna study until trial_summary_<version>.json + target_reached."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import optuna
from optuna.trial import TrialState

REPO_ROOT = Path(__file__).resolve().parents[3]
import sys

sys.path.insert(0, str(REPO_ROOT))
from dpo.train.paths import trial_summary_path


def trial_budget_min(params: dict) -> float:
    """Expected wall minutes for a trial (heuristic)."""
    epochs = int(params.get("num_train_epochs", 1))
    lm = params.get("length_mode", "ld_0.3")
    batch = params.get("batch_combo", "1x8")
    base = 22.0
    mult = 1.0
    if epochs >= 2:
        mult *= 1.75
    if lm == "ld_0.5":
        mult *= 1.6
    elif lm in ("ld_0.2", "ld_0.1"):
        mult *= 1.15
    if batch in ("1x16",):
        mult *= 1.35
    elif batch in ("2x8", "2x4"):
        mult *= 1.25
    elif batch in ("2x2",):
        mult *= 1.15
    return base * epochs * mult


def log_mtime_age_s(path: Path) -> float | None:
    if not path.exists():
        return None
    return time.time() - path.stat().st_mtime


def parse_worker_log_tail(log_path: Path, tail_chars: int = 12000) -> dict:
    """Extract last stage, training heartbeat, val_diag progress from log tail."""
    out: dict = {
        "last_stage": None,
        "last_hb_step": None,
        "last_val_diag": None,
    }
    if not log_path.is_file():
        return out
    text = log_path.read_text(encoding="utf-8", errors="replace")[-tail_chars:]
    stages = re.findall(r"stage=([a-z0-9_]+)", text)
    if stages:
        out["last_stage"] = stages[-1]
    hb = re.findall(r"heartbeat step=(\d+)", text)
    if hb:
        out["last_hb_step"] = int(hb[-1])
    vd = re.findall(r"val_diag trial=\d+ row=(\d+)/(\d+)", text)
    if vd:
        out["last_val_diag"] = f"{vd[-1][0]}/{vd[-1][1]}"
    vd_done = re.findall(r"stage=val_diagnostics done", text)
    if vd_done:
        out["val_diag_done"] = True
    return out


def find_worker_log(run_dir: Path, trial: optuna.trial.FrozenTrial) -> Path | None:
    wid = trial.user_attrs.get("worker_id")
    if wid is not None:
        p = run_dir / f"worker_{wid}.log"
        if p.is_file():
            return p
    for p in sorted(run_dir.glob("worker_*.log")):
        if f"TRIAL {trial.number} START" in p.read_text(encoding="utf-8", errors="replace"):
            return p
    return None


def kill_study_processes(run_dir: Path) -> None:
    rd = str(run_dir)
    subprocess.run(["pkill", "-f", rd], check=False)
    time.sleep(3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--study-name", default="steering-dpo-v1.1-v4-seed42")
    ap.add_argument("--study-version", default="v1.1")
    ap.add_argument("--target-complete", type=int, default=20)
    ap.add_argument("--poll-min", type=float, default=20.0)
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()
    storage = f"sqlite:///{run_dir / 'optuna_study.db'}"
    summary_path = trial_summary_path(run_dir, args.study_version)
    legacy = run_dir / "trial_summary.json"
    if legacy.is_file() and not summary_path.is_file():
        summary_path = legacy

    print(
        f"Babysit {run_dir} target={args.target_complete} summary={summary_path.name}",
        flush=True,
    )
    db_path = run_dir / "optuna_study.db"
    while True:
        now = datetime.now().isoformat(timespec="seconds")
        if summary_path.is_file():
            s = json.loads(summary_path.read_text())
            if s.get("target_reached"):
                print(f"{now} DONE target_reached ({summary_path.name})", flush=True)
                return
        if not db_path.is_file():
            print(f"{now} waiting for {db_path.name}...", flush=True)
            time.sleep(5)
            continue
        try:
            study = optuna.load_study(study_name=args.study_name, storage=storage)
        except KeyError:
            print(f"{now} waiting for study {args.study_name}...", flush=True)
            time.sleep(5)
            continue
        from collections import Counter

        c = Counter(t.state for t in study.trials)
        complete = c.get(TrialState.COMPLETE, 0)
        fail = c.get(TrialState.FAIL, 0)
        running = [t for t in study.trials if t.state == TrialState.RUNNING]
        print(
            f"{now} COMPLETE={complete}/{args.target_complete} "
            f"FAIL={fail} RUNNING={len(running)}",
            flush=True,
        )
        for t in study.trials:
            if t.state in (TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED):
                fr = (t.user_attrs.get("failure_reason") or "")[:50]
                ls = t.user_attrs.get("last_stage", "")
                print(
                    f"  #{t.number} {t.state.name} val={t.value} "
                    f"stage={ls} {fr}",
                    flush=True,
                )

        sleep_min = args.poll_min
        stale_kill = False
        for t in running:
            params = dict(t.params)
            budget_min = trial_budget_min(params)
            sleep_min = max(sleep_min, min(budget_min * 0.15, 25.0))
            log_path = find_worker_log(run_dir, t)
            age = log_mtime_age_s(log_path) if log_path else None
            age_min = (age or 0) / 60.0
            hard_cap = max(budget_min * 2.5, 180.0)
            last_stage = t.user_attrs.get("last_stage") or "?"
            val_prog = t.user_attrs.get("val_diag_progress")
            log_meta: dict = {}
            if log_path:
                log_meta = parse_worker_log_tail(log_path)
                if log_meta.get("last_stage"):
                    last_stage = log_meta["last_stage"]
                if log_meta.get("last_val_diag"):
                    val_prog = log_meta["last_val_diag"]

            print(
                f"  RUN #{t.number} budget~{budget_min:.0f}m log_age={age_min:.0f}m "
                f"cap={hard_cap:.0f}m last_stage={last_stage} "
                f"val_diag={val_prog or '-'} "
                f"{params.get('length_mode')} ep={params.get('num_train_epochs')}",
                flush=True,
            )

            # Post-train val diagnostics still updates logs / user_attrs.
            in_val_diag = last_stage == "val_diagnostics" or (
                val_prog is not None and not log_meta.get("val_diag_done")
            )
            if in_val_diag and age is not None and age_min < hard_cap:
                print(
                    f"    val_diagnostics in progress (log_age={age_min:.0f}m), not stale",
                    flush=True,
                )
                continue

            if age is not None and age_min > hard_cap:
                print(
                    f"  KILL stale trial #{t.number} "
                    f"(log silent {age_min:.0f}m > {hard_cap:.0f}m, stage={last_stage})",
                    flush=True,
                )
                stale_kill = True
            if log_path and log_path.exists():
                tail = log_path.read_text(errors="replace")[-8000:]
                if "WATCHDOG PRUNE" in tail or "step_stall" in tail:
                    print("  watchdog pruned on worker log", flush=True)
                m = re.findall(r"step_wall_s=(\d+)", tail)
                if m and int(m[-1]) > 3600:
                    print(f"  KILL pathological step_wall>{m[-1]}s", flush=True)
                    stale_kill = True
                if "val_diagnostics_timeout" in tail:
                    print("  val_diagnostics_timeout in log", flush=True)

        if stale_kill:
            kill_study_processes(run_dir)
            try:
                optuna.storages.fail_stale_trials(study)
                print("fail_stale_trials: marked zombie RUNNING trials FAIL", flush=True)
            except Exception as e:
                print(f"fail_stale_trials skipped: {e}", flush=True)
            print("Killed stale run; workers retry on next poll.", flush=True)
            time.sleep(10)
            continue

        if complete >= args.target_complete and not running:
            for _ in range(12):
                if summary_path.is_file():
                    if json.loads(summary_path.read_text()).get("target_reached"):
                        print(f"{now} DONE", flush=True)
                        return
                time.sleep(30)
        if complete >= args.target_complete:
            time.sleep(60)
            continue

        time.sleep(sleep_min * 60)


if __name__ == "__main__":
    main()
