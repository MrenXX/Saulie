#!/usr/bin/env python3
"""Babysit parallel Optuna study until trial_summary.json + target_reached."""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import optuna
from optuna.trial import TrialState


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


def parse_running_trials(run_dir: Path) -> dict[int, dict]:
    """trial_number -> {worker, started_line, last_heartbeat_ts}."""
    info: dict[int, dict] = {}
    hb_re = re.compile(
        r"\[(\d{2}:\d{2}:\d{2})\] W(\d) heartbeat step=(\d+)"
    )
    start_re = re.compile(r"\[(\d{2}:\d{2}:\d{2})\] W(\d) TRIAL (\d+) START")
    for wlog in sorted(run_dir.glob("worker_*.log")):
        wid = int(wlog.stem.split("_")[1])
        text = wlog.read_text(encoding="utf-8", errors="replace")
        for m in start_re.finditer(text):
            tnum = int(m.group(3))
            info[tnum] = {"worker": wid, "start": m.group(1), "log": wlog}
        for m in hb_re.finditer(text):
            tnum = None
            for tn, meta in info.items():
                if meta.get("worker") == wid:
                    tnum = tn
            if tnum is not None:
                info[tnum]["last_hb"] = m.group(1)
                info[tnum]["last_step"] = int(m.group(3))
    return info


def kill_study_processes(run_dir: Path) -> None:
    rd = str(run_dir)
    subprocess.run(["pkill", "-f", rd], check=False)
    time.sleep(3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--study-name", default="steering-dpo-v1.1-v4-seed42")
    ap.add_argument("--target-complete", type=int, default=20)
    ap.add_argument("--poll-min", type=float, default=20.0)
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()
    storage = f"sqlite:///{run_dir / 'optuna_study.db'}"
    summary_path = run_dir / "trial_summary.json"

    print(f"Babysit {run_dir} target={args.target_complete}", flush=True)
    while True:
        now = datetime.now().isoformat(timespec="seconds")
        if summary_path.is_file():
            import json

            s = json.loads(summary_path.read_text())
            if s.get("target_reached"):
                print(f"{now} DONE target_reached", flush=True)
                return
        study = optuna.load_study(study_name=args.study_name, storage=storage)
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
                print(f"  #{t.number} {t.state.name} val={t.value} {fr}", flush=True)

        sleep_min = args.poll_min
        stale_kill = False
        for t in running:
            params = dict(t.params)
            budget_min = trial_budget_min(params)
            sleep_min = max(sleep_min, min(budget_min * 0.15, 25.0))
            log_path = run_dir / f"worker_{t.user_attrs.get('worker_id', 0)}.log"
            if not log_path.exists():
                for p in run_dir.glob("worker_*.log"):
                    if f"TRIAL {t.number} START" in p.read_text(errors="replace"):
                        log_path = p
                        break
            age = log_mtime_age_s(log_path)
            age_min = (age or 0) / 60.0
            hard_cap = max(budget_min * 2.5, 180.0)
            print(
                f"  RUN #{t.number} budget~{budget_min:.0f}m log_age={age_min:.0f}m "
                f"cap={hard_cap:.0f}m {params.get('length_mode')} ep={params.get('num_train_epochs')}",
                flush=True,
            )
            if age is not None and age_min > hard_cap:
                print(
                    f"  KILL stale trial #{t.number} (log silent {age_min:.0f}m > {hard_cap:.0f}m)",
                    flush=True,
                )
                stale_kill = True
            # Cliff in log: step_wall_s logged
            if log_path.exists():
                tail = log_path.read_text(errors="replace")[-8000:]
                if "WATCHDOG PRUNE" in tail or "step_stall" in tail:
                    print(f"  watchdog pruned on worker log", flush=True)
                m = re.findall(r"step_wall_s=(\d+)", tail)
                if m and int(m[-1]) > 3600:
                    print(f"  KILL pathological step_wall>{m[-1]}s", flush=True)
                    stale_kill = True

        if stale_kill:
            kill_study_processes(run_dir)
            print("Killed stale run; fail_stale_trials on next worker start.", flush=True)
            time.sleep(10)
            continue

        if complete >= args.target_complete and not running:
            # wait for summary
            for _ in range(12):
                if summary_path.is_file():
                    import json

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
