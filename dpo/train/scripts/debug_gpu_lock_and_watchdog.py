#!/usr/bin/env python3
"""Quick debug checks: GPU train lock serialization + stall watchdog logic (no full Optuna)."""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from dpo.train.debug_probe import probe  # noqa: E402
from dpo.train.dpo_trainer_compat import StepWatchdogCallback, TrialWallTimeout  # noqa: E402
from dpo.train.gpu_train_lock import gpu_train_lock  # noqa: E402
from transformers.trainer_callback import TrainerControl, TrainerState  # noqa: E402
from transformers.training_args import TrainingArguments  # noqa: E402


def _lock_worker(run_dir: Path, worker_id: int, hold_s: float, out: mp.Queue) -> None:
    t0 = time.monotonic()
    with gpu_train_lock(run_dir, worker_id):
        waited = time.monotonic() - t0
        probe(
            "A",
            "debug_gpu_lock_and_watchdog.py:worker",
            "lock_held",
            {"worker_id": worker_id, "waited_s": round(waited, 3), "hold_s": hold_s},
        )
        time.sleep(hold_s)
    out.put({"worker_id": worker_id, "waited_s": waited})


def test_gpu_lock_serializes() -> None:
    run_dir = REPO / "dpo/train/models/_debug_lock_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    out: mp.Queue = mp.Queue()
    p0 = mp.Process(target=_lock_worker, args=(run_dir, 0, 1.5, out))
    p1 = mp.Process(target=_lock_worker, args=(run_dir, 1, 0.2, out))
    p0.start()
    time.sleep(0.2)
    p1.start()
    p0.join(timeout=30)
    p1.join(timeout=30)
    results = [out.get(timeout=5) for _ in range(2)]
    w1 = next(r for r in results if r["worker_id"] == 1)
    assert w1["waited_s"] >= 1.0, f"worker 1 should wait for worker 0, got {w1}"
    print("OK gpu_lock: worker1 waited", round(w1["waited_s"], 2), "s")


def test_stall_watchdog_cliff() -> None:
    """Simulate W0 step 8->9 cliff: 6s baseline then 368s step should prune."""
    cb = StepWatchdogCallback(trial_number=99, worker_id=0)
    args = TrainingArguments(output_dir="/tmp/dpo_dbg", max_steps=20)
    state = TrainerState()
    control = TrainerControl()
    for i, dt in enumerate([6.0, 6.0, 6.0, 6.0, 368.0], start=1):
        state.global_step = i
        cb._prev_log_mono = time.monotonic() - dt
        try:
            cb.on_log(args, state, control, logs={"loss": 0.5, "num_tokens": 1000 * i})
        except TrialWallTimeout as e:
            if i == 5:
                print("OK stall_watchdog: pruned cliff at step", i, "->", e)
                return
            raise
    raise AssertionError("expected TrialWallTimeout on cliff step")


def test_slow_uniform_steps_not_pruned() -> None:
    """Uniform 45 min/step (2700s) should NOT trip 8x median stall (only hard 3h step cap)."""
    cb = StepWatchdogCallback(trial_number=98, worker_id=0)
    args = TrainingArguments(output_dir="/tmp/dpo_dbg", max_steps=10)
    state = TrainerState()
    control = TrainerControl()
    slow = 45 * 60
    for i in range(1, 5):
        state.global_step = i
        cb._prev_log_mono = time.monotonic() - slow
        cb.on_log(args, state, control, logs={"loss": 0.5})
    print("OK stall_watchdog: uniform 45min steps not cliff-pruned (steps=", i, ")")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    test_gpu_lock_serializes()
    test_stall_watchdog_cliff()
    test_slow_uniform_steps_not_pruned()
    print("All debug checks passed.")
