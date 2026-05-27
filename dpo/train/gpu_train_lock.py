"""Exclusive GPU training lock for parallel Optuna workers on one GPU."""

from __future__ import annotations

import fcntl
import time
from contextlib import contextmanager
from pathlib import Path

@contextmanager
def gpu_train_lock(run_dir: Path, worker_id: int | None):
    """Only one worker may run trainer.train() at a time per run_dir."""
    lock_path = run_dir / ".gpu_train.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    t_wait = time.monotonic()
    label = f"W{worker_id}" if worker_id is not None else "solo"
    with lock_path.open("w", encoding="utf-8") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        waited_s = time.monotonic() - t_wait
        if waited_s >= 5.0:
            print(
                f"[{label}] gpu_train_lock acquired after {waited_s:.0f}s wait",
                flush=True,
            )
        try:
            yield
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
