"""Redirect: DPO seq-length analysis lives in dpo/train/analyze_seq_len.py."""

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
runpy.run_path(str(REPO_ROOT / "dpo/train/analyze_seq_len.py"), run_name="__main__")
