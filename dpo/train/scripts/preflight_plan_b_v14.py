#!/usr/bin/env python3
"""Preflight Plan B v1.4: SFT adapter + unmerged DPO stack loads."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from dpo.train.paths import MODEL_ID_BF16, SFT_ADAPTER
from dpo.train.train_dpo import build_dpo_peft_model


def main() -> None:
    if not MODEL_ID_BF16.is_dir():
        raise SystemExit(f"Missing base: {MODEL_ID_BF16}")
    if not SFT_ADAPTER.is_dir():
        raise SystemExit(f"Missing SFT adapter: {SFT_ADAPTER}")
    print("Building unmerged BnB + SFT + DPO stack (trial-0 anchor ranks)...")
    model = build_dpo_peft_model(lora_r=8, lora_alpha=16, lora_dropout=0.05)
    stack = getattr(model, "_saulie_dpo_stack", None)
    print(f"  stack attr: {stack!r} (expect None or not 'baked')")
    if stack == "baked":
        raise SystemExit("Expected unmerged stack, got baked")
    print("OK — v1.4 preflight passed")


if __name__ == "__main__":
    main()
