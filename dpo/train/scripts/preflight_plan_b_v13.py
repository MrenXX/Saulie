#!/usr/bin/env python3
"""Preflight Plan B v1.3: merged base exists, baked model builder loads."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from dpo.train.paths import MODEL_ID_SFT_MERGED_BF16
from dpo.train.train_dpo import build_dpo_peft_model_baked


def main() -> None:
    if not MODEL_ID_SFT_MERGED_BF16.is_dir():
        raise SystemExit(f"Missing merged base: {MODEL_ID_SFT_MERGED_BF16}")
    print("Building baked DPO stack (smoke, no train)...")
    model = build_dpo_peft_model_baked(lora_r=8, lora_alpha=16, lora_dropout=0.05)
    adapters = list(model.peft_config.keys())
    print(f"  adapters: {adapters}")
    assert "default" in adapters and "dpo" in adapters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable: {trainable:,}")
    print("OK")


if __name__ == "__main__":
    main()
