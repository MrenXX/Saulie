#!/usr/bin/env python3
"""Preflight v1.5: RPO loss bundle + unmerged DPO stack."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from trl import DPOConfig

from dpo.train.paths import MODEL_ID_BF16, SFT_ADAPTER
from dpo.train.train_dpo import build_dpo_peft_model, make_dpo_config, rpo_loss_bundle


def main() -> None:
    if not MODEL_ID_BF16.is_dir():
        raise SystemExit(f"Missing base: {MODEL_ID_BF16}")
    if not SFT_ADAPTER.is_dir():
        raise SystemExit(f"Missing SFT adapter: {SFT_ADAPTER}")

    loss_types, loss_weights = rpo_loss_bundle(["sigmoid"], 0.5)
    assert loss_types == ["sigmoid", "sft"] and loss_weights == [1.0, 0.5]
    cfg = make_dpo_config(
        Path("/tmp/dpo_v15_preflight"),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1.5e-5,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.1,
        max_grad_norm=0.3,
        weight_decay=0.05,
        beta=0.05,
        loss_type=["sigmoid"],
        ld_alpha=0.3,
        use_weighting=False,
        label_smoothing=0.0,
        neftune_noise_alpha=0.0,
        rpo_alpha=0.5,
    )
    assert cfg.loss_type == ["sigmoid", "sft"]
    assert list(cfg.loss_weights) == [1.0, 0.5]
    print("DPOConfig RPO bundle OK")

    print("Building unmerged BnB + SFT + DPO stack...")
    model = build_dpo_peft_model(lora_r=16, lora_alpha=32, lora_dropout=0.05)
    if getattr(model, "_saulie_dpo_stack", None) == "baked":
        raise SystemExit("Expected unmerged stack, got baked")
    print("OK — v1.5 preflight passed")


if __name__ == "__main__":
    main()
