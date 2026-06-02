#!/usr/bin/env python3
"""Preflight Plan A v1.2: TRL IPO loss + ld_0.5 loss types, data, SFT path."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    import trl
    from trl import DPOConfig

    from dpo.train.paths import DATA_PATH, SFT_ADAPTER
    from dpo.train.train_dpo import parse_length_mode

    print(f"TRL version: {trl.__version__}")
    print(f"Data: {DATA_PATH} exists={DATA_PATH.is_file()}")
    print(f"SFT adapter: {SFT_ADAPTER} exists={SFT_ADAPTER.is_dir()}")

    for mode in ("ld_0.5", "ipo"):
        loss_type, ld_alpha, use_weighting = parse_length_mode(mode)
        cfg = DPOConfig(
            output_dir="/tmp/dpo_v12_preflight",
            loss_type=loss_type,
            ld_alpha=ld_alpha,
            use_weighting=use_weighting,
            beta=0.20,
            max_length=512,
            report_to="none",
        )
        print(f"  {mode}: loss_type={loss_type} ld_alpha={ld_alpha} use_weighting={use_weighting} OK")

    print("\nPreflight PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
