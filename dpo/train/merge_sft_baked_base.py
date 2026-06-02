"""
Merge frozen SFT trial-17 LoRA into the dense BF16 Qwen base (Plan B).

Must use the unquantized instruct checkpoint — do not merge against FP8 or BnB-loaded
weights. Output is a standalone dense checkpoint (no active SFT adapter at inference).

  source /root/miniconda3/etc/profile.d/conda.sh && conda activate saulgman
  python dpo/train/merge_sft_baked_base.py
  python dpo/train/merge_sft_baked_base.py --output /path/to/Qwen3-4B-Instruct-2507-SFT-MERGED-BF16
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import peft
import torch
import transformers
from peft import PeftModel
from transformers import AutoTokenizer

from dpo.train.model_load import load_bf16_base
from dpo.train.paths import MODEL_ID_BF16, MODEL_ID_SFT_MERGED_BF16, SFT_ADAPTER
from dpo.train.train_dpo import load_tokenizer

DEFAULT_OUTPUT = MODEL_ID_SFT_MERGED_BF16


def merge_sft_into_dense_base(
    *,
    base_path: Path = MODEL_ID_BF16,
    sft_adapter: Path = SFT_ADAPTER,
    output_dir: Path = DEFAULT_OUTPUT,
) -> Path:
    if not base_path.is_dir():
        raise FileNotFoundError(f"BF16 base not found: {base_path}")
    if not (sft_adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"SFT adapter not found: {sft_adapter}")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dense base: {base_path}")
    base = load_bf16_base()
    print(f"Loading SFT adapter: {sft_adapter}")
    model = PeftModel.from_pretrained(
        base,
        str(sft_adapter),
        is_trainable=False,
    )
    print("Merging SFT LoRA into base weights (merge_and_unload)...")
    merged = model.merge_and_unload()
    merged.eval()

    print(f"Saving merged checkpoint to {output_dir}")
    merged.save_pretrained(str(output_dir), safe_serialization=True)

    print("Saving tokenizer (patched chat template)...")
    tokenizer = load_tokenizer()
    tokenizer.save_pretrained(str(output_dir))

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plan": "plan_b_sft_baked_base",
        "raw_base_path": str(base_path.resolve()),
        "sft_adapter_path": str(sft_adapter.resolve()),
        "output_dir": str(output_dir),
        "dtype": "bfloat16",
        "merge_method": "peft.merge_and_unload",
        "peft_version": peft.__version__,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "note": "Dense merge only; do not run against MODEL_ID_FP8 or BnB-quantized load.",
    }
    meta_path = output_dir / "merge_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {meta_path}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge SFT trial-17 LoRA into dense BF16 Qwen3 base (Plan B)"
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=MODEL_ID_BF16,
        help="Unquantized instruct checkpoint (default: MODEL_ID_BF16)",
    )
    parser.add_argument(
        "--sft-adapter",
        type=Path,
        default=SFT_ADAPTER,
        help="SFT best_adapter directory (default: trial-17)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for merged weights + tokenizer",
    )
    args = parser.parse_args()
    out = merge_sft_into_dense_base(
        base_path=args.base_path,
        sft_adapter=args.sft_adapter,
        output_dir=args.output,
    )
    print(f"\nDone. SFT-baked base: {out}")
    print("REPL (conda env saulgman):")
    print("  python dpo/eval/chat_policy_stack.py --sft-baked")
    print("  python dpo/eval/chat_policy_stack.py --sft-baked --base bf16")


if __name__ == "__main__":
    main()
