"""Plan B Part 1: 10-skeleton HF smoke on SFT-baked base (+ optional DPO adapters)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dpo.train.merge_sft_dpo_lora import (
    load_baked_dpo_stack,
    resolve_dpo_adapter_path,
)
from dpo.train.model_load import load_sft_baked_base
from dpo.train.qwen3_decode import add_decode_argparse
from dpo.train.smoke_policy_stack_hf import activate_adapter_mode, run_skeleton
from dpo.train.train_dpo import load_tokenizer

SKELETON_IDS = (
    "eval_A4_002,eval_A4_003,eval_A4_004,eval_A6_002,eval_B8_001,"
    "eval_B8_002,eval_C4_001,eval_C8_005,eval_D6_002,eval_D8_002"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan B 10-skeleton HF smoke (Qwen3 sample decode @ w=1.0)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "dpo/eval/plan_b_part1_w1_10skel.jsonl",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="v1.3 optuna run dir; generates all trial-N/best_adapter",
    )
    parser.add_argument(
        "--dpo-adapter",
        type=Path,
        default=None,
        help="Single DPO adapter (trial-N/best_adapter); omit with --run-dir",
    )
    parser.add_argument(
        "--no-baked-control",
        action="store_true",
        help="Skip sft_baked_control baseline rows (default: include control)",
    )
    add_decode_argparse(parser)
    args = parser.parse_args()

    sk_path = REPO_ROOT / "dpo/eval/eval_skeletons.json"
    with sk_path.open(encoding="utf-8") as f:
        by_id = {s["id"]: s for s in json.load(f)}
    skeletons = [by_id[i.strip()] for i in SKELETON_IDS.split(",")]

    candidates: list[tuple[str, Path | None]] = []
    if not args.no_baked_control:
        candidates.append(("sft_baked_control", None))
    if args.run_dir is not None:
        run_dir = args.run_dir.expanduser().resolve()
        for trial_dir in sorted(run_dir.glob("trial-*/best_adapter")):
            if trial_dir.is_dir():
                label = trial_dir.parent.name.replace("trial-", "v13_trial")
                candidates.append((label, trial_dir))
    elif args.dpo_adapter is not None:
        candidates.append(("custom_dpo", args.dpo_adapter))
    else:
        raise SystemExit("Pass --run-dir or --dpo-adapter")

    tokenizer = load_tokenizer()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = len(candidates) * len(skeletons)
    done = 0

    with args.output.open("w", encoding="utf-8") as out:
        for trial_name, adapter_path in candidates:
            print(f"\n=== {trial_name} w=1.0 ===")
            if adapter_path is None:
                model = load_sft_baked_base("bnb")
                adapter_mode = "baked"
            else:
                dpo_path = resolve_dpo_adapter_path(adapter_path)
                model = load_baked_dpo_stack(dpo_path)
                adapter_mode = "dpo"
            model.eval()

            for sk in skeletons:
                messages = run_skeleton(
                    model,
                    tokenizer,
                    sk,
                    adapter_mode=adapter_mode,
                    decode=args.decode,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                )
                row = {
                    "skeleton_id": sk["id"],
                    "trial": trial_name,
                    "stack": "plan_b_baked",
                    "dpo_weight": 1.0,
                    "messages": messages,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                done += 1
                print(f"  [{done}/{total}] {sk['id']}")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nWrote {args.output} ({done} rows)")
    print("Score: python dpo/eval/score_rescue_smoke.py --jsonl", args.output)


if __name__ == "__main__":
    main()
