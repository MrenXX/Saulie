"""
HF BnB policy stack grid: trial × dpo_weight × skeleton → JSONL (one line per run).

Training setup only: load SFT + DPO, set_adapter([default,dpo]), scale DPO residual.
Reloads model per (trial, weight) so scaling is not compounded.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dpo.train.merge_sft_dpo_lora import load_stacked_for_merge, resolve_dpo_adapter_path
from dpo.train.smoke_policy_stack_hf import (
    DPO_ADAPTER_NAME,
    activate_adapter_mode,
    run_skeleton,
    scale_adapter_residual,
)
from dpo.train.train_dpo import load_tokenizer

TRIALS = {
    "trial29": REPO_ROOT
    / "dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-29/best_adapter",
    "trial10": REPO_ROOT
    / "dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-10/best_adapter",
    "trial1_v10": REPO_ROOT
    / "dpo/train/models/steering-dpo-v1.0/optuna-run-20260523-041252/trial-1/best_adapter",
}
WEIGHTS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00]
SKELETONS_PATH = REPO_ROOT / "dpo/eval/eval_skeletons.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "dpo/eval/train_setup_grid_10skel.jsonl")
    parser.add_argument("--n-skeletons", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decode", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--skeleton-ids", type=str, default=None, help="comma list; overrides random pick")
    args = parser.parse_args()

    with SKELETONS_PATH.open(encoding="utf-8") as f:
        all_sk = json.load(f)
    by_id = {s["id"]: s for s in all_sk}

    if args.skeleton_ids:
        sk_ids = [s.strip() for s in args.skeleton_ids.split(",")]
    else:
        rng = random.Random(args.seed)
        sk_ids = sorted(rng.sample(list(by_id.keys()), args.n_skeletons))

    skeletons = [by_id[i] for i in sk_ids]
    print(f"Skeletons ({len(sk_ids)}): {', '.join(sk_ids)}")

    tokenizer = load_tokenizer()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = len(TRIALS) * len(WEIGHTS) * len(skeletons)
    done = 0

    with args.output.open("w", encoding="utf-8") as out:
        for trial_name, adapter_path in TRIALS.items():
            dpo_path = resolve_dpo_adapter_path(adapter_path)
            for w in WEIGHTS:
                print(f"\n=== {trial_name} w={w} ===")
                model = load_stacked_for_merge(dpo_path)
                if w != 1.0:
                    scale_adapter_residual(model, DPO_ADAPTER_NAME, w)
                activate_adapter_mode(model, "policy")
                model.eval()

                for sk in skeletons:
                    messages = run_skeleton(
                        model,
                        tokenizer,
                        sk,
                        adapter_mode="policy",
                        decode=args.decode,
                        max_new_tokens=350,
                        temperature=0.7,
                        top_p=0.8,
                    )
                    row = {
                        "skeleton_id": sk["id"],
                        "trial": trial_name,
                        "dpo_weight": w,
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


if __name__ == "__main__":
    main()
