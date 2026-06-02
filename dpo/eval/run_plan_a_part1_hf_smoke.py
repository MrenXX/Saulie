"""Plan A Part 1: HF exact-stack smoke at dpo_weight=1.0 on the 10-skeleton set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dpo.train.merge_sft_dpo_lora import load_stacked_for_merge, resolve_dpo_adapter_path
from dpo.train.qwen3_decode import add_decode_argparse
from dpo.train.smoke_policy_stack_hf import activate_adapter_mode, run_skeleton
from dpo.train.train_dpo import load_tokenizer

# plan_a_existing_trials_and_final_stack_retrains.md — Part 1 slate
PLAN_A_TRIALS = {
    "v11_trial13": REPO_ROOT
    / "dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-13/best_adapter",
    "v10_trial23": REPO_ROOT
    / "dpo/train/models/steering-dpo-v1.0/optuna-run-20260523-041252/trial-23/best_adapter",
    "v11_trial0": REPO_ROOT
    / "dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-0/best_adapter",
    "v11_trial17": REPO_ROOT
    / "dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-17/best_adapter",
    "v11_trial15": REPO_ROOT
    / "dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-15/best_adapter",
    "v11_trial10": REPO_ROOT
    / "dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-10/best_adapter",
}

SKELETON_IDS = (
    "eval_A4_002,eval_A4_003,eval_A4_004,eval_A6_002,eval_B8_001,"
    "eval_B8_002,eval_C4_001,eval_C8_005,eval_D6_002,eval_D8_002"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "dpo/eval/plan_a_part1_w1_10skel.jsonl",
    )
    add_decode_argparse(parser)
    args = parser.parse_args()

    sk_path = REPO_ROOT / "dpo/eval/eval_skeletons.json"
    with sk_path.open(encoding="utf-8") as f:
        by_id = {s["id"]: s for s in json.load(f)}
    sk_ids = [s.strip() for s in SKELETON_IDS.split(",")]
    skeletons = [by_id[i] for i in sk_ids]

    tokenizer = load_tokenizer()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = len(PLAN_A_TRIALS) * len(skeletons)
    done = 0

    with args.output.open("w", encoding="utf-8") as out:
        for trial_name, adapter_path in PLAN_A_TRIALS.items():
            if not adapter_path.is_dir():
                raise FileNotFoundError(adapter_path)
            dpo_path = resolve_dpo_adapter_path(adapter_path)
            print(f"\n=== {trial_name} w=1.0 ===")
            model = load_stacked_for_merge(dpo_path)
            activate_adapter_mode(model, "policy")
            model.eval()

            for sk in skeletons:
                messages = run_skeleton(
                    model,
                    tokenizer,
                    sk,
                    adapter_mode="policy",
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


if __name__ == "__main__":
    main()
