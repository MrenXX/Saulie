#!/usr/bin/env python3
"""
HF BnB8 batch generation (local / merge debug only).

NOT used for the final judge packet — production eval uses vLLM FP8:
  dpo/eval/vllm_scripts/eval_generate_vllm.py
  dpo/eval/run_v15_final_eval.py

SFT baseline: BnB8 + frozen SFT trial-17
DPO finalists: BnB8 + cat-merged sft_dpo_cat

Locked eval sampling (DPO_FINAL_EVAL_EXECUTION_PLAN.md):
  sample temp=0.7 top_p=0.8 top_k=20 repetition_penalty=1.05 max_new_tokens=256

Examples:
  # Limit test
  python dpo/eval/eval_generate_hf.py \\
    --round 1 --skeleton-ids eval_A4_001,eval_B8_001,eval_O4_001 \\
    --models sft,trial-16 \\
    --output dpo/eval/generations_limit_test.json

  # Full Round 1 (blind + unblind sidecar)
  python dpo/eval/eval_generate_hf.py --round 1 --anonymize \\
    --output dpo/eval/generations_round1.json \\
    --unblind-output dpo/eval/generations_round1_unblind.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dpo.train.merge_sft_dpo_lora import load_cat_merged_adapter, load_sft_stack
from dpo.train.paths import MODEL_ID_BF16, SFT_ADAPTER
from dpo.train.qwen3_decode import (
    DECODE_SAMPLE,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    generation_metadata,
)
from dpo.train.smoke_policy_stack_hf import run_skeleton
from dpo.train.train_dpo import load_tokenizer

DEFAULT_SKELETONS = REPO_ROOT / "dpo/eval/eval_skeletons.json"
V15_RUN = REPO_ROOT / "dpo/train/models/steering-dpo-v1.5/optuna-run-20260602-052732"

# Eval-locked defaults (override qwen3_decode REPETITION_PENALTY_DEFAULT=1.0)
EVAL_MAX_NEW_TOKENS = 256
EVAL_REPETITION_PENALTY = 1.05

BASELINE_MODEL_NAME = "steering-sft-v1.1_trial-17"
FINALIST_TRIALS = (19, 16, 8, 27, 20, 4)

ROUND1_STEERING_IDS = [
    "eval_A4_001", "eval_A4_003", "eval_A6_001", "eval_A6_003", "eval_A8_001", "eval_A10_001",
    "eval_B6_001", "eval_B8_001", "eval_B8_003", "eval_B8_005", "eval_B10_001", "eval_B10_003", "eval_B10_005",
    "eval_C4_001", "eval_C6_001", "eval_C6_003", "eval_C8_001", "eval_C8_003", "eval_C10_001",
    "eval_D4_001", "eval_D6_001", "eval_D6_003", "eval_D8_001", "eval_D8_002", "eval_D10_001", "eval_D10_003",
]

ROUND1_ORDINARY_IDS = [
    "eval_O4_001", "eval_O4_002", "eval_O6_001", "eval_O6_002", "eval_O6_003",
    "eval_O8_001", "eval_O8_002", "eval_O8_003",
]

ROUND2_STEERING_IDS = [
    "eval_A4_002", "eval_A4_004", "eval_A4_005", "eval_A6_002", "eval_A6_004", "eval_A6_005", "eval_A8_002",
    "eval_B6_002", "eval_B8_002", "eval_B8_004", "eval_B8_006", "eval_B10_002", "eval_B10_004",
    "eval_C4_002", "eval_C6_002", "eval_C6_004", "eval_C8_002", "eval_C8_004", "eval_C8_005", "eval_C10_002",
    "eval_D4_002", "eval_D6_002", "eval_D6_004", "eval_D8_003", "eval_D8_004", "eval_D10_002",
]

ROUND2_ORDINARY_IDS = list(ROUND1_ORDINARY_IDS)

CANDIDATE_LETTERS = ("A", "B", "C", "D", "E", "F")


def round_skeleton_ids(round_num: int) -> list[str]:
    if round_num == 1:
        return ROUND1_STEERING_IDS + ROUND1_ORDINARY_IDS
    if round_num == 2:
        return ROUND2_STEERING_IDS + ROUND2_ORDINARY_IDS
    raise ValueError(f"round must be 1 or 2, got {round_num}")


def skeleton_eval_kind(skeleton: dict) -> str:
    return skeleton.get("eval_kind") or (
        "ordinary_conversation" if skeleton.get("opening_type") == "O" else "steering"
    )


def load_skeletons(path: Path, wanted_ids: list[str]) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        all_rows = json.load(f)
    by_id = {row["id"]: row for row in all_rows}
    missing = [sid for sid in wanted_ids if sid not in by_id]
    if missing:
        raise SystemExit(f"Missing skeleton ids in {path}: {missing}")
    return [by_id[sid] for sid in wanted_ids]


def default_finalist_specs(run_dir: Path = V15_RUN) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "key": "sft",
            "model_name": BASELINE_MODEL_NAME,
            "kind": "sft_baseline",
            "trial_number": 17,
            "adapter_mode": "sft",
            "cat_adapter": None,
        }
    ]
    for i, trial in enumerate(FINALIST_TRIALS):
        specs.append(
            {
                "key": f"trial-{trial}",
                "model_name": f"steering-dpo-v1.5_trial-{trial}_sft_dpo_cat",
                "kind": "dpo_merged",
                "trial_number": trial,
                "adapter_mode": "cat",
                "cat_adapter": run_dir / f"trial-{trial}" / "sft_dpo_cat",
                "candidate_letter": CANDIDATE_LETTERS[i],
            }
        )
    return specs


def parse_models_arg(models_arg: str | None, run_dir: Path) -> list[dict[str, Any]]:
    all_specs = {s["key"]: s for s in default_finalist_specs(run_dir)}
    all_specs["baseline"] = all_specs["sft"]
    if models_arg is None:
        return default_finalist_specs(run_dir)
    keys = [k.strip().lower() for k in models_arg.split(",") if k.strip()]
    out: list[dict[str, Any]] = []
    for key in keys:
        if key not in all_specs:
            raise SystemExit(f"Unknown model key {key!r}; choose from: {sorted(all_specs)}")
        spec = dict(all_specs[key])
        out.append(spec)
    return out


def load_model_for_spec(spec: dict[str, Any]):
    if spec["adapter_mode"] == "sft":
        return load_sft_stack(base="bnb")
    cat_path = spec["cat_adapter"]
    if cat_path is None or not Path(cat_path).is_dir():
        raise FileNotFoundError(f"Missing cat adapter for {spec['key']}: {cat_path}")
    return load_cat_merged_adapter(Path(cat_path), base="bnb")


def release_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def blind_model_name(spec: dict[str, Any], anonymize: bool) -> str:
    if not anonymize or spec["kind"] == "sft_baseline":
        return spec["model_name"]
    letter = spec.get("candidate_letter")
    if not letter:
        raise ValueError(f"Missing candidate_letter for {spec['key']}")
    return f"candidate_{letter}"


def build_model_result(
    spec: dict[str, Any],
    *,
    anonymize: bool,
    conversations: list[dict],
    elapsed_s: float,
    gen_meta: dict,
) -> dict[str, Any]:
    public_name = blind_model_name(spec, anonymize)
    row: dict[str, Any] = {
        "model": public_name,
        "is_baseline": spec["kind"] == "sft_baseline",
        "kind": spec["kind"],
        "adapter_mode": spec["adapter_mode"],
        "generation": gen_meta,
        "elapsed_seconds": round(elapsed_s, 2),
        "conversations": conversations,
    }
    if not anonymize or spec["kind"] == "sft_baseline":
        row["manifest"] = {
            "model_name": spec["model_name"],
            "trial_number": spec.get("trial_number"),
            "cat_adapter": str(spec["cat_adapter"]) if spec.get("cat_adapter") else None,
        }
    return row


def build_unblind_mapping(specs: list[dict[str, Any]]) -> dict[str, Any]:
    mapping = {}
    for spec in specs:
        if spec["kind"] == "sft_baseline":
            continue
        letter = spec.get("candidate_letter")
        mapping[f"candidate_{letter}"] = {
            "trial_number": spec["trial_number"],
            "model_name": spec["model_name"],
            "cat_adapter": str(spec["cat_adapter"]),
        }
    return mapping


def run_generation(
    *,
    specs: list[dict[str, Any]],
    skeletons: list[dict],
    decode: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    anonymize: bool,
) -> tuple[dict[str, dict], dict[str, Any]]:
    tokenizer = load_tokenizer()
    gen_meta = generation_metadata(
        decode=decode,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    all_results: dict[str, dict] = {}

    for idx, spec in enumerate(specs):
        public_name = blind_model_name(spec, anonymize)
        print(f"\n{'=' * 60}")
        print(f"[{idx + 1}/{len(specs)}] Loading {spec['key']} -> {public_name}")
        print(f"{'=' * 60}")

        t0 = time.time()
        model = load_model_for_spec(spec)
        model.eval()
        load_s = time.time() - t0
        print(f"  loaded in {load_s:.1f}s")

        conversations: list[dict] = []
        for sk_idx, skeleton in enumerate(skeletons):
            print(
                f"\n  [{sk_idx + 1}/{len(skeletons)}] {skeleton['id']} "
                f"type={skeleton['opening_type']} eval_kind={skeleton_eval_kind(skeleton)}"
            )
            messages = run_skeleton(
                model,
                tokenizer,
                skeleton,
                adapter_mode=spec["adapter_mode"],
                decode=decode,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
            conversations.append(
                {
                    "skeleton_id": skeleton["id"],
                    "opening_type": skeleton["opening_type"],
                    "eval_kind": skeleton_eval_kind(skeleton),
                    "target_turns": skeleton["target_turns"],
                    "actual_turns": len(messages),
                    "messages": messages,
                }
            )

        elapsed = time.time() - t0
        all_results[public_name] = build_model_result(
            spec,
            anonymize=anonymize,
            conversations=conversations,
            elapsed_s=elapsed,
            gen_meta=gen_meta,
        )
        release_model(model)

    unblind = build_unblind_mapping(specs)
    return all_results, unblind


def main() -> None:
    parser = argparse.ArgumentParser(description="HF BnB8 DPO final eval batch generation")
    parser.add_argument("--round", type=int, choices=(1, 2), default=1)
    parser.add_argument("--skeletons", type=Path, default=DEFAULT_SKELETONS)
    parser.add_argument(
        "--skeleton-ids",
        type=str,
        default=None,
        help="Comma-separated skeleton ids (overrides --round list)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated keys: sft, trial-16, trial-19, ... (default: full slate)",
    )
    parser.add_argument("--run-dir", type=Path, default=V15_RUN)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--unblind-output",
        type=Path,
        default=None,
        help="Write trial mapping sidecar when --anonymize",
    )
    parser.add_argument(
        "--anonymize",
        action="store_true",
        help="Blind DPO models to candidate_A..F; keep SFT baseline name",
    )
    parser.add_argument("--decode", default=DECODE_SAMPLE, choices=(DECODE_SAMPLE,))
    parser.add_argument("--max-new-tokens", type=int, default=EVAL_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=EVAL_REPETITION_PENALTY,
    )
    args = parser.parse_args()

    if args.skeleton_ids:
        wanted = [s.strip() for s in args.skeleton_ids.split(",") if s.strip()]
    else:
        wanted = round_skeleton_ids(args.round)

    skeletons = load_skeletons(args.skeletons, wanted)
    specs = parse_models_arg(args.models, args.run_dir)

    print(f"Round: {args.round}")
    print(f"Skeletons: {len(skeletons)}")
    print(f"Models: {[s['key'] for s in specs]}")
    print(
        f"Sampling: decode={args.decode} temp={args.temperature} top_p={args.top_p} "
        f"top_k={args.top_k} rep_penalty={args.repetition_penalty} max_new_tokens={args.max_new_tokens}"
    )
    print(f"Anonymize: {args.anonymize}")

    total_start = time.time()
    all_results, unblind = run_generation(
        specs=specs,
        skeletons=skeletons,
        decode=args.decode,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        anonymize=args.anonymize,
    )
    total_elapsed = time.time() - total_start

    payload = {
        "generated_at": datetime.now().isoformat(),
        "eval_round": args.round,
        "policy": "hf_bnb8_sft_or_cat",
        "base_model": str(MODEL_ID_BF16),
        "sft_adapter": str(SFT_ADAPTER),
        "run_dir": str(args.run_dir.resolve()),
        "skeletons_path": str(args.skeletons.resolve()),
        "skeleton_ids": wanted,
        "anonymized": args.anonymize,
        "generation": generation_metadata(
            decode=args.decode,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        ),
        "models": all_results,
        "wall_seconds": round(total_elapsed, 2),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {args.output}")

    if args.anonymize and args.unblind_output:
        sidecar = {
            "generated_at": payload["generated_at"],
            "eval_round": args.round,
            "baseline_model": BASELINE_MODEL_NAME,
            "candidate_mapping": unblind,
            "note": "Do not send this file to the LLM judge.",
        }
        args.unblind_output.parent.mkdir(parents=True, exist_ok=True)
        args.unblind_output.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        print(f"Wrote {args.unblind_output}")

    total_convs = sum(len(r["conversations"]) for r in all_results.values())
    print(f"Models: {len(all_results)} | conversations: {total_convs} | wall: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
