#!/usr/bin/env python3
"""
Orchestrate v1.5 DPO final eval on vLLM FP8.

  1. Optionally deploy (DEPLOY=1)
  2. Limit test or full Round 1 with runtime LoRA swap per DPO trial

Example:
  python dpo/eval/run_v15_final_eval.py --limit-test
  python dpo/eval/run_v15_final_eval.py --round 1 --anonymize
  python dpo/eval/run_v15_final_eval.py --round 1 --anonymize   # resumes if checkpoint exists
  python dpo/eval/run_v15_final_eval.py --round 1 --anonymize --fresh  # start over
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from dpo.eval.v15_eval_config import MANIFEST_PATH, VLLM_MAX_LORA_RANK

DEPLOY_SH = EVAL_DIR / "vllm_scripts" / "deploy_qwenie_eval.sh"
GEN_PY = EVAL_DIR / "vllm_scripts" / "eval_generate_vllm.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy", action="store_true", help="Run deploy_qwenie_eval.sh first")
    parser.add_argument("--limit-test", action="store_true")
    parser.add_argument("--round", type=int, choices=(1, 2), default=1)
    parser.add_argument("--anonymize", action="store_true")
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--skip-deploy", action="store_true")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing generations_roundN.json and restart from scratch",
    )
    args = parser.parse_args()

    if args.deploy and not args.skip_deploy:
        print(f"Deploying vLLM (MAX_LORA_RANK={VLLM_MAX_LORA_RANK})...")
        subprocess.run(["bash", str(DEPLOY_SH)], check=True)

    out = EVAL_DIR / (
        "generations_limit_test_vllm.json"
        if args.limit_test
        else f"generations_round{args.round}.json"
    )
    unblind = (
        EVAL_DIR / f"generations_round{args.round}_unblind.json"
        if args.anonymize and not args.limit_test
        else None
    )

    cmd = [
        sys.executable,
        str(GEN_PY),
        "--candidate-manifest",
        str(MANIFEST_PATH),
        "--output",
        str(out),
    ]
    if args.limit_test:
        cmd.extend(
            [
                "--round",
                "1",
                "--skeleton-ids",
                "eval_A4_001,eval_B8_001,eval_O4_001",
                "--models",
                args.models or "sft,trial-16",
            ]
        )
    else:
        cmd.extend(["--round", str(args.round)])
        if args.models:
            cmd.extend(["--models", args.models])
    if args.anonymize:
        cmd.append("--anonymize")
        if unblind:
            cmd.extend(["--unblind-output", str(unblind)])
    if getattr(args, "fresh", False):
        cmd.append("--fresh")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
