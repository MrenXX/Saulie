#!/usr/bin/env python3
"""
Prod ladder eval: base FP8 → SFT trial-17 → prod DPO trial-4.

Deploys each model to vLLM sequentially, generates all 60 skeletons, and
checkpoints after every skeleton so a crash does not restart completed work.

Resume (default):
  python dpo/eval/run_prod_ladder_eval.py

Smoke (3 skeletons):
  python dpo/eval/run_prod_ladder_eval.py --limit-test

Single model retry:
  python dpo/eval/run_prod_ladder_eval.py --models base

Fresh start:
  python dpo/eval/run_prod_ladder_eval.py --fresh
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from dpo.eval.v15_eval_config import (
    EVAL_INFERENCE_SYSTEM_PROMPT,
    PROD_LADDER_MANIFEST_PATH,
    PROD_LADDER_OUTPUT,
    VLLM_API_KEY,
)

GEN_PY = EVAL_DIR / "vllm_scripts" / "eval_generate_vllm.py"
HEALTH_URL = "http://127.0.0.1:8000/health"
LIMIT_SKELETONS = "eval_A4_001,eval_B8_001,eval_O4_001"
MAX_DEPLOY_WAIT_S = 240
MAX_GEN_RETRIES = 3


def load_manifest(path: Path) -> list[dict]:
    entries = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_checkpoint(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def model_complete_in_checkpoint(checkpoint: dict, model_name: str, expected: int) -> bool:
    m = (checkpoint.get("models") or {}).get(model_name)
    if not m:
        return False
    return len(m.get("conversations") or []) >= expected


def wait_for_vllm(timeout_s: int = MAX_DEPLOY_WAIT_S) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if requests.get(HEALTH_URL, timeout=3).ok:
                print(f"[ok] vLLM healthy @ {HEALTH_URL}")
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError(f"vLLM did not become healthy within {timeout_s}s")


def deploy_model(entry: dict) -> None:
    script = REPO_ROOT / entry["deploy_script"]
    if not script.is_file():
        raise FileNotFoundError(f"Deploy script missing: {script}")
    print(f"\n[*] Deploying {entry['manifest_key']} via {script}")
    subprocess.run(["bash", str(script)], check=True)
    wait_for_vllm()


def run_generation(
    *,
    entry: dict,
    output: Path,
    limit_test: bool,
    fresh: bool,
    no_resume: bool,
    models_total: int,
) -> None:
    cmd = [
        sys.executable,
        str(GEN_PY),
        "--candidate-manifest",
        str(PROD_LADDER_MANIFEST_PATH),
        "--output",
        str(output),
        "--models",
        entry["manifest_key"],
        "--skip-runtime-load",
        "--study",
        "prod_ladder_v1",
        "--system-prompt-file",
        str(EVAL_INFERENCE_SYSTEM_PROMPT),
        "--models-total",
        str(models_total),
    ]
    if limit_test:
        cmd.extend(["--skeleton-ids", LIMIT_SKELETONS])
    else:
        cmd.append("--all-skeletons")
    if fresh:
        cmd.append("--fresh")
    if no_resume:
        cmd.append("--no-resume")

    print("[*] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def expected_skeleton_count(limit_test: bool) -> int:
    return 3 if limit_test else 60


def main() -> None:
    parser = argparse.ArgumentParser(description="Prod ladder eval orchestrator")
    parser.add_argument("--limit-test", action="store_true")
    parser.add_argument("--models", type=str, default=None, help="base,sft,prod")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--skip-deploy",
        action="store_true",
        help="Assume vLLM already serves the target model",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROD_LADDER_OUTPUT,
    )
    args = parser.parse_args()

    manifest = load_manifest(PROD_LADDER_MANIFEST_PATH)
    if args.models:
        keys = {k.strip() for k in args.models.split(",") if k.strip()}
        manifest = [e for e in manifest if e.get("manifest_key") in keys]
        if not manifest:
            raise SystemExit(f"No manifest entries for keys: {keys}")

    expected = expected_skeleton_count(args.limit_test)
    checkpoint = None if args.fresh else load_checkpoint(args.output)
    full_manifest = load_manifest(PROD_LADDER_MANIFEST_PATH)
    models_total = len(full_manifest)
    fresh_once = args.fresh

    for entry in manifest:
        api_name = entry["model_name"]
        if checkpoint and model_complete_in_checkpoint(checkpoint, api_name, expected):
            print(f"[skip] {entry['manifest_key']} already complete in {args.output}")
            continue

        attempt = 0
        while attempt < MAX_GEN_RETRIES:
            attempt += 1
            try:
                if not args.skip_deploy:
                    deploy_model(entry)
                run_generation(
                    entry=entry,
                    output=args.output,
                    limit_test=args.limit_test,
                    fresh=fresh_once,
                    no_resume=args.no_resume,
                    models_total=models_total,
                )
                fresh_once = False
                checkpoint = load_checkpoint(args.output)
                if checkpoint and model_complete_in_checkpoint(checkpoint, api_name, expected):
                    print(f"[ok] {entry['manifest_key']} complete")
                    break
                raise RuntimeError(f"{entry['manifest_key']} finished but checkpoint incomplete")
            except (subprocess.CalledProcessError, RuntimeError) as exc:
                print(f"[!] {entry['manifest_key']} attempt {attempt}/{MAX_GEN_RETRIES} failed: {exc}")
                if attempt >= MAX_GEN_RETRIES:
                    raise
                print("[*] Retrying after brief pause (checkpoint preserved)...")
                time.sleep(5)
                checkpoint = load_checkpoint(args.output)

    final = load_checkpoint(args.output)
    if final:
        n_done = final.get("models_complete", 0)
        n_total = final.get("models_total") or len(manifest)
        complete = final.get("checkpoint_complete", False)
        print(f"\n[done] {args.output}")
        print(f"       models_complete={n_done}/{n_total} checkpoint_complete={complete}")


if __name__ == "__main__":
    main()
