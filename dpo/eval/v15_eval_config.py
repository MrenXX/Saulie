"""Shared constants for v1.5 DPO final eval (vLLM FP8 + cat LoRA)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

V15_RUN = REPO_ROOT / "dpo/train/models/steering-dpo-v1.5/optuna-run-20260602-052732"
SFT_ADAPTER_HOST = REPO_ROOT / "sft/models/steering-sft-v1.1/trial-17/best_adapter"
DEFAULT_SKELETONS = REPO_ROOT / "dpo/eval/eval_skeletons.json"
MANIFEST_PATH = REPO_ROOT / "dpo/eval/v15_final_eval_manifest.jsonl"
PROD_LADDER_MANIFEST_PATH = REPO_ROOT / "dpo/eval/prod_ladder_manifest.jsonl"
PROD_LADDER_OUTPUT = REPO_ROOT / "dpo/eval/generations_prod_ladder.json"
EVAL_INFERENCE_SYSTEM_PROMPT = REPO_ROOT / "dpo/eval/eval_inference_system_prompt.md"
STARTUP_MANIFEST_PATH = REPO_ROOT / "dpo/eval/v15_deploy_startup_manifest.jsonl"

VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_API_KEY = os.getenv("VLLM_API_KEY")
if not VLLM_API_KEY:
    raise RuntimeError("VLLM_API_KEY not set — add it to .env (see .env.example)")
VLLM_LOAD_LORA_URL = "http://localhost:8000/v1/load_lora_adapter"
VLLM_UNLOAD_LORA_URL = "http://localhost:8000/v1/unload_lora_adapter"

# Container bind-mount root (see deploy_qwenie_eval.sh)
LORA_HOST_MOUNT = "/models/lora-host"

# Cat-merge: one adapter with r = r_sft + r_dpo (e.g. 16+16=32 or 16+24=40 on trials 8/27).
# vLLM only allows max_lora_rank in {8,16,32,64,...} — use 64 to cover slate max r=40.
SFT_LORA_R = 16
CAT_MERGED_LORA_R_MAX = 40
VLLM_MAX_LORA_RANK = 64

# Locked eval sampling (DPO_FINAL_EVAL_EXECUTION_PLAN + chat_policy_stack)
EVAL_MAX_TOKENS = 256
EVAL_TEMPERATURE = 0.7
EVAL_TOP_P = 0.8
EVAL_TOP_K = 20
EVAL_REPETITION_PENALTY = 1.05

BASELINE_MODEL_NAME = "steering-sft-v1.1_trial-17"
FINALIST_TRIALS = (19, 16, 8, 27, 20, 4)
CANDIDATE_LETTERS = ("A", "B", "C", "D", "E", "F")

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


def vllm_extra_body() -> dict:
    """vLLM-only sampling params (not in standard OpenAI client kwargs)."""
    return {
        "top_k": EVAL_TOP_K,
        "repetition_penalty": EVAL_REPETITION_PENALTY,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def generation_metadata() -> dict:
    return {
        "backend": "vllm_fp8",
        "temperature": EVAL_TEMPERATURE,
        "top_p": EVAL_TOP_P,
        "top_k": EVAL_TOP_K,
        "repetition_penalty": EVAL_REPETITION_PENALTY,
        "max_tokens": EVAL_MAX_TOKENS,
        "base_url": VLLM_BASE_URL,
        "enable_thinking": False,
        "note": "top_k, repetition_penalty, chat_template_kwargs via OpenAI extra_body for vLLM",
    }


def build_manifest_entries() -> list[dict]:
    entries: list[dict] = [
        {
            "model_name": BASELINE_MODEL_NAME,
            "judge_id": None,
            "kind": "sft_baseline",
            "trial_number": 17,
            "lora_rank": SFT_LORA_R,
            "adapter_path": str(SFT_ADAPTER_HOST.resolve()),
            "container_path": f"{LORA_HOST_MOUNT}/sft17",
            "preload_at_startup": True,
            "runtime_load": False,
        }
    ]
    for i, trial in enumerate(FINALIST_TRIALS):
        host = (V15_RUN / f"trial-{trial}" / "sft_dpo_cat").resolve()
        judge_id = f"candidate_{CANDIDATE_LETTERS[i]}"
        cfg = json.loads((host / "adapter_config.json").read_text(encoding="utf-8"))
        entries.append(
            {
                "model_name": judge_id,
                "judge_id": judge_id,
                "kind": "dpo_merged",
                "trial_number": trial,
                "lora_rank": cfg["r"],
                "adapter_path": str(host),
                "container_path": f"{LORA_HOST_MOUNT}/v15-run/trial-{trial}/sft_dpo_cat",
                "preload_at_startup": False,
                "runtime_load": True,
            }
        )
    return entries
