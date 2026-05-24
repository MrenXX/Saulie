"""Paths for DPO training under dpo/train/."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DPO_TRAIN_DIR = Path(__file__).resolve().parent

DATA_PATH = REPO_ROOT / "dpo/dataset/DPO_522_prompt_a_and_prompt_b_V4_repaired.jsonl"
VALIDATION_REPORT = REPO_ROOT / "dpo/dataset/DPO_522_prompt_a_and_prompt_b_V4_validation_report.json"
SPLIT_DIR = DPO_TRAIN_DIR / "dataset"
CACHE_DIR = DPO_TRAIN_DIR / "cache" / "dpo_v4"
OUTPUT_BASE = DPO_TRAIN_DIR / "models"
MLRUNS_DIR = DPO_TRAIN_DIR / "mlruns"

MODEL_ID_BF16 = REPO_ROOT / "Qwen3-4B-Instruct-2507"
MODEL_ID_FP8 = REPO_ROOT / "Qwen3-4B-Instruct-2507-FP8"
SFT_ADAPTER = REPO_ROOT / "train/models/steering-sft-v1.1/trial-17/best_adapter"

EXPERIMENT_NAME = "steering-dpo-v1.0"
