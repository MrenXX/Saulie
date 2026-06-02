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
MODEL_ID_SFT_MERGED_BF16 = REPO_ROOT / "Qwen3-4B-Instruct-2507-SFT-MERGED-BF16"
SFT_ADAPTER = REPO_ROOT / "train/models/steering-sft-v1.1/trial-17/best_adapter"

EXPERIMENT_NAME = "steering-dpo-v1.0"
EXPERIMENT_NAME_V1_1 = "steering-dpo-v1.1"
EXPERIMENT_NAME_V1_2 = "steering-dpo-v1.2"
# Plan B: DPO on dense SFT-merged base (not raw base + frozen SFT LoRA)
EXPERIMENT_NAME_V1_3 = "steering-dpo-v1.3-sft-merged"
# Plan B configs replicated on raw BnB base + frozen SFT LoRA + trainable DPO LoRA
EXPERIMENT_NAME_V1_4 = "steering-dpo-v1.4"
EXPERIMENT_NAME_V1_5 = "steering-dpo-v1.5"
STUDY_RESULTS_DIR = REPO_ROOT / "dpo" / "study_results"


def trial_summary_filename(study_version: str) -> str:
    """Versioned trial summary basename, e.g. trial_summary_v1.5.json."""
    return f"trial_summary_{study_version}.json"


def trial_summary_path(run_dir: Path, study_version: str) -> Path:
    return run_dir / trial_summary_filename(study_version)


def experiment_name_for_version(study_version: str) -> str:
    if study_version == "v1.5":
        return EXPERIMENT_NAME_V1_5
    if study_version == "v1.4":
        return EXPERIMENT_NAME_V1_4
    if study_version == "v1.3":
        return EXPERIMENT_NAME_V1_3
    if study_version == "v1.2":
        return EXPERIMENT_NAME_V1_2
    if study_version == "v1.1":
        return EXPERIMENT_NAME_V1_1
    return EXPERIMENT_NAME


def output_experiment_dir(study_version: str) -> Path:
    return OUTPUT_BASE / experiment_name_for_version(study_version)
