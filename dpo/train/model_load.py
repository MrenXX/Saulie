"""Shared model loading for DPO train and merge (training base = BnB 8-bit)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from dpo.train.paths import MODEL_ID_BF16, MODEL_ID_SFT_MERGED_BF16

BaseKind = Literal["bnb", "bf16"]


def default_device_map() -> str | dict:
    return {"": 0} if torch.cuda.is_available() else "cpu"


def load_bnb_8bit_base(*, device_map: str | dict | None = None) -> AutoModelForCausalLM:
    """
    Same base as DPO training: HF Qwen3-4B checkpoint with BitsAndBytes 8-bit.

    device_map defaults to GPU 0 when CUDA is available, else CPU.
    """
    if device_map is None:
        device_map = default_device_map()
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    return AutoModelForCausalLM.from_pretrained(
        str(MODEL_ID_BF16),
        quantization_config=bnb,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )


def load_base(kind: BaseKind = "bnb", *, device_map: str | dict | None = None) -> AutoModelForCausalLM:
    if kind == "bnb":
        return load_bnb_8bit_base(device_map=device_map)
    if kind == "bf16":
        return load_bf16_base(device_map=device_map)
    raise ValueError(f"unknown base kind: {kind!r}")


def load_bf16_base(
    *,
    device_map: str | dict | None = None,
    model_path: Path | None = None,
) -> AutoModelForCausalLM:
    """
    Full-precision (no LoRA) Qwen3-4B-Instruct checkpoint as bfloat16 on GPU.

    Same weights path as training; not 8-bit quantized.
    model_path overrides MODEL_ID_BF16 (e.g. SFT-merged baked base).
    """
    if device_map is None:
        device_map = default_device_map()
    path = model_path if model_path is not None else MODEL_ID_BF16
    return AutoModelForCausalLM.from_pretrained(
        str(path),
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )


def load_sft_baked_base(
    kind: BaseKind = "bnb",
    *,
    device_map: str | dict | None = None,
    merged_path: Path | None = None,
) -> AutoModelForCausalLM:
    """
    Plan B Control B: dense SFT-merged checkpoint, optionally BnB 8-bit for training parity.

    Requires merge_sft_baked_base.py output at MODEL_ID_SFT_MERGED_BF16 (or merged_path).
    """
    path = merged_path if merged_path is not None else MODEL_ID_SFT_MERGED_BF16
    if not path.is_dir():
        raise FileNotFoundError(
            f"SFT-baked base not found at {path}. "
            "Run: python dpo/train/merge_sft_baked_base.py"
        )
    if device_map is None:
        device_map = default_device_map()
    if kind == "bnb":
        bnb = BitsAndBytesConfig(load_in_8bit=True)
        return AutoModelForCausalLM.from_pretrained(
            str(path),
            quantization_config=bnb,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )
    if kind == "bf16":
        return AutoModelForCausalLM.from_pretrained(
            str(path),
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )
    raise ValueError(f"unknown base kind: {kind!r}")
