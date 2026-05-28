"""Shared model loading for DPO train and merge (training base = BnB 8-bit)."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from dpo.train.paths import MODEL_ID_BF16


def load_bnb_8bit_base(*, device_map: str | dict | None = None) -> AutoModelForCausalLM:
    """
    Same base as DPO training: HF Qwen3-4B checkpoint with BitsAndBytes 8-bit.

    device_map defaults to GPU 0 when CUDA is available, else CPU.
    """
    if device_map is None:
        device_map = {"": 0} if torch.cuda.is_available() else "cpu"
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    return AutoModelForCausalLM.from_pretrained(
        str(MODEL_ID_BF16),
        quantization_config=bnb,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )
