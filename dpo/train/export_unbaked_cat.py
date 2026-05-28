"""
Export cat-stacked LoRA without scaling baked into A; lora_alpha = sft_alpha + dpo_alpha.

PEFT cat merge bakes scaling into A and sets lora_alpha=r (48). Some runtimes may
expect unbaked A/B with lora_alpha = sum(alphas) so effective scale = alpha/r = 2.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel
from peft.tuners.lora.layer import LoraLayer
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dpo.train.model_load import load_bnb_8bit_base
from dpo.train.merge_sft_dpo_lora import (
    CAT_ADAPTER_NAME,
    DPO_ADAPTER_NAME,
    POLICY_ADAPTER_STACK,
    SFT_ADAPTER_NAME,
    flatten_adapter_dir,
    resolve_dpo_adapter_path,
)
from dpo.train.paths import SFT_ADAPTER

SFT_ALPHA = 32
DPO_ALPHA = 64
MERGED_R = 16 + 32
MERGED_ALPHA = SFT_ALPHA + DPO_ALPHA  # 96 -> scale 2.0 at r=48


def export_unbaked_cat(model: PeftModel, output_dir: Path) -> dict:
    """Build sft_dpo_cat weights by unscaled concat; write adapter_config for vLLM."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tensors = {}
    layers = 0

    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        if not all(ad in module.lora_A for ad in POLICY_ADAPTER_STACK):
            continue

        a_parts = []
        b_parts = []
        for ad in POLICY_ADAPTER_STACK:
            a_parts.append(module.lora_A[ad].weight.detach().float().cpu())
            b_parts.append(module.lora_B[ad].weight.detach().float().cpu())

        a_cat = torch.cat(a_parts, dim=0)
        b_cat = torch.cat(b_parts, dim=1)

        key_a = name.replace(".base_layer", "") + ".lora_A.weight"
        key_b = name.replace(".base_layer", "") + ".lora_B.weight"
        tensors[key_a] = a_cat.to(torch.bfloat16)
        tensors[key_b] = b_cat.to(torch.bfloat16)
        layers += 1

    save_file(tensors, output_dir / "adapter_model.safetensors")

    cfg = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": MERGED_R,
        "lora_alpha": MERGED_ALPHA,
        "lora_dropout": 0.0,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "bias": "none",
        "base_model_name_or_path": str(REPO_ROOT / "Qwen3-4B-Instruct-2507"),
        "inference_mode": True,
        "init_lora_weights": True,
    }
    (output_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    return {
        "layers": layers,
        "merged_r": MERGED_R,
        "merged_alpha": MERGED_ALPHA,
        "effective_scale": MERGED_ALPHA / MERGED_R,
        "method": "unbaked_concat",
    }


def main():
    dpo_path = resolve_dpo_adapter_path(
        REPO_ROOT
        / "dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-29/best_adapter"
    )
    out = (
        REPO_ROOT
        / "dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-29/sft_dpo_cat_unbaked"
    )

    print("Loading BnB stack for unbaked export...")
    base = load_bnb_8bit_base()
    model = PeftModel.from_pretrained(
        base, str(SFT_ADAPTER), adapter_name=SFT_ADAPTER_NAME, is_trainable=False
    )
    model.load_adapter(str(dpo_path), adapter_name=DPO_ADAPTER_NAME, is_trainable=False)

    meta = export_unbaked_cat(model, out)
    print(f"Wrote unbaked cat to {out}")
    print(meta)

    meta_path = out / "merge_meta.json"
    meta_path.write_text(
        json.dumps({"export": meta, "note": "unbaked A/B, lora_alpha=96, r=48"}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
