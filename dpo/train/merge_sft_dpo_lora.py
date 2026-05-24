"""
Cat-merge frozen SFT + trained DPO LoRA into one adapter for vLLM (inference only).

Training keeps adapters separate; run this after Optuna/selection:

  python dpo/train/merge_sft_dpo_lora.py \\
    --dpo-adapter dpo/train/models/steering-dpo-v1.0/dummy-run/best_adapter \\
    --output dpo/train/models/steering-dpo-v1.0/dummy-run/sft_dpo_cat \\
    --check-logps
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from dpo.train.paths import MODEL_ID_BF16, SFT_ADAPTER
from train.train_sft import patch_chat_template_for_assistant_loss

SFT_ADAPTER_NAME = "default"
DPO_ADAPTER_NAME = "dpo"
CAT_ADAPTER_NAME = "sft_dpo_cat"
LOGIT_TOL = 1e-3

FIXED_CHAT_CONVERSATIONS = [
    [{"role": "user", "content": "What running shoes do you recommend for marathon training?"}],
    [
        {"role": "user", "content": "I'm looking at the Nike Pegasus vs Brooks Ghost."},
        {"role": "assistant", "content": "Both are solid daily trainers. What's your weekly mileage?"},
        {"role": "user", "content": "About 40 miles, neutral gait."},
    ],
]


def validate_merge_compatibility(model: PeftModel, dpo_adapter_dir: Path) -> dict:
    """Pre-merge checks: lineage, targets, ranks, no modules_to_save conflict."""
    sft_cfg = model.peft_config[SFT_ADAPTER_NAME]
    dpo_cfg = model.peft_config[DPO_ADAPTER_NAME]
    issues: list[str] = []

    if sft_cfg.peft_type != dpo_cfg.peft_type:
        issues.append(f"peft_type mismatch: {sft_cfg.peft_type} vs {dpo_cfg.peft_type}")
    if set(sft_cfg.target_modules) != set(dpo_cfg.target_modules):
        issues.append(
            f"target_modules mismatch: {sft_cfg.target_modules} vs {dpo_cfg.target_modules}"
        )
    sft_save = getattr(sft_cfg, "modules_to_save", None) or []
    dpo_save = getattr(dpo_cfg, "modules_to_save", None) or []
    if sft_save or dpo_save:
        issues.append(f"modules_to_save present sft={sft_save} dpo={dpo_save}")

    dpo_config_path = dpo_adapter_dir / "adapter_config.json"
    meta = {
        "sft_path": str(SFT_ADAPTER),
        "dpo_path": str(dpo_adapter_dir),
        "sft_rank": sft_cfg.r,
        "dpo_rank": dpo_cfg.r,
        "sft_alpha": sft_cfg.lora_alpha,
        "dpo_alpha": dpo_cfg.lora_alpha,
        "target_modules": list(sft_cfg.target_modules),
        "peft_type": sft_cfg.peft_type.value if hasattr(sft_cfg.peft_type, "value") else str(sft_cfg.peft_type),
        "dpo_adapter_config_exists": dpo_config_path.exists(),
        "issues": issues,
        "pass": len(issues) == 0,
    }
    if issues:
        raise ValueError("Merge compatibility failed: " + "; ".join(issues))
    return meta


def load_stacked_for_merge(dpo_adapter_dir: Path) -> PeftModel:
    base = AutoModelForCausalLM.from_pretrained(
        str(MODEL_ID_BF16),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(
        base,
        str(SFT_ADAPTER),
        adapter_name=SFT_ADAPTER_NAME,
        is_trainable=False,
    )
    model.load_adapter(str(dpo_adapter_dir), adapter_name=DPO_ADAPTER_NAME, is_trainable=False)
    return model


def merge_cat(model: PeftModel) -> PeftModel:
    if SFT_ADAPTER_NAME not in model.peft_config or DPO_ADAPTER_NAME not in model.peft_config:
        raise ValueError(
            f"Expected adapters {SFT_ADAPTER_NAME} and {DPO_ADAPTER_NAME}, "
            f"got {list(model.peft_config.keys())}"
        )
    model.add_weighted_adapter(
        adapters=[SFT_ADAPTER_NAME, DPO_ADAPTER_NAME],
        adapter_name=CAT_ADAPTER_NAME,
        weights=[1.0, 1.0],
        combination_type="cat",
    )
    model.set_adapter(CAT_ADAPTER_NAME)
    return model


def compare_logps_chat(
    stacked: PeftModel,
    cat: PeftModel,
    tokenizer: AutoTokenizer,
    conversations: list[list[dict]],
) -> dict:
    """Next-token logit check via apply_chat_template (Qwen chat format)."""
    stacked.eval()
    cat.eval()
    diffs = []
    per_prompt = []

    for messages in conversations:
        ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            stacked.base_model.set_adapter([SFT_ADAPTER_NAME, DPO_ADAPTER_NAME])
            logits_s = stacked(ids).logits[0, -1]
            cat.set_adapter(CAT_ADAPTER_NAME)
            logits_c = cat(ids).logits[0, -1]
        diff = (logits_s - logits_c).abs().max().item()
        diffs.append(diff)
        per_prompt.append({"messages": len(messages), "max_abs_logit_diff": diff})

    return {
        "prompts": len(conversations),
        "tolerance": LOGIT_TOL,
        "max_abs_logit_diff": max(diffs),
        "mean_abs_logit_diff": sum(diffs) / len(diffs),
        "pass": max(diffs) <= LOGIT_TOL,
        "per_prompt": per_prompt,
    }


def main():
    parser = argparse.ArgumentParser(description="Cat-merge SFT + DPO LoRA for vLLM")
    parser.add_argument("--dpo-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-logps", action="store_true", help="Run stacked vs cat logit check")
    args = parser.parse_args()

    print("Loading SFT + DPO adapters on CPU base (no BnB merge)...")
    model = load_stacked_for_merge(args.dpo_adapter)
    print(f"  adapters before cat: {list(model.peft_config.keys())}")

    compat = validate_merge_compatibility(model, args.dpo_adapter)
    merged = merge_cat(model)
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.output), selected_adapters=[CAT_ADAPTER_NAME])
    print(f"Saved cat adapter to {args.output}")
    print("  vLLM: load this directory as the single LoRA adapter (--max-lora-rank = r_sft + r_dpo)")

    meta = {
        "combination_type": "cat",
        "weights": [1.0, 1.0],
        "source_adapters": [SFT_ADAPTER_NAME, DPO_ADAPTER_NAME],
        "output_adapter_dir": str(args.output.resolve()),
        "adapter_config_json": str((args.output / "adapter_config.json").resolve()),
        "vllm_load_path": str(args.output.resolve()),
        **compat,
    }
    if args.check_logps:
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_ID_BF16))
        patch_chat_template_for_assistant_loss(tokenizer)
        meta["logit_check"] = compare_logps_chat(
            model, merged, tokenizer, FIXED_CHAT_CONVERSATIONS
        )
        print(f"  logit check: {meta['logit_check']}")
        if not meta["logit_check"]["pass"]:
            raise ValueError(
                f"Logit diff {meta['logit_check']['max_abs_logit_diff']:.6f} > tol {LOGIT_TOL}"
            )

    with (args.output / "merge_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
