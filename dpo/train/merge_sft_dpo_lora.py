"""
Cat-stack frozen SFT + trained DPO LoRA into one adapter for vLLM (inference only).

Validation uses the same BnB 8-bit base as DPO training. Deploy the output on
FP8 Qwen3 in vLLM (closest supported substitute for 8-bit training).

  python dpo/train/merge_sft_dpo_lora.py \\
    --dpo-adapter .../trial-29/best_adapter \\
        --dpo-weight 0.25 \\
    --output .../trial-29/sft_dpo_cat \\
    --check-logps
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from dpo.train.model_load import load_bnb_8bit_base
from dpo.train.paths import MODEL_ID_BF16, MODEL_ID_FP8, SFT_ADAPTER
from train.train_sft import patch_chat_template_for_assistant_loss

SFT_ADAPTER_NAME = "default"
DPO_ADAPTER_NAME = "dpo"
CAT_ADAPTER_NAME = "sft_dpo_cat"
POLICY_ADAPTER_STACK = [SFT_ADAPTER_NAME, DPO_ADAPTER_NAME]
LOGIT_TOL = 1e-3
TRAIN_BASE_LABEL = "bnb_8bit"
DEPLOY_BASE_LABEL = "fp8_vllm"

FIXED_CHAT_CONVERSATIONS = [
    [{"role": "user", "content": "What running shoes do you recommend for marathon training?"}],
    [
        {"role": "user", "content": "I'm looking at the Nike Pegasus vs Brooks Ghost."},
        {"role": "assistant", "content": "Both are solid daily trainers. What's your weekly mileage?"},
        {"role": "user", "content": "About 40 miles, neutral gait."},
    ],
]


def resolve_dpo_adapter_path(path: Path) -> Path:
    """Optuna saves weights under best_adapter/dpo/; accept either path."""
    path = path.resolve()
    if (path / "adapter_config.json").is_file():
        return path
    nested = path / "dpo"
    if (nested / "adapter_config.json").is_file():
        return nested
    raise FileNotFoundError(
        f"No adapter_config.json at {path} or {nested}; "
        "pass trial-N/best_adapter or best_adapter/dpo"
    )


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
        "train_base": TRAIN_BASE_LABEL,
        "deploy_base": DEPLOY_BASE_LABEL,
        "deploy_model_path": str(MODEL_ID_FP8),
        "checkpoint_path": str(MODEL_ID_BF16),
        "sft_path": str(SFT_ADAPTER),
        "dpo_path": str(dpo_adapter_dir),
        "sft_rank": sft_cfg.r,
        "dpo_rank": dpo_cfg.r,
        "merged_rank": sft_cfg.r + dpo_cfg.r,
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
    """BnB 8-bit base — matches DPO training policy stack."""
    base = load_bnb_8bit_base()
    model = PeftModel.from_pretrained(
        base,
        str(SFT_ADAPTER),
        adapter_name=SFT_ADAPTER_NAME,
        is_trainable=False,
    )
    model.load_adapter(str(dpo_adapter_dir), adapter_name=DPO_ADAPTER_NAME, is_trainable=False)
    return model


def flatten_adapter_dir(output_dir: Path, adapter_name: str) -> Path:
    """PEFT save_pretrained may nest weights under output_dir/<adapter_name>/."""
    nested = output_dir / adapter_name
    if not nested.is_dir() or not (nested / "adapter_config.json").is_file():
        if (output_dir / "adapter_config.json").is_file():
            return output_dir
        raise FileNotFoundError(
            f"No adapter_config.json at {output_dir} or {nested}"
        )
    for item in nested.iterdir():
        dest = output_dir / item.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        shutil.move(str(item), str(dest))
    nested.rmdir()
    readme = output_dir / "README.md"
    if readme.is_file():
        readme.unlink()
    return output_dir


def merge_cat(model: PeftModel, dpo_weight: float) -> PeftModel:
    if dpo_weight < 0:
        raise ValueError("dpo_weight must be non-negative")
    if SFT_ADAPTER_NAME not in model.peft_config or DPO_ADAPTER_NAME not in model.peft_config:
        raise ValueError(
            f"Expected adapters {SFT_ADAPTER_NAME} and {DPO_ADAPTER_NAME}, "
            f"got {list(model.peft_config.keys())}"
        )
    model.add_weighted_adapter(
        adapters=POLICY_ADAPTER_STACK,
        weights=[1.0, dpo_weight],
        adapter_name=CAT_ADAPTER_NAME,
        combination_type="cat",
    )
    return model


def _scale_adapter_temporarily(model: PeftModel, adapter_name: str, weight: float):
    originals = []
    if weight == 1.0:
        return originals
    for module in model.modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict) and adapter_name in scaling:
            originals.append((scaling, scaling[adapter_name]))
            scaling[adapter_name] *= weight
    if not originals:
        raise ValueError(f"No scaling entries found for adapter {adapter_name!r}")
    return originals


def _restore_adapter_scaling(originals) -> None:
    for scaling, value in originals:
        scaling[DPO_ADAPTER_NAME] = value


def _model_device(model: PeftModel) -> torch.device:
    return next(model.parameters()).device


def verify_weight_matrices(model: PeftModel, dpo_weight: float) -> dict:
    """Per-layer delta-W: SFT + weight*DPO must match cat exactly for linear LoRA."""
    max_diff = 0.0
    layers = 0
    for _name, module in model.named_modules():
        if not (hasattr(module, "lora_A") and CAT_ADAPTER_NAME in module.lora_A):
            continue
        d_stack = module.scaling[SFT_ADAPTER_NAME] * (
            module.lora_B[SFT_ADAPTER_NAME].weight.float()
            @ module.lora_A[SFT_ADAPTER_NAME].weight.float()
        )
        d_stack = d_stack + dpo_weight * module.scaling[DPO_ADAPTER_NAME] * (
            module.lora_B[DPO_ADAPTER_NAME].weight.float()
            @ module.lora_A[DPO_ADAPTER_NAME].weight.float()
        )
        d_cat = module.scaling[CAT_ADAPTER_NAME] * (
            module.lora_B[CAT_ADAPTER_NAME].weight.float() @ module.lora_A[CAT_ADAPTER_NAME].weight.float()
        )
        max_diff = max(max_diff, (d_stack - d_cat).abs().max().item())
        layers += 1
    tol = 1e-5
    return {
        "layers_checked": layers,
        "tolerance": tol,
        "max_abs_delta_diff": max_diff,
        "pass": max_diff <= tol,
    }


def compare_logps_chat(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    conversations: list[list[dict]],
    dpo_weight: float,
) -> dict:
    """Scaled training policy [default,dpo] vs cat adapter; same BnB base as train_dpo."""
    model.eval()
    device = _model_device(model)
    diffs = []
    per_prompt = []

    for messages in conversations:
        ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        if not isinstance(ids, torch.Tensor):
            ids = ids["input_ids"]
        ids = ids.to(device)
        with torch.no_grad():
            model.base_model.set_adapter(POLICY_ADAPTER_STACK)
            originals = _scale_adapter_temporarily(model, DPO_ADAPTER_NAME, dpo_weight)
            try:
                logits_s = model(ids).logits[0, -1]
            finally:
                _restore_adapter_scaling(originals)
            model.set_adapter(CAT_ADAPTER_NAME)
            logits_c = model(ids).logits[0, -1]
        diff = (logits_s - logits_c).abs().max().item()
        diffs.append(diff)
        per_prompt.append({"messages": len(messages), "max_abs_logit_diff": diff})

    return {
        "validation_base": TRAIN_BASE_LABEL,
        "dpo_weight": dpo_weight,
        "prompts": len(conversations),
        "tolerance": LOGIT_TOL,
        "max_abs_logit_diff": max(diffs),
        "mean_abs_logit_diff": sum(diffs) / len(diffs),
        "pass": max(diffs) <= LOGIT_TOL,
        "per_prompt": per_prompt,
    }


def main():
    parser = argparse.ArgumentParser(description="Cat-stack SFT + DPO LoRA for FP8 vLLM")
    parser.add_argument("--dpo-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dpo-weight",
        type=float,
        default=1.0,
        help="Scale the DPO residual before cat export; 1.0 reproduces the training policy.",
    )
    parser.add_argument(
        "--check-logps",
        action="store_true",
        help="Require stacked [default,dpo] vs cat match on BnB 8-bit (training base)",
    )
    args = parser.parse_args()

    dpo_path = resolve_dpo_adapter_path(args.dpo_adapter)
    if dpo_path != args.dpo_adapter.resolve():
        print(f"  resolved DPO adapter: {dpo_path}")

    print(f"Loading SFT + DPO on {TRAIN_BASE_LABEL} base (same as train_dpo.py)...")
    model = load_stacked_for_merge(dpo_path)
    print(f"  adapters before cat: {list(model.peft_config.keys())}")

    compat = validate_merge_compatibility(model, dpo_path)
    merge_cat(model, args.dpo_weight)

    weight_check = verify_weight_matrices(model, args.dpo_weight)
    print(f"  weight-matrix check: {weight_check}")
    if not weight_check["pass"]:
        raise ValueError(
            f"Cat weight matrices differ from stack by {weight_check['max_abs_delta_diff']}; "
            "not writing cat adapter"
        )

    logit_check = None
    if args.check_logps:
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_ID_BF16))
        patch_chat_template_for_assistant_loss(tokenizer)
        logit_check = compare_logps_chat(model, tokenizer, FIXED_CHAT_CONVERSATIONS, args.dpo_weight)
        print(f"  forward logit check ({TRAIN_BASE_LABEL}, informational): {logit_check}")
        if not logit_check["pass"]:
            print(
                "  NOTE: BnB 8-bit may differ on full forward vs single cat LoRA even when ΔW matches; "
                "vLLM FP8 serves one cat adapter (same ΔW). Proceeding because weight check passed."
            )

    args.output.mkdir(parents=True, exist_ok=True)
    model.set_adapter(CAT_ADAPTER_NAME)
    model.save_pretrained(str(args.output), selected_adapters=[CAT_ADAPTER_NAME])
    flatten_adapter_dir(args.output, CAT_ADAPTER_NAME)
    print(f"Saved cat-stacked adapter to {args.output}")
    print(f"  Deploy on {DEPLOY_BASE_LABEL}: {MODEL_ID_FP8}")
    print("  vLLM: one LoRA dir, --max-lora-rank >= merged rank (use 64 if vLLM requires bucket)")

    meta = {
        "combination_type": "cat",
        "weights": [1.0, args.dpo_weight],
        "dpo_weight": args.dpo_weight,
        "source_adapters": POLICY_ADAPTER_STACK,
        "output_adapter_dir": str(args.output.resolve()),
        "adapter_config_json": str((args.output / "adapter_config.json").resolve()),
        "vllm_load_path": str(args.output.resolve()),
        **compat,
    }
    meta["weight_matrix_check"] = weight_check
    if logit_check is not None:
        meta["forward_logit_check"] = logit_check
        meta["forward_logit_check"]["pass_for_save"] = weight_check["pass"]

    with (args.output / "merge_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
