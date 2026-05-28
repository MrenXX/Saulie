"""
Smoke-generate with the exact DPO training policy forward (no cat merge).

  HF Qwen3-4B + BnB 8-bit + frozen SFT (default) + trained DPO (dpo)
  model.base_model.set_adapter(["default", "dpo"])

Example:
  python dpo/train/smoke_policy_stack_hf.py \\
    --dpo-adapter dpo/train/models/.../trial-29/best_adapter \\
        --dpo-weight 0.25 \\
        --decode greedy \\
    --skeleton-ids eval_A4_001,eval_B8_001 \\
    --output dpo/eval/dpo_phase1_policy_stack_hf_smoke.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from peft import PeftModel
from transformers import AutoTokenizer

from dpo.train.dpo_trainer_compat import (
    DPO_ADAPTER_NAME,
    POLICY_ADAPTER_STACK,
    SFT_ADAPTER_NAME,
    collect_adapter_diagnostics,
    ensure_policy_adapter_stack,
)
from dpo.train.merge_sft_dpo_lora import load_stacked_for_merge, resolve_dpo_adapter_path
from dpo.train.paths import MODEL_ID_BF16, SFT_ADAPTER
from dpo.train.train_dpo import load_tokenizer

MAX_TOKENS = 350
TEMPERATURE = 0.7
TOP_P = 0.8
ADAPTER_MODES = ("policy", "sft", "dpo")
DECODE_MODES = ("sample", "greedy")
DEFAULT_SKELETONS = REPO_ROOT / "dpo/eval/eval_skeletons.json"
DEFAULT_DPO = (
    REPO_ROOT
    / "dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-29/best_adapter"
)


def _device(model: PeftModel) -> torch.device:
    return next(model.parameters()).device


def scale_adapter_residual(model: PeftModel, adapter_name: str, weight: float) -> int:
    """Scale one LoRA adapter's runtime residual by multiplying PEFT scaling values."""
    if weight < 0:
        raise ValueError("adapter weight must be non-negative")
    changed = 0
    if weight == 1.0:
        return changed
    for module in model.modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict) and adapter_name in scaling:
            scaling[adapter_name] *= weight
            changed += 1
    if changed == 0:
        raise ValueError(f"No scaling entries found for adapter {adapter_name!r}")
    return changed


def activate_adapter_mode(model: PeftModel, mode: str) -> None:
    if mode == "policy":
        ensure_policy_adapter_stack(model)
    elif mode == "sft":
        model.set_adapter(SFT_ADAPTER_NAME)
    elif mode == "dpo":
        model.set_adapter(DPO_ADAPTER_NAME)
    else:
        raise ValueError(f"unknown adapter mode: {mode}")


@torch.inference_mode()
def generate_turn(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    messages: list[dict],
    *,
    adapter_mode: str,
    decode: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    activate_adapter_mode(model, adapter_mode)
    ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
    ids = ids.to(_device(model))
    prompt_len = ids.shape[1]
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if decode == "sample":
        generate_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
    else:
        generate_kwargs.update({"do_sample": False})
    out = model.generate(ids, **generate_kwargs)
    new_ids = out[0, prompt_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def run_skeleton(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    skeleton: dict,
    *,
    adapter_mode: str,
    decode: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[dict]:
    messages: list[dict] = []
    for i, user_msg in enumerate(skeleton["user_turns"]):
        messages.append({"role": "user", "content": user_msg})
        text = generate_turn(
            model,
            tokenizer,
            messages,
            adapter_mode=adapter_mode,
            decode=decode,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        messages.append({"role": "assistant", "content": text})
        status = "FINAL" if i == len(skeleton["user_turns"]) - 1 else "intermediate"
        print(f"    Turn {i + 1}/{len(skeleton['user_turns'])} [{status}]: {len(text)} chars")
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description="HF BnB policy-stack smoke generation")
    parser.add_argument("--dpo-adapter", type=Path, default=DEFAULT_DPO)
    parser.add_argument("--skeletons", type=Path, default=DEFAULT_SKELETONS)
    parser.add_argument("--skeleton-ids", type=str, default="eval_A4_001,eval_B8_001")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "dpo/eval/dpo_phase1_policy_stack_hf_smoke.json")
    parser.add_argument("--adapter-mode", choices=ADAPTER_MODES, default="policy")
    parser.add_argument("--dpo-weight", type=float, default=1.0)
    parser.add_argument("--decode", choices=DECODE_MODES, default="sample")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    args = parser.parse_args()

    dpo_path = resolve_dpo_adapter_path(args.dpo_adapter)
    wanted = {s.strip() for s in args.skeleton_ids.split(",")}
    with args.skeletons.open(encoding="utf-8") as f:
        skeletons = [s for s in json.load(f) if s["id"] in wanted]
    if not skeletons:
        raise SystemExit(f"No skeletons matched {wanted}")

    print("Loading tokenizer (train_dpo.load_tokenizer)...")
    tokenizer = load_tokenizer()

    print(f"Loading BnB stack: SFT={SFT_ADAPTER}")
    print(f"  DPO adapter: {dpo_path}")
    t0 = time.time()
    model = load_stacked_for_merge(dpo_path)
    load_s = time.time() - t0
    scaled_modules = 0
    if args.adapter_mode in {"policy", "dpo"}:
        scaled_modules = scale_adapter_residual(model, DPO_ADAPTER_NAME, args.dpo_weight)
    activate_adapter_mode(model, args.adapter_mode)
    model.eval()

    diag = collect_adapter_diagnostics(model)
    print(f"  load time: {load_s:.1f}s")
    print(f"  active adapters: {diag.get('active_adapters')}")
    print(f"  adapter mode: {args.adapter_mode}")
    print(f"  dpo weight: {args.dpo_weight} ({scaled_modules} modules scaled)")

    conversations = []
    for sk in skeletons:
        print(f"\n  Skeleton {sk['id']} (type={sk['opening_type']})")
        messages = run_skeleton(
            model,
            tokenizer,
            sk,
            adapter_mode=args.adapter_mode,
            decode=args.decode,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        conversations.append(
            {
                "skeleton_id": sk["id"],
                "opening_type": sk["opening_type"],
                "target_turns": sk["target_turns"],
                "actual_turns": len(sk["user_turns"]) * 2,
                "messages": messages,
            }
        )

    payload = {
        "generated_at": datetime.now().isoformat(),
        "policy": f"hf_bnb_{args.adapter_mode}",
        "description": "BnB 8-bit base with selectable PEFT adapter mode; policy means set_adapter([default, dpo])",
        "base_model": str(MODEL_ID_BF16),
        "sft_adapter": str(SFT_ADAPTER),
        "dpo_adapter": str(dpo_path),
        "adapter_mode": args.adapter_mode,
        "active_adapters": diag.get("active_adapters"),
        "policy_stack": POLICY_ADAPTER_STACK,
        "dpo_weight": args.dpo_weight,
        "dpo_scaled_modules": scaled_modules,
        "adapter_diagnostics": diag,
        "generation": {
            "decode": args.decode,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature if args.decode == "sample" else None,
            "top_p": args.top_p if args.decode == "sample" else None,
        },
        "skeletons_path": str(args.skeletons),
        "conversations": conversations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
