"""
Cat-stack frozen SFT + trained DPO LoRA into one adapter for vLLM (inference only).

Export is gated only by fp32 Delta-W correctness. Forward drift and generation smoke
are non-blocking diagnostics (see dpo/eval/MERGE_SCRIPT_VALIDATION_FIX_PLAN.md).

  python dpo/train/merge_sft_dpo_lora.py \\
    --dpo-adapter .../trial-N/best_adapter \\
    --output .../trial-N/sft_dpo_cat \\
    --audit-forward-drift --generation-smoke
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from dpo.train.dpo_trainer_compat import ensure_policy_adapter_stack
from dpo.train.model_load import BaseKind, load_base, load_sft_baked_base
from dpo.train.paths import MODEL_ID_BF16, MODEL_ID_FP8, MODEL_ID_SFT_MERGED_BF16, SFT_ADAPTER
from dpo.train.qwen3_decode import (
    DECODE_SAMPLE,
    REPETITION_PENALTY_DEFAULT,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    build_generate_kwargs,
)
from train.train_sft import patch_chat_template_for_assistant_loss

SFT_ADAPTER_NAME = "default"
DPO_ADAPTER_NAME = "dpo"
CAT_ADAPTER_NAME = "sft_dpo_cat"
POLICY_ADAPTER_STACK = [SFT_ADAPTER_NAME, DPO_ADAPTER_NAME]
DELTA_W_TOL = 1e-5
LEGACY_LOGIT_TOL = 1e-3
TRAIN_BASE_LABEL = "bnb_8bit"
DEPLOY_BASE_LABEL = "fp8_vllm"

# Fixed 7-prompt smoke set (all trials use the same prompts for comparability).
MERGE_SMOKE_PROMPTS: list[dict] = [
    {
        "id": "chat_01",
        "category": "normal_english_chat",
        "messages": [
            {
                "role": "user",
                "content": "What running shoes do you recommend for marathon training?",
            }
        ],
    },
    {
        "id": "chat_02",
        "category": "normal_english_chat",
        "messages": [
            {
                "role": "user",
                "content": "Give me three bullet tips for recovering after a long run.",
            }
        ],
    },
    {
        "id": "instr_01",
        "category": "short_instruction",
        "messages": [
            {
                "role": "user",
                "content": "Rewrite this sentence to be shorter: 'I am planning to go for a run tomorrow morning if the weather is good.'",
            }
        ],
    },
    {
        "id": "task_01",
        "category": "task_style",
        "messages": [
            {
                "role": "user",
                "content": "I'm choosing between the Nike Pegasus and Brooks Ghost for 40 miles per week with a neutral gait. Which would you pick and why?",
            }
        ],
    },
    {
        "id": "style_01",
        "category": "style_sensitive",
        "messages": [
            {
                "role": "user",
                "content": "Answer in two short paragraphs with no bullet points: how should a beginner build up to a half marathon?",
            }
        ],
    },
    {
        "id": "collapse_01",
        "category": "collapse_probe",
        "messages": [
            {
                "role": "user",
                "content": "Say hello in one sentence, then stop.",
            }
        ],
    },
    {
        "id": "long_01",
        "category": "longish_prompt",
        "messages": [
            {
                "role": "user",
                "content": "I'm training for my first marathon in 16 weeks. I can run 25 miles per week now, mostly easy pace. Outline a simple week-by-week progression for the next four weeks only, keeping it practical.",
            }
        ],
    },
]

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


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


def load_prompt_set(prompts_path: Path | None) -> list[dict]:
    if prompts_path is None:
        return MERGE_SMOKE_PROMPTS
    rows = []
    for line in prompts_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "messages" not in row:
            raise ValueError(f"prompt row missing messages: {row}")
        rows.append(row)
    return rows


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

    if issues:
        raise ValueError("Merge compatibility failed: " + "; ".join(issues))

    return {
        "sft_rank": sft_cfg.r,
        "dpo_rank": dpo_cfg.r,
        "merged_rank": sft_cfg.r + dpo_cfg.r,
        "sft_alpha": sft_cfg.lora_alpha,
        "dpo_alpha": dpo_cfg.lora_alpha,
        "target_modules": list(sft_cfg.target_modules),
        "peft_type": sft_cfg.peft_type.value
        if hasattr(sft_cfg.peft_type, "value")
        else str(sft_cfg.peft_type),
        "dpo_path": str(dpo_adapter_dir.resolve()),
    }


def load_sft_stack(*, base: BaseKind = "bnb") -> PeftModel:
    """Frozen SFT trial-17 on selectable base (default BnB 8-bit = DPO training)."""
    model = PeftModel.from_pretrained(
        load_base(base),
        str(SFT_ADAPTER),
        adapter_name=SFT_ADAPTER_NAME,
        is_trainable=False,
    )
    return model


def load_stacked_for_merge(dpo_adapter_dir: Path, *, base: BaseKind = "bnb") -> PeftModel:
    """SFT + DPO policy stack on selectable base (default BnB 8-bit = DPO training)."""
    model = load_sft_stack(base=base)
    model.load_adapter(str(dpo_adapter_dir), adapter_name=DPO_ADAPTER_NAME, is_trainable=False)
    return model


def load_cat_merged_adapter(cat_adapter_dir: Path, *, base: BaseKind = "bnb") -> PeftModel:
    """Single cat-merged SFT+DPO adapter on base (HF REPL / vLLM export)."""
    cat_dir = cat_adapter_dir.resolve()
    if not (cat_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"No adapter_config.json in {cat_dir}; run merge_sft_dpo_lora.py first"
        )
    meta_path = cat_dir / "merge_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        mc = meta.get("merge_correctness") or meta.get("weight_matrix_check") or {}
        print(
            f"  merge_meta: pass={mc.get('pass')} max_diff={mc.get('max_abs_delta_diff')}"
        )
        if mc and not mc.get("pass"):
            print("  WARNING: merge_meta reports failed Delta-W check")
    model = PeftModel.from_pretrained(
        load_base(base),
        str(cat_dir),
        is_trainable=False,
    )
    if base == "fp8":
        _harmonize_lora_dtypes(model, torch.bfloat16)
    return model


def _harmonize_lora_dtypes(model: PeftModel, dtype: torch.dtype) -> None:
    for param in model.parameters():
        if param.dtype in (torch.float16, torch.bfloat16) and param.dtype != dtype:
            param.data = param.data.to(dtype)


def load_baked_dpo_stack(
    dpo_adapter_dir: Path,
    *,
    base: BaseKind = "bnb",
    merged_path: Path | None = None,
) -> PeftModel:
    """Plan B inference: BnB(SFT-merged base) + trained DPO adapter only."""
    dpo_path = resolve_dpo_adapter_path(dpo_adapter_dir)
    path = merged_path if merged_path is not None else MODEL_ID_SFT_MERGED_BF16
    model = PeftModel.from_pretrained(
        load_sft_baked_base(base, merged_path=path),
        str(dpo_path),
        adapter_name=DPO_ADAPTER_NAME,
        is_trainable=False,
    )
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


def _set_cat_adapter(model: PeftModel) -> None:
    """Cat export may register as default on reload; stacked merge uses sft_dpo_cat."""
    names = list(model.peft_config.keys())
    if CAT_ADAPTER_NAME in names:
        model.set_adapter(CAT_ADAPTER_NAME)
    elif len(names) == 1:
        model.set_adapter(names[0])
    else:
        raise ValueError(
            f"Cannot select cat adapter among {names!r}; expected {CAT_ADAPTER_NAME!r}"
        )


def verify_weight_matrices(model: PeftModel, dpo_weight: float) -> dict:
    """Hard gate: fp32 raw LoRA Delta-W reconstruction must match cat."""
    max_diff = 0.0
    worst_module: str | None = None
    layers = 0
    for name, module in model.named_modules():
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
            module.lora_B[CAT_ADAPTER_NAME].weight.float()
            @ module.lora_A[CAT_ADAPTER_NAME].weight.float()
        )
        layer_diff = (d_stack - d_cat).abs().max().item()
        if layer_diff > max_diff:
            max_diff = layer_diff
            worst_module = name
        layers += 1
    return {
        "blocking": True,
        "method": "fp32 raw LoRA Delta-W reconstruction",
        "pass": max_diff <= DELTA_W_TOL,
        "tolerance": DELTA_W_TOL,
        "max_abs_delta_diff": max_diff,
        "layers_checked": layers,
        "worst_module": worst_module,
    }


def _topk_agreement(logits_a: torch.Tensor, logits_b: torch.Tensor, k: int) -> float:
    top_a = logits_a.topk(k).indices.tolist()
    top_b = logits_b.topk(k).indices.tolist()
    if k == 1:
        return 1.0 if top_a[0] == top_b[0] else 0.0
    return len(set(top_a) & set(top_b)) / float(k)


def audit_forward_drift(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    prompts: list[dict],
    dpo_weight: float,
) -> dict:
    """Non-blocking BnB8 stack vs cat last-token logit drift."""
    model.eval()
    device = _model_device(model)
    diffs: list[float] = []
    top1: list[float] = []
    top5: list[float] = []
    per_prompt: list[dict] = []

    for row in prompts:
        messages = row["messages"]
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
        t1 = _topk_agreement(logits_s, logits_c, 1)
        t5 = _topk_agreement(logits_s, logits_c, 5)
        diffs.append(diff)
        top1.append(t1)
        top5.append(t5)
        per_prompt.append(
            {
                "id": row.get("id"),
                "category": row.get("category"),
                "max_abs_logit_diff": diff,
                "top1_agreement": t1,
                "top5_overlap": t5,
            }
        )

    return {
        "blocking": False,
        "status": "measured",
        "reference": "bnb8_stack",
        "candidate": "bnb8_cat",
        "legacy_max_abs_logit_tolerance": LEGACY_LOGIT_TOL,
        "summary": {
            "prompts": len(prompts),
            "max_abs_logit_diff": max(diffs) if diffs else None,
            "mean_abs_logit_diff": sum(diffs) / len(diffs) if diffs else None,
            "top1_agreement_mean": sum(top1) / len(top1) if top1 else None,
            "top5_overlap_mean": sum(top5) / len(top5) if top5 else None,
        },
        "per_prompt": per_prompt,
    }


def _encode_prompt_ids(
    tokenizer: AutoTokenizer, messages: list[dict], device: torch.device
) -> torch.Tensor:
    ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
    return ids.to(device)


@torch.inference_mode()
def collect_cat_last_token_logits(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    prompts: list[dict],
) -> list[torch.Tensor]:
    """Last-token logits with cat adapter active (CPU tensors for cross-base compare)."""
    model.eval()
    device = _model_device(model)
    _set_cat_adapter(model)
    logits_list: list[torch.Tensor] = []
    for row in prompts:
        ids = _encode_prompt_ids(tokenizer, row["messages"], device)
        logits_list.append(model(ids).logits[0, -1].detach().cpu())
    return logits_list


def compare_last_token_logit_drift(
    reference_logits: list[torch.Tensor],
    candidate_logits: list[torch.Tensor],
    prompts: list[dict],
    *,
    reference: str,
    candidate: str,
    note: str | None = None,
) -> dict:
    diffs: list[float] = []
    top1: list[float] = []
    top5: list[float] = []
    per_prompt: list[dict] = []

    for row, log_a, log_b in zip(prompts, reference_logits, candidate_logits):
        diff = (log_a - log_b).abs().max().item()
        t1 = _topk_agreement(log_a, log_b, 1)
        t5 = _topk_agreement(log_a, log_b, 5)
        diffs.append(diff)
        top1.append(t1)
        top5.append(t5)
        per_prompt.append(
            {
                "id": row.get("id"),
                "category": row.get("category"),
                "max_abs_logit_diff": diff,
                "top1_agreement": t1,
                "top5_overlap": t5,
            }
        )

    out: dict = {
        "blocking": False,
        "status": "measured",
        "reference": reference,
        "candidate": candidate,
        "summary": {
            "prompts": len(prompts),
            "max_abs_logit_diff": max(diffs) if diffs else None,
            "mean_abs_logit_diff": sum(diffs) / len(diffs) if diffs else None,
            "top1_agreement_mean": sum(top1) / len(top1) if top1 else None,
            "top5_overlap_mean": sum(top5) / len(top5) if top5 else None,
        },
        "per_prompt": per_prompt,
    }
    if note:
        out["note"] = note
    return out


def audit_deploy_fp8_drift(
    bnb_cat_logits: list[torch.Tensor],
    fp8_model: PeftModel,
    tokenizer: AutoTokenizer,
    prompts: list[dict],
) -> dict:
    """Non-blocking BnB8 cat vs HF FP8+cat (deploy proxy for vLLM FP8+LoRA)."""
    fp8_model.eval()
    _set_cat_adapter(fp8_model)
    device = _model_device(fp8_model)
    fp8_logits: list[torch.Tensor] = []
    for row in prompts:
        ids = _encode_prompt_ids(tokenizer, row["messages"], device)
        with torch.no_grad():
            fp8_logits.append(fp8_model(ids).logits[0, -1].detach().cpu())
    return compare_last_token_logit_drift(
        bnb_cat_logits,
        fp8_logits,
        prompts,
        reference="bnb8_cat",
        candidate="fp8_cat",
        note="hf_fp8_proxy_not_vllm",
    )


@torch.inference_mode()
def _generate_one(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    messages: list[dict],
    *,
    mode: str,
    dpo_weight: float,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> str:
    originals = []
    if mode == "stack":
        ensure_policy_adapter_stack(model)
        originals = _scale_adapter_temporarily(model, DPO_ADAPTER_NAME, dpo_weight)
    elif mode == "cat":
        _set_cat_adapter(model)
    else:
        raise ValueError(f"unknown generation mode: {mode}")

    ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
    ids = ids.to(_model_device(model))
    prompt_len = ids.shape[1]
    gen_kw = build_generate_kwargs(
        tokenizer,
        decode=DECODE_SAMPLE,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    try:
        out = model.generate(ids, **gen_kw)
    finally:
        if mode == "stack":
            _restore_adapter_scaling(originals)
    new_ids = out[0, prompt_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def tag_generation(text: str, *, english_prompt: bool = True) -> list[str]:
    tags: list[str] = []
    stripped = text.strip()
    if not stripped:
        tags.append("empty_output")
        return tags
    if english_prompt and _CJK_RE.search(stripped):
        tags.append("unexpected_cjk")
    words = stripped.split()
    if len(words) >= 12:
        bigrams = [" ".join(words[i : i + 2]) for i in range(len(words) - 1)]
        if bigrams:
            from collections import Counter

            most_common = Counter(bigrams).most_common(1)[0]
            if most_common[1] >= 4:
                tags.append("looping")
    non_alnum = sum(1 for ch in stripped if not ch.isalnum() and not ch.isspace())
    if len(stripped) > 80 and non_alnum / len(stripped) > 0.35:
        tags.append("token_soup")
    if len(words) >= 6 and len(set(words)) / len(words) < 0.35:
        tags.append("looping")
    return tags


def run_generation_smoke(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    prompts: list[dict],
    dpo_weight: float,
    *,
    max_new_tokens: int = 256,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    top_k: int = TOP_K,
    repetition_penalty: float = 1.05,
) -> dict:
    """Non-blocking A=stack vs B=cat generation on the tiny prompt set."""
    per_prompt: list[dict] = []
    stack_fail_tags: set[str] = set()
    cat_fail_tags: set[str] = set()
    stack_ok = 0
    cat_ok = 0

    for row in prompts:
        messages = row["messages"]
        stack_text = _generate_one(
            model,
            tokenizer,
            messages,
            mode="stack",
            dpo_weight=dpo_weight,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        cat_text = _generate_one(
            model,
            tokenizer,
            messages,
            mode="cat",
            dpo_weight=dpo_weight,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        stack_tags = tag_generation(stack_text)
        cat_tags = tag_generation(cat_text)
        stack_fail_tags.update(stack_tags)
        cat_fail_tags.update(cat_tags)
        if not stack_tags:
            stack_ok += 1
        if not cat_tags:
            cat_ok += 1
        per_prompt.append(
            {
                "id": row.get("id"),
                "category": row.get("category"),
                "stack": {"text": stack_text[:500], "tags": stack_tags},
                "cat": {"text": cat_text[:500], "tags": cat_tags},
            }
        )

    n = len(prompts) or 1
    return {
        "blocking_for_export": False,
        "status": "measured",
        "sampling": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
        },
        "summary": {
            "prompts": len(prompts),
            "stack_clean_prompts": stack_ok,
            "cat_clean_prompts": cat_ok,
            "stack_failure_tags": sorted(stack_fail_tags),
            "cat_failure_tags": sorted(cat_fail_tags),
            "stack_clean_rate": stack_ok / n,
            "cat_clean_rate": cat_ok / n,
        },
        "per_prompt": per_prompt,
    }


def run_deploy_behavior_smoke(
    fp8_model: PeftModel,
    tokenizer: AutoTokenizer,
    prompts: list[dict],
    bnb_cat_smoke: dict,
    *,
    max_new_tokens: int = 256,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    top_k: int = TOP_K,
    repetition_penalty: float = 1.05,
) -> dict:
    """Non-blocking BnB8 cat vs FP8+cat generation on the tiny prompt set."""
    bnb_by_id = {r["id"]: r for r in bnb_cat_smoke.get("per_prompt", [])}
    per_prompt: list[dict] = []
    fp8_fail_tags: set[str] = set()
    bnb_fail_tags: set[str] = set()
    fp8_ok = 0
    bnb_ok = 0
    regressions = 0

    for row in prompts:
        pid = row.get("id")
        bnb_row = bnb_by_id.get(pid, {})
        bnb_text = (bnb_row.get("cat") or {}).get("text", "")
        bnb_tags = list((bnb_row.get("cat") or {}).get("tags", []))
        fp8_text = _generate_one(
            fp8_model,
            tokenizer,
            row["messages"],
            mode="cat",
            dpo_weight=1.0,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        fp8_tags = tag_generation(fp8_text)
        bnb_fail_tags.update(bnb_tags)
        fp8_fail_tags.update(fp8_tags)
        if not bnb_tags:
            bnb_ok += 1
        if not fp8_tags:
            fp8_ok += 1
        new_fp8_only = [t for t in fp8_tags if t not in bnb_tags]
        if new_fp8_only:
            regressions += 1
        per_prompt.append(
            {
                "id": pid,
                "category": row.get("category"),
                "bnb8_cat": {"text": bnb_text[:500], "tags": bnb_tags},
                "fp8_cat": {"text": fp8_text[:500], "tags": fp8_tags},
                "fp8_only_regression_tags": new_fp8_only,
            }
        )

    n = len(prompts) or 1
    return {
        "blocking_for_export": False,
        "status": "measured",
        "reference": "bnb8_cat",
        "candidate": "fp8_cat",
        "note": "hf_fp8_proxy_not_vllm",
        "sampling": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
        },
        "summary": {
            "prompts": len(prompts),
            "bnb8_cat_clean_prompts": bnb_ok,
            "fp8_cat_clean_prompts": fp8_ok,
            "bnb8_cat_failure_tags": sorted(bnb_fail_tags),
            "fp8_cat_failure_tags": sorted(fp8_fail_tags),
            "bnb8_cat_clean_rate": bnb_ok / n,
            "fp8_cat_clean_rate": fp8_ok / n,
            "fp8_only_regression_prompts": regressions,
        },
        "per_prompt": per_prompt,
    }


def _release_model(model: PeftModel | None) -> None:
    if model is None:
        return
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_merge_meta(
    *,
    dpo_adapter: Path,
    output: Path,
    dpo_weight: float,
    compat: dict,
    merge_correctness: dict,
    forward_drift: dict | None,
    behavior_smoke: dict | None,
    deploy_forward_drift: dict | None,
    deploy_behavior_smoke: dict | None,
    saved: bool,
) -> dict:
    reason = (
        "Delta-W merge correctness passed. Local drift/smoke checks are diagnostics."
        if saved
        else "Export blocked: Delta-W merge correctness failed."
    )
    verdict = "reject_export"
    if saved and behavior_smoke and behavior_smoke.get("status") == "measured":
        s = behavior_smoke["summary"]
        stack_rate = s.get("stack_clean_rate", 0.0)
        cat_rate = s.get("cat_clean_rate", 0.0)
        if stack_rate > 0 and cat_rate > 0:
            verdict = "keep_cat_for_eval"
        elif stack_rate <= 0 and cat_rate <= 0:
            verdict = "policy_likely_bad"
        elif stack_rate > 0 and cat_rate <= 0:
            verdict = "cat_runtime_suspicious"
    elif saved:
        verdict = "export_ok_smoke_not_run"

    return {
        "candidate": {
            "dpo_adapter": str(dpo_adapter.resolve()),
            "cat_adapter_dir": str(output.resolve()),
            "cat_adapter_name": CAT_ADAPTER_NAME,
            "stack_adapters": POLICY_ADAPTER_STACK,
            "dpo_weight": dpo_weight,
            "train_base": TRAIN_BASE_LABEL,
            "deploy_base": DEPLOY_BASE_LABEL,
            "deploy_model_path": str(MODEL_ID_FP8),
            "sft_path": str(SFT_ADAPTER),
            **compat,
        },
        "merge_correctness": merge_correctness,
        "local_forward_drift": forward_drift
        or {"blocking": False, "status": "not_run"},
        "local_behavior_smoke": behavior_smoke
        or {"blocking_for_export": False, "status": "not_run"},
        "deploy_forward_drift": deploy_forward_drift
        or {"blocking": False, "status": "not_run"},
        "deploy_behavior_smoke": deploy_behavior_smoke
        or {"blocking_for_export": False, "status": "not_run"},
        "export_decision": {
            "saved_adapter": saved,
            "reason": reason,
            "verdict": verdict,
            "deploy_fp8_note": _deploy_fp8_note(deploy_forward_drift, deploy_behavior_smoke),
        },
    }


def _deploy_fp8_note(
    deploy_forward_drift: dict | None,
    deploy_behavior_smoke: dict | None,
) -> str | None:
    parts: list[str] = []
    if deploy_forward_drift and deploy_forward_drift.get("status") == "measured":
        s = deploy_forward_drift["summary"]
        parts.append(
            f"fp8_logits max={s.get('max_abs_logit_diff'):.4f} "
            f"top1={s.get('top1_agreement_mean'):.2f}"
        )
    if deploy_behavior_smoke and deploy_behavior_smoke.get("status") == "measured":
        s = deploy_behavior_smoke["summary"]
        parts.append(
            f"fp8_gen clean={s.get('fp8_cat_clean_prompts')}/{s.get('prompts')} "
            f"regressions={s.get('fp8_only_regression_prompts')}"
        )
    return "; ".join(parts) if parts else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cat-merge SFT+DPO LoRA (Delta-W gated export; drift/smoke are diagnostic)"
    )
    parser.add_argument("--dpo-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dpo-weight",
        type=float,
        default=1.0,
        help="Scale DPO residual before cat export (1.0 = training policy).",
    )
    parser.add_argument(
        "--audit-forward-drift",
        action="store_true",
        help="Non-blocking BnB8 stack-vs-cat logit drift on the tiny prompt set.",
    )
    parser.add_argument(
        "--check-logps",
        action="store_true",
        help="Alias for --audit-forward-drift (legacy name).",
    )
    parser.add_argument(
        "--generation-smoke",
        action="store_true",
        help="Non-blocking stack vs cat generation smoke on the tiny prompt set.",
    )
    parser.add_argument(
        "--prompts-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL with {id, category, messages} per line.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Generation smoke max_new_tokens.",
    )
    parser.add_argument(
        "--audit-fp8-deploy",
        action="store_true",
        help="Non-blocking BnB8 cat vs HF FP8+cat deploy drift (proxy for vLLM FP8+LoRA).",
    )
    parser.add_argument(
        "--audit-fp8-drift",
        action="store_true",
        help="Alias for --audit-fp8-deploy.",
    )
    args = parser.parse_args()
    audit = args.audit_forward_drift or args.check_logps
    audit_fp8 = args.audit_fp8_deploy or args.audit_fp8_drift
    prompts = load_prompt_set(args.prompts_jsonl)

    dpo_path = resolve_dpo_adapter_path(args.dpo_adapter)
    if dpo_path != args.dpo_adapter.resolve():
        print(f"  resolved DPO adapter: {dpo_path}")

    print(f"Loading SFT + DPO on {TRAIN_BASE_LABEL} base...")
    model = load_stacked_for_merge(dpo_path)
    print(f"  adapters before cat: {list(model.peft_config.keys())}")

    compat = validate_merge_compatibility(model, dpo_path)
    merge_cat(model, args.dpo_weight)

    merge_correctness = verify_weight_matrices(model, args.dpo_weight)
    print(f"  Delta-W (export gate): pass={merge_correctness['pass']} "
          f"max_diff={merge_correctness['max_abs_delta_diff']:.2e}")
    if not merge_correctness["pass"]:
        meta = build_merge_meta(
            dpo_adapter=args.dpo_adapter,
            output=args.output,
            dpo_weight=args.dpo_weight,
            compat=compat,
            merge_correctness=merge_correctness,
            forward_drift=None,
            behavior_smoke=None,
            deploy_forward_drift=None,
            deploy_behavior_smoke=None,
            saved=False,
        )
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "merge_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        raise ValueError(
            f"Delta-W failed (max_diff={merge_correctness['max_abs_delta_diff']}); "
            "cat adapter not exported"
        )

    tokenizer = None
    if audit or args.generation_smoke:
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_ID_BF16))
        patch_chat_template_for_assistant_loss(tokenizer)

    forward_drift = None
    if audit and tokenizer is not None:
        forward_drift = audit_forward_drift(
            model, tokenizer, prompts, args.dpo_weight
        )
        s = forward_drift["summary"]
        print(
            f"  forward drift (non-blocking): max={s['max_abs_logit_diff']:.4f} "
            f"mean={s['mean_abs_logit_diff']:.4f} top1={s['top1_agreement_mean']:.2f} "
            f"top5={s['top5_overlap_mean']:.2f}"
        )

    behavior_smoke = None
    if args.generation_smoke and tokenizer is not None:
        print(f"  generation smoke ({len(prompts)} prompts, non-blocking)...")
        behavior_smoke = run_generation_smoke(
            model,
            tokenizer,
            prompts,
            args.dpo_weight,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=1.05,
        )
        bs = behavior_smoke["summary"]
        print(
            f"  smoke: stack_clean={bs['stack_clean_prompts']}/{bs['prompts']} "
            f"cat_clean={bs['cat_clean_prompts']}/{bs['prompts']} "
            f"verdict pending"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    model.set_adapter(CAT_ADAPTER_NAME)
    model.save_pretrained(str(args.output), selected_adapters=[CAT_ADAPTER_NAME])
    flatten_adapter_dir(args.output, CAT_ADAPTER_NAME)
    print(f"Exported cat adapter -> {args.output}")
    print("  Export gated by Delta-W only; use cat adapter for judge generation (plan B).")

    deploy_forward_drift = None
    deploy_behavior_smoke = None
    if audit_fp8:
        fp8_tokenizer = AutoTokenizer.from_pretrained(str(MODEL_ID_FP8))
        patch_chat_template_for_assistant_loss(fp8_tokenizer)
        print("  deploy FP8 audit: collecting BnB8 cat logits (FP8 tokenizer)...")
        bnb_cat_logits = collect_cat_last_token_logits(model, fp8_tokenizer, prompts)
        bnb_model = model
        model = None
        _release_model(bnb_model)

        print(f"  loading FP8 base + cat from {args.output}...")
        fp8_model = None
        try:
            fp8_model = load_cat_merged_adapter(args.output, base="fp8")
            deploy_forward_drift = audit_deploy_fp8_drift(
                bnb_cat_logits, fp8_model, fp8_tokenizer, prompts
            )
            ds = deploy_forward_drift["summary"]
            print(
                f"  deploy forward drift (non-blocking): max={ds['max_abs_logit_diff']:.4f} "
                f"mean={ds['mean_abs_logit_diff']:.4f} top1={ds['top1_agreement_mean']:.2f} "
                f"top5={ds['top5_overlap_mean']:.2f}"
            )
            if behavior_smoke is not None:
                print(f"  deploy generation smoke ({len(prompts)} prompts)...")
                deploy_behavior_smoke = run_deploy_behavior_smoke(
                    fp8_model,
                    fp8_tokenizer,
                    prompts,
                    behavior_smoke,
                    max_new_tokens=args.max_new_tokens,
                    repetition_penalty=1.05,
                )
                dbs = deploy_behavior_smoke["summary"]
                print(
                    f"  deploy smoke: bnb_cat_clean={dbs['bnb8_cat_clean_prompts']}/"
                    f"{dbs['prompts']} fp8_cat_clean={dbs['fp8_cat_clean_prompts']}/"
                    f"{dbs['prompts']} fp8_only_regressions={dbs['fp8_only_regression_prompts']}"
                )
        except Exception as exc:
            print(f"  WARNING: deploy FP8 audit failed: {exc}")
            deploy_forward_drift = {
                "blocking": False,
                "status": "error",
                "reference": "bnb8_cat",
                "candidate": "fp8_cat",
                "error": str(exc),
            }
        finally:
            _release_model(fp8_model)

    meta = build_merge_meta(
        dpo_adapter=args.dpo_adapter,
        output=args.output,
        dpo_weight=args.dpo_weight,
        compat=compat,
        merge_correctness=merge_correctness,
        forward_drift=forward_drift,
        behavior_smoke=behavior_smoke,
        deploy_forward_drift=deploy_forward_drift,
        deploy_behavior_smoke=deploy_behavior_smoke,
        saved=True,
    )
    (args.output / "merge_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"  verdict: {meta['export_decision']['verdict']}")
    if meta["export_decision"].get("deploy_fp8_note"):
        print(f"  deploy_fp8: {meta['export_decision']['deploy_fp8_note']}")


if __name__ == "__main__":
    main()
