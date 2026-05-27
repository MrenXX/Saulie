"""Optuna trial diagnostics per Diagnostic_Metrics.md."""

from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from trl.trainer.utils import selective_log_softmax

from dpo.train.dpo_data import MAX_LENGTH, SPLIT_SEED, manifest_path_for_seed, manifest_sha256
from dpo.train.dpo_trainer_compat import ensure_policy_adapter_stack
from dpo.train.paths import DATA_PATH, OUTPUT_BASE

_PROVENANCE: dict[str, Any] | None = None


def set_run_provenance(
    *,
    data_hash: str,
    length_stats: dict,
    mask_audit_pass: bool,
    trl_version: str,
    dummy_report_path: Path | None = None,
) -> None:
    global _PROVENANCE
    adapter_diag = {}
    if dummy_report_path and dummy_report_path.exists():
        report = json.loads(dummy_report_path.read_text(encoding="utf-8"))
        adapter_diag = report.get("adapter_diagnostics", {})
    _PROVENANCE = {
        "data_hash": data_hash,
        "split_manifest_sha256": manifest_sha256(manifest_path_for_seed(SPLIT_SEED)),
        "max_length": MAX_LENGTH,
        "max_observed_length": length_stats.get("max_observed"),
        "mask_audit_pass": mask_audit_pass,
        "only_dpo_trainable": adapter_diag.get("only_dpo_trainable", True),
        "non_dpo_trainable_count": adapter_diag.get("non_dpo_trainable_count", 0),
        "trl_version": trl_version,
    }


def get_provenance() -> dict[str, Any]:
    return dict(_PROVENANCE or {})


def log_line(prefix: str, msg: str, *, also_print: bool = True) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {prefix} {msg}"
    if also_print:
        print(line, flush=True)


def worker_prefix(worker_id: int | None, solo: bool = False) -> str:
    if solo:
        return "SOLO"
    if worker_id is not None:
        return f"W{worker_id}"
    return "MAIN"


@torch.no_grad()
def _per_batch_dpo_scores(trainer, batch: dict) -> tuple[float, float, float]:
    """Return chosen_reward, rejected_reward, margin for a batch of 1 pair."""
    trainer.model.eval()
    model = trainer.accelerator.unwrap_model(trainer.model)
    ensure_policy_adapter_stack(model)

    inputs = trainer._prepare_inputs(batch)
    _non_model_keys = {"completion_mask", "ref_chosen_logps", "ref_rejected_logps"}
    model_kwargs = {k: v for k, v in inputs.items() if k not in _non_model_keys}
    model_kwargs["use_cache"] = False

    outputs = model(**model_kwargs)
    input_ids = inputs["input_ids"]
    completion_mask = inputs["completion_mask"]
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_completion_mask = completion_mask[..., 1:].contiguous()
    per_token_logps = selective_log_softmax(shift_logits, shift_labels)
    per_token_logps[shift_completion_mask == 0] = 0.0

    if trainer.ld_alpha is None:
        logps = per_token_logps.sum(dim=1)
    else:
        comp_pos = shift_completion_mask.cumsum(dim=1)
        comp_lens = shift_completion_mask.sum(dim=1).long()
        chosen_lens, rejected_lens = comp_lens.chunk(2, dim=0)
        shared_lens = torch.minimum(chosen_lens, rejected_lens)
        shared_lens = torch.cat([shared_lens, shared_lens], dim=0).to(trainer.accelerator.device)
        shared_mask = (comp_pos > 0) & (comp_pos <= shared_lens.unsqueeze(1))
        tail_mask = comp_pos > shared_lens.unsqueeze(1)
        shared_logps = (per_token_logps * shared_mask).sum(dim=1)
        tail_logps = (per_token_logps * tail_mask).sum(dim=1)
        logps = shared_logps + trainer.ld_alpha * tail_logps

    chosen_logps, rejected_logps = logps.chunk(2, dim=0)
    ref_chosen = inputs["ref_chosen_logps"]
    ref_rejected = inputs["ref_rejected_logps"]
    chosen_lr = chosen_logps - ref_chosen
    rejected_lr = rejected_logps - ref_rejected
    chosen_reward = (trainer.beta * chosen_lr).item()
    rejected_reward = (trainer.beta * rejected_lr).item()
    return chosen_reward, rejected_reward, chosen_reward - rejected_reward


def _bucket_stats(records: list[dict], key_fn: Callable[[dict], str]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[key_fn(r)].append(r)
    out = {}
    for key, items in sorted(groups.items()):
        correct = [1.0 if x["correct"] else 0.0 for x in items]
        margins = [x["margin"] for x in items]
        out[key] = {
            "count": len(items),
            "accuracy": sum(correct) / len(correct) if correct else 0.0,
            "mean_margin": sum(margins) / len(margins) if margins else 0.0,
        }
    return out


def _macro_accuracy(buckets: dict[str, dict]) -> float:
    accs = [b["accuracy"] for b in buckets.values() if b["count"] > 0]
    return sum(accs) / len(accs) if accs else 0.0


def _safe_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    c = np.corrcoef(xs, ys)
    if math.isnan(c[0, 1]):
        return None
    return float(c[0, 1])


def _val_diag_wall_limit_s() -> float:
    raw = os.environ.get("DPO_MAX_VAL_DIAG_WALL_S")
    if raw is None:
        return 3600.0
    try:
        return float(raw)
    except ValueError:
        return 3600.0


def _vram_diag_snap() -> str:
    if not torch.cuda.is_available():
        return ""
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    return f" vram_alloc={alloc:.2f}GB vram_reserved={reserved:.2f}GB"


def compute_val_diagnostics(
    trainer,
    val_dataset,
    collator,
    *,
    log_fn: Callable[[str], None] | None = None,
    trial_number: int | None = None,
    log_every: int = 5,
    max_wall_s: float | None = None,
    heartbeat_fn: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Per-row val rewards + bucket / length-bias metrics."""
    n_rows = len(val_dataset)
    wall_limit = max_wall_s if max_wall_s is not None else _val_diag_wall_limit_s()
    t0 = time.monotonic()
    label = f"trial={trial_number}" if trial_number is not None else "trial=?"
    if log_fn:
        log_fn(f"stage=val_diagnostics start rows={n_rows} {label}")

    rows: list[dict] = []
    for i in range(n_rows):
        elapsed = time.monotonic() - t0
        if elapsed > wall_limit:
            from dpo.train.dpo_trainer_compat import TrialWallTimeout

            raise TrialWallTimeout(
                f"val_diagnostics_timeout elapsed={elapsed:.0f}s > {wall_limit:.0f}s "
                f"at row={i}/{n_rows} {label}"
            )
        if heartbeat_fn is not None:
            heartbeat_fn(i, n_rows)
        if log_fn and (i == 0 or (i + 1) % log_every == 0 or i + 1 == n_rows):
            log_fn(
                f"val_diag {label} row={i + 1}/{n_rows} elapsed={elapsed:.0f}s"
                f"{_vram_diag_snap()}"
            )
        ex = val_dataset[i]
        batch = collator([ex])
        chosen_r, rejected_r, margin = _per_batch_dpo_scores(trainer, batch)
        c_len = int(ex.get("chosen_scored_len", 0))
        r_len = int(ex.get("rejected_scored_len", 0))
        rows.append(
            {
                "id": ex.get("id"),
                "correct": chosen_r > rejected_r,
                "margin": margin,
                "chosen_scored_len": c_len,
                "rejected_scored_len": r_len,
                "length_delta": c_len - r_len,
                "dpo_source": ex.get("dpo_source", "unknown"),
                "category": ex.get("category", "unknown"),
                "source_family": ex.get("source_family", "unknown"),
            }
        )

    by_source = _bucket_stats(rows, lambda r: r["dpo_source"])
    by_family = _bucket_stats(rows, lambda r: r["source_family"])
    by_category = _bucket_stats(rows, lambda r: r["category"])
    by_family_cat = _bucket_stats(
        rows, lambda r: f"{r['source_family']}|{r['category']}"
    )

    deltas = [r["length_delta"] for r in rows]
    abs_deltas = [abs(d) for d in deltas]
    margins = [r["margin"] for r in rows]
    chosen_longer = [r for r in rows if r["length_delta"] > 0]
    rejected_longer = [r for r in rows if r["length_delta"] < 0]

    def _acc_mean(subset: list[dict]) -> tuple[float | None, float | None]:
        if not subset:
            return None, None
        acc = sum(1.0 if r["correct"] else 0.0 for r in subset) / len(subset)
        m = sum(r["margin"] for r in subset) / len(subset)
        return acc, m

    acc_cl, marg_cl = _acc_mean(chosen_longer)
    acc_rl, marg_rl = _acc_mean(rejected_longer)

    if log_fn:
        log_fn(
            f"stage=val_diagnostics done rows={n_rows} "
            f"elapsed={time.monotonic() - t0:.0f}s {label}"
        )

    return {
        "val_rows": len(rows),
        "by_dpo_source": by_source,
        "by_source_family": by_family,
        "by_category": by_category,
        "by_source_family_x_category": by_family_cat,
        "macro_accuracy_by_source_family": _macro_accuracy(by_family),
        "macro_accuracy_by_category": _macro_accuracy(by_category),
        "macro_accuracy_by_source_family_category": _macro_accuracy(by_family_cat),
        "chosen_scored_len_mean": float(np.mean([r["chosen_scored_len"] for r in rows])),
        "rejected_scored_len_mean": float(np.mean([r["rejected_scored_len"] for r in rows])),
        "length_delta_mean": float(np.mean(deltas)),
        "abs_length_delta_mean": float(np.mean(abs_deltas)),
        "margin_vs_length_delta_corr": _safe_corr(margins, deltas),
        "margin_vs_abs_length_delta_corr": _safe_corr(margins, abs_deltas),
        "accuracy_when_chosen_longer": acc_cl,
        "accuracy_when_rejected_longer": acc_rl,
        "mean_margin_when_chosen_longer": marg_cl,
        "mean_margin_when_rejected_longer": marg_rl,
    }


def build_trial_scorecard(
    metrics: dict,
    *,
    train_loss: float,
    vram: dict,
    runtime_seconds: float,
    saved_adapter_path: str,
    failure_reason: str | None = None,
    ref_cache: dict | None = None,
) -> dict[str, Any]:
    return {
        "eval_rewards_accuracy": metrics.get("eval_rewards/accuracies"),
        "eval_rewards_margin": metrics.get("eval_rewards/margins"),
        "eval_loss": metrics.get("eval_loss"),
        "train_loss": train_loss,
        "eval_rewards_chosen": metrics.get("eval_rewards/chosen"),
        "eval_rewards_rejected": metrics.get("eval_rewards/rejected"),
        "eval_logps_chosen": metrics.get("eval_logps/chosen"),
        "eval_logps_rejected": metrics.get("eval_logps/rejected"),
        "peak_vram_allocated_gb": vram.get("peak_allocated_gb"),
        "peak_vram_reserved_gb": vram.get("peak_reserved_gb"),
        "runtime_seconds": runtime_seconds,
        "saved_adapter_path": saved_adapter_path,
        "failure_reason": failure_reason,
        "ref_cache": ref_cache or {},
    }


def apply_trial_diagnostics(
    trial,
    *,
    scorecard: dict,
    val_diag: dict | None,
    provenance: dict,
    params: dict,
    derived: dict,
    adapter_diag: dict,
) -> dict[str, Any]:
    """Store diagnostics on Optuna trial; return full payload for JSON artifact."""
    payload: dict[str, Any] = {
        "scorecard": scorecard,
        "val_diagnostics": val_diag,
        "provenance": {**provenance, "length_mode": params.get("length_mode"), "effective_batch": derived.get("effective_batch")},
        "adapter_only_dpo_trainable": adapter_diag.get("only_dpo_trainable"),
        "non_dpo_trainable_count": adapter_diag.get("non_dpo_trainable_count"),
    }
    for k, v in scorecard.items():
        if v is not None:
            trial.set_user_attr(k, v)
    if val_diag:
        trial.set_user_attr("macro_accuracy_by_source_family", val_diag["macro_accuracy_by_source_family"])
        trial.set_user_attr("macro_accuracy_by_category", val_diag["macro_accuracy_by_category"])
        trial.set_user_attr(
            "macro_accuracy_by_source_family_category",
            val_diag["macro_accuracy_by_source_family_category"],
        )
        trial.set_user_attr("margin_vs_length_delta_corr", val_diag.get("margin_vs_length_delta_corr"))
        trial.set_user_attr("margin_vs_abs_length_delta_corr", val_diag.get("margin_vs_abs_length_delta_corr"))
        trial.set_user_attr("val_diagnostics_json", json.dumps(val_diag))
    for k, v in payload["provenance"].items():
        if v is not None:
            trial.set_user_attr(k, v)
    return payload


def compute_hybrid_score_v1_1(
    *,
    accuracy: float,
    macro_family_category: float | None,
    margin: float | None,
    eval_loss: float | None,
    len_corr: float | None,
    abs_len_corr: float | None,
) -> float:
    """Scalar objective for Optuna v1.1 (accuracy-first with guardrail penalties)."""
    score = float(accuracy)
    if macro_family_category is not None:
        score -= 0.50 * max(0.0, 0.95 - float(macro_family_category))
    if len_corr is not None:
        score -= 0.15 * max(0.0, abs(float(len_corr)) - 0.35)
    if abs_len_corr is not None:
        score -= 0.15 * max(0.0, abs(float(abs_len_corr)) - 0.40)
    if eval_loss is not None:
        score -= 0.03 * max(0.0, float(eval_loss) - 0.50)
    if margin is not None:
        m = float(margin)
        if m < 0:
            score -= 0.20
        elif m < 0.50:
            score -= 0.05 * (0.50 - m)
        elif m > 20:
            score -= min(0.20, 0.002 * (m - 20))
    return score


def log_trial_scorecard(prefix: str, trial_number: int, scorecard: dict, val_diag: dict | None) -> None:
    log_line(prefix, f"{'=' * 56}")
    log_line(prefix, f"TRIAL {trial_number} COMPLETE")
    log_line(prefix, f"  accuracy={scorecard.get('eval_rewards_accuracy'):.4f} margin={scorecard.get('eval_rewards_margin'):.4f}")
    log_line(prefix, f"  eval_loss={scorecard.get('eval_loss'):.4f} train_loss={scorecard.get('train_loss'):.4f}")
    log_line(prefix, f"  VRAM peak alloc={scorecard.get('peak_vram_allocated_gb'):.2f}GB reserved={scorecard.get('peak_vram_reserved_gb'):.2f}GB")
    log_line(prefix, f"  runtime={scorecard.get('runtime_seconds'):.0f}s adapter={scorecard.get('saved_adapter_path')}")
    if val_diag:
        log_line(
            prefix,
            f"  macro_acc family={val_diag['macro_accuracy_by_source_family']:.3f} "
            f"category={val_diag['macro_accuracy_by_category']:.3f} "
            f"family_x_cat={val_diag['macro_accuracy_by_source_family_category']:.3f}",
        )
        corr = val_diag.get("margin_vs_length_delta_corr")
        if corr is not None:
            flag = " SUSPECT" if abs(corr) > 0.5 else ""
            log_line(prefix, f"  margin_vs_length_delta_corr={corr:.3f}{flag}")
    log_line(prefix, f"{'=' * 56}")


def build_study_review(study) -> dict[str, Any]:
    """Top-trial diagnostic view for final summary."""
    from optuna.trial import TrialState

    complete = [t for t in study.trials if t.state == TrialState.COMPLETE]

    def _trial_row(t):
        ua = t.user_attrs
        raw_acc = ua.get("eval_rewards_accuracy")
        return {
            "trial": t.number,
            "eval_rewards_accuracy": raw_acc if raw_acc is not None else t.value,
            "hybrid_score_v1_1": ua.get("hybrid_score_v1_1", t.value),
            "eval_rewards_margin": ua.get("eval_rewards_margin"),
            "macro_accuracy_by_source_family_category": ua.get(
                "macro_accuracy_by_source_family_category"
            ),
            "margin_vs_length_delta_corr": ua.get("margin_vs_length_delta_corr"),
            "solo_retry": ua.get("solo_retry", False),
            "parallel_oom_recovered": ua.get("parallel_oom_recovered", False),
            "failure_reason": ua.get("failure_reason"),
            "length_mode": t.params.get("length_mode"),
        }

    suspicious = [
        r for r in (_trial_row(t) for t in complete)
        if r.get("margin_vs_length_delta_corr") is not None
        and abs(r["margin_vs_length_delta_corr"]) > 0.5
    ]
    weak_buckets = [
        r for r in (_trial_row(t) for t in complete)
        if r.get("macro_accuracy_by_source_family_category") is not None
        and r["macro_accuracy_by_source_family_category"] < 0.85
        and (r.get("eval_rewards_accuracy") or 0) > 0.9
    ]
    oom_trials = [
        {
            "trial": t.number,
            "failure_reason": t.user_attrs.get("failure_reason"),
            "queued_for_solo_retry": t.user_attrs.get("queued_for_solo_retry"),
            "solo_retry": t.user_attrs.get("solo_retry"),
        }
        for t in study.trials
        if t.user_attrs.get("queued_for_solo_retry") or t.user_attrs.get("solo_retry")
    ]

    duplicate_pruned = [
        {
            "trial": t.number,
            "duplicate_of": t.user_attrs.get("duplicate_of"),
            "failure_reason": t.user_attrs.get("failure_reason"),
        }
        for t in study.trials
        if t.user_attrs.get("failure_reason") == "duplicate_params"
    ]

    top_hybrid = sorted(
        complete,
        key=lambda t: t.user_attrs.get("hybrid_score_v1_1", t.value or -1),
        reverse=True,
    )[:5]

    return {
        "top_by_accuracy": [_trial_row(t) for t in sorted(
            complete,
            key=lambda t: t.user_attrs.get("eval_rewards_accuracy", t.value or -1) or -1,
            reverse=True,
        )[:5]],
        "top_by_hybrid_score": [_trial_row(t) for t in top_hybrid],
        "duplicate_pruned": duplicate_pruned,
        "top_by_macro_family_category": sorted(
            [_trial_row(t) for t in complete],
            key=lambda r: r.get("macro_accuracy_by_source_family_category") or 0,
            reverse=True,
        )[:5],
        "top_by_margin": sorted(
            [_trial_row(t) for t in complete],
            key=lambda r: r.get("eval_rewards_margin") or 0,
            reverse=True,
        )[:5],
        "suspicious_length_correlation": suspicious,
        "weak_source_category_buckets": weak_buckets,
        "oom_and_solo_recovered": oom_trials,
    }
