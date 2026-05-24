"""Build dummy-run report with Optuna-readiness gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_gates(report: dict) -> dict[str, Any]:
    """Return per-gate pass/fail for corrected dummy approval."""
    gates: dict[str, Any] = {}

    mask = report.get("mask_audit", {})
    gates["mask_audit_pass"] = {
        "pass": mask.get("pass") is True,
        "detail": mask.get("violations", []),
    }

    length = report.get("length_stats", {})
    gates["no_overlength"] = {
        "pass": length.get("overlength_count", 1) == 0,
        "detail": length.get("overlength_ids", []),
    }

    adapter = report.get("adapter_diagnostics", {})
    gates["only_dpo_trainable"] = {
        "pass": adapter.get("only_dpo_trainable") is True,
        "detail": adapter.get("non_dpo_trainable_count", -1),
    }

    metrics = report
    acc = metrics.get("eval_rewards_accuracies") or metrics.get("eval_rewards/accuracies")
    eval_loss = metrics.get("eval_loss")
    gates["finite_metrics"] = {
        "pass": acc is not None and eval_loss is not None
        and all(
            v == v and abs(v) < 1e6
            for v in (acc, eval_loss, metrics.get("train_loss", 0))
            if v is not None
        ),
        "detail": {"acc": acc, "eval_loss": eval_loss},
    }

    vram = report.get("vram", {})
    peak = vram.get("peak_allocated_gb", 99)
    gates["vram_envelope"] = {
        "pass": 7.0 <= peak <= 12.0,
        "detail": {"peak_allocated_gb": peak},
    }

    adapter_path = report.get("saved_adapter_path")
    gates["adapter_saved"] = {
        "pass": bool(adapter_path) and Path(adapter_path).joinpath("adapter_config.json").exists(),
        "detail": adapter_path,
    }

    trl = report.get("trl_version", "")
    gates["trl_sigmoid_norm"] = {
        "pass": _trl_supports_sigmoid_norm(trl),
        "detail": trl,
    }

    all_pass = all(g["pass"] for g in gates.values())
    gates["all_pass"] = {"pass": all_pass, "detail": None}
    return gates


def _trl_supports_sigmoid_norm(version: str) -> bool:
    try:
        from packaging.version import Version

        return Version(version) >= Version("1.0.0")
    except Exception:
        return "sigmoid_norm" in str(version)


def write_report(report: dict, path: Path) -> None:
    report["gates"] = evaluate_gates(report)
    report["optuna_ready"] = report["gates"]["all_pass"]["pass"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
