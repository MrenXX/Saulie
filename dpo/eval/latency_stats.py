"""Shared percentile helpers for latency benchmarks."""

from __future__ import annotations

import statistics
from typing import Any


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    xs = sorted(values)
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def summarize_ms(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None, "min_ms": None, "max_ms": None}
    return {
        "n": len(values),
        "mean_ms": round(statistics.mean(values), 2),
        "p50_ms": round(percentile(values, 50) or 0, 2),
        "p95_ms": round(percentile(values, 95) or 0, 2),
        "min_ms": round(min(values), 2),
        "max_ms": round(max(values), 2),
    }


def summarize_tokens_per_s(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean_tps": None, "p50_tps": None, "p95_tps": None}
    return {
        "n": len(values),
        "mean_tps": round(statistics.mean(values), 2),
        "p50_tps": round(percentile(values, 50) or 0, 2),
        "p95_tps": round(percentile(values, 95) or 0, 2),
    }


def check_targets(actual: dict[str, Any], targets: dict[str, Any]) -> list[dict[str, Any]]:
    """Return pass/fail rows comparing actual p95/mean to target thresholds."""
    rows = []
    for key, spec in targets.items():
        if key not in actual:
            continue
        val = actual[key]
        if val is None:
            rows.append({"metric": key, "status": "no_data", "actual": None, "target": spec})
            continue
        op = spec.get("op", "lte")
        threshold = spec["value"]
        stat_key = spec.get("stat", "p95_ms")
        if stat_key == "rate" and isinstance(val, dict):
            actual_val = val.get("rate")
        else:
            actual_val = val.get(stat_key) if isinstance(val, dict) else val
        if actual_val is None:
            rows.append({"metric": key, "status": "no_data", "actual": None, "target": spec})
            continue
        if op == "lte":
            ok = actual_val <= threshold
        elif op == "gte":
            ok = actual_val >= threshold
        else:
            ok = False
        rows.append({
            "metric": key,
            "status": "pass" if ok else "fail",
            "actual": actual_val,
            "threshold": threshold,
            "stat": stat_key,
            "unit": spec.get("unit", "ms"),
        })
    return rows
