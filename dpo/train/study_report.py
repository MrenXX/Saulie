"""Generate a self-contained HTML review report from trial_summary.json."""

from __future__ import annotations

import argparse
import html
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def _esc(value: Any) -> str:
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _fmt_num(value: Any, digits: int = 3) -> str:
    number = _num(value)
    if number is None:
        return "n/a"
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    if abs(number) >= 100:
        return f"{number:,.1f}"
    if abs(number) >= 1:
        return f"{number:,.{digits}f}"
    if abs(number) >= 0.01:
        return f"{number:.4f}"
    return f"{number:.2e}"


def _fmt_pct(value: Any, digits: int = 1) -> str:
    number = _num(value)
    if number is None:
        return "n/a"
    return f"{number * 100:.{digits}f}%"


def _json_from_attr(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _trial_number(trial: dict[str, Any]) -> Any:
    return trial.get("trial_number", trial.get("number", "?"))


def _state_class(state: Any) -> str:
    text = str(state or "unknown").lower()
    return "".join(ch if ch.isalnum() else "-" for ch in text)


def _user_attrs(trial: dict[str, Any]) -> dict[str, Any]:
    attrs = trial.get("user_attrs") or {}
    return attrs if isinstance(attrs, dict) else {}


def _params(trial: dict[str, Any]) -> dict[str, Any]:
    params = trial.get("params") or {}
    return params if isinstance(params, dict) else {}


def _derived(trial: dict[str, Any]) -> dict[str, Any]:
    derived = trial.get("derived") or _user_attrs(trial).get("derived") or {}
    return derived if isinstance(derived, dict) else {}


def _val_diag(trial: dict[str, Any]) -> dict[str, Any]:
    return _json_from_attr(_user_attrs(trial).get("val_diagnostics_json"))


def _metric(trial: dict[str, Any], key: str) -> Any:
    attrs = _user_attrs(trial)
    if key in attrs:
        return attrs[key]
    return trial.get(key)


def _hybrid(trial: dict[str, Any]) -> float | None:
    h = _num(_metric(trial, "hybrid_score_v1_1"))
    return h if h is not None else _num(trial.get("value"))


def _accuracy(trial: dict[str, Any]) -> float | None:
    accuracy = _num(_metric(trial, "eval_rewards_accuracy"))
    return accuracy if accuracy is not None else _num(trial.get("value"))


def _macro(trial: dict[str, Any]) -> float | None:
    return _num(_metric(trial, "macro_accuracy_by_source_family_category"))


def _margin(trial: dict[str, Any]) -> float | None:
    return _num(_metric(trial, "eval_rewards_margin"))


def _loss(trial: dict[str, Any]) -> float | None:
    return _num(_metric(trial, "eval_loss"))


def _length_corr(trial: dict[str, Any]) -> float | None:
    return _num(_metric(trial, "margin_vs_length_delta_corr"))


def _abs_length_corr(trial: dict[str, Any]) -> float | None:
    return _num(_metric(trial, "margin_vs_abs_length_delta_corr"))


def _runtime(trial: dict[str, Any]) -> float | None:
    return _num(_metric(trial, "runtime_seconds"))


def _vram(trial: dict[str, Any]) -> float | None:
    return _num(_metric(trial, "peak_vram_allocated_gb"))


def _effective_batch(trial: dict[str, Any]) -> Any:
    params = _params(trial)
    derived = _derived(trial)
    return derived.get("effective_batch") or _metric(trial, "effective_batch") or (
        params.get("per_device_train_batch_size"),
        params.get("gradient_accumulation_steps"),
    )


def _state_counts(trials: list[dict[str, Any]], summary_counts: dict[str, Any]) -> dict[str, int]:
    counts = Counter(str(t.get("state", "UNKNOWN")) for t in trials)
    for key, value in summary_counts.items():
        if key not in {"total", "attempted"}:
            number = _num(value)
            if number is not None:
                counts[key] = int(number)
    counts["total"] = len(trials) or int(_num(summary_counts.get("total")) or 0)
    return dict(counts)


def _all_trials(summary: dict[str, Any]) -> list[dict[str, Any]]:
    trials = summary.get("trials") or []
    if isinstance(trials, list) and trials:
        return [t for t in trials if isinstance(t, dict)]

    fallback = []
    for row in summary.get("complete_trials") or []:
        if not isinstance(row, dict):
            continue
        fallback.append(
            {
                "trial_number": row.get("number"),
                "state": "COMPLETE",
                "value": row.get("value"),
                "params": row.get("params") or {},
                "solo_retry": row.get("solo_retry", False),
                "parallel_oom_recovered": row.get("parallel_oom_recovered", False),
                "user_attrs": {"eval_rewards_accuracy": row.get("value")},
            }
        )
    return fallback


def _sort_key(trial: dict[str, Any], *, use_hybrid: bool = False) -> tuple[Any, ...]:
    state_order = {"COMPLETE": 0, "PRUNED": 1, "FAIL": 2, "RUNNING": 3, "WAITING": 4}
    state = str(trial.get("state", "UNKNOWN"))
    primary = _hybrid(trial) if use_hybrid else _accuracy(trial)
    return (
        state_order.get(state, 9),
        -(primary if primary is not None else -1.0),
        -(_macro(trial) if _macro(trial) is not None else -1.0),
        _loss(trial) if _loss(trial) is not None else 999999.0,
        _trial_number(trial) if isinstance(_trial_number(trial), int) else 999999,
    )


def _scalar_attrs(attrs: dict[str, Any]) -> list[tuple[str, Any]]:
    skip = {"adapter_diagnostics", "derived", "ref_cache", "val_diagnostics_json", "vram"}
    rows = []
    for key in sorted(attrs):
        if key in skip:
            continue
        value = attrs[key]
        if isinstance(value, (dict, list)):
            continue
        rows.append((key, value))
    return rows


def _score_class(value: float | None, high_good: bool = True) -> str:
    if value is None:
        return "neutral"
    if high_good:
        if value >= 0.95:
            return "good"
        if value >= 0.85:
            return "watch"
        return "bad"
    if value <= 0.25:
        return "good"
    if value <= 0.5:
        return "watch"
    return "bad"


def _corr_class(value: float | None) -> str:
    if value is None:
        return "neutral"
    if abs(value) <= 0.35:
        return "good"
    if abs(value) <= 0.5:
        return "watch"
    return "bad"


def _bar(value: Any, *, high_good: bool = True) -> str:
    number = _num(value)
    if number is None:
        return '<span class="na">n/a</span>'
    pct = max(0.0, min(100.0, number * 100.0))
    klass = _score_class(number, high_good=high_good)
    return (
        f'<span class="bar-value"><span>{_fmt_pct(number)}</span>'
        f'<i class="bar {klass}"><b style="width:{pct:.1f}%"></b></i></span>'
    )


def _sparkline(values: list[float | None], *, low_good: bool = False) -> str:
    points = [(i, v) for i, v in enumerate(values) if v is not None]
    if len(points) < 2:
        return '<span class="na">not enough data</span>'
    width = 260
    height = 54
    pad = 3
    ys = [v for _, v in points]
    lo = min(ys)
    hi = max(ys)
    if hi == lo:
        hi = lo + 1.0
    denom_x = max(1, len(values) - 1)
    coords = []
    for i, value in points:
        x = pad + (width - pad * 2) * (i / denom_x)
        ratio = (value - lo) / (hi - lo)
        if low_good:
            ratio = 1.0 - ratio
        y = pad + (height - pad * 2) * (1.0 - ratio)
        coords.append(f"{x:.1f},{y:.1f}")
    first = coords[0]
    last = coords[-1]
    polyline = " ".join(coords)
    first_x, first_y = first.split(",")
    last_x, last_y = last.split(",")
    return (
        f'<svg class="spark" viewBox="0 0 {width} {height}" aria-hidden="true">'
        f'<polyline points="{polyline}" fill="none" stroke="currentColor" stroke-width="1.4"/>'
        f'<circle cx="{first_x}" cy="{first_y}" r="1.8"/>'
        f'<circle cx="{last_x}" cy="{last_y}" r="2.2" class="end"/>'
        "</svg>"
    )


def _metric_value(value: float | None, kind: str) -> str:
    if value is None:
        return "n/a"
    if kind == "pct":
        return _fmt_pct(value)
    if kind == "seconds":
        return f"{_fmt_num(value, digits=1)}s"
    if kind == "gb":
        return f"{_fmt_num(value, digits=2)} GB"
    return _fmt_num(value)


def _spark_panel(
    values: list[float | None],
    *,
    kind: str = "number",
    low_good: bool = False,
) -> str:
    nums = [v for v in values if v is not None]
    if not nums:
        return '<span class="na">not enough data</span>'
    latest = next((v for v in reversed(values) if v is not None), None)
    avg = sum(nums) / len(nums)
    direction = "lower better" if low_good else "higher better"
    return f"""
<div class="spark-panel">
  {_sparkline(values, low_good=low_good)}
  <dl class="spark-stats">
        <div><dt>min</dt><dd>{_metric_value(min(nums), kind)}</dd></div>
    <div><dt>avg</dt><dd>{_metric_value(avg, kind)}</dd></div>
        <div><dt>max</dt><dd>{_metric_value(max(nums), kind)}</dd></div>
        <div><dt>latest</dt><dd>{_metric_value(latest, kind)}</dd></div>
  </dl>
  <p class="spark-note">{direction}; plotted by trial number.</p>
</div>"""


def _batch_label(trial: dict[str, Any]) -> str:
    params = _params(trial)
    if params.get("batch_combo"):
        eff = _effective_batch(trial)
        combo = params["batch_combo"]
        return f"{_esc(combo)}={_esc(eff)}" if eff and not isinstance(eff, tuple) else _esc(combo)
    batch = params.get("per_device_train_batch_size")
    accum = params.get("gradient_accumulation_steps")
    effective = _effective_batch(trial)
    if isinstance(effective, tuple):
        effective = None
    if batch is None and accum is None:
        return _esc(effective) if effective else "n/a"
    if effective:
        return f"{_esc(batch)}x{_esc(accum)}={_esc(effective)}"
    return f"{_esc(batch)}x{_esc(accum)}"


def _objective_label(trial: dict[str, Any]) -> str:
    params = _params(trial)
    derived = _derived(trial)
    length = params.get("length_mode") or _metric(trial, "length_mode") or "n/a"
    loss = derived.get("loss_type") or "dpo"
    ld_alpha = derived.get("ld_alpha")
    if ld_alpha is not None:
        return f"{_esc(length)} (ld={_esc(ld_alpha)})"
    return f"{_esc(length)} ({_esc(loss)})"


def _gate_flags(trial: dict[str, Any]) -> list[str]:
    attrs = _user_attrs(trial)
    flags = []
    if attrs.get("only_dpo_trainable") is True or attrs.get("non_dpo_trainable_count") == 0:
        flags.append("adapter ok")
    elif "only_dpo_trainable" in attrs or "non_dpo_trainable_count" in attrs:
        flags.append("adapter check")

    if attrs.get("mask_audit_pass") is True:
        flags.append("mask ok")
    elif "mask_audit_pass" in attrs:
        flags.append("mask check")

    max_len = _num(attrs.get("max_length"))
    max_obs = _num(attrs.get("max_observed_length"))
    if max_len is not None and max_obs is not None:
        flags.append("len ok" if max_obs <= max_len else "len over")
    return flags


def _trial_flags(trial: dict[str, Any]) -> list[tuple[str, str]]:
    flags: list[tuple[str, str]] = []
    state = str(trial.get("state", ""))
    if trial.get("failure_reason"):
        flags.append(("bad", str(trial["failure_reason"])[:80]))
    if trial.get("queued_for_solo_retry"):
        flags.append(("watch", "queued solo"))
    if trial.get("parallel_oom_recovered"):
        flags.append(("good", "OOM recovered solo"))
    if trial.get("solo_retry"):
        flags.append(("neutral", "solo retry"))
    if _user_attrs(trial).get("anchor_trial"):
        flags.append(("good", "anchor"))
    if _user_attrs(trial).get("duplicate_of") is not None:
        flags.append(("watch", f"dup #{_user_attrs(trial).get('duplicate_of')}"))
    if state == "PRUNED":
        flags.append(("watch", "pruned"))
    if state == "FAIL":
        flags.append(("bad", "failed"))

    corr = _length_corr(trial)
    if corr is not None and abs(corr) > 0.5:
        flags.append(("bad", "length corr"))
    elif corr is not None and abs(corr) > 0.35:
        flags.append(("watch", "length drift"))

    margin = _margin(trial)
    if margin is not None and margin < 0:
        flags.append(("bad", "negative margin"))
    elif margin is not None and margin > 100:
        flags.append(("watch", "margin outlier"))

    for flag in _gate_flags(trial):
        flags.append(("good" if flag.endswith("ok") else "watch", flag))
    return flags


def _flag_html(flags: list[tuple[str, str]]) -> str:
    if not flags:
        return '<span class="na">n/a</span>'
    return "".join(f'<span class="pill {klass}">{_esc(label)}</span>' for klass, label in flags)


def _bucket_table(title: str, buckets: dict[str, Any]) -> str:
    if not buckets:
        return ""
    rows = []
    for name, stats in sorted(buckets.items()):
        if not isinstance(stats, dict):
            continue
        count = stats.get("count", stats.get("n", "n/a"))
        acc = _num(stats.get("accuracy"))
        margin = stats.get("mean_margin")
        rows.append(
            "<tr>"
            f"<td>{_esc(name)}</td>"
            f"<td>{_esc(count)}</td>"
            f"<td>{_bar(acc)}</td>"
            f"<td>{_fmt_num(margin)}</td>"
            "</tr>"
        )
    if not rows:
        return ""
    return f"""
<div class="bucket-block">
  <h4>{_esc(title)}</h4>
  <table class="bucket-table">
    <thead><tr><th>bucket</th><th>n</th><th>accuracy</th><th>mean margin</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</div>"""


def _length_diag_table(diag: dict[str, Any]) -> str:
    if not diag:
        return ""
    fields = [
        ("chosen scored len mean", diag.get("chosen_scored_len_mean")),
        ("rejected scored len mean", diag.get("rejected_scored_len_mean")),
        ("length delta mean", diag.get("length_delta_mean")),
        ("abs length delta mean", diag.get("abs_length_delta_mean")),
        ("margin vs length corr", diag.get("margin_vs_length_delta_corr")),
        ("margin vs abs length corr", diag.get("margin_vs_abs_length_delta_corr")),
        ("acc when chosen longer", diag.get("accuracy_when_chosen_longer")),
        ("acc when rejected longer", diag.get("accuracy_when_rejected_longer")),
        ("margin when chosen longer", diag.get("mean_margin_when_chosen_longer")),
        ("margin when rejected longer", diag.get("mean_margin_when_rejected_longer")),
    ]
    rows = "".join(
        f"<tr><td>{_esc(label)}</td><td>{_fmt_num(value)}</td></tr>" for label, value in fields
    )
    return f"""
<div class="bucket-block">
  <h4>length diagnostics</h4>
  <table class="kv"><tbody>{rows}</tbody></table>
</div>"""


def _trial_detail(trial: dict[str, Any]) -> str:
    attrs = _user_attrs(trial)
    params = _params(trial)
    derived = _derived(trial)
    diag = _val_diag(trial)
    scalar_rows = "".join(
        f"<tr><td>{_esc(key)}</td><td>{_esc(value)}</td></tr>" for key, value in _scalar_attrs(attrs)
    )
    param_rows = "".join(
        f"<tr><td>{_esc(key)}</td><td>{_esc(value)}</td></tr>" for key, value in sorted(params.items())
    )
    derived_rows = "".join(
        f"<tr><td>{_esc(key)}</td><td>{_esc(value)}</td></tr>" for key, value in sorted(derived.items())
    )
    adapter_path = attrs.get("saved_adapter_path")
    adapter = f'<p class="path"><b>adapter:</b> {_esc(adapter_path)}</p>' if adapter_path else ""
    buckets = "".join(
        [
            _bucket_table("source family", diag.get("by_source_family") or {}),
            _bucket_table("raw DPO source", diag.get("by_dpo_source") or {}),
            _bucket_table("category", diag.get("by_category") or {}),
            _bucket_table("family x category", diag.get("by_source_family_x_category") or {}),
            _length_diag_table(diag),
        ]
    )
    return f"""
<div class="detail-grid">
  <div>{adapter}<h4>params</h4><table class="kv"><tbody>{param_rows}</tbody></table></div>
  <div><h4>derived</h4><table class="kv"><tbody>{derived_rows}</tbody></table></div>
  <div><h4>scalar logs</h4><table class="kv"><tbody>{scalar_rows}</tbody></table></div>
  <div class="buckets">{buckets or '<p class="na">No validation diagnostics were logged for this trial.</p>'}</div>
</div>"""


def _trial_row(trial: dict[str, Any]) -> str:
    number = _trial_number(trial)
    state = str(trial.get("state", "UNKNOWN"))
    params = _params(trial)
    acc = _accuracy(trial)
    macro = _macro(trial)
    margin = _margin(trial)
    loss = _loss(trial)
    corr = _length_corr(trial)
    abs_corr = _abs_length_corr(trial)
    runtime = _runtime(trial)
    vram = _vram(trial)
    lr = params.get("learning_rate")
    beta = params.get("beta")
    lora_r = params.get("lora_r")
    dropout = params.get("lora_dropout")
    neftune = params.get("neftune_noise_alpha")
    scheduler = params.get("lr_scheduler_type", "")
    row_flags = _trial_flags(trial)
    row_class = " ".join({klass for klass, _ in row_flags if klass in {"watch", "bad"}})
    detail = _trial_detail(trial)
    return f"""
<tr class="trial-row {row_class}" data-state="{_esc(state)}" data-trial="{_esc(number)}">
  <td class="trial-id">#{_esc(number)}</td>
  <td data-sort="{_esc(state)}"><span class="state {_state_class(state)}">{_esc(state)}</span></td>
  <td data-sort="{acc if acc is not None else -1}">{_bar(acc)}</td>
  <td data-sort="{macro if macro is not None else -1}">{_bar(macro)}</td>
  <td data-sort="{margin if margin is not None else -999999}">{_fmt_num(margin)}</td>
  <td data-sort="{loss if loss is not None else 999999}">{_fmt_num(loss)}</td>
  <td data-sort="{corr if corr is not None else 999999}" class="{_corr_class(corr)}">{_fmt_num(corr)}</td>
  <td data-sort="{abs_corr if abs_corr is not None else 999999}" class="{_corr_class(abs_corr)}">{_fmt_num(abs_corr)}</td>
  <td>{_objective_label(trial)}</td>
  <td>{_fmt_num(beta)}</td>
  <td>{_fmt_num(lr, digits=2)}</td>
  <td>r{_esc(lora_r)} / d{_fmt_num(dropout)}</td>
  <td>{_batch_label(trial)}</td>
  <td>{_esc(scheduler)}</td>
  <td>{_fmt_num(neftune)}</td>
  <td data-sort="{runtime if runtime is not None else -1}">{_fmt_num(runtime, digits=1)}s</td>
  <td data-sort="{vram if vram is not None else -1}">{_fmt_num(vram, digits=2)} GB</td>
  <td class="flags">{_flag_html(row_flags)}</td>
</tr>
<tr class="detail-row" data-for-trial="{_esc(number)}" hidden>
  <td colspan="18">{detail}</td>
</tr>"""


def _summary_card(title: str, value: str, body: str = "", klass: str = "") -> str:
    klass_attr = f" {klass}" if klass else ""
    return f"""
<section class="summary-card{klass_attr}">
  <h3>{_esc(title)}</h3>
  <div class="big-number">{value}</div>
  {body}
</section>"""


def _candidate_rows(trials: list[dict[str, Any]], *, watchlist: bool = False) -> str:
    rows = []
    for trial in trials[:6]:
        number = _trial_number(trial)
        flags = _trial_flags(trial)
        flag_text = ", ".join(label for _, label in flags[:3]) or "clean"
        rows.append(
            "<tr>"
            f"<td>#{_esc(number)}</td>"
            f"<td>{_fmt_pct(_accuracy(trial))}</td>"
            f"<td>{_fmt_pct(_macro(trial))}</td>"
            f"<td>{_fmt_num(_margin(trial))}</td>"
            f"<td>{_fmt_num(_loss(trial))}</td>"
            f"<td>{_fmt_num(_length_corr(trial))}</td>"
            f"<td>{_esc(flag_text)}</td>"
            "</tr>"
        )
    if not rows:
        text = "No watchlist entries" if watchlist else "No candidates matched the clean-candidate gate"
        return f'<p class="na">{text}.</p>'
    return (
        '<table class="mini-table"><thead><tr><th>trial</th><th>acc</th><th>macro</th>'
        '<th>margin</th><th>loss</th><th>len corr</th><th>note</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _best_candidates(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    complete = [t for t in trials if str(t.get("state")) == "COMPLETE" and _accuracy(t) is not None]
    if not complete:
        return []
    best_acc = max(_accuracy(t) or 0.0 for t in complete)
    threshold = max(0.0, best_acc - 0.02)
    clean = []
    for trial in complete:
        acc = _accuracy(trial) or 0.0
        macro = _macro(trial) or 0.0
        corr = _length_corr(trial)
        flags = [label for _, label in _trial_flags(trial)]
        if acc >= threshold and macro >= 0.95 and (corr is None or abs(corr) <= 0.5):
            if "adapter check" not in flags and "mask check" not in flags and "len over" not in flags:
                clean.append(trial)
    return sorted(
        clean,
        key=lambda t: (
            -(_accuracy(t) or -1.0),
            -(_macro(t) or -1.0),
            _loss(t) if _loss(t) is not None else 999999.0,
            -(_margin(t) or -999999.0),
        ),
    )


def _watchlist(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for trial in trials:
        state = str(trial.get("state", ""))
        acc = _accuracy(trial)
        margin = _margin(trial)
        corr = _length_corr(trial)
        flags = _trial_flags(trial)
        if state in {"FAIL", "PRUNED"} or trial.get("failure_reason"):
            rows.append(trial)
        elif acc is not None and acc < 0.85:
            rows.append(trial)
        elif margin is not None and (margin < 0 or margin > 100):
            rows.append(trial)
        elif corr is not None and abs(corr) > 0.5:
            rows.append(trial)
        elif any(label in {"adapter check", "mask check", "len over"} for _, label in flags):
            rows.append(trial)
    return sorted(rows, key=lambda t: (str(t.get("state")) == "COMPLETE", _accuracy(t) or -1.0))


def _review_shortlists(summary: dict[str, Any], trials: list[dict[str, Any]]) -> str:
    candidates = _best_candidates(trials)
    watch = _watchlist(trials)
    review = summary.get("study_review") or {}
    top_margin = []
    top_hybrid = []
    for item in review.get("top_by_hybrid_score") or []:
        trial_num = item.get("trial", item.get("trial_number"))
        match = next((t for t in trials if _trial_number(t) == trial_num), None)
        if match:
            top_hybrid.append(match)
    for item in review.get("top_by_margin") or []:
        trial_num = item.get("trial", item.get("trial_number"))
        match = next((t for t in trials if _trial_number(t) == trial_num), None)
        if match:
            top_margin.append(match)
    return f"""
<section class="review-grid">
  <div>
    <h2>Candidate Shortlist</h2>
    <p class="section-note">Trials within two accuracy points of the best run, with macro accuracy at least 95% and no hard gate warnings.</p>
    {_candidate_rows(candidates)}
  </div>
  <div>
    <h2>Watchlist</h2>
    <p class="section-note">Low accuracy, failed/pruned attempts, length-correlation risk, negative margins, or margin outliers.</p>
    {_candidate_rows(watch, watchlist=True)}
  </div>
  <div>
    <h2>Hybrid Leaders</h2>
    <p class="section-note">Optuna v1.1 objective (accuracy with macro/length/margin guardrails).</p>
    {_candidate_rows(top_hybrid or sorted(trials, key=lambda t: -(_hybrid(t) or -999999.0)), watchlist=True)}
  </div>
  <div>
    <h2>Raw Accuracy Leaders</h2>
    <p class="section-note">Pairwise reward accuracy only — separate from the hybrid objective.</p>
    {_candidate_rows(sorted(trials, key=lambda t: -(_accuracy(t) or -999999.0))[:5], watchlist=True)}
  </div>
  <div>
    <h2>Margin Leaders</h2>
    <p class="section-note">High margin can be useful, but it is not the selection criterion by itself.</p>
    {_candidate_rows(top_margin or sorted(trials, key=lambda t: -(_margin(t) or -999999.0)), watchlist=True)}
  </div>
</section>"""


def _evidence_cards(summary: dict[str, Any], trials: list[dict[str, Any]]) -> str:
    complete = [t for t in trials if str(t.get("state")) == "COMPLETE"]
    ordered = sorted(trials, key=lambda t: _trial_number(t) if isinstance(_trial_number(t), int) else 999999)
    acc_values = [_accuracy(t) for t in ordered]
    loss_values = [_loss(t) for t in ordered]
    vram_values = [_vram(t) for t in ordered]
    runtime_values = [_runtime(t) for t in ordered]
    best = min(complete, key=_sort_key) if complete else None
    best_value = f"#{_esc(_trial_number(best))}" if best else "n/a"
    best_body = ""
    if best:
        best_body = (
            f'<p>acc <b>{_fmt_pct(_accuracy(best))}</b>, macro <b>{_fmt_pct(_macro(best))}</b>, '
            f'margin <b>{_fmt_num(_margin(best))}</b>, loss <b>{_fmt_num(_loss(best))}</b></p>'
        )
    counts = _state_counts(trials, summary.get("counts") or {})
    target = summary.get("target_complete_trials", "n/a")
    length_corrs = [_length_corr(t) for t in complete if _length_corr(t) is not None]
    high_corr = sum(1 for c in length_corrs if abs(c) > 0.5)
    mask_ok = sum(1 for t in complete if _user_attrs(t).get("mask_audit_pass") is True)
    adapter_ok = sum(
        1
        for t in complete
        if _user_attrs(t).get("only_dpo_trainable") is True
        or _user_attrs(t).get("non_dpo_trainable_count") == 0
    )
    return "".join(
        [
            _summary_card("Best Trial", best_value, best_body, "accent"),
            _summary_card(
                "Completion",
                f"{counts.get('COMPLETE', 0)} / {target}",
                f"<p>{counts.get('PRUNED', 0)} pruned, {counts.get('FAIL', 0)} failed, {counts.get('total', len(trials))} total attempts.</p>",
            ),
            _summary_card(
                "Safety Gates",
                f"{adapter_ok}/{len(complete)} adapters",
                f"<p>{mask_ok}/{len(complete)} complete trials passed mask audit logs.</p>",
            ),
            _summary_card(
                "Length Bias",
                f"{high_corr} flagged",
                f"<p>{len(length_corrs)} complete trials logged margin-vs-length correlation.</p>",
                "warn" if high_corr else "",
            ),
            _summary_card(
                "Accuracy Path",
                _spark_panel(acc_values, kind="pct"),
            ),
            _summary_card(
                "Loss Path",
                _spark_panel(loss_values, low_good=True),
            ),
            _summary_card(
                "Runtime Path",
                _spark_panel(runtime_values, kind="seconds", low_good=True),
            ),
            _summary_card(
                "VRAM Path",
                _spark_panel(vram_values, kind="gb", low_good=True),
            ),
        ]
    )


def _provenance(summary: dict[str, Any]) -> str:
    keys = [
        "study_name",
        "trl_version",
        "dataset_hash",
        "split_manifest_sha256",
        "parallel_workers",
        "target_complete_trials",
        "max_attempted_trials",
        "parallel_oom_queued_count",
        "solo_recovered_complete_count",
        "solo_intrinsic_oom_count",
        "finished_at",
        "study_version",
        "optuna_base_seed",
        "sampler_settings",
        "duplicate_pruned_count",
        "unique_complete_config_count",
        "run_dir",
        "study_storage",
        "solo_retry_queue",
        "dummy_report_path",
    ]
    rows = "".join(
        f"<tr><td>{_esc(key)}</td><td>{_esc(summary.get(key, 'n/a'))}</td></tr>" for key in keys
    )
    return f"""
<section>
  <h2>Provenance</h2>
  <table class="kv provenance"><tbody>{rows}</tbody></table>
</section>"""


def _trial_table(trials: list[dict[str, Any]], *, use_hybrid: bool = False) -> str:
    rows = "".join(
        _trial_row(t) for t in sorted(trials, key=lambda t: _sort_key(t, use_hybrid=use_hybrid))
    )
    return f"""
<section>
  <div class="table-head">
    <div>
      <h2>Trial Ledger</h2>
      <p class="section-note">A dense table for final selection: objective metrics first, then length-bias checks, hyperparams, runtime, VRAM, and gates.</p>
    </div>
    <div class="controls">
      <input id="search" type="search" placeholder="filter text">
      <button type="button" class="filter active" data-filter="all">all</button>
      <button type="button" class="filter" data-filter="COMPLETE">complete</button>
      <button type="button" class="filter" data-filter="PRUNED">pruned</button>
      <button type="button" class="filter" data-filter="FAIL">fail</button>
      <button type="button" id="toggle-details">details</button>
    </div>
  </div>
  <div class="table-wrap">
    <table id="trials-table">
      <thead>
        <tr>
          <th data-col="0">trial</th>
          <th data-col="1">state</th>
          <th data-col="2">accuracy</th>
          <th data-col="3">macro</th>
          <th data-col="4">margin</th>
          <th data-col="5">eval loss</th>
          <th data-col="6">len corr</th>
          <th data-col="7">abs len corr</th>
          <th data-col="8">objective</th>
          <th data-col="9">beta</th>
          <th data-col="10">lr</th>
          <th data-col="11">lora</th>
          <th data-col="12">batch</th>
          <th data-col="13">scheduler</th>
          <th data-col="14">neftune</th>
          <th data-col="15">runtime</th>
          <th data-col="16">vram</th>
          <th data-col="17">flags</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</section>"""


def render_study_report_html(summary: dict[str, Any]) -> str:
    trials = _all_trials(summary)
    counts = _state_counts(trials, summary.get("counts") or {})
    use_hybrid = summary.get("study_version") == "v1.1"
    title = summary.get("study_name", "DPO Optuna study")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DPO study report - {_esc(title)}</title>
<style>
:root {{
    color-scheme: dark;
    --paper: #0d1117;
    --ink: #d8dee9;
    --muted: #8b96a3;
    --line: #2a333d;
    --soft: #111821;
    --softer: #161c23;
    --good: #3ecf76;
    --good-bg: #0a2318;
    --watch: #e09b20;
    --watch-bg: #271c04;
    --bad: #e05a48;
    --bad-bg: #270d09;
    --blue: #5fa8d8;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--paper); color: var(--ink); font-family: Georgia, "Times New Roman", serif; line-height: 1.42; }}
main {{ padding: 32px 24px 56px; }}
h1 {{ font-size: 30px; line-height: 1.1; margin: 0 0 8px; font-weight: 600; }}
h2 {{ font: 700 13px/1.2 system-ui, sans-serif; text-transform: uppercase; letter-spacing: 0; color: var(--muted); margin: 0 0 10px; }}
h3 {{ font: 700 12px/1.2 system-ui, sans-serif; text-transform: uppercase; letter-spacing: 0; color: var(--muted); margin: 0 0 8px; }}
h4 {{ font: 700 12px/1.2 system-ui, sans-serif; color: var(--muted); margin: 10px 0 6px; }}
p {{ margin: 0 0 8px; }}
.deck {{ color: var(--muted); margin-bottom: 18px; }}
.lede {{ border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); padding: 12px 0; margin-bottom: 18px; display: flex; flex-wrap: wrap; gap: 18px; font: 13px/1.35 system-ui, sans-serif; color: var(--muted); }}
.lede b {{ color: var(--ink); }}
.summary-grid {{ display: grid; grid-template-columns: repeat(4, minmax(190px, 1fr)); gap: 10px; margin: 16px 0 28px; }}
.summary-card {{ border: 1px solid var(--line); background: var(--softer); border-radius: 6px; padding: 12px; min-height: 122px; }}
.summary-card.accent {{ border-color: var(--blue); }}
.summary-card.warn {{ border-color: var(--watch); background: var(--watch-bg); }}
.big-number {{ font: 700 26px/1.05 system-ui, sans-serif; margin-bottom: 8px; }}
.summary-card p {{ font: 13px/1.35 system-ui, sans-serif; color: var(--muted); }}
.spark {{ width: 100%; height: 58px; color: var(--ink); display: block; margin: 2px 0 8px; }}
.spark .end {{ fill: var(--bad); }}
.spark-panel {{ font: 13px/1.35 system-ui, sans-serif; }}
.spark-stats {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 6px; margin: 0; }}
.spark-stats div {{ border-top: 1px solid var(--line); padding-top: 5px; min-width: 0; }}
.spark-stats dt {{ color: var(--muted); font: 700 10px/1.2 system-ui, sans-serif; text-transform: uppercase; }}
.spark-stats dd {{ margin: 2px 0 0; color: var(--ink); font: 700 12px/1.2 system-ui, sans-serif; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.spark-note {{ margin-top: 7px; }}
.review-grid {{ display: grid; grid-template-columns: repeat(3, minmax(280px, 1fr)); gap: 18px; margin-bottom: 30px; }}
.section-note {{ color: var(--muted); font: 13px/1.35 system-ui, sans-serif; margin-bottom: 8px; }}
table {{ border-collapse: collapse; width: 100%; }}
th {{ font: 700 12px/1.2 system-ui, sans-serif; color: var(--muted); text-align: left; white-space: nowrap; }}
td {{ font: 13px/1.35 system-ui, sans-serif; vertical-align: top; }}
.mini-table th, .mini-table td {{ border-bottom: 1px solid var(--line); padding: 5px 6px; }}
.table-head {{ display: flex; justify-content: space-between; gap: 16px; align-items: end; margin-top: 12px; }}
.controls {{ display: flex; flex-wrap: wrap; gap: 6px; justify-content: flex-end; align-items: center; font: 13px system-ui, sans-serif; }}
input[type="search"], button {{ border: 1px solid var(--line); background: var(--paper); color: var(--ink); border-radius: 5px; padding: 6px 9px; font: 13px system-ui, sans-serif; }}
button {{ cursor: pointer; }}
button.active {{ border-color: var(--blue); color: var(--blue); background: #0e2030; }}
.table-wrap {{ overflow-x: auto; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }}
#trials-table {{ min-width: 1280px; }}
#trials-table thead {{ position: sticky; top: 0; background: var(--paper); z-index: 2; }}
#trials-table th {{ border-bottom: 1px solid var(--line); padding: 8px 7px; cursor: pointer; }}
#trials-table td {{ border-bottom: 1px solid var(--line); padding: 7px; }}
#trials-table tr.trial-row:hover td {{ background: var(--soft); }}
#trials-table tr.watch td {{ background: rgba(39, 28, 4, 0.55); }}
#trials-table tr.bad td {{ background: rgba(39, 13, 9, 0.55); }}
.trial-id {{ font-weight: 700; }}
.state, .pill {{ display: inline-block; border-radius: 4px; padding: 2px 6px; font: 700 11px/1.2 system-ui, sans-serif; white-space: nowrap; }}
.state.complete, .pill.good {{ background: var(--good-bg); color: var(--good); }}
.state.pruned, .pill.watch {{ background: var(--watch-bg); color: var(--watch); }}
.state.fail, .pill.bad {{ background: var(--bad-bg); color: var(--bad); }}
.state.running, .state.waiting, .pill.neutral {{ background: var(--soft); color: var(--muted); }}
.flags {{ min-width: 160px; }}
.flags .pill {{ margin: 0 3px 3px 0; }}
.bar-value {{ min-width: 92px; display: inline-grid; grid-template-columns: 42px 46px; gap: 6px; align-items: center; }}
.bar {{ display: block; height: 5px; background: #2a333d; border-radius: 999px; overflow: hidden; }}
.bar b {{ display: block; height: 100%; background: var(--muted); }}
.bar.good b {{ background: var(--good); }}
.bar.watch b {{ background: var(--watch); }}
.bar.bad b {{ background: var(--bad); }}
.good {{ color: var(--good); }}
.watch {{ color: var(--watch); }}
.bad {{ color: var(--bad); }}
.neutral, .na {{ color: var(--muted); }}
.detail-row td {{ background: var(--softer); padding: 14px 16px 18px; }}
.detail-grid {{ display: grid; grid-template-columns: minmax(240px, 0.9fr) minmax(190px, 0.7fr) minmax(260px, 1fr) minmax(320px, 1.3fr); gap: 16px; }}
.kv td, .bucket-table td, .bucket-table th {{ border-bottom: 1px solid var(--line); padding: 4px 6px; }}
.kv td:first-child {{ color: var(--muted); white-space: nowrap; width: 1%; padding-right: 14px; }}
.kv td:last-child {{ word-break: break-all; overflow-wrap: anywhere; width: 100%; }}
.path {{ font: 12px/1.35 system-ui, sans-serif; color: var(--muted); overflow-wrap: anywhere; }}
.bucket-block {{ margin-bottom: 12px; }}
.provenance {{ margin-bottom: 24px; }}
@media (max-width: 980px) {{
  main {{ padding: 22px 14px 42px; }}
  .summary-grid, .review-grid, .detail-grid {{ grid-template-columns: 1fr; }}
  .table-head {{ align-items: stretch; flex-direction: column; }}
  .controls {{ justify-content: flex-start; }}
}}
</style>
</head>
<body>
<main>
  <header>
    <h1>DPO Optuna Study Report</h1>
    <p class="deck">{_esc(title)}. Compact evidence for reviewing trial quality: reward accuracy, macro bucket behavior, margins, losses, length-bias diagnostics, adapter/mask gates, runtime, VRAM, and exact hyperparameters.</p>
    <div class="lede">
      <span>complete <b>{counts.get('COMPLETE', 0)}</b></span>
      <span>pruned <b>{counts.get('PRUNED', 0)}</b></span>
      <span>failed <b>{counts.get('FAIL', 0)}</b></span>
      <span>target <b>{_esc(summary.get('target_complete_trials', 'n/a'))}</b></span>
      <span>workers <b>{_esc(summary.get('parallel_workers', 'n/a'))}</b></span>
      <span>dataset <b>{_esc(summary.get('dataset_hash', 'n/a'))}</b></span>
    </div>
  </header>

  <section class="summary-grid">{_evidence_cards(summary, trials)}</section>
  {_review_shortlists(summary, trials)}
  {_trial_table(trials, use_hybrid=use_hybrid)}
  {_provenance(summary)}
</main>
<script>
(function() {{
  const table = document.getElementById('trials-table');
  const tbody = table.querySelector('tbody');
  const search = document.getElementById('search');
  let sortCol = 2;
  let sortAsc = false;
  let activeFilter = 'all';
  let detailsOpen = false;

  function trialRows() {{ return Array.from(tbody.querySelectorAll('tr.trial-row')); }}
  function detailFor(row) {{ return tbody.querySelector('tr.detail-row[data-for-trial="' + row.dataset.trial + '"]'); }}
  function sortValue(cell) {{
    if (!cell) return '';
    const raw = cell.dataset.sort !== undefined ? cell.dataset.sort : cell.textContent.trim();
    const asNum = Number(raw);
    return Number.isNaN(asNum) ? String(raw).toLowerCase() : asNum;
  }}
  function sortRows() {{
    const rows = trialRows();
    rows.sort((a, b) => {{
      const av = sortValue(a.children[sortCol]);
      const bv = sortValue(b.children[sortCol]);
      if (av < bv) return sortAsc ? -1 : 1;
      if (av > bv) return sortAsc ? 1 : -1;
      return 0;
    }});
    rows.forEach(row => {{
      const detail = detailFor(row);
      tbody.appendChild(row);
      if (detail) tbody.appendChild(detail);
    }});
  }}
  function applyFilters() {{
    const query = (search.value || '').toLowerCase();
    trialRows().forEach(row => {{
      const detail = detailFor(row);
      const stateOk = activeFilter === 'all' || row.dataset.state === activeFilter;
      const textOk = !query || row.textContent.toLowerCase().includes(query) || (detail && detail.textContent.toLowerCase().includes(query));
      const show = stateOk && textOk;
      row.hidden = !show;
      if (detail) detail.hidden = !show || !detailsOpen;
    }});
  }}
  table.querySelectorAll('thead th').forEach(th => {{
    th.addEventListener('click', () => {{
      const col = Number(th.dataset.col);
      if (sortCol === col) sortAsc = !sortAsc; else {{ sortCol = col; sortAsc = true; }}
      sortRows();
      applyFilters();
    }});
  }});
  trialRows().forEach(row => {{
    row.addEventListener('click', () => {{
      const detail = detailFor(row);
      if (detail) detail.hidden = !detail.hidden;
    }});
  }});
  document.querySelectorAll('.filter').forEach(btn => {{
    btn.addEventListener('click', () => {{
      document.querySelectorAll('.filter').forEach(other => other.classList.remove('active'));
      btn.classList.add('active');
      activeFilter = btn.dataset.filter;
      applyFilters();
    }});
  }});
  search.addEventListener('input', applyFilters);
  document.getElementById('toggle-details').addEventListener('click', event => {{
    detailsOpen = !detailsOpen;
    event.currentTarget.classList.toggle('active', detailsOpen);
    applyFilters();
  }});
}})();
</script>
</body>
</html>"""


def write_study_report(summary_path: Path, output_path: Path | None = None) -> Path:
    summary_path = summary_path.resolve()
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    out = output_path.resolve() if output_path else summary_path.with_name("study_report.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_study_report_html(data), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HTML study report from trial_summary.json")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    out = write_study_report(args.summary, args.output)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()