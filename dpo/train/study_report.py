"""Generate a self-contained HTML report from trial_summary.json."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def _esc(x: Any) -> str:
    if x is None:
        return ""
    return html.escape(str(x))


def _fmt_num(v: Any, digits: int = 4) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
        if abs(f) >= 100:
            return f"{f:.1f}"
        if abs(f) >= 1:
            return f"{f:.3f}"
        if abs(f) >= 0.01:
            return f"{f:.4f}"
        return f"{f:.2e}"
    except (TypeError, ValueError):
        return _esc(v)


def _ref_cache_label(ua: dict) -> str:
    rc = ua.get("ref_cache") or {}
    splits = rc.get("splits") or {}
    if not splits:
        return "—"
    parts = []
    for name in ("train", "eval"):
        meta = splits.get(name) or {}
        hit = meta.get("hit")
        if hit is True:
            parts.append(f"{name[0].upper()}:H")
        elif hit is False:
            parts.append(f"{name[0].upper()}:M")
        else:
            parts.append(f"{name[0].upper()}:?")
    return "/".join(parts)


def _trial_sort_key(t: dict) -> tuple:
    state = t.get("state", "")
    ua = t.get("user_attrs") or {}
    acc = ua.get("eval_rewards_accuracy")
    if acc is None:
        acc = t.get("value")
    state_order = {"COMPLETE": 0, "PRUNED": 1, "FAIL": 2, "RUNNING": 3}.get(state, 4)
    return (state_order, -(acc or -1), t.get("trial_number", 0))


def _shortlist_card(title: str, rows: list[dict], highlight: str | None = None) -> str:
    if not rows:
        return f'<div class="card"><h3>{_esc(title)}</h3><p class="muted">None</p></div>'
    body = ['<table class="mini"><thead><tr><th>Trial</th><th>Acc</th><th>Margin</th><th>Macro</th><th>LenΔ corr</th></tr></thead><tbody>']
    for r in rows:
        trial = r.get("trial", r.get("trial_number", "?"))
        corr = r.get("margin_vs_length_delta_corr")
        corr_cls = "warn" if corr is not None and abs(float(corr)) > 0.5 else ""
        body.append(
            f'<tr class="{corr_cls}"><td>#{_esc(trial)}</td>'
            f'<td>{_fmt_num(r.get("eval_rewards_accuracy"))}</td>'
            f'<td>{_fmt_num(r.get("eval_rewards_margin"))}</td>'
            f'<td>{_fmt_num(r.get("macro_accuracy_by_source_family_category"))}</td>'
            f'<td>{_fmt_num(corr)}</td></tr>'
        )
    body.append("</tbody></table></div>")
    hcls = f' class="{highlight}"' if highlight else ""
    return f'<div class="card{hcls}"><h3>{_esc(title)}</h3>{"".join(body)}'


def _detail_attrs(ua: dict) -> str:
    skip = {
        "adapter_diagnostics",
        "derived",
        "val_diagnostics_json",
        "ref_cache",
        "vram",
    }
    lines = []
    for k in sorted(ua.keys()):
        if k in skip:
            continue
        v = ua[k]
        if isinstance(v, (dict, list)):
            continue
        lines.append(f"<tr><td class='k'>{_esc(k)}</td><td>{_esc(v)}</td></tr>")
    return "".join(lines)


def _trial_table_row(t: dict, run_dir: Path) -> str:
    n = t.get("trial_number", "?")
    state = t.get("state", "?")
    p = t.get("params") or {}
    d = t.get("derived") or {}
    ua = t.get("user_attrs") or {}
    acc = ua.get("eval_rewards_accuracy", t.get("value"))
    margin = ua.get("eval_rewards_margin")
    macro = ua.get("macro_accuracy_by_source_family_category")
    corr = ua.get("margin_vs_length_delta_corr")
    eff = d.get("effective_batch") or ua.get("effective_batch")
    batch = p.get("per_device_train_batch_size", "?")
    accum = p.get("gradient_accumulation_steps", "?")
    runtime = ua.get("runtime_seconds")
    vram = ua.get("peak_vram_allocated_gb")
    adapter = ua.get("saved_adapter_path") or str(run_dir / f"trial-{n}" / "best_adapter")
    flags = []
    if t.get("solo_retry"):
        flags.append("solo")
    if t.get("parallel_oom_recovered"):
        flags.append("oom→solo")
    if t.get("failure_reason"):
        flags.append(_esc(t["failure_reason"])[:24])
    if corr is not None and abs(float(corr)) > 0.5:
        flags.append("⚠ len-corr")
    flag_str = ", ".join(flags) if flags else "—"
    state_cls = state.lower()
    warn_cls = " row-warn" if corr is not None and abs(float(corr)) > 0.5 else ""
    diag_path = run_dir / f"trial-{n}" / "diagnostics.json"
    detail = _detail_attrs(ua)
    return f"""
<tr class="trial-row{warn_cls}" data-state="{_esc(state)}" data-trial="{n}">
  <td>#{n}</td>
  <td><span class="badge {state_cls}">{_esc(state)}</span></td>
  <td data-sort="{acc if acc is not None else -1}">{_fmt_num(acc)}</td>
  <td data-sort="{margin if margin is not None else -1}">{_fmt_num(margin)}</td>
  <td data-sort="{macro if macro is not None else -1}">{_fmt_num(macro)}</td>
  <td>{_fmt_num(p.get('beta'))}</td>
  <td>{_fmt_num(p.get('learning_rate'), 2)}</td>
  <td>{_esc(p.get('lora_r'))}</td>
  <td>{batch}×{accum}={_esc(eff)}</td>
  <td>{_esc(p.get('length_mode'))}</td>
  <td>{_esc(p.get('lr_scheduler_type', ''))[:12]}</td>
  <td>{_fmt_num(p.get('neftune_noise_alpha'), 1)}</td>
  <td data-sort="{runtime if runtime is not None else -1}">{_fmt_num(runtime, 0) if runtime else '—'}s</td>
  <td data-sort="{vram if vram is not None else -1}">{_fmt_num(vram, 2) if vram else '—'}</td>
  <td>{_ref_cache_label(ua)}</td>
  <td class="flags">{flag_str}</td>
</tr>
<tr class="detail-row" data-for-trial="{n}" style="display:none">
  <td colspan="16">
    <details open>
      <summary>Details — trial #{n}</summary>
      <p><b>Adapter:</b> <code>{_esc(adapter)}</code>
         {' | <a href="file://' + _esc(str(diag_path)) + '">diagnostics.json</a>' if diag_path.exists() else ''}</p>
      <table class="kv"><tbody>{detail}</tbody></table>
    </details>
  </td>
</tr>"""


def render_study_report_html(summary: dict) -> str:
    run_dir = Path(summary.get("run_dir", "."))
    counts = summary.get("counts") or {}
    review = summary.get("study_review") or {}
    trials = sorted(summary.get("trials") or [], key=_trial_sort_key)
    best = summary.get("best_trial")
    best_acc = summary.get("best_accuracy")
    best_params = summary.get("best_params") or {}
    best_params_html = "<br>".join(f"{_esc(k)}={_esc(v)}" for k, v in best_params.items())

    cards = [
        _shortlist_card("Top by accuracy", review.get("top_by_accuracy") or []),
        _shortlist_card(
            "Top by macro (family×category)",
            review.get("top_by_macro_family_category") or [],
        ),
        _shortlist_card("Top by margin", review.get("top_by_margin") or []),
        _shortlist_card(
            "Suspicious length correlation (|ρ|>0.5)",
            review.get("suspicious_length_correlation") or [],
            highlight="alert",
        ),
        _shortlist_card(
            "Weak buckets (macro&lt;0.85, acc&gt;0.9)",
            review.get("weak_source_category_buckets") or [],
        ),
        _shortlist_card("OOM / solo recovered", review.get("oom_and_solo_recovered") or []),
    ]

    rows_html = "".join(_trial_table_row(t, run_dir) for t in trials)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DPO study — {_esc(summary.get('study_name', 'study'))}</title>
<style>
:root {{ --bg:#0f1419; --card:#1a2332; --text:#e6edf3; --muted:#8b949e;
  --ok:#3fb950; --warn:#d29922; --bad:#f85149; --pruned:#a371f7; --border:#30363d; }}
* {{ box-sizing: border-box; }}
body {{ font-family: system-ui, sans-serif; background: var(--bg); color: var(--text);
  margin: 0; padding: 1rem 1.5rem 3rem; line-height: 1.45; }}
h1 {{ font-size: 1.35rem; margin: 0 0 0.5rem; }}
h3 {{ font-size: 0.95rem; margin: 0 0 0.5rem; color: var(--muted); }}
.muted {{ color: var(--muted); font-size: 0.9rem; }}
.header {{ margin-bottom: 1.25rem; padding-bottom: 1rem; border-bottom: 1px solid var(--border); }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 0.75rem; margin-bottom: 1.5rem; }}
.card {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 0.75rem; }}
.card.alert {{ border-color: var(--warn); }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.82rem; }}
.mini td, .mini th {{ padding: 0.25rem 0.4rem; text-align: left; }}
.mini tr.warn td {{ color: var(--warn); }}
#main-table {{ background: var(--card); border-radius: 8px; overflow: hidden; border: 1px solid var(--border); }}
#main-table thead {{ position: sticky; top: 0; background: #21262d; z-index: 1; }}
#main-table th {{ cursor: pointer; user-select: none; padding: 0.5rem 0.35rem; text-align: left;
  border-bottom: 1px solid var(--border); white-space: nowrap; }}
#main-table th:hover {{ background: #30363d; }}
#main-table td {{ padding: 0.35rem; border-bottom: 1px solid var(--border); }}
#main-table tr.trial-row:nth-child(4n+1) {{ background: rgba(255,255,255,0.02); }}
#main-table tr.row-warn td {{ background: rgba(210,153,34,0.08); }}
.badge {{ display: inline-block; padding: 0.1rem 0.4rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }}
.badge.complete {{ background: rgba(63,185,80,0.2); color: var(--ok); }}
.badge.pruned {{ background: rgba(163,113,247,0.2); color: var(--pruned); }}
.badge.fail {{ background: rgba(248,81,73,0.2); color: var(--bad); }}
.badge.running {{ background: rgba(210,153,34,0.2); color: var(--warn); }}
.toolbar {{ margin: 1rem 0; display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center; }}
.toolbar button {{ background: var(--card); color: var(--text); border: 1px solid var(--border);
  padding: 0.35rem 0.75rem; border-radius: 6px; cursor: pointer; }}
.toolbar button.active {{ border-color: var(--ok); color: var(--ok); }}
.detail-row td {{ background: #161b22; font-size: 0.8rem; }}
.kv .k {{ color: var(--muted); padding-right: 1rem; vertical-align: top; }}
code {{ font-size: 0.78rem; word-break: break-all; }}
.flags {{ font-size: 0.75rem; color: var(--muted); max-width: 8rem; }}
a {{ color: #58a6ff; }}
</style>
</head>
<body>
<div class="header">
  <h1>DPO Optuna study report</h1>
  <p class="muted"><b>{_esc(summary.get('study_name'))}</b></p>
  <p class="muted">Run dir: <code>{_esc(run_dir)}</code></p>
  <p class="muted">Optuna DB: <code>{_esc(summary.get('study_storage'))}</code></p>
  <p>COMPLETE <b>{counts.get('COMPLETE', 0)}</b> · PRUNED <b>{counts.get('PRUNED', 0)}</b> ·
     FAIL <b>{counts.get('FAIL', 0)}</b> · target {summary.get('target_complete_trials', '?')}
     {' ✓ reached' if summary.get('target_reached') else ''}</p>
  <p>Best trial <b>#{_esc(best)}</b> accuracy <b>{_fmt_num(best_acc)}</b><br>{best_params_html}</p>
</div>
<div class="cards">{''.join(cards)}</div>
<div class="toolbar">
  <span>Filter:</span>
  <button type="button" class="filter-btn active" data-filter="all">All</button>
  <button type="button" class="filter-btn" data-filter="COMPLETE">COMPLETE</button>
  <button type="button" class="filter-btn" data-filter="PRUNED">PRUNED</button>
  <button type="button" class="filter-btn" data-filter="FAIL">FAIL</button>
  <span class="muted" style="margin-left:1rem">Click column headers to sort. Click a row to expand details.</span>
</div>
<div id="main-table">
<table id="trials-table">
<thead><tr>
  <th data-col="0">Trial</th><th data-col="1">State</th><th data-col="2">Acc</th>
  <th data-col="3">Margin</th><th data-col="4">Macro</th><th data-col="5">β</th>
  <th data-col="6">LR</th><th data-col="7">LoRA r</th><th data-col="8">Batch</th>
  <th data-col="9">Length</th><th data-col="10">Sched</th><th data-col="11">Neftune</th>
  <th data-col="12">Runtime</th><th data-col="13">VRAM</th><th data-col="14">Ref</th>
  <th data-col="15">Flags</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>
</div>
<script>
(function() {{
  const table = document.getElementById('trials-table');
  const tbody = table.querySelector('tbody');
  let sortCol = 2, sortAsc = false;

  function getTrialRows() {{
    return [...tbody.querySelectorAll('tr.trial-row')];
  }}

  function pairFor(row) {{
    const n = row.dataset.trial;
    return [row, tbody.querySelector('tr.detail-row[data-for-trial="'+n+'"]')];
  }}

  function sortRows() {{
    const rows = getTrialRows();
    rows.sort((a, b) => {{
      const ac = a.children[sortCol];
      const bc = b.children[sortCol];
      let av = ac.dataset.sort !== undefined ? parseFloat(ac.dataset.sort) : ac.textContent.trim();
      let bv = bc.dataset.sort !== undefined ? parseFloat(bc.dataset.sort) : bc.textContent.trim();
      if (typeof av === 'string') av = av.toLowerCase();
      if (typeof bv === 'string') bv = bv.toLowerCase();
      if (av < bv) return sortAsc ? -1 : 1;
      if (av > bv) return sortAsc ? 1 : -1;
      return 0;
    }});
    rows.forEach(r => {{
      const [, detail] = pairFor(r);
      tbody.appendChild(r);
      if (detail) tbody.appendChild(detail);
    }});
  }}

  table.querySelectorAll('thead th').forEach(th => {{
    th.addEventListener('click', () => {{
      const col = parseInt(th.dataset.col, 10);
      if (sortCol === col) sortAsc = !sortAsc; else {{ sortCol = col; sortAsc = true; }}
      sortRows();
    }});
  }});

  getTrialRows().forEach(row => {{
    row.style.cursor = 'pointer';
    row.addEventListener('click', () => {{
      const [, detail] = pairFor(row);
      if (!detail) return;
      detail.style.display = detail.style.display === 'none' ? '' : 'none';
    }});
  }});

  document.querySelectorAll('.filter-btn').forEach(btn => {{
    btn.addEventListener('click', () => {{
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const f = btn.dataset.filter;
      getTrialRows().forEach(row => {{
        const [, detail] = pairFor(row);
        const show = f === 'all' || row.dataset.state === f;
        row.style.display = show ? '' : 'none';
        if (detail && !show) detail.style.display = 'none';
      }});
    }});
  }});
}})();
</script>
</body>
</html>"""


def write_study_report(summary_path: Path) -> Path:
    summary_path = summary_path.resolve()
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    run_dir = Path(data.get("run_dir", summary_path.parent))
    out = run_dir / "study_report.html"
    out.write_text(render_study_report_html(data), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HTML study report from trial_summary.json")
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    out = write_study_report(args.summary)
    print(f"Wrote {out}")
    print(f"Open: file://{out.resolve()}")


if __name__ == "__main__":
    main()
