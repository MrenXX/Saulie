#!/usr/bin/env python3
"""Merge latest vLLM/RAG/E2E latency runs into LATENCY_REPORT.md with SLO pass/fail."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from dpo.eval.latency_stats import check_targets
RUNS_DIR = Path(__file__).resolve().parent / "latency_runs"
TARGETS_PATH = Path(__file__).resolve().parent / "latency_targets.json"
REPORT_PATH = Path(__file__).resolve().parent / "LATENCY_REPORT.md"


def latest_glob(pattern: str) -> Path | None:
    files = sorted(RUNS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_json(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_summary_block(title: str, summary: dict) -> list[str]:
    lines = [f"### {title}", ""]
    if not summary:
        lines.append("_No data._")
        lines.append("")
        return lines
    for key, val in summary.items():
        if isinstance(val, dict) and "n" in val:
            if "p50_tps" in val:
                lines.append(
                    f"- **{key}**: n={val['n']} mean={val.get('mean_tps')} "
                    f"p50={val.get('p50_tps')} p95={val.get('p95_tps')} tok/s"
                )
            elif "rate" in val:
                lines.append(f"- **{key}**: {val}")
            else:
                lines.append(
                    f"- **{key}**: n={val['n']} mean={val.get('mean_ms')}ms "
                    f"p50={val.get('p50_ms')}ms p95={val.get('p95_ms')}ms"
                )
        else:
            lines.append(f"- **{key}**: {val}")
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm", type=Path, default=None)
    parser.add_argument("--rag", type=Path, default=None)
    parser.add_argument("--e2e", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    vllm = load_json(args.vllm or latest_glob("vllm_latency_*.json"))
    rag = load_json(args.rag or latest_glob("rag_latency_*.json"))
    e2e = load_json(args.e2e or latest_glob("prod_e2e_latency_*.json"))
    targets = json.loads(TARGETS_PATH.read_text(encoding="utf-8"))

    actual_for_checks = {
        "ttft_short_prompt": (vllm or {}).get("summary", {}).get("ttft_short_prompt"),
        "ttft_long_prompt": (vllm or {}).get("summary", {}).get("ttft_long_prompt"),
        "decode_tokens_per_s": (vllm or {}).get("summary", {}).get("decode_tokens_per_s"),
        "embed_ms": (rag or {}).get("summary", {}).get("embed_ms"),
        "qdrant_ms": (rag or {}).get("summary", {}).get("qdrant_ms"),
        "search_total_ms": (rag or {}).get("summary", {}).get("search_total_ms"),
        "probe_turn_ms": (e2e or {}).get("summary", {}).get("probe_turn_ms"),
        "tool_turn_total_ms": (e2e or {}).get("summary", {}).get("tool_turn_total_ms"),
        "tool_rag_ms": (e2e or {}).get("summary", {}).get("tool_rag_ms"),
    }
    tool_hit = (e2e or {}).get("summary", {}).get("tool_hit_rate", {})
    if tool_hit.get("rate") is not None:
        actual_for_checks["tool_hit_rate"] = {"rate": tool_hit["rate"]}

    slo_rows = []
    for section, specs in targets.items():
        if section == "description":
            continue
        if not isinstance(specs, dict):
            continue
        slo_rows.extend(check_targets(actual_for_checks, specs))

    lines = [
        "# Saulie Latency Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Source runs",
        "",
        f"- vLLM: `{vllm and (args.vllm or latest_glob('vllm_latency_*.json'))}`",
        f"- RAG: `{rag and (args.rag or latest_glob('rag_latency_*.json'))}`",
        f"- E2E: `{e2e and (args.e2e or latest_glob('prod_e2e_latency_*.json'))}`",
        "",
        "## SLO pass/fail",
        "",
        "| Metric | Status | Actual | Threshold |",
        "|--------|--------|--------|-----------|",
    ]
    for row in slo_rows:
        actual = row.get("actual")
        thresh = row.get("threshold", row.get("target", {}).get("value") if isinstance(row.get("target"), dict) else None)
        unit = row.get("unit", "")
        lines.append(
            f"| {row['metric']} | {row['status']} | {actual}{unit} | {thresh}{unit} |"
        )
    lines.append("")
    lines.append("## vLLM (isolated)")
    lines.append("")
    lines.extend(fmt_summary_block("Summary", (vllm or {}).get("summary")))
    lines.append("## RAG (isolated)")
    lines.append("")
    lines.extend(fmt_summary_block("Summary", (rag or {}).get("summary")))
    lines.append("## Prod E2E (agent API)")
    lines.append("")
    lines.extend(fmt_summary_block("Summary", (e2e or {}).get("summary")))

    if e2e:
        lines.append("#### E2E tool-turn decomposition (p50/p95)")
        lines.append("")
        s = e2e["summary"]
        for key in ("tool_llm1_ms", "tool_rag_ms", "tool_llm2_ms", "tool_turn_total_ms"):
            v = s.get(key, {})
            lines.append(f"- {key}: p50={v.get('p50_ms')}ms p95={v.get('p95_ms')}ms")
        lines.append("")

    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
