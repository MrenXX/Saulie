#!/usr/bin/env python3
"""Build MODEL_PROMPT_MATRIX_COMPARISON.md from matrix_runs/*.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_runs(runs_dir: Path) -> list[dict]:
    runs = []
    for path in sorted(runs_dir.glob("*.json")):
        runs.append(json.loads(path.read_text(encoding="utf-8")))
    return runs


def english_score(run: dict) -> str:
    """Heuristic label from hallucination + listening failures."""
    halluc = run.get("summary", {}).get("hallucination_turns") or []
    if len(halluc) >= 2:
        return "poor"
    if len(halluc) == 1:
        return "mixed"
    # scan for obvious ignore-user patterns in kitchen scenario
    for sc in run.get("scenarios", []):
        if sc.get("scenario") != "kitchen":
            continue
        for t in sc.get("turns", []):
            if t.get("turn") == 3 and "prep" in (t.get("assistant") or "").lower():
                if "cleanup" in (t.get("user") or "").lower():
                    return "mixed"
    return "ok"


def tool_label(run: dict) -> str:
    s = run.get("summary", {})
    if s.get("any_tool_executed"):
        return "**EXECUTED**"
    if s.get("any_tool_emitted"):
        return "emitted (not executed)"
    return "none"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = load_runs(args.runs_dir)
    if not runs:
        args.output.write_text("# Matrix comparison\n\nNo runs found.\n", encoding="utf-8")
        return

    tool_cells = [r for r in runs if r.get("summary", {}).get("any_tool_emitted")]
    exec_cells = [r for r in runs if r.get("summary", {}).get("any_tool_executed")]

    lines = [
        "# Model × Prompt Matrix Comparison",
        "",
        "Focus: **tool call emission/execution** (broken RAG results are OK for this pass) and **English clarity**.",
        "",
        "## Tool calling summary",
        "",
        f"- Cells with **any tool emitted** ({len(tool_cells)}/9): "
        + (", ".join(f"`{r['model_key']}/{r['prompt']}`" for r in tool_cells) if tool_cells else "none"),
        "",
        f"- Cells with **tool executed** ({len(exec_cells)}/9): "
        + (", ".join(f"`{r['model_key']}/{r['prompt']}`" for r in exec_cells) if exec_cells else "none"),
        "",
        "## Full grid",
        "",
        "| Model | Prompt | Tool | English | Emit turns | Exec turns |",
        "|-------|--------|------|---------|------------|------------|",
    ]

    for r in runs:
        s = r.get("summary", {})
        lines.append(
            f"| {r.get('model_key')} | {r.get('prompt')} | {tool_label(r)} | {english_score(r)} | "
            f"{', '.join(s.get('tool_emit_turns') or []) or '—'} | "
            f"{', '.join(s.get('tool_exec_turns') or []) or '—'} |"
        )

    lines.extend(["", "## Cells that called the tool (detail)", ""])

    if not tool_cells:
        lines.append("**No configuration emitted a tool call in any scenario.**")
    else:
        for r in tool_cells:
            lines.append(f"### `{r['model_key']}` + `{r['prompt']}`")
            lines.append("")
            for sc in r.get("scenarios", []):
                for t in sc.get("turns", []):
                    if t.get("tool_emitted") or t.get("tool_executed"):
                        flag = "EXEC" if t.get("tool_executed") else "EMIT"
                        lines.append(
                            f"- **{sc.get('scenario')} turn {t.get('turn')}** [{flag}] "
                            f"user: *{t.get('user', '')[:60]}*"
                        )
                        preview = (t.get("assistant") or "")[:200].replace("\n", " ")
                        lines.append(f"  - assistant: {preview}")
            lines.append("")

    lines.extend(["", "## English / hallucination notes", ""])
    for r in runs:
        hall = r.get("summary", {}).get("hallucination_turns") or []
        if hall:
            lines.append(
                f"- `{r['model_key']}/{r['prompt']}`: likely invented product/spec at {', '.join(hall)}"
            )

    lines.extend(["", "## Recommendation placeholder", ""])
    if exec_cells:
        best = exec_cells[0]
        lines.append(
            f"Start deploy experiments with **`{best['model_key']}` + `{best['prompt']}`** "
            f"(tool executed in: {', '.join(best['summary'].get('tool_exec_turns') or [])})."
        )
    elif tool_cells:
        best = tool_cells[0]
        lines.append(
            f"Only emission, no execution: try **`{best['model_key']}` + `{best['prompt']}`** — "
            "fix RAG/harness next."
        )
    else:
        lines.append(
            "No tool calls in any cell. Next: restore forced tool_choice harness or SFT tool-call SFT pass."
        )

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
