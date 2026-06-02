#!/usr/bin/env python3
"""Score rescue smoke JSONs for pass/fail heuristics from DPO_POLICY_RESCUE_PLAN.md."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")
BAD_TEMPLATE_RE = re.compile(r"(不是.+是|三样|不靠.+是)", re.IGNORECASE)


def score_file(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    issues: list[str] = []
    turns = 0
    for conv in data.get("conversations", []):
        for msg in conv.get("messages", []):
            if msg.get("role") != "assistant":
                continue
            turns += 1
            text = msg.get("content", "")
            if CJK_RE.search(text):
                issues.append(f"{conv.get('skeleton_id')}: CJK in assistant turn")
            if BAD_TEMPLATE_RE.search(text):
                issues.append(f"{conv.get('skeleton_id')}: bad template pattern")
    return {
        "file": str(path),
        "dpo_weight": data.get("dpo_weight"),
        "dpo_adapter": data.get("dpo_adapter"),
        "decode": data.get("generation", {}).get("decode"),
        "assistant_turns": turns,
        "pass": len(issues) == 0,
        "issues": issues[:10],
        "issue_count": len(issues),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    rows = [score_file(p) for p in args.paths]
    for r in rows:
        status = "PASS" if r["pass"] else "FAIL"
        w = r.get("dpo_weight", "?")
        print(f"{status}  w={w}  {Path(r['file']).name}  issues={r['issue_count']}")
        for issue in r["issues"]:
            print(f"    - {issue}")
    passed = sum(1 for r in rows if r["pass"])
    print(f"\n{passed}/{len(rows)} passed")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
