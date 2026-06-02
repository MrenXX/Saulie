#!/usr/bin/env python3
"""Hard-fail heuristics for Plan B Part 1 JSONL (extends score_rescue_smoke patterns)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")
BAD_TEMPLATE_RE = re.compile(r"(不是.+是|三样|不靠.+是)", re.IGNORECASE)
WORD_RE = re.compile(r"[a-zA-Z']{3,}")


def _ngram_jaccard(a: str, b: str, n: int = 4) -> float:
    def grams(text: str) -> set[str]:
        words = WORD_RE.findall(text.lower())
        if len(words) < n:
            return {" ".join(words)} if words else set()
        return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}

    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def score_jsonl(path: Path, *, repeat_threshold: float = 0.72) -> dict:
    issues: list[str] = []
    by_trial_skeleton: dict[str, list[str]] = defaultdict(list)
    rows = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows += 1
            row = json.loads(line)
            sk = row.get("skeleton_id", "?")
            trial = row.get("trial", "?")
            key = f"{trial}|{sk}"
            prev_assistant: list[str] = []
            for msg in row.get("messages", []):
                if msg.get("role") != "assistant":
                    continue
                text = (msg.get("content") or "").strip()
                if not text:
                    issues.append(f"{trial}/{sk}: empty assistant turn")
                    continue
                if CJK_RE.search(text):
                    issues.append(f"{trial}/{sk}: CJK in assistant turn")
                if BAD_TEMPLATE_RE.search(text):
                    issues.append(f"{trial}/{sk}: bad template pattern")
                if len(text) > 2800:
                    issues.append(f"{trial}/{sk}: extreme length ({len(text)} chars)")
                for prev in prev_assistant:
                    j = _ngram_jaccard(prev, text)
                    if j >= repeat_threshold:
                        issues.append(
                            f"{trial}/{sk}: repeated assistant turn (jaccard={j:.2f})"
                        )
                        break
                prev_assistant.append(text)
            by_trial_skeleton[trial].append(sk)

    trial_pass: dict[str, bool] = {}
    for trial, skels in by_trial_skeleton.items():
        trial_issues = [i for i in issues if i.startswith(f"{trial}/")]
        trial_pass[trial] = len(trial_issues) == 0

    return {
        "file": str(path),
        "rows": rows,
        "trials": sorted(by_trial_skeleton.keys()),
        "pass": len(issues) == 0,
        "issues": issues[:30],
        "issue_count": len(issues),
        "trial_pass": trial_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--repeat-threshold", type=float, default=0.72)
    args = parser.parse_args()
    r = score_jsonl(args.jsonl, repeat_threshold=args.repeat_threshold)
    print(f"{'PASS' if r['pass'] else 'FAIL'}  {args.jsonl.name}  issues={r['issue_count']}  rows={r['rows']}")
    for trial, ok in sorted(r["trial_pass"].items()):
        print(f"  {trial}: {'PASS' if ok else 'FAIL'}")
    for issue in r["issues"]:
        print(f"    - {issue}")
    sys.exit(0 if r["pass"] else 1)


if __name__ == "__main__":
    main()
