"""Check public documentation and saved evidence without GPU dependencies."""

import collections
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def jsonl(path):
    return [json.loads(line) for line in (ROOT / path).read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path):
    # Match Git's LF representation in older Windows checkouts as well.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def headings(path):
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"^```.*?^```[^\n]*", "", text, flags=re.M | re.S)
    result, counts = set(), collections.Counter()
    for heading in re.findall(r"^#{1,6}\s+(.+)$", text, flags=re.M):
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        suffix = f"-{counts[slug]}" if counts[slug] else ""
        result.add(slug + suffix)
        counts[slug] += 1
    return result


def main():
    errors = []
    links = 0
    docs = sorted(p for p in ROOT.rglob("*.md") if ".git" not in p.parts)
    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        text = re.sub(r"^```.*?^```[^\n]*", "", text, flags=re.M | re.S)
        for target in re.findall(r"\[[^\]\n]+\]\(([^)\n]+)\)", text):
            target = target.strip().split(' "', 1)[0].strip("<>")
            parts = urlsplit(target)
            if parts.scheme or parts.netloc:
                continue
            path = (doc.parent / unquote(parts.path)).resolve() if parts.path else doc
            links += 1
            if not path.exists():
                errors.append(f"{doc.relative_to(ROOT)}: missing {target}")
            elif parts.fragment and path.suffix == ".md" and unquote(parts.fragment) not in headings(path):
                errors.append(f"{doc.relative_to(ROOT)}: missing anchor {target}")

    for item in load("docs/evidence/export-manifest.json")["files"]:
        path = ROOT / item["published_path"]
        if not path.is_file() or digest(path) != item["sha256"]:
            errors.append(f"Export hash mismatch: {item['published_path']}")

    data_path = "dpo/dataset/DPO_522_prompt_a_and_prompt_b_V4_repaired.jsonl"
    split_path = "dpo/train/dataset/dpo_v4_split_seed_42.jsonl"
    data, split = jsonl(data_path), jsonl(split_path)
    checks = {
        "522 unique preference IDs": len(data) == len({r["id"] for r in data}) == 522,
        "source counts": collections.Counter(r["dpo_source"] for r in data) == {"prompt_a": 330, "prompt_b_repaired": 92, "prompt_b_exp500": 100},
        "category counts": collections.Counter(r["category"] for r in data) == {"style": 220, "steering": 121, "product_fit": 181},
        "209 branch-local user rows": sum(any(m["role"] == "user" for k in ("chosen", "rejected") for m in r[k]) for r in data) == 209,
        "split covers each ID once": len(split) == len({r["id"] for r in split}) == 522 and {r["id"] for r in split} == {r["id"] for r in data},
        "split sizes": collections.Counter(r["split"] for r in split) == {"train": 418, "val": 52, "test": 52},
        "split seed": all(r["split_seed"] == 42 for r in split),
    }
    summary = load("dpo/study_results/trial_summary_v1.5.json")
    checks["DPO data matches training hash"] = digest(ROOT / data_path).startswith(summary["dataset_hash"])
    checks["DPO split matches training hash"] = digest(ROOT / split_path) == summary["split_manifest_sha256"]
    checks["DPO trial accounting"] = len(summary["trials"]) == 31 and len(summary["complete_trials"]) == 21 and summary["unique_complete_config_count"] == 11
    checks["DPO unique configurations"] = len({json.dumps(r["params"], sort_keys=True) for r in summary["complete_trials"]}) == 11
    checks["DPO objective winner"] = summary["best_trial"] == 16
    for version, size, best in [("v1.1", 20, 16), ("v1.2", 7, 13)]:
        trials = load(f"docs/evidence/sft-study-{version}.json")
        checks[f"SFT {version} ranking"] = len(trials) == size and min(trials, key=lambda r: r["eval_loss"])["trial"] == best
    sft = load("docs/evidence/sft-conversations.json")
    checks["SFT comparison coverage"] = len(sft) == 11 and all(len(m["conversations"]) == 14 for m in sft.values())
    for path, models, scenarios in [("dpo/eval/generations_round1.json", 7, 34), ("dpo/eval/generations_prod_ladder.json", 3, 60)]:
        result = load(path)
        checks[path + " coverage"] = result["checkpoint_complete"] is True and len(result["models"]) == models and all(len(m["conversations"]) == scenarios for m in result["models"].values())
    for label, passed in checks.items():
        if not passed:
            errors.append("Evidence check failed: " + label)
    if errors:
        print("\n".join(errors))
        return 1
    print(f"OK: {len(docs)} Markdown files, {links} local links, 5 export hashes, {len(checks)} evidence checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
