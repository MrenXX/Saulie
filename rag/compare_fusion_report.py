#!/usr/bin/env python3
"""
Centralized RRF vs DBSF comparison across Indian and McAuley collections.

Scoring (18 queries, same keyword rubric for all runs):
  #1 relevant     — 1 if the top result clearly matches the query, else 0  (max 18)
  Relevant top-3  — count of clearly relevant hits in positions 1–3       (max 54)
"""
import json
from pathlib import Path

INDIAN_PATH = Path(__file__).parent / "fusion_benchmark_results.json"
MCCAULEY_PATH = Path(__file__).parent / "fusion_benchmark_results_v2.json"
REPORT_PATH = Path(__file__).parent / "fusion_comparison_report.md"

NUM_QUERIES = 18
MAX_TOP1 = NUM_QUERIES  # 18
MAX_TOP3 = NUM_QUERIES * 3  # 54

RUBRIC = {
    "wireless earbuds noise cancelling": {
        "keywords": ["earbud", "earphone", "headphone", "airpod", "buds"],
        "anti": ["sneaker", "chana", "chhole", "lentil", "shoe"],
    },
    "bluetooth speaker portable waterproof": {
        "keywords": ["speaker", "jbl", "bluetooth"],
        "anti": ["apron", "backpack", "bag"],
    },
    "gaming laptop RTX": {
        "keywords": ["laptop", "notebook", "gaming pc"],
        "anti": ["jeans", "keychain", "thermal", "motherboard", "ram", "graphics card"],
    },
    "32 inch smart TV 4K": {
        "keywords": ["tv", "television", "smart tv", "led tv"],
        "anti": [],
    },
    "men's running shoes lightweight": {
        "keywords": ["running", "shoe", "sneaker", "walking", "athletic"],
        "anti": ["football stud", "formal"],
    },
    "women's winter coat warm": {
        "keywords": ["coat", "jacket", "winter", "parka", "puffer"],
        "anti": ["kurti", "salwar", "lab coat", "palazzo", "socks"],
    },
    "cotton bed sheets king size": {
        "keywords": ["bedsheet", "bed sheet", "sheet", "bedding"],
        "anti": ["jewellery", "organiser", "mortar", "stone"],
    },
    "stainless steel cookware set": {
        "keywords": ["cookware", "stainless", "pot", "pan", "dinnerware", "flatware"],
        "anti": ["lunch box"],
    },
    "yoga mat non slip thick": {
        "keywords": ["yoga mat", "yogamat"],
        "anti": [],
    },
    "protein powder whey chocolate": {
        "keywords": ["whey", "protein powder", "protein"],
        "anti": [],
    },
    "baby diaper pants large pack": {
        "keywords": ["diaper", "pampers", "huggies"],
        "anti": [],
    },
    "dog food dry adult": {
        "keywords": ["dog food", "dry dog", "canine"],
        "anti": ["wet food", "cat food", "gravy"],
    },
    "car phone mount dashboard": {
        "keywords": ["mount", "holder", "phone mount", "dashboard"],
        "anti": ["a/c cleaner", "perfume", "charger"],
    },
    "mechanical keyboard RGB gaming": {
        "keywords": ["keyboard"],
        "anti": ["mouse", "mice"],
    },
    "men's formal leather belt": {
        "keywords": ["belt", "leather belt"],
        "anti": ["clutch", "wallet", "bag"],
    },
    "kids school backpack waterproof": {
        "keywords": ["backpack", "school bag", "bag"],
        "anti": ["raincoat"],
    },
    "air fryer large capacity": {
        "keywords": ["air fryer"],
        "anti": ["scale", "purifier", "deep fryer", "mixer"],
    },
    "face moisturizer dry skin": {
        "keywords": ["moistur", "cream", "lotion", "cetaphil"],
        "anti": ["sunscreen"],
    },
}


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def is_relevant(query: str, name: str) -> bool:
    rub = RUBRIC.get(query, {"keywords": [], "anti": []})
    n = name.lower()
    for anti in rub["anti"]:
        if anti in n:
            return False
    return any(kw in n for kw in rub["keywords"])


def top1_relevant(query: str, hits: list) -> int:
    if not hits:
        return 0
    return 1 if is_relevant(query, hits[0].get("name", "")) else 0


def count_relevant_top3(query: str, hits: list) -> int:
    return sum(1 for h in hits[:3] if is_relevant(query, h.get("name", "")))


def top1_name(hits: list) -> str:
    if not hits:
        return "(no results)"
    return (hits[0].get("name") or "")[:55]


def aggregate_block(data: dict, fusion: str) -> tuple[int, int]:
    top1_total = top3_total = 0
    for row in data[fusion]:
        q = row["query"]
        top1_total += top1_relevant(q, row["hits"])
        top3_total += count_relevant_top3(q, row["hits"])
    return top1_total, top3_total


def fusion_score(data: dict, fusion: str, query: str) -> tuple[int, int]:
    row = next(r for r in data[fusion] if r["query"] == query)
    return top1_relevant(query, row["hits"]), count_relevant_top3(query, row["hits"])


def best_fusion_label(data: dict) -> str:
    rrf1, rrf3 = aggregate_block(data, "rrf")
    dbsf1, dbsf3 = aggregate_block(data, "dbsf")
    if (rrf1, rrf3) > (dbsf1, dbsf3):
        return "RRF"
    if (dbsf1, dbsf3) > (rrf1, rrf3):
        return "DBSF"
    return "Tie"


def build_report() -> str:
    indian = load_results(INDIAN_PATH)
    mccauley = load_results(MCCAULEY_PATH)

    lines = [
        "# Fusion & Dataset Comparison Report",
        "",
        "## How scoring works",
        "",
        f"- **{NUM_QUERIES} test queries** (same intents; category filters differ per dataset).",
        f"- **#1 relevant (max {MAX_TOP1})** — For each query, **1** if the **first** result clearly matches "
        "the product intent (keyword rubric), **0** otherwise. Summed across all queries.",
        f"- **Relevant in top-3 (max {MAX_TOP3})** — For each query, count how many of the **3** results are "
        "clearly relevant (**0**, **1**, **2**, or **3**). Summed across all queries.",
        "- Both metrics use the **same** relevance rule; they are not weighted differently.",
        "- **Fusion winner** per dataset: higher #1 relevant wins; tie-break on relevant-in-top-3.",
        "",
        "## Aggregate scores",
        "",
        f"| Dataset | Fusion | #1 relevant (/{MAX_TOP1}) | Relevant in top-3 (/{MAX_TOP3}) |",
        "|---------|--------|-------------------------|-------------------------------|",
    ]

    aggregates = {}
    for label, data in [("Indian", indian), ("McAuley", mccauley)]:
        for fusion in ("rrf", "dbsf"):
            t1, t3 = aggregate_block(data, fusion)
            aggregates[(label, fusion)] = (t1, t3)
            lines.append(f"| {label} | {fusion.upper()} | {t1} | {t3} |")

    lines += [
        "",
        "## Per-query comparison",
        "",
        "| Query | Indian RRF #1 | Indian DBSF #1 | McAuley RRF #1 | McAuley DBSF #1 | Notes |",
        "|-------|---------------|----------------|----------------|-----------------|-------|",
    ]

    indian_rrf = {r["query"]: r for r in indian["rrf"]}
    indian_dbsf = {r["query"]: r for r in indian["dbsf"]}
    mc_rrf = {r["query"]: r for r in mccauley["rrf"]}
    mc_dbsf = {r["query"]: r for r in mccauley["dbsf"]}

    indian_wins = mccauley_wins = ties = 0

    def esc(s):
        return s.replace("|", "/")

    for query in RUBRIC:
        ir = indian_rrf[query]["hits"]
        id_ = indian_dbsf[query]["hits"]
        mr = mc_rrf[query]["hits"]
        md = mc_dbsf[query]["hits"]

        ind_best = max(
            (top1_relevant(query, ir), count_relevant_top3(query, ir)),
            (top1_relevant(query, id_), count_relevant_top3(query, id_)),
        )
        mc_best = max(
            (top1_relevant(query, mr), count_relevant_top3(query, mr)),
            (top1_relevant(query, md), count_relevant_top3(query, md)),
        )

        if ind_best > mc_best:
            indian_wins += 1
            note = "Indian better"
        elif mc_best > ind_best:
            mccauley_wins += 1
            note = "McAuley better"
        else:
            ties += 1
            note = "Tie"

        lines.append(
            f"| {query[:40]} | {esc(top1_name(ir))} | {esc(top1_name(id_))} | "
            f"{esc(top1_name(mr))} | {esc(top1_name(md))} | {note} |"
        )

    best_indian = best_fusion_label(indian)
    best_mc = best_fusion_label(mccauley)

    irrf1, irrf3 = aggregates[("Indian", "rrf")]
    idbsf1, idbsf3 = aggregates[("Indian", "dbsf")]
    mrrf1, mrrf3 = aggregates[("McAuley", "rrf")]
    mdbsf1, mdbsf3 = aggregates[("McAuley", "dbsf")]

    indian_best = max((irrf1, irrf3), (idbsf1, idbsf3))
    mc_best = max((mrrf1, mrrf3), (mdbsf1, mdbsf3))
    best_dataset = "McAuley" if mc_best > indian_best else ("Indian" if indian_best > mc_best else "Tie")

    rec_fusion = best_mc.lower() if best_dataset == "McAuley" else best_indian.lower()
    if rec_fusion == "tie":
        rec_fusion = "rrf"

    lines += [
        "",
        "## Verdict",
        "",
        f"- **Per-query dataset wins:** Indian {indian_wins}, McAuley {mccauley_wins}, ties {ties}",
        f"- **Best fusion on Indian:** {best_indian} "
        f"(RRF: {irrf1}/{MAX_TOP1} #1, {irrf3}/{MAX_TOP3} top-3 | "
        f"DBSF: {idbsf1}/{MAX_TOP1} #1, {idbsf3}/{MAX_TOP3} top-3)",
        f"- **Best fusion on McAuley:** {best_mc} "
        f"(RRF: {mrrf1}/{MAX_TOP1} #1, {mrrf3}/{MAX_TOP3} top-3 | "
        f"DBSF: {mdbsf1}/{MAX_TOP1} #1, {mdbsf3}/{MAX_TOP3} top-3)",
        f"- **Overall dataset winner:** {best_dataset}",
        "",
        "### Recommended defaults",
        "",
        f"- `QDRANT_COLLECTION=amazon_products_v2` when using McAuley",
        f"- `QDRANT_COLLECTION=amazon_products` when using Indian CSV",
        f"- `FUSION_METHOD={rec_fusion}`",
    ]

    return "\n".join(lines) + "\n"


def main():
    report = build_report()
    REPORT_PATH.write_text(report)
    print(report)
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
