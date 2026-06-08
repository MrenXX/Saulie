#!/usr/bin/env python3
"""
Centralized RRF vs DBSF comparison across Indian and McAuley collections.
Loads fusion_benchmark_results.json (Indian) and fusion_benchmark_results_v2.json (McAuley).
"""
import json
import re
from pathlib import Path

INDIAN_PATH = Path("/root/rag/fusion_benchmark_results.json")
MCCAULEY_PATH = Path("/root/rag/fusion_benchmark_results_v2.json")
REPORT_PATH = Path("/root/rag/fusion_comparison_report.md")

# Manual relevance rubric: (top1_grade, top3_good_count, rrf_top1_short, dbsf_top1_short, note)
# Grades: G=good P=partial B=bad
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
        "anti": ["jeans", "keychain", "thermal", "motherboard", "ram"],
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
        "anti": ["kurti", "salwar", "lab coat", "palazzo"],
    },
    "cotton bed sheets king size": {
        "keywords": ["bedsheet", "bed sheet", "sheet", "bedding"],
        "anti": ["jewellery", "organiser", "mortar", "stone"],
    },
"stainless steel cookware set": {
        "keywords": ["cookware", "stainless", "pot", "pan", "dinnerware"],
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


def hit_text(hit: dict) -> str:
    return (hit.get("name") or "").lower()


def score_hit(query: str, name: str) -> str:
    rub = RUBRIC.get(query, {"keywords": [], "anti": []})
    n = name.lower()
    for anti in rub["anti"]:
        if anti in n:
            return "B"
    for kw in rub["keywords"]:
        if kw in n:
            return "G"
    return "P" if rub["keywords"] else "B"


def grade_top1(query: str, hits: list) -> str:
    if not hits:
        return "B"
    return score_hit(query, hits[0].get("name", ""))


def count_good_top3(query: str, hits: list) -> int:
    return sum(1 for h in hits[:3] if score_hit(query, h.get("name", "")) == "G")


def top1_name(hits: list) -> str:
    if not hits:
        return "(no results)"
    return (hits[0].get("name") or "")[:55]


def aggregate_block(data: dict, fusion: str) -> tuple[int, int]:
    score_map = {"G": 2, "P": 1, "B": 0}
    top1_total = top3_total = 0
    for row in data[fusion]:
        g = grade_top1(row["query"], row["hits"])
        top1_total += score_map[g]
        top3_total += count_good_top3(row["query"], row["hits"])
    return top1_total, top3_total


def build_report() -> str:
    indian = load_results(INDIAN_PATH)
    mccauley = load_results(MCCAULEY_PATH)

    lines = [
        "# Fusion & Dataset Comparison Report",
        "",
        "## Aggregate scores (18 queries)",
        "",
        "| Dataset | Fusion | Top-1 score (max 36) | Top-3 good hits (max 54) |",
        "|---------|--------|----------------------|--------------------------|",
    ]

    for label, data in [("Indian", indian), ("McAuley", mccauley)]:
        for fusion in ("rrf", "dbsf"):
            t1, t3 = aggregate_block(data, fusion)
            lines.append(f"| {label} | {fusion.upper()} | {t1} | {t3} |")

    lines += [
        "",
        "_Top-1: G=2, P=1, B=0. Top-3: count of clearly relevant products (keyword rubric)._",
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
    rrf_wins_indian = dbsf_wins_indian = 0
    rrf_wins_mc = dbsf_wins_mc = 0

    for query in RUBRIC:
        ir = indian_rrf[query]["hits"]
        id_ = indian_dbsf[query]["hits"]
        mr = mc_rrf[query]["hits"]
        md = mc_dbsf[query]["hits"]

        best_indian = max(
            grade_top1(query, ir) + str(count_good_top3(query, ir)),
            grade_top1(query, id_) + str(count_good_top3(query, id_)),
        )
        best_mc = max(
            grade_top1(query, mr) + str(count_good_top3(query, mr)),
            grade_top1(query, md) + str(count_good_top3(query, md)),
        )

        gi = grade_top1(query, ir)
        gd_i = grade_top1(query, id_)
        gm_r = grade_top1(query, mr)
        gm_d = grade_top1(query, md)

        score_map = {"G": 2, "P": 1, "B": 0}
        ind_score = max(score_map[gi] + count_good_top3(query, ir), score_map[gd_i] + count_good_top3(query, id_))
        mc_score = max(score_map[gm_r] + count_good_top3(query, mr), score_map[gm_d] + count_good_top3(query, md))

        if ind_score > mc_score:
            indian_wins += 1
            note = "Indian better"
        elif mc_score > ind_score:
            mccauley_wins += 1
            note = "McAuley better"
        else:
            ties += 1
            note = "Tie"

        if score_map[gi] + count_good_top3(query, ir) > score_map[gd_i] + count_good_top3(query, id_):
            rrf_wins_indian += 1
        elif score_map[gd_i] + count_good_top3(query, id_) > score_map[gi] + count_good_top3(query, ir):
            dbsf_wins_indian += 1

        if score_map[gm_r] + count_good_top3(query, mr) > score_map[gm_d] + count_good_top3(query, md):
            rrf_wins_mc += 1
        elif score_map[gm_d] + count_good_top3(query, md) > score_map[gm_r] + count_good_top3(query, mr):
            dbsf_wins_mc += 1

        def esc(s):
            return s.replace("|", "/")

        lines.append(
            f"| {query[:40]} | {esc(top1_name(ir))} | {esc(top1_name(id_))} | "
            f"{esc(top1_name(mr))} | {esc(top1_name(md))} | {note} |"
        )

    irrf_t1, irrf_t3 = aggregate_block(indian, "rrf")
    idbsf_t1, idbsf_t3 = aggregate_block(indian, "dbsf")
    mrrf_t1, mrrf_t3 = aggregate_block(mccauley, "rrf")
    mdbsf_t1, mdbsf_t3 = aggregate_block(mccauley, "dbsf")

    best_indian_fusion = "DBSF" if idbsf_t1 > irrf_t1 else ("RRF" if irrf_t1 > idbsf_t1 else "Tie")
    best_mc_fusion = "DBSF" if mdbsf_t1 > mrrf_t1 else ("RRF" if mrrf_t1 > mdbsf_t1 else "Tie")

    indian_total = max(irrf_t1 + irrf_t3, idbsf_t1 + idbsf_t3)
    mc_total = max(mrrf_t1 + mrrf_t3, mdbsf_t1 + mdbsf_t3)
    best_dataset = "McAuley" if mc_total > indian_total else ("Indian" if indian_total > mc_total else "Tie")

    lines += [
        "",
        "## Verdict",
        "",
        f"- **Per-query dataset wins:** Indian {indian_wins}, McAuley {mccauley_wins}, ties {ties}",
        f"- **Best fusion on Indian:** {best_indian_fusion} (RRF top-1={irrf_t1}, DBSF top-1={idbsf_t1})",
        f"- **Best fusion on McAuley:** {best_mc_fusion} (RRF top-1={mrrf_t1}, DBSF top-1={mdbsf_t1})",
        f"- **Overall dataset winner:** {best_dataset}",
        "",
        "### Recommended defaults",
        "",
        f"- `QDRANT_COLLECTION=amazon_products_v2` if using McAuley ({best_mc_fusion})",
        f"- `QDRANT_COLLECTION=amazon_products` if keeping Indian ({best_indian_fusion})",
        f"- `FUSION_METHOD={best_mc_fusion.lower() if best_dataset == 'McAuley' else best_indian_fusion.lower()}`",
    ]

    return "\n".join(lines) + "\n"


def main():
    report = build_report()
    REPORT_PATH.write_text(report)
    print(report)
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
