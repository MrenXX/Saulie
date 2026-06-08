#!/usr/bin/env python3
"""
Stream local McAuley meta_*.jsonl.gz files, filter/rank, write mccauley_products_500k.csv.
Requires: bash download_mccauley_meta.sh (at least one category for smoke tests).
"""
import gzip
import json
import os
import re
import subprocess
from pathlib import Path

import pandas as pd

META_DIR = Path(os.getenv("MCCAULEY_META_DIR", "/root/rag/mccauley_meta"))
OUT_CSV = Path(os.getenv("MCCAULEY_CSV", "/root/rag/mccauley_products_500k.csv"))

# Per-category caps (sum ~455k; buffer via quality filters)
CATEGORY_CAPS = {
    "Electronics": 40_000,
    "Cell_Phones_and_Accessories": 35_000,
    "Clothing_Shoes_and_Jewelry": 50_000,
    "Home_and_Kitchen": 50_000,
    "Sports_and_Outdoors": 35_000,
    "Beauty_and_Personal_Care": 30_000,
    "Health_and_Household": 30_000,
    "Toys_and_Games": 30_000,
    "Tools_and_Home_Improvement": 25_000,
    "Automotive": 25_000,
    "Pet_Supplies": 25_000,
    "Patio_Lawn_and_Garden": 25_000,
    "Baby_Products": 20_000,
    "Office_Products": 20_000,
    "All_Beauty": 15_000,
}

MIN_TITLE_LEN = 20
MIN_RATING_NUMBER = 50
MIN_AVG_RATING = 3.5
BRAND_ONLY_RE = re.compile(r"^[A-Za-z0-9 ]{2,12}$")


def parse_price(val) -> float:
    if val is None or val == "None" or val == "":
        return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def has_embed_text(row: dict) -> bool:
    features = row.get("features") or []
    desc = row.get("description") or []
    if isinstance(features, list) and len(features) > 0:
        return True
    if isinstance(desc, list) and len(" ".join(str(x) for x in desc).strip()) > 20:
        return True
    return False


def build_embed_text(row: dict) -> str:
    title = str(row.get("title", "")).strip()
    cat = str(row.get("main_category", "")).strip()
    parts = [f"{cat}: {title}" if cat else title]
    features = row.get("features") or []
    if features:
        parts.append(" | ".join(str(f) for f in features[:5]))
    desc = row.get("description") or []
    if desc:
        parts.append(" ".join(str(d) for d in desc)[:400])
    return " ".join(parts)[:2000]


def passes_quality(row: dict) -> bool:
    title = str(row.get("title", "")).strip()
    if len(title) < MIN_TITLE_LEN or title.endswith("..."):
        return False
    if BRAND_ONLY_RE.fullmatch(title):
        return False
    rn = int(row.get("rating_number") or 0)
    if rn < MIN_RATING_NUMBER:
        return False
    ar = float(row.get("average_rating") or 0)
    if ar < MIN_AVG_RATING:
        return False
    if parse_price(row.get("price")) <= 0:
        return False
    if not has_embed_text(row):
        return False
    if not str(row.get("parent_asin", "")).strip():
        return False
    return True


def sub_category(row: dict) -> str:
    cats = row.get("categories") or []
    if isinstance(cats, list) and cats:
        return str(cats[-1])
    return ""


def row_to_record(row: dict) -> dict:
    price = parse_price(row.get("price"))
    asin = str(row.get("parent_asin", "")).strip().upper()
    return {
        "parent_asin": asin,
        "name": str(row.get("title", "")).strip(),
        "main_category": str(row.get("main_category", "")).strip(),
        "sub_category": sub_category(row),
        "store": str(row.get("store") or ""),
        "ratings": float(row.get("average_rating") or 0),
        "no_of_ratings": int(row.get("rating_number") or 0),
        "actual_price": price,
        "discount_price": price,
        "embed_text": build_embed_text(row),
    }


def load_category(meta_path: Path, cap: int) -> list[dict]:
    candidates: list[dict] = []
    seen_asin: set[str] = set()

    with gzip.open(meta_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = str(row.get("parent_asin", "")).strip().upper()
            if not asin or asin in seen_asin:
                continue
            if not passes_quality(row):
                continue
            seen_asin.add(asin)
            candidates.append(row_to_record(row))

    candidates.sort(key=lambda r: (r["no_of_ratings"], r["ratings"]), reverse=True)
    selected = candidates[:cap]
    print(f"  {meta_path.name}: scanned → {len(candidates):,} passed → kept {len(selected):,}")
    return selected


def gzip_complete(path: Path) -> bool:
    """Skip partial downloads still being written by download_mccauley_meta.sh."""
    r = subprocess.run(["gzip", "-t", str(path)], capture_output=True)
    return r.returncode == 0


def main():
    if not META_DIR.exists():
        raise SystemExit(f"Meta dir not found: {META_DIR}. Run download_mccauley_meta.sh first.")

    all_rows: list[dict] = []
    for cat, cap in CATEGORY_CAPS.items():
        path = META_DIR / f"meta_{cat}.jsonl.gz"
        if not path.exists():
            print(f"[skip] missing {path.name}")
            continue
        if not gzip_complete(path):
            print(f"[skip] incomplete/corrupt {path.name} (download may still be running)")
            continue
        print(f"[*] Processing {cat} (cap {cap:,})")
        all_rows.extend(load_category(path, cap))

    if not all_rows:
        raise SystemExit("No products selected — download at least one meta category.")

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["parent_asin"], keep="first")
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[ok] Wrote {len(df):,} products → {OUT_CSV}")
    print(df["main_category"].value_counts().head(15).to_string())


if __name__ == "__main__":
    main()
