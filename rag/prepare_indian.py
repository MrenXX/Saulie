#!/usr/bin/env python3
"""
Clean the Indian Amazon CSV for hybrid indexing.

Fixes:
  - Drop truncated titles (...), junk 'stores' category, low-rated rows
  - Dedupe by parent_asin (not name) — keep highest review count
  - Parse comma-formatted no_of_ratings
  - Extract parent_asin from amazon.in link
  - Build category-enriched embed_text (not name-only)

Output: amazon_indian_clean.csv (default)
"""
import os
import re
from pathlib import Path

import pandas as pd

_RAG_ROOT = Path(os.getenv("RAG_ROOT", Path(__file__).resolve().parent))
IN_CSV = Path(os.getenv("INDIAN_CSV_IN", _RAG_ROOT / "Amazon-Products_fixed_deduped.csv"))
OUT_CSV = Path(os.getenv("INDIAN_CSV_OUT", _RAG_ROOT / "amazon_indian_clean.csv"))

MIN_TITLE_LEN = 15
MIN_RATINGS = 3.0
MIN_NO_OF_RATINGS = 10
ASIN_RE = re.compile(r"/dp/([A-Z0-9]{10})", re.I)
SKIP_CATEGORIES = {"stores"}


def parse_int(val) -> int:
    try:
        return int(str(val).replace(",", "").strip())
    except (TypeError, ValueError):
        return 0


def parse_float(val) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def asin_from_link(link: str) -> str:
    m = ASIN_RE.search(str(link or ""))
    return m.group(1).upper() if m else ""


def build_embed_text(row: dict) -> str:
    name = str(row.get("name", "")).strip()
    cat = str(row.get("main_category", "")).strip()
    sub = str(row.get("sub_category", "")).strip()
    parts = [f"{cat}: {name}" if cat else name]
    if sub and sub.lower() not in ("nan", cat.lower()):
        parts.append(f"Category: {sub}")
    return " ".join(parts)[:2000]


def passes_quality(row: pd.Series) -> bool:
    name = str(row.get("name", "")).strip()
    cat = str(row.get("main_category", "")).strip().lower()
    if len(name) < MIN_TITLE_LEN or name.endswith("..."):
        return False
    if cat in SKIP_CATEGORIES:
        return False
    if parse_float(row.get("ratings")) < MIN_RATINGS:
        return False
    if parse_int(row.get("no_of_ratings")) < MIN_NO_OF_RATINGS:
        return False
    if not asin_from_link(row.get("link", "")):
        return False
    return True


def row_to_record(row: pd.Series) -> dict:
    asin = asin_from_link(row.get("link", ""))
    actual = parse_float(row.get("actual_price"))
    discount = parse_float(row.get("discount_price"))
    rec = {
        "parent_asin": asin,
        "name": str(row.get("name", "")).strip(),
        "main_category": str(row.get("main_category", "")).strip(),
        "sub_category": str(row.get("sub_category", "")).strip(),
        "link": str(row.get("link", "")).strip(),
        "ratings": parse_float(row.get("ratings")),
        "no_of_ratings": parse_int(row.get("no_of_ratings")),
        "actual_price": actual,
        "discount_price": discount if discount > 0 else actual,
    }
    rec["embed_text"] = build_embed_text(rec)
    return rec


def main():
    if not IN_CSV.exists():
        raise SystemExit(f"Input not found: {IN_CSV}")

    print(f"Reading {IN_CSV} in chunks...")
    best: dict[str, dict] = {}
    scanned = filtered = 0

    for chunk in pd.read_csv(IN_CSV, chunksize=100_000, low_memory=False):
        for _, row in chunk.iterrows():
            scanned += 1
            if not passes_quality(row):
                filtered += 1
                continue
            rec = row_to_record(row)
            asin = rec["parent_asin"]
            prev = best.get(asin)
            if prev is None or rec["no_of_ratings"] > prev["no_of_ratings"]:
                best[asin] = rec

    if not best:
        raise SystemExit("No rows passed filters.")

    df = pd.DataFrame(best.values())
    df = df.sort_values("no_of_ratings", ascending=False)
    df.to_csv(OUT_CSV, index=False)

    print(f"\n[ok] Scanned {scanned:,}  filtered {filtered:,}  wrote {len(df):,} → {OUT_CSV}")
    print("\nTop categories:")
    print(df["main_category"].value_counts().head(12).to_string())
    trunc_left = df["name"].astype(str).str.endswith("...").sum()
    print(f"\nTruncated titles remaining: {trunc_left} (should be 0)")


if __name__ == "__main__":
    main()
