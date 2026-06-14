#!/usr/bin/env bash
# Download McAuley Amazon Reviews 2023 PRODUCT metadata (meta_*), one category at a time.
# Resume-friendly for slow/unreliable connections (WSL2).
#
# Usage:
#   bash download_mccauley_meta.sh              # all categories
#   bash download_mccauley_meta.sh All_Beauty   # single category (smoke test)
#
# Files land in $RAG_ROOT/mccauley_meta/meta_{Category}.jsonl.gz

set -euo pipefail

OUT="${MCCAULEY_META_DIR:-${RAG_ROOT:-/root/saulie/rag}/mccauley_meta}"
# Verified 2025: datarepo .jsonl.gz 404; mcauleylab .jsonl.gz works; HF resolve works as fallback
BASE_PRIMARY="https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories"
BASE_HF="https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories"

# 15 shopping categories (~500k products after filtering in prepare_mccauley.py)
CATEGORIES=(
  All_Beauty
  Electronics
  Cell_Phones_and_Accessories
  Clothing_Shoes_and_Jewelry
  Home_and_Kitchen
  Sports_and_Outdoors
  Beauty_and_Personal_Care
  Health_and_Household
  Toys_and_Games
  Tools_and_Home_Improvement
  Automotive
  Pet_Supplies
  Patio_Lawn_and_Garden
  Baby_Products
  Office_Products
)

mkdir -p "$OUT"

download_one() {
  local cat="$1"
  local dest="$OUT/meta_${cat}.jsonl.gz"
  local fname="meta_${cat}.jsonl.gz"

  if [[ -f "$dest" ]] && gzip -t "$dest" 2>/dev/null; then
    echo "[skip] $fname already complete ($(du -h "$dest" | cut -f1))"
    return 0
  fi

  echo "[*] Downloading $fname ..."
  local url_primary="$BASE_PRIMARY/$fname"
  local url_hf="$BASE_HF/${fname%.gz}"  # HF serves uncompressed .jsonl

  # Primary: mcauleylab gzip (resume-friendly)
  if curl --fail --location --retry 999 --retry-delay 5 --retry-all-errors \
       --continue-at - --connect-timeout 30 --max-time 0 \
       -o "$dest" "$url_primary"; then
    if gzip -t "$dest" 2>/dev/null; then
      echo "[ok] $fname from mcauleylab ($(du -h "$dest" | cut -f1))"
      return 0
    fi
    echo "[!] corrupt gzip from mcauleylab"
    rm -f "$dest"
  fi

  # Fallback: HuggingFace uncompressed .jsonl → gzip locally
  local tmp_jsonl="$OUT/${fname%.gz}"
  echo "[*] Trying HF fallback: ${fname%.gz}"
  if curl --fail --location --retry 999 --retry-delay 5 --retry-all-errors \
       --continue-at - --connect-timeout 30 --max-time 0 \
       -o "$tmp_jsonl" "$url_hf"; then
    gzip -c "$tmp_jsonl" > "$dest"
    rm -f "$tmp_jsonl"
    if gzip -t "$dest" 2>/dev/null; then
      echo "[ok] $fname from HF ($(du -h "$dest" | cut -f1))"
      return 0
    fi
    rm -f "$dest"
  fi
  rm -f "$tmp_jsonl"
  echo "[x] FAILED $fname — re-run script to resume"
  return 1
}

if [[ $# -gt 0 ]]; then
  download_one "$1"
else
  failed=0
  for cat in "${CATEGORIES[@]}"; do
    download_one "$cat" || failed=$((failed + 1))
  done
  echo "────────────────────────────────────────"
  if [[ $failed -eq 0 ]]; then
    echo "[ok] All categories downloaded to $OUT"
  else
    echo "[!] $failed category download(s) failed — re-run to resume"
    exit 1
  fi
fi
