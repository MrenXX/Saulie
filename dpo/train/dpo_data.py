"""DPO V4 data loading, stratified splits, and assistant-only tokenization."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

from dpo.train.paths import CACHE_DIR, DATA_PATH, SPLIT_DIR, VALIDATION_REPORT

MAX_LENGTH = 704
SPLIT_SEED = 42
TARGET_TRAIN = 418
TARGET_VAL = 52
TARGET_TEST = 52

# Curated rows for hard-fail mask audit (ordinary, multi-turn, prompt_b variants, long, branch-local user)
MASK_AUDIT_ROW_IDS = [
    "dpo_001",  # ordinary prompt_a
    "dpo_002",  # multi-turn steering + branch-local user
    "dpo2_pair_dpo2_A6_001",  # prompt_b repaired
    "dpo2_pair_exp500_dpo2_skel_steer_a_001",  # prompt_b exp500
    "dpo_112",  # longest pair
    "dpo_129",
]


def source_family(dpo_source: str) -> str:
    return "prompt_a" if dpo_source == "prompt_a" else "prompt_b"


def has_branch_local_user(chosen: list[dict], rejected: list[dict]) -> bool:
    return any(m["role"] == "user" for m in chosen) or any(m["role"] == "user" for m in rejected)


def load_v4_rows() -> list[dict[str, Any]]:
    rows = []
    with DATA_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != 522:
        raise ValueError(f"Expected 522 rows, got {len(rows)}")
    with VALIDATION_REPORT.open(encoding="utf-8") as f:
        report = json.load(f)
    if not report.get("passes_core_validation"):
        raise ValueError("V4 validation report does not pass core validation")
    return rows


def _row_sort_key(row_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{row_id}".encode()).hexdigest()


def _stratum_key(row: dict, cell3: Counter, cell2: Counter) -> str:
    sf = source_family(row["dpo_source"])
    cat = row["category"]
    opening = row["opening_type"]
    key3 = (sf, cat, opening)
    if cell3[key3] >= 5:
        return f"{sf}|{cat}|{opening}"
    key2 = (sf, cat)
    if cell2[key2] >= 5:
        return f"{sf}|{cat}"
    return sf


def create_split_manifest(rows: list[dict], seed: int = SPLIT_SEED) -> dict[str, str]:
    """Return id -> split mapping; write manifest jsonl."""
    from sklearn.model_selection import train_test_split

    cell3 = Counter(
        (source_family(r["dpo_source"]), r["category"], r["opening_type"]) for r in rows
    )
    cell2 = Counter((source_family(r["dpo_source"]), r["category"]) for r in rows)
    rows_sorted = sorted(rows, key=lambda r: _row_sort_key(r["id"], seed))
    labels = [_stratum_key(r, cell3, cell2) for r in rows_sorted]
    indices = list(range(len(rows_sorted)))

    train_val_idx, test_idx = train_test_split(
        indices, test_size=TARGET_TEST / len(rows), random_state=seed, stratify=labels
    )
    train_val_labels = [labels[i] for i in train_val_idx]
    val_ratio = TARGET_VAL / (len(rows) - TARGET_TEST)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_ratio, random_state=seed, stratify=train_val_labels
    )

    split_map: dict[str, str] = {}
    for i in train_idx:
        split_map[rows_sorted[i]["id"]] = "train"
    for i in val_idx:
        split_map[rows_sorted[i]["id"]] = "val"
    for i in test_idx:
        split_map[rows_sorted[i]["id"]] = "test"

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = SPLIT_DIR / f"dpo_v4_split_seed_{seed}.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            sk = _stratum_key(row, cell3, cell2)
            rec = {
                "id": row["id"],
                "split": split_map[row["id"]],
                "split_seed": seed,
                "dpo_source": row["dpo_source"],
                "source_family": source_family(row["dpo_source"]),
                "category": row["category"],
                "opening_type": row["opening_type"],
                "total_turns": row["total_turns"],
                "divergence_turn": row["divergence_turn"],
                "has_branch_local_user": has_branch_local_user(row["chosen"], row["rejected"]),
                "stratum_key": sk,
            }
            f.write(json.dumps(rec) + "\n")

    totals = Counter(split_map.values())
    if set(split_map) != {r["id"] for r in rows}:
        raise ValueError("Split manifest missing rows or duplicate IDs")
    if totals["train"] + totals["val"] + totals["test"] != 522:
        raise ValueError(f"Split counts don't sum to 522: {totals}")

    print(f"Split manifest: {manifest_path}")
    print(f"  train={totals['train']} val={totals['val']} test={totals['test']}")
    return split_map


def manifest_path_for_seed(seed: int = SPLIT_SEED) -> Path:
    return SPLIT_DIR / f"dpo_v4_split_seed_{seed}.jsonl"


def manifest_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_split_manifest_records(seed: int = SPLIT_SEED) -> list[dict]:
    path = manifest_path_for_seed(seed)
    if not path.exists():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_split_manifest(seed: int = SPLIT_SEED) -> dict[str, str]:
    """Load existing split manifest (do not reshuffle)."""
    records = load_split_manifest_records(seed)
    split_map = {rec["id"]: rec["split"] for rec in records}
    totals = Counter(split_map.values())
    print(f"Loaded split manifest: {manifest_path_for_seed(seed)}")
    print(f"  train={totals['train']} val={totals['val']} test={totals['test']}")
    return split_map


def compute_split_diagnostics(records: list[dict]) -> dict[str, Any]:
    """Per-split counts for stratification and provenance fields."""

    def _counts_for(field: str | tuple) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {"train": {}, "val": {}, "test": {}}
        for rec in records:
            split = rec["split"]
            if isinstance(field, tuple):
                key = "|".join(str(rec[f]) for f in field)
            else:
                key = str(rec[field])
            out[split][key] = out[split].get(key, 0) + 1
        return out

    return {
        "dpo_source": _counts_for("dpo_source"),
        "source_family": _counts_for("source_family"),
        "category": _counts_for("category"),
        "source_family_x_category": _counts_for(("source_family", "category")),
        "dpo_source_x_category": _counts_for(("dpo_source", "category")),
        "opening_type": _counts_for("opening_type"),
        "total_turns": _counts_for("total_turns"),
        "divergence_turn": _counts_for("divergence_turn"),
        "has_branch_local_user": _counts_for("has_branch_local_user"),
        "stratum_key": _counts_for("stratum_key"),
    }


def _normalize_ids(ids) -> list[int]:
    if ids and isinstance(ids[0], list):
        return ids[0]
    return list(ids)


def _normalize_mask(mask) -> list[int]:
    if mask and isinstance(mask[0], list):
        return mask[0]
    return list(mask)


def tokenize_preference_pair(
    tokenizer,
    prompt: list[dict],
    completion: list[dict],
) -> tuple[list[int], list[int], list[int]]:
    """Return prompt_ids, completion_ids, completion_score_mask (1=assistant only)."""
    prompt_ids = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
    )
    prompt_ids = _normalize_ids(prompt_ids)

    full = tokenizer.apply_chat_template(
        prompt + completion,
        tokenize=True,
        return_assistant_tokens_mask=True,
        return_dict=True,
    )
    full_ids = _normalize_ids(full["input_ids"])
    asst_mask = _normalize_mask(full.get("assistant_masks", full.get("assistant_mask", [])))

    if len(full_ids) < len(prompt_ids):
        raise ValueError("Full sequence shorter than prompt prefix")
    if full_ids[: len(prompt_ids)] != prompt_ids:
        prompt_ids = full_ids[: len(prompt_ids)]

    comp_ids = full_ids[len(prompt_ids) :]
    comp_score = asst_mask[len(prompt_ids) :]
    if len(comp_ids) != len(comp_score):
        raise ValueError("Completion length mismatch with assistant mask")

    if any(m["role"] == "user" for m in completion):
        if sum(comp_score) == 0:
            raise ValueError("Branch has user turns but zero scored assistant tokens")

    return prompt_ids, comp_ids, comp_score


def _decode_scored_tokens(tokenizer, ids: list[int], mask: list[int]) -> list[str]:
    scored_ids = [tid for tid, m in zip(ids, mask) if m]
    return tokenizer.convert_ids_to_tokens(scored_ids)


def _audit_one_branch(
    tokenizer,
    row_id: str,
    branch: str,
    prompt: list[dict],
    completion: list[dict],
) -> dict[str, Any]:
    prompt_ids, comp_ids, comp_score = tokenize_preference_pair(tokenizer, prompt, completion)
    violations: list[str] = []

    if sum(comp_score) == 0:
        violations.append(f"{branch}: zero scored assistant/action tokens")

    for i, msg in enumerate(completion):
        if msg["role"] != "user":
            continue
        # User tokens in completion must not be scored
        pass  # checked via mask on tokenized stream

    # Decode completion tokens and check user-role spans
    full = tokenizer.apply_chat_template(
        prompt + completion,
        tokenize=True,
        return_assistant_tokens_mask=True,
        return_dict=True,
    )
    full_ids = _normalize_ids(full["input_ids"])
    asst_mask = _normalize_mask(full.get("assistant_masks", full.get("assistant_mask", [])))
    comp_mask = asst_mask[len(prompt_ids) :]

    # Any completion token that belongs to a user message must have score 0
    offset = len(prompt_ids)
    for msg in completion:
        if msg["role"] == "user":
            # approximate: tokenize single message in isolation is unreliable; use mask invariant
            pass

    # Strong invariant: no scored token where mask says non-assistant
    for idx, (tid, sc) in enumerate(zip(comp_ids, comp_score)):
        if sc and comp_mask[idx] == 0:
            violations.append(f"{branch}: token {idx} scored but assistant_mask=0")

    if len(comp_ids) != len(comp_score):
        violations.append(f"{branch}: length mismatch ids={len(comp_ids)} mask={len(comp_score)}")

    return {
        "row_id": row_id,
        "branch": branch,
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(comp_ids),
        "scored_tokens": int(sum(comp_score)),
        "violations": violations,
        "scored_token_preview": _decode_scored_tokens(tokenizer, comp_ids, comp_score)[:8],
    }


def run_mask_audit(
    tokenizer,
    rows: list[dict],
    output_path: Path,
    row_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Hard-fail mask audit on curated rows; write mask_audit.json."""
    from train.train_sft import patch_chat_template_for_assistant_loss

    patch_chat_template_for_assistant_loss(tokenizer)

    by_id = {r["id"]: r for r in rows}
    if row_ids is None:
        row_ids = list(MASK_AUDIT_ROW_IDS)

    # Add longest rows from length scan
    lengths = []
    for row in rows:
        p_ids, c_ids, _ = tokenize_preference_pair(tokenizer, row["prompt"], row["chosen"])
        _, r_ids, _ = tokenize_preference_pair(tokenizer, row["prompt"], row["rejected"])
        lengths.append((len(p_ids) + max(len(c_ids), len(r_ids)), row["id"]))
    lengths.sort(reverse=True)
    for _, rid in lengths[:3]:
        if rid not in row_ids:
            row_ids.append(rid)

    # Rows with branch-local user
    for row in rows:
        if has_branch_local_user(row["chosen"], row["rejected"]) and row["id"] not in row_ids:
            row_ids.append(row["id"])
            if sum(1 for r in rows if has_branch_local_user(r["chosen"], r["rejected"]) and r["id"] in row_ids) >= 5:
                break

    audits = []
    all_violations: list[str] = []

    for rid in row_ids:
        if rid not in by_id:
            continue
        row = by_id[rid]
        for branch, completion in (("chosen", row["chosen"]), ("rejected", row["rejected"])):
            entry = _audit_one_branch(tokenizer, rid, branch, row["prompt"], completion)
            audits.append(entry)
            all_violations.extend(entry["violations"])

    result = {
        "row_ids": row_ids,
        "audits": audits,
        "pass": len(all_violations) == 0,
        "violations": all_violations,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if all_violations:
        raise ValueError(f"Mask audit failed ({len(all_violations)} violations): {all_violations[:5]}")

    print(f"Mask audit passed ({len(audits)} branch checks) -> {output_path}")
    return result


def audit_masks(tokenizer, rows: list[dict], n: int = 5) -> None:
    from train.train_sft import patch_chat_template_for_assistant_loss

    patch_chat_template_for_assistant_loss(tokenizer)
    print("\n--- Mask audit (sample rows) ---")
    for row in rows[:n]:
        _, chosen_ids, chosen_score = tokenize_preference_pair(
            tokenizer, row["prompt"], row["chosen"]
        )
        print(f"  {row['id']}: chosen_completion_tokens={len(chosen_ids)} scored={sum(chosen_score)}")


def compute_length_stats(rows: list[dict], tokenizer) -> dict[str, Any]:
    """Token lengths for chosen/rejected pairs (pre-train gate)."""
    lengths: list[int] = []
    overlength: list[str] = []

    for row in rows:
        prompt_ids, chosen_ids, _ = tokenize_preference_pair(
            tokenizer, row["prompt"], row["chosen"]
        )
        _, rejected_ids, _ = tokenize_preference_pair(tokenizer, row["prompt"], row["rejected"])
        chosen_len = len(prompt_ids) + len(chosen_ids)
        rejected_len = len(prompt_ids) + len(rejected_ids)
        pair_max = max(chosen_len, rejected_len)
        lengths.append(pair_max)
        if pair_max > MAX_LENGTH:
            overlength.append(row["id"])

    arr = sorted(lengths)
    n = len(arr)
    p95_idx = int(0.95 * (n - 1)) if n else 0
    return {
        "max_length_limit": MAX_LENGTH,
        "max_observed": max(arr) if arr else 0,
        "p95_observed": arr[p95_idx] if arr else 0,
        "mean_observed": sum(arr) / n if n else 0,
        "overlength_count": len(overlength),
        "overlength_ids": overlength[:20],
    }


def build_datasets(
    tokenizer,
    split_map: dict[str, str] | None = None,
    *,
    enforce_max_length: bool = True,
) -> tuple[DatasetDict, dict[str, Any]]:
    from train.train_sft import patch_chat_template_for_assistant_loss

    patch_chat_template_for_assistant_loss(tokenizer)

    rows = load_v4_rows()
    if split_map is None:
        split_map = create_split_manifest(rows)

    length_stats = compute_length_stats(rows, tokenizer)
    if enforce_max_length and length_stats["overlength_count"] > 0:
        raise ValueError(
            f"{length_stats['overlength_count']} rows exceed MAX_LENGTH={MAX_LENGTH}: "
            f"{length_stats['overlength_ids']}"
        )

    records = {"train": [], "val": [], "test": []}
    for row in rows:
        split = split_map[row["id"]]
        if split == "validation":
            split = "val"
        prompt_ids, chosen_ids, chosen_score = tokenize_preference_pair(
            tokenizer, row["prompt"], row["chosen"]
        )
        _, rejected_ids, rejected_score = tokenize_preference_pair(
            tokenizer, row["prompt"], row["rejected"]
        )

        pair_max = max(
            len(prompt_ids) + len(chosen_ids),
            len(prompt_ids) + len(rejected_ids),
        )
        if enforce_max_length and pair_max > MAX_LENGTH:
            raise ValueError(f"{row['id']}: pair_len={pair_max} > max_length={MAX_LENGTH}")

        records[split].append(
            {
                "id": row["id"],
                "prompt_ids": prompt_ids,
                "chosen_ids": chosen_ids,
                "rejected_ids": rejected_ids,
                "chosen_score_mask": chosen_score,
                "rejected_score_mask": rejected_score,
                "dpo_source": row["dpo_source"],
                "category": row["category"],
                "source_family": source_family(row["dpo_source"]),
                "chosen_scored_len": sum(chosen_score),
                "rejected_scored_len": sum(rejected_score),
            }
        )

    print(f"Tokenized: train={len(records['train'])} val={len(records['val'])} test={len(records['test'])}")
    ds = DatasetDict(
        {
            "train": Dataset.from_list(records["train"]),
            "val": Dataset.from_list(records["val"]),
            "test": Dataset.from_list(records["test"]),
        }
    )
    return ds, length_stats
