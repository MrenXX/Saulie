"""Stable on-disk cache for TRL reference log probabilities across Optuna trials."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import trl

from dpo.train.dpo_data import MAX_LENGTH, SPLIT_SEED, manifest_sha256, manifest_path_for_seed
from dpo.train.paths import CACHE_DIR, MODEL_ID_SFT_MERGED_BF16, SFT_ADAPTER

_REF_CACHE_META: dict[str, Any] = {"splits": {}}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sft_adapter_hash() -> str:
    """Hash SFT reference weights (not per-trial DPO)."""
    parts: list[str] = []
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        p = SFT_ADAPTER / name
        if p.exists():
            parts.append(_sha256_file(p))
    if not parts:
        raise FileNotFoundError(f"No SFT adapter files under {SFT_ADAPTER}")
    return hashlib.sha256("".join(parts).encode()).hexdigest()[:16]


def sft_merged_base_hash() -> str:
    """Hash dense SFT-baked checkpoint (Plan B reference policy)."""
    weights = MODEL_ID_SFT_MERGED_BF16 / "model.safetensors"
    if weights.is_file():
        return _sha256_file(weights)[:16]
    meta = MODEL_ID_SFT_MERGED_BF16 / "merge_meta.json"
    if meta.is_file():
        return _sha256_file(meta)[:16]
    raise FileNotFoundError(
        f"No SFT-merged base at {MODEL_ID_SFT_MERGED_BF16}; run merge_sft_baked_base.py"
    )


def reference_policy_hash(*, baked_base: bool = False) -> str:
    if baked_base:
        return f"merged:{sft_merged_base_hash()}"
    return f"sft_lora:{sft_adapter_hash()}"


def ref_cache_key(
    *,
    dataset_fingerprint: str,
    precompute_batch_size: int,
    baked_base: bool = False,
) -> str:
    manifest_path = manifest_path_for_seed(SPLIT_SEED)
    return hashlib.sha256(
        "|".join(
            [
                manifest_sha256(manifest_path),
                dataset_fingerprint,
                reference_policy_hash(baked_base=baked_base),
                str(MAX_LENGTH),
                str(precompute_batch_size),
                trl.__version__,
            ]
        ).encode()
    ).hexdigest()[:24]


def cache_path(key: str, split_name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"ref_{key}_{split_name}.npz"


def set_last_ref_cache_meta(meta: dict[str, Any]) -> None:
    split = meta.get("split", "unknown")
    _REF_CACHE_META["splits"][split] = meta


def get_last_ref_cache_meta() -> dict[str, Any]:
    splits = _REF_CACHE_META.get("splits", {})
    hits = [s.get("hit") for s in splits.values()]
    return {
        "splits": splits,
        "all_hits": bool(hits) and all(hits),
        "any_miss": any(h is False for h in hits),
    }


def reset_ref_cache_meta() -> None:
    _REF_CACHE_META["splits"] = {}


def load_ref_logps(path: Path) -> tuple[np.ndarray, np.ndarray]:
    loaded = np.load(path)
    return loaded["ref_chosen_logps"], loaded["ref_rejected_logps"]


def save_ref_logps(
    path: Path,
    ref_chosen_logps: np.ndarray,
    ref_rejected_logps: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ref_chosen_logps=ref_chosen_logps,
        ref_rejected_logps=ref_rejected_logps,
    )
