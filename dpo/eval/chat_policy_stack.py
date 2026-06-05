#!/usr/bin/env python3
"""Interactive REPL: HF Qwen3 base (BnB or bf16) with optional SFT/DPO adapters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dpo.train.merge_sft_dpo_lora import (
    load_baked_dpo_stack,
    load_cat_merged_adapter,
    load_sft_stack,
    load_stacked_for_merge,
    resolve_dpo_adapter_path,
)
from dpo.train.model_load import BaseKind, load_base, load_sft_baked_base
from dpo.train.paths import MODEL_ID_BF16, MODEL_ID_SFT_MERGED_BF16, SFT_ADAPTER
from dpo.train.dpo_trainer_compat import DPO_ADAPTER_NAME, collect_adapter_diagnostics
from dpo.train.qwen3_decode import add_decode_argparse
from dpo.train.smoke_policy_stack_hf import (
    activate_adapter_mode,
    generate_turn,
    scale_adapter_residual,
)
from dpo.train.train_dpo import load_tokenizer

TRIAL13 = (
    REPO_ROOT
    / "dpo/train/models/steering-dpo-v1.1/optuna-run-20260526-042551/trial-13/best_adapter"
)
V12_LATEST_POINTER = REPO_ROOT / "dpo/train/models/steering-dpo-v1.2/LATEST_RUN_DIR.txt"
V12_KNOWN_RUN = (
    REPO_ROOT / "dpo/train/models/steering-dpo-v1.2/optuna-run-20260530-064345"
)

TRIAL_ALIASES: dict[str, tuple[str, str]] = {
    "0": ("trial-0", "plan_a_minimal_dpo (ld_0.5)"),
    "minimal": ("trial-0", "plan_a_minimal_dpo (ld_0.5)"),
    "dpo": ("trial-0", "plan_a_minimal_dpo (ld_0.5)"),
    "plan_a_minimal_dpo": ("trial-0", "plan_a_minimal_dpo (ld_0.5)"),
    "1": ("trial-1", "plan_a_ipo"),
    "ipo": ("trial-1", "plan_a_ipo"),
    "plan_a_ipo": ("trial-1", "plan_a_ipo"),
}

BASE_LABELS = {"bnb": "BnB 8-bit", "bf16": "bfloat16"}


def resolve_v12_run_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        run_dir = explicit.expanduser().resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"--run-dir is not a directory: {run_dir}")
        return run_dir
    if V12_LATEST_POINTER.is_file():
        run_dir = Path(V12_LATEST_POINTER.read_text().strip()).expanduser().resolve()
        if run_dir.is_dir():
            return run_dir
    if V12_KNOWN_RUN.is_dir():
        return V12_KNOWN_RUN.resolve()
    raise FileNotFoundError(
        "No v1.2 run dir: pass --run-dir or write LATEST_RUN_DIR.txt under steering-dpo-v1.2/"
    )


def adapter_in_run(run_dir: Path, trial: str) -> tuple[Path, str]:
    key = trial.strip().lower()
    if key not in TRIAL_ALIASES:
        choices = ", ".join(sorted(TRIAL_ALIASES))
        raise ValueError(f"Unknown --trial {trial!r}; choose from: {choices}")
    subdir, label = TRIAL_ALIASES[key]
    adapter = run_dir / subdir / "best_adapter"
    if not adapter.is_dir():
        raise FileNotFoundError(f"Missing adapter dir: {adapter}")
    return adapter, label


def resolve_dpo_path(args: argparse.Namespace) -> tuple[Path, str]:
    if args.trial is not None:
        run_dir = resolve_v12_run_dir(args.run_dir)
        adapter, label = adapter_in_run(run_dir, args.trial)
        return adapter, f"{label} @ {run_dir}"
    if args.dpo_adapter is not None:
        return args.dpo_adapter, str(args.dpo_adapter)
    return TRIAL13, "v1.1 trial-13 (default)"


def normalize_args(args: argparse.Namespace) -> tuple[BaseKind, str]:
    """Return (base_kind, stack_mode): base_only | sft_only | sft_baked | sft_baked_dpo | policy | cat."""
    if args.base_bnb and args.base_bf16:
        raise SystemExit("Use only one of --base-bnb and --base-bf16")
    base: BaseKind = args.base
    if args.cat_adapter is not None:
        return base, "cat"
    if args.sft_baked and args.dpo_adapter is not None:
        stack_mode = "sft_baked_dpo"
    elif args.sft_baked:
        stack_mode = "sft_baked"
    elif args.base_bnb:
        base = "bnb"
        stack_mode = "base_only"
    elif args.base_bf16:
        base = "bf16"
        stack_mode = "base_only"
    elif args.base_only:
        stack_mode = "base_only"
    elif args.sft_only:
        stack_mode = "sft_only"
    else:
        stack_mode = "policy"
    return base, stack_mode


def main() -> None:
    epilog = """
Base precision (--base applies to every stack, default bnb = DPO training match):
  bnb   BitsAndBytes 8-bit
  bf16  full bfloat16 weights (more VRAM; adapters still work)

Bare base (no adapters):
  python dpo/eval/chat_policy_stack.py --base-only
  python dpo/eval/chat_policy_stack.py --base bf16 --base-only
  python dpo/eval/chat_policy_stack.py --base-bnb   # same as --base-only
  python dpo/eval/chat_policy_stack.py --base-bf16  # same as --base bf16 --base-only

SFT trial-17 only:
  python dpo/eval/chat_policy_stack.py --sft-only
  python dpo/eval/chat_policy_stack.py --base bf16 --sft-only

Policy stack (SFT + DPO):
  python dpo/eval/chat_policy_stack.py --trial minimal
  python dpo/eval/chat_policy_stack.py --base bf16 --trial ipo --dpo-weight 1.0

Plan B — SFT baked into dense base (merge first):
  python dpo/train/merge_sft_baked_base.py
  python dpo/eval/chat_policy_stack.py --sft-baked
  python dpo/eval/chat_policy_stack.py --sft-baked --dpo-adapter .../trial-0/best_adapter

Cat-merged SFT+DPO (one adapter, after merge_sft_dpo_lora.py):
  python dpo/train/merge_sft_dpo_lora.py --dpo-adapter .../trial-3/best_adapter \\
    --output .../trial-3/sft_dpo_cat --check-logps
  python dpo/eval/chat_policy_stack.py --cat-adapter .../trial-3/sft_dpo_cat
"""
    parser = argparse.ArgumentParser(
        description="Chat with Qwen3 base and optional SFT/DPO adapters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    parser.add_argument(
        "--base",
        choices=("bnb", "bf16"),
        default="bnb",
        help="Underlying model load (default: bnb = DPO training stack)",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Load base weights only (no SFT/DPO adapters)",
    )
    parser.add_argument(
        "--base-bnb",
        action="store_true",
        help="Shortcut: --base bnb --base-only",
    )
    parser.add_argument(
        "--base-bf16",
        action="store_true",
        help="Shortcut: --base bf16 --base-only",
    )
    parser.add_argument(
        "--dpo-adapter",
        type=Path,
        default=None,
        help="trial-N/best_adapter; default v1.1 trial-13 if no --trial",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="v1.2 optuna run dir (with --trial)",
    )
    parser.add_argument(
        "--trial",
        choices=sorted(TRIAL_ALIASES),
        default=None,
        help="trial-0 (minimal DPO) or trial-1 (IPO) from a v1.2 run",
    )
    parser.add_argument(
        "--sft-only",
        action="store_true",
        help="SFT trial-17 only (no DPO adapter)",
    )
    parser.add_argument(
        "--sft-baked",
        action="store_true",
        help="Dense SFT-merged base (Plan B Control B); run merge_sft_baked_base.py first",
    )
    parser.add_argument(
        "--sft-merged-path",
        type=Path,
        default=None,
        help="Override merged checkpoint dir (default: Qwen3-4B-Instruct-2507-SFT-MERGED-BF16)",
    )
    parser.add_argument("--dpo-weight", type=float, default=1.0)
    parser.add_argument(
        "--cat-adapter",
        type=Path,
        default=None,
        help="Merged sft_dpo_cat dir from merge_sft_dpo_lora.py (base + single cat LoRA)",
    )
    add_decode_argparse(parser)
    args = parser.parse_args()

    try:
        base_kind, stack_mode = normalize_args(args)
    except SystemExit as e:
        parser.error(str(e))

    if args.cat_adapter is not None:
        exclusive = (
            args.base_only,
            args.sft_only,
            args.sft_baked,
            args.base_bnb,
            args.base_bf16,
            args.dpo_adapter is not None,
            args.trial is not None,
        )
        if any(exclusive):
            parser.error("--cat-adapter cannot be combined with other stack flags or --dpo-adapter")
    if stack_mode == "base_only" and (args.dpo_adapter is not None or args.trial is not None):
        parser.error("--base-only cannot be used with --dpo-adapter or --trial")
    if stack_mode == "sft_only" and (args.dpo_adapter is not None or args.trial is not None):
        parser.error("--sft-only cannot be used with --dpo-adapter or --trial")
    if stack_mode == "sft_baked" and args.trial is not None:
        parser.error("--sft-baked cannot be used with --trial")
    if stack_mode == "sft_baked_dpo" and args.trial is not None:
        parser.error("--sft-baked with --dpo-adapter cannot be used with --trial")
    if args.dpo_adapter is not None and args.trial is not None:
        parser.error("Use either --dpo-adapter or --trial, not both")
    mode_flags = (args.base_only, args.sft_only, args.sft_baked, args.base_bnb, args.base_bf16)
    if sum(mode_flags) > 1:
        parser.error(
            "Use only one of --base-only, --sft-only, --sft-baked, --base-bnb, --base-bf16"
        )

    merged_path = (
        args.sft_merged_path.expanduser().resolve()
        if args.sft_merged_path is not None
        else MODEL_ID_SFT_MERGED_BF16
    )
    base_label = BASE_LABELS[base_kind]
    print("Loading tokenizer...")
    tokenizer = load_tokenizer()
    if stack_mode == "sft_baked":
        print(f"SFT-baked checkpoint: {merged_path}")
    else:
        print(f"Base: {base_label} ({MODEL_ID_BF16})")
    print("(first load can take ~30s)\n")

    if stack_mode == "cat":
        cat_path = args.cat_adapter.expanduser().resolve()
        stack_label = f"{base_label} + cat-merged SFT+DPO"
        adapter_mode = "cat"
        print(f"Stack: {stack_label}")
        print(f"Cat adapter: {cat_path}")
        model = load_cat_merged_adapter(cat_path, base=base_kind)
        activate_adapter_mode(model, adapter_mode)
    elif stack_mode == "sft_baked_dpo":
        dpo_path = resolve_dpo_adapter_path(args.dpo_adapter)
        stack_label = f"{base_label} SFT-baked + DPO (Plan B)"
        adapter_mode = "dpo"
        print(f"Stack: {stack_label}")
        print(f"DPO adapter: {dpo_path}")
        model = load_baked_dpo_stack(dpo_path, base=base_kind, merged_path=merged_path)
    elif stack_mode == "sft_baked":
        stack_label = f"{base_label} SFT-baked (merged trial-17, no adapter)"
        adapter_mode = "baked"
        print(f"Stack: {stack_label}")
        model = load_sft_baked_base(base_kind, merged_path=merged_path)
    elif stack_mode == "base_only":
        stack_label = f"{base_label} base only"
        adapter_mode = "base"
        print(f"Stack: {stack_label}")
        model = load_base(base_kind)
    elif stack_mode == "sft_only":
        stack_label = f"{base_label} + SFT trial-17"
        adapter_mode = "sft"
        print(f"Stack: {stack_label}")
        model = load_sft_stack(base=base_kind)
    else:
        adapter_arg, trial_label = resolve_dpo_path(args)
        dpo_path = resolve_dpo_adapter_path(adapter_arg)
        stack_label = f"{base_label} + SFT + DPO ({trial_label})"
        adapter_mode = "policy"
        print(f"Stack: {stack_label}")
        print(f"DPO adapter: {dpo_path}")
        model = load_stacked_for_merge(dpo_path, base=base_kind)
        if args.dpo_weight != 1.0:
            n = scale_adapter_residual(model, DPO_ADAPTER_NAME, args.dpo_weight)
            print(f"Scaled DPO residual: weight={args.dpo_weight} ({n} modules)")
        activate_adapter_mode(model, adapter_mode)

    model.eval()
    diag = collect_adapter_diagnostics(model)
    dpo_note = "n/a" if adapter_mode in ("sft", "base", "baked", "cat") else str(args.dpo_weight)
    decode_note = args.decode
    if args.decode == "sample":
        decode_note = (
            f"sample temp={args.temperature} top_p={args.top_p} top_k={args.top_k} "
            f"rep_penalty={args.repetition_penalty}"
        )
    print(
        f"active_adapters={diag.get('active_adapters')} "
        f"mode={adapter_mode} base={base_kind} decode={decode_note} dpo_weight={dpo_note}"
    )
    print("Commands: /reset  /quit  /weight <0-1>")
    print("-" * 60)

    messages: list[dict] = []

    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not user:
            continue
        low = user.lower()
        if low in ("/quit", "/exit", "/q"):
            break
        if low == "/reset":
            messages.clear()
            print("(conversation cleared)")
            continue
        if low.startswith("/weight"):
            parts = user.split()
            if len(parts) != 2:
                print("Usage: /weight 0.25  (requires restart to apply — reload script)")
                continue
            print("Change --dpo-weight on the command line and restart this script.")
            continue

        messages.append({"role": "user", "content": user})
        reply = generate_turn(
            model,
            tokenizer,
            messages,
            adapter_mode=adapter_mode,
            decode=args.decode,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        messages.append({"role": "assistant", "content": reply})
        print(f"\nAssistant:\n{reply}")


if __name__ == "__main__":
    main()
