#!/usr/bin/env python3
"""
Multi-turn eval generation via vLLM FP8 + LoRA (production path).

Sampling (locked for DPO final eval):
  temperature=0.7 top_p=0.8 max_tokens=256
  top_k=20 repetition_penalty=1.05 via OpenAI extra_body (vLLM-specific)

Prerequisites:
  bash dpo/eval/vllm_scripts/deploy_qwenie_eval.sh
  pip install openai colorama requests
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from colorama import Fore, Style, init as colorama_init

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from dpo.eval.v15_eval_config import (
    BASELINE_MODEL_NAME,
    DEFAULT_SKELETONS,
    MANIFEST_PATH,
    VLLM_API_KEY,
    VLLM_BASE_URL,
    generation_metadata,
    round_skeleton_ids,
    skeleton_eval_kind,
    vllm_extra_body,
    EVAL_MAX_TOKENS,
    EVAL_TEMPERATURE,
    EVAL_TOP_P,
)
from dpo.eval.vllm_scripts.vllm_lora_runtime import (
    ensure_dpo_adapter_loaded,
    ensure_dpo_adapter_unloaded,
    wait_for_model,
)

colorama_init(autoreset=True)

OUTPUT_DIR = REPO_ROOT / "dpo/eval"
BASE_MODEL_NAME = "Saulie"  # engine name; not used in final eval list


def model_is_complete(model_result: dict, expected_skeleton_count: int) -> bool:
    convs = model_result.get("conversations") or []
    return len(convs) == expected_skeleton_count


def load_checkpoint(output_path: Path) -> dict | None:
    if not output_path.is_file():
        return None
    try:
        return json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"{Fore.YELLOW}WARNING: corrupt checkpoint {output_path}: {exc}{Style.RESET_ALL}")
        return None


def checkpoint_skeleton_ids(skeletons: list[dict]) -> list[str]:
    return [s["id"] for s in skeletons]


def validate_checkpoint_for_resume(
    checkpoint: dict,
    *,
    skeletons: list[dict],
    anonymize: bool,
    manifest_path: Path,
) -> None:
    expected_ids = checkpoint_skeleton_ids(skeletons)
    saved_ids = checkpoint.get("skeleton_ids")
    if saved_ids != expected_ids:
        raise SystemExit(
            "Checkpoint skeleton set does not match this run "
            f"(checkpoint has {len(saved_ids or [])} ids, run wants {len(expected_ids)}). "
            "Use --fresh to ignore the checkpoint."
        )
    if checkpoint.get("anonymized") != anonymize:
        raise SystemExit(
            "Checkpoint anonymize flag differs from this run. Use --fresh to start over."
        )
    saved_manifest = checkpoint.get("candidate_manifest")
    if saved_manifest and saved_manifest != str(manifest_path.resolve()):
        print(
            f"{Fore.YELLOW}WARNING: checkpoint manifest {saved_manifest!r} "
            f"!= {manifest_path.resolve()!r}{Style.RESET_ALL}"
        )


def build_payload(
    *,
    all_results: dict[str, dict],
    skeletons: list[dict],
    anonymize: bool,
    manifest_path: Path,
    eval_round: int | None,
    total_start: float,
    complete: bool,
    study: str | None = None,
    system_prompt_file: str | None = None,
    system_prompt_scope: str | None = None,
) -> dict:
    payload = {
        "generated_at": datetime.now().isoformat(),
        "backend": "vllm_fp8",
        "checkpoint_complete": complete,
        "skeletons_path": str(DEFAULT_SKELETONS),
        "skeleton_ids": checkpoint_skeleton_ids(skeletons),
        "skeleton_count": len(skeletons),
        "eval_round": eval_round,
        "candidate_manifest": str(manifest_path.resolve()),
        "anonymized": anonymize,
        "generation": generation_metadata(),
        "models": all_results,
        "models_complete": sum(
            1 for m in all_results.values() if model_is_complete(m, len(skeletons))
        ),
        "models_total": None,  # filled by caller if known
        "wall_seconds": round(time.time() - total_start, 2),
    }
    if study:
        payload["study"] = study
    if system_prompt_file:
        payload["system_prompt_file"] = system_prompt_file
        payload["system_prompt_scope"] = system_prompt_scope
    return payload


def save_checkpoint(
    output_path: Path,
    payload: dict,
    *,
    unblind_path: Path | None,
    unblind_mapping: dict[str, dict],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(output_path)
    n_done = payload.get("models_complete", len(payload.get("models", {})))
    n_total = payload.get("models_total") or "?"
    print(
        f"{Fore.GREEN}  checkpoint saved -> {output_path} "
        f"({n_done}/{n_total} models){Style.RESET_ALL}"
    )
    if unblind_path is not None:
        sidecar = {
            "generated_at": payload["generated_at"],
            "baseline_model": BASELINE_MODEL_NAME,
            "candidate_mapping": unblind_mapping,
            "checkpoint_complete": payload.get("checkpoint_complete", False),
            "note": "Do not send to LLM judge.",
        }
        utmp = unblind_path.with_suffix(unblind_path.suffix + ".tmp")
        utmp.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        utmp.replace(unblind_path)


def get_available_models(client: OpenAI) -> list[str]:
    return [m.id for m in client.models.list().data]


def load_candidate_manifest(path: Path) -> list[dict]:
    entries = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_skeletons(path: Path, wanted_ids: list[str] | None) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        all_rows = json.load(f)
    if wanted_ids is None:
        return all_rows
    by_id = {row["id"]: row for row in all_rows}
    missing = [sid for sid in wanted_ids if sid not in by_id]
    if missing:
        raise SystemExit(f"Missing skeleton ids: {missing}")
    return [by_id[sid] for sid in wanted_ids]


def blind_public_name(entry: dict, anonymize: bool) -> str:
    if not anonymize or entry.get("kind") == "sft_baseline":
        return entry["model_name"]
    return entry["judge_id"] or entry["model_name"]


def generate_response(client: OpenAI, model_name: str, messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=EVAL_MAX_TOKENS,
        temperature=EVAL_TEMPERATURE,
        top_p=EVAL_TOP_P,
        extra_body=vllm_extra_body(),
    )
    return (response.choices[0].message.content or "").strip()


def run_skeleton(
    client: OpenAI,
    model_name: str,
    skeleton: dict,
    *,
    system_prompt: str | None = None,
) -> list[dict]:
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for i, user_msg in enumerate(skeleton["user_turns"]):
        messages.append({"role": "user", "content": user_msg})
        response = generate_response(client, model_name, messages)
        messages.append({"role": "assistant", "content": response})
        is_last = i == len(skeleton["user_turns"]) - 1
        tag = "FINAL" if is_last else "intermediate"
        print(f"    Turn {i + 1}/{len(skeleton['user_turns'])} [{tag}]: {len(response)} chars")
    return messages


def parse_model_keys(keys_arg: str | None, entries: list[dict]) -> list[dict]:
    if not keys_arg:
        return entries
    by_key: dict[str, dict] = {}
    for e in entries:
        mk = e.get("manifest_key")
        if mk:
            by_key[mk] = e
        kind = e.get("kind")
        if kind == "sft_baseline":
            by_key["sft"] = e
            by_key["baseline"] = e
        elif kind == "base_fp8":
            by_key["base"] = e
        elif kind == "prod_dpo":
            by_key["prod"] = e
        elif e.get("trial_number") is not None:
            by_key[f"trial-{e['trial_number']}"] = e
    out = []
    for k in [x.strip() for x in keys_arg.split(",") if x.strip()]:
        if k not in by_key:
            raise SystemExit(
                f"Unknown model key {k!r}; use base, sft, prod, or trial-N"
            )
        out.append(by_key[k])
    return out


def build_unblind_mapping(entries: list[dict]) -> dict[str, dict]:
    mapping = {}
    for e in entries:
        if e.get("kind") != "dpo_merged":
            continue
        pub = e.get("judge_id") or e["model_name"]
        mapping[pub] = {
            "trial_number": e["trial_number"],
            "adapter_path": e.get("adapter_path"),
            "container_path": e.get("container_path"),
        }
    return mapping


def run_eval(
    *,
    entries: list[dict],
    skeletons: list[dict],
    output_path: Path,
    anonymize: bool,
    unblind_path: Path | None,
    skip_runtime_load: bool,
    manifest_path: Path,
    eval_round: int | None,
    resume: bool,
    fresh: bool,
    system_prompt_text: str | None = None,
    system_prompt_file: Path | None = None,
    study: str | None = None,
    models_total: int | None = None,
) -> None:
    client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    print(f"\n{Fore.CYAN}vLLM @ {VLLM_BASE_URL}{Style.RESET_ALL}")
    print(f"  sampling: temp={EVAL_TEMPERATURE} top_p={EVAL_TOP_P} "
          f"top_k={vllm_extra_body()['top_k']} "
          f"rep_penalty={vllm_extra_body()['repetition_penalty']} "
          f"max_tokens={EVAL_MAX_TOKENS}")

    if fresh and output_path.is_file():
        print(f"{Fore.YELLOW}--fresh: removing existing checkpoint {output_path}{Style.RESET_ALL}")
        output_path.unlink(missing_ok=True)
        if unblind_path:
            unblind_path.unlink(missing_ok=True)

    all_results: dict[str, dict] = {}
    if resume and output_path.is_file():
        checkpoint = load_checkpoint(output_path)
        if checkpoint:
            validate_checkpoint_for_resume(
                checkpoint, skeletons=skeletons, anonymize=anonymize, manifest_path=manifest_path
            )
            all_results = dict(checkpoint.get("models") or {})
            n_ok = sum(
                1 for m in all_results.values() if model_is_complete(m, len(skeletons))
            )
            print(
                f"{Fore.CYAN}Resuming from checkpoint: {output_path} "
                f"({n_ok} complete model(s) loaded){Style.RESET_ALL}"
            )

    unblind_mapping = build_unblind_mapping(entries)
    total_start = time.time()
    expected_n = len(skeletons)

    sys_prompt_file_str = str(system_prompt_file.resolve()) if system_prompt_file else None
    sys_prompt_scope = "base_only" if system_prompt_text else None

    def flush_checkpoint(*, complete: bool) -> None:
        payload = build_payload(
            all_results=all_results,
            skeletons=skeletons,
            anonymize=anonymize,
            manifest_path=manifest_path,
            eval_round=eval_round,
            total_start=total_start,
            complete=complete,
            study=study,
            system_prompt_file=sys_prompt_file_str,
            system_prompt_scope=sys_prompt_scope,
        )
        payload["models_total"] = models_total if models_total is not None else len(entries)
        save_checkpoint(
            output_path,
            payload,
            unblind_path=unblind_path,
            unblind_mapping=unblind_mapping,
        )

    for idx, entry in enumerate(entries):
        api_name = entry["model_name"]
        public_name = blind_public_name(entry, anonymize)
        print(f"\n{Fore.CYAN}{'=' * 60}")
        print(f" [{idx + 1}/{len(entries)}] {public_name} (api={api_name})")
        print(f"{'=' * 60}{Style.RESET_ALL}")

        existing = all_results.get(public_name)
        if existing and model_is_complete(existing, expected_n):
            print(
                f"{Fore.GREEN}  SKIP: checkpoint already has {expected_n} conversations{Style.RESET_ALL}"
            )
            continue
        use_sys = bool(entry.get("use_system_prompt") and system_prompt_text)
        if existing and not model_is_complete(existing, expected_n):
            got = len(existing.get("conversations") or [])
            print(
                f"{Fore.YELLOW}  RESUME incomplete checkpoint ({got}/{expected_n} skeletons){Style.RESET_ALL}"
            )
        elif existing:
            got = 0

        if entry.get("runtime_load") and not skip_runtime_load:
            ensure_dpo_adapter_loaded(entry)
            wait_for_model(client, api_name)

        model_start = time.time()
        conversations: list[dict] = []
        done_ids: set[str] = set()
        if existing and not model_is_complete(existing, expected_n):
            conversations = list(existing.get("conversations") or [])
            done_ids = {c["skeleton_id"] for c in conversations}

        all_results[public_name] = {
            "model": public_name,
            "api_model_name": api_name,
            "is_baseline": entry.get("kind") == "sft_baseline",
            "kind": entry.get("kind"),
            "system_prompt_used": use_sys,
            "conversations": conversations,
            "elapsed_seconds": existing.get("elapsed_seconds", 0) if existing else 0,
        }
        if not anonymize or entry.get("kind") in ("sft_baseline", "base_fp8", "prod_dpo"):
            all_results[public_name]["manifest"] = {
                k: entry.get(k)
                for k in (
                    "manifest_key",
                    "trial_number",
                    "lora_rank",
                    "adapter_path",
                    "container_path",
                    "deploy_script",
                )
                if entry.get(k) is not None
            }

        for sk_idx, skeleton in enumerate(skeletons):
            if skeleton["id"] in done_ids:
                continue
            ek = skeleton_eval_kind(skeleton)
            print(
                f"\n  {Fore.YELLOW}[{sk_idx + 1}/{len(skeletons)}] {skeleton['id']} "
                f"type={skeleton['opening_type']} eval_kind={ek}{Style.RESET_ALL}"
            )
            messages = run_skeleton(
                client,
                api_name,
                skeleton,
                system_prompt=system_prompt_text if use_sys else None,
            )
            conversations.append(
                {
                    "skeleton_id": skeleton["id"],
                    "opening_type": skeleton["opening_type"],
                    "eval_kind": ek,
                    "target_turns": skeleton["target_turns"],
                    "actual_turns": len(messages),
                    "messages": messages,
                }
            )
            all_results[public_name]["conversations"] = conversations
            all_results[public_name]["elapsed_seconds"] = round(
                time.time() - model_start, 2
            )
            flush_checkpoint(complete=False)

        if entry.get("runtime_load") and not skip_runtime_load:
            ensure_dpo_adapter_unloaded(entry)

        flush_checkpoint(complete=False)

    flush_checkpoint(complete=True)
    print(f"\n{Fore.GREEN}Final output: {output_path}{Style.RESET_ALL}")
    if unblind_path:
        print(f"Unblind sidecar: {unblind_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM FP8 DPO final eval generation")
    parser.add_argument("--candidate-manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--skeletons", type=Path, default=DEFAULT_SKELETONS)
    parser.add_argument("--round", type=int, choices=(1, 2), default=None)
    parser.add_argument("--all-skeletons", action="store_true")
    parser.add_argument("--skeleton-ids", type=str, default=None)
    parser.add_argument("--study", type=str, default=None)
    parser.add_argument(
        "--models-total",
        type=int,
        default=None,
        help="Total models in full study (when running one model at a time)",
    )
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        default=None,
        help="Steering system prompt applied when manifest entry has use_system_prompt=true",
    )
    parser.add_argument("--models", type=str, default=None, help="sft,trial-16,...")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unblind-output", type=Path, default=None)
    parser.add_argument("--anonymize", action="store_true")
    parser.add_argument(
        "--skip-runtime-load",
        action="store_true",
        help="Adapters already loaded on server (or preloaded at startup)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing checkpoint; do not skip completed models",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing output/unblind files before running",
    )
    args = parser.parse_args()

    if args.skeleton_ids:
        wanted = [s.strip() for s in args.skeleton_ids.split(",") if s.strip()]
        eval_round = None
    elif args.all_skeletons:
        wanted = None
        eval_round = None
    elif args.round is not None:
        wanted = round_skeleton_ids(args.round)
        eval_round = args.round
    else:
        wanted = round_skeleton_ids(1)
        eval_round = 1

    skeletons = load_skeletons(args.skeletons, wanted)
    entries = load_candidate_manifest(args.candidate_manifest)
    entries = parse_model_keys(args.models, entries)

    system_prompt_text = None
    if args.system_prompt_file:
        system_prompt_text = args.system_prompt_file.read_text(encoding="utf-8").strip()

    run_eval(
        entries=entries,
        skeletons=skeletons,
        output_path=args.output,
        anonymize=args.anonymize,
        unblind_path=args.unblind_output,
        skip_runtime_load=args.skip_runtime_load,
        manifest_path=args.candidate_manifest,
        eval_round=eval_round,
        resume=not args.no_resume,
        fresh=args.fresh,
        system_prompt_text=system_prompt_text,
        system_prompt_file=args.system_prompt_file,
        study=args.study,
        models_total=args.models_total,
    )


if __name__ == "__main__":
    main()
