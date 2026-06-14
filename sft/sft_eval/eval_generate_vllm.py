"""
Multi-Turn Eval Generation Script (vLLM API version)
=====================================================
Connects to a running vLLM server with multiple LoRA adapters,
generates assistant responses turn-by-turn for the top N trial
adapters from each experiment + the base model.

Output is saved for LLM-as-judge scoring.

Prerequisites:
    pip install openai colorama

Usage:
    # Start vLLM server first:
    bash deploy_qwenie.sh

    # Top 5 from each experiment (default):
    python eval_generate.py

    # Top 3 from each experiment:
    python eval_generate.py --top_n 3

    # All adapters (no filtering):
    python eval_generate.py --top_n 0

    # Only one experiment:
    python eval_generate.py --filter "v1.1"

    # Skip base model:
    python eval_generate.py --skip_base
"""

import json
import time
import argparse
from pathlib import Path
from datetime import datetime

from openai import OpenAI
from colorama import Fore, Style, init as colorama_init
colorama_init(autoreset=True)


# ============================================================
# CONFIG
# ============================================================

ABS_PATH        = Path(r"/root/saulie")
SKELETONS_PATH  = ABS_PATH / "train" / "sft_eval" / "eval_skeletons.json"
OUTPUT_DIR      = ABS_PATH / "train" / "eval_results"
OUTPUT_BASE     = ABS_PATH / "train" / "models"

# Experiment directories to scan for trial_summary.json
# Adapter names in vLLM follow: {experiment_name}_trial-{N}
EXPERIMENT_NAMES = [
    "steering-sft-v1.1",
    "steering-sft-v1.2",
]

VLLM_BASE_URL   = "http://localhost:8000/v1"
VLLM_API_KEY    = "dipshit"  # Must match deploy_qwenie.sh

BASE_MODEL_NAME = "Saulie"  # --served-model-name in deploy script

# Generation params — match these to what you want at inference time
MAX_TOKENS      = 350
TEMPERATURE     = 0.7
TOP_P           = 0.8


# ============================================================
# HELPERS
# ============================================================

def get_available_models(client):
    """Query vLLM for all registered models (base + LoRA adapters)."""
    response = client.models.list()
    return [m.id for m in response.data]


def load_candidate_manifest(path: Path) -> list[dict]:
    entries = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def get_top_n_adapter_names(top_n):
    """
    Read trial_summary.json from each experiment, pick top N by eval_loss,
    and return a dict mapping vLLM adapter name -> eval_loss.

    Adapter names follow the deploy script convention:
        {experiment_name}_trial-{N}

    If top_n == 0, returns all trials (no filtering).
    """
    adapter_info = {}  # adapter_name -> {"eval_loss": float, "params": dict}

    for exp_name in EXPERIMENT_NAMES:
        summary_path = OUTPUT_BASE / exp_name / "trial_summary.json"
        if not summary_path.exists():
            print(f"{Fore.YELLOW}trial_summary.json not found for {exp_name}, "
                  f"skipping top_n filtering for this experiment.{Style.RESET_ALL}")
            continue

        with open(summary_path) as f:
            trial_summary = json.load(f)

        # trial_summary is already sorted by eval_loss from train_sft.py
        trials = trial_summary if top_n == 0 else trial_summary[:top_n]

        for t in trials:
            adapter_name = f"{exp_name}_trial-{t['trial']}"
            adapter_info[adapter_name] = {
                "eval_loss": t["eval_loss"],
                "params": t.get("params", {}),
            }

        print(f"  {exp_name}: selected {len(trials)}/{len(trial_summary)} trials"
              f" (top {top_n if top_n > 0 else 'all'} by eval_loss)")

    return adapter_info


def generate_response(client, model_name, messages):
    """Generate a single assistant response via the OpenAI-compatible API."""
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return response.choices[0].message.content.strip()


def run_skeleton(client, model_name, skeleton):
    """
    Run a single conversation skeleton: feed user turns one at a time,
    generate assistant responses in between.

    Returns the full messages list (alternating user/assistant).
    """
    messages = []
    user_turns = skeleton["user_turns"]

    for i, user_msg in enumerate(user_turns):
        messages.append({"role": "user", "content": user_msg})

        response = generate_response(client, model_name, messages)
        messages.append({"role": "assistant", "content": response})

        is_last = (i == len(user_turns) - 1)
        status = "FINAL (should contain recommendation)" if is_last else "intermediate"
        print(f"    Turn {i+1}/{len(user_turns)} [{status}]: {len(response)} chars")

    return messages


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-turn eval generation (vLLM API)")
    parser.add_argument("--candidate-manifest", type=str, default=None,
                        help="JSONL manifest of models to evaluate (DPO final eval)")
    parser.add_argument("--skeletons", type=str, default=None,
                        help="Path to eval_skeletons.json")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output JSON")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Evaluate only this manifest model name")
    parser.add_argument("--skeleton-ids", type=str, default=None,
                        help="Comma-separated skeleton ids to run (default: all)")
    parser.add_argument("--limit-models", type=int, default=None,
                        help="Use first N manifest models only")
    parser.add_argument("--limit-skeletons", type=int, default=None,
                        help="Use first N skeletons only")
    parser.add_argument("--top_n", type=int, default=5,
                        help="Top N trials per experiment by eval_loss (0 = all)")
    parser.add_argument("--filter", type=str, default=None,
                        help="Only eval adapters whose name contains this substring (e.g. 'v1.1')")
    parser.add_argument("--skip_base", action="store_true",
                        help="Skip the base model (no adapter) evaluation")
    parser.add_argument("--output", type=str, default=None,
                        help="Custom output filename (default: auto-generated)")
    args = parser.parse_args()

    client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    skeletons_path = Path(args.skeletons) if args.skeletons else SKELETONS_PATH
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    manifest_entries: list[dict] = []
    models_to_eval: list[str] = []
    model_meta: dict[str, dict] = {}

    print(f"\n{Fore.CYAN}Querying vLLM server at {VLLM_BASE_URL}...{Style.RESET_ALL}")
    try:
        all_models = get_available_models(client)
    except Exception as e:
        print(f"{Fore.RED}Failed to connect to vLLM server: {e}{Style.RESET_ALL}")
        print("Make sure deploy_qwenie_eval.sh is running and the API is healthy.")
        return

    print(f"Server has {len(all_models)} model(s): {all_models}")

    if args.candidate_manifest:
        manifest_path = Path(args.candidate_manifest)
        if not manifest_path.is_file():
            print(f"{Fore.RED}Manifest not found: {manifest_path}{Style.RESET_ALL}")
            return
        manifest_entries = load_candidate_manifest(manifest_path)
        if args.model_name:
            manifest_entries = [e for e in manifest_entries if e["model_name"] == args.model_name]
        elif args.filter:
            manifest_entries = [e for e in manifest_entries if args.filter in e["model_name"]]
        if args.limit_models is not None:
            manifest_entries = manifest_entries[: args.limit_models]
        models_to_eval = [e["model_name"] for e in manifest_entries]
        model_meta = {e["model_name"]: e for e in manifest_entries}
        missing = [m for m in models_to_eval if m not in all_models]
        if missing:
            print(f"{Fore.RED}Manifest models not registered in vLLM: {missing}{Style.RESET_ALL}")
            return
    else:
        print(f"\n{Fore.CYAN}Loading trial summaries (top_n={args.top_n})...{Style.RESET_ALL}")
        adapter_info = get_top_n_adapter_names(args.top_n)
        adapter_models = [m for m in all_models if m != BASE_MODEL_NAME and m in adapter_info]
        adapter_models.sort()
        wanted_but_missing = set(adapter_info.keys()) - set(all_models)
        if wanted_but_missing:
            print(f"{Fore.YELLOW}Top-N adapters not found on server: {wanted_but_missing}{Style.RESET_ALL}")
        if args.filter:
            adapter_models = [m for m in adapter_models if args.filter in m]
        models_to_eval = []
        if not args.skip_base:
            models_to_eval.append(BASE_MODEL_NAME)
        models_to_eval.extend(adapter_models)
        model_meta = {
            m: {"eval_loss": adapter_info[m]["eval_loss"], "params": adapter_info[m]["params"]}
            for m in adapter_models
        }

    if not models_to_eval:
        print(f"{Fore.RED}No models to evaluate after filtering.{Style.RESET_ALL}")
        return

    if not skeletons_path.exists():
        print(f"{Fore.RED}Skeletons file not found: {skeletons_path}{Style.RESET_ALL}")
        return

    with open(skeletons_path) as f:
        skeletons = json.load(f)

    if args.skeleton_ids:
        wanted = {s.strip() for s in args.skeleton_ids.split(",")}
        skeletons = [s for s in skeletons if s["id"] in wanted]
        if not skeletons:
            print(f"{Fore.RED}No skeletons matched --skeleton-ids {wanted}{Style.RESET_ALL}")
            return
    elif args.limit_skeletons is not None:
        skeletons = skeletons[: args.limit_skeletons]

    print(f"\n{Fore.GREEN}{'='*60}")
    print(f" Multi-Turn Eval Generation (vLLM)")
    print(f" Skeletons:  {len(skeletons)}")
    print(f" Models:     {len(models_to_eval)}")
    print(f"{'='*60}{Style.RESET_ALL}")
    print(f"\nModels to evaluate:")
    for m in models_to_eval:
        meta = model_meta.get(m, {})
        kind = meta.get("kind", "legacy")
        print(f"  - {m} ({kind})")

    # ── Generate conversations ──────────────────────────────────────
    all_results = {}
    total_start = time.time()

    for model_idx, model_name in enumerate(models_to_eval):
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f" [{model_idx+1}/{len(models_to_eval)}] Generating: {model_name}")
        print(f"{'='*60}{Style.RESET_ALL}")

        model_start = time.time()

        meta = model_meta.get(model_name, {})
        model_results = {
            "model": model_name,
            "is_base": model_name == BASE_MODEL_NAME,
            "manifest": meta,
            "eval_loss": meta.get("eval_loss") or meta.get("metrics", {}).get("eval_loss"),
            "params": meta.get("params", {}),
            "conversations": [],
        }

        for skel_idx, skeleton in enumerate(skeletons):
            print(f"\n  {Fore.YELLOW}[{skel_idx+1}/{len(skeletons)}] "
                  f"Skeleton: {skeleton['id']} "
                  f"(type={skeleton['opening_type']}, "
                  f"target_turns={skeleton['target_turns']}){Style.RESET_ALL}")

            messages = run_skeleton(client, model_name, skeleton)

            model_results["conversations"].append({
                "skeleton_id": skeleton["id"],
                "opening_type": skeleton["opening_type"],
                "target_turns": skeleton["target_turns"],
                "actual_turns": len(messages),
                "messages": messages,
            })

        model_elapsed = time.time() - model_start
        print(f"\n  {Fore.GREEN}{model_name} done in {model_elapsed:.1f}s{Style.RESET_ALL}")

        all_results[model_name] = model_results

    total_elapsed = time.time() - total_start

    # ── Save results ────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_file = output_dir / args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filter_tag = f"_{args.filter}" if args.filter else ""
        top_tag = f"_top{args.top_n}" if args.top_n > 0 else "_all"
        output_file = output_dir / f"eval_generations{filter_tag}{top_tag}_{timestamp}.json"

    payload = {
        "generated_at": datetime.now().isoformat(),
        "skeletons_path": str(skeletons_path),
        "manifest": str(args.candidate_manifest) if args.candidate_manifest else None,
        "generation": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS,
            "base_url": VLLM_BASE_URL,
        },
        "models": all_results,
    }
    with open(output_file, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # ── Summary ─────────────────────────────────────────────────────
    total_convs = sum(len(r["conversations"]) for r in all_results.values())
    total_turns = sum(
        len(c["messages"])
        for r in all_results.values()
        for c in r["conversations"]
    )
    preview_results = all_results

    print(f"\n{Fore.GREEN}{'='*60}")
    print(f" GENERATION COMPLETE")
    print(f" Output:              {output_file}")
    print(f" Models evaluated:    {len(all_results)}")
    print(f" Total conversations: {total_convs}")
    print(f" Total turns:         {total_turns}")
    print(f" Wall time:           {total_elapsed:.1f}s")
    print(f"{'='*60}{Style.RESET_ALL}")

    # ── Quick preview ───────────────────────────────────────────────
    print(f"\n{Fore.CYAN}Preview of final assistant turns:{Style.RESET_ALL}")
    for model_name, results in preview_results.items():
        print(f"\n  {Fore.GREEN}{model_name}{Style.RESET_ALL}")
        for conv in results["conversations"][:3]:
            last_assistant = [m for m in conv["messages"] if m["role"] == "assistant"][-1]
            preview = last_assistant["content"][:120]
            if len(last_assistant["content"]) > 120:
                preview += "..."
            print(f"    {conv['skeleton_id']}: {preview}")
        if len(results["conversations"]) > 3:
            print(f"    ... and {len(results['conversations']) - 3} more")

    print(f"\nSend {output_file} to an LLM judge for scoring.")


if __name__ == "__main__":
    main()
