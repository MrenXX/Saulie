"""
Multi-Turn Eval Generation Script
==================================
Loads conversation skeletons (pre-scripted user turns) and generates
assistant responses turn-by-turn for the top N Optuna trial adapters.

Output is saved for LLM-as-judge scoring.

Usage:
    python eval_generate.py [--n_top 5] [--study_dir PATH]
"""

import os
import gc
import json
import argparse
from pathlib import Path
from datetime import datetime

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from colorama import Fore, Style, init as colorama_init
colorama_init(autoreset=True)


# ============================================================
# CONFIG — match these to train_sft.py
# ============================================================

ABS_PATH        = Path(r"/root/saulie")
OUTPUT_BASE     = ABS_PATH / "train" / "models"
MODEL_ID_FP8    = ABS_PATH / "Qwen3-4B-Instruct-2507-FP8"
MODEL_ID_BF16   = ABS_PATH / "Qwen3-4B-Instruct-2507"
USE_QUANT_VERSION = False

SKELETONS_PATH  = ABS_PATH / "train" / "sft_eval" / "eval_skeletons.json"
EXPERIMENT_NAME = "steering-sft-v1.2"

# Generation params
MAX_NEW_TOKENS  = 350
TEMPERATURE     = 0.7
TOP_P           = 0.8
TOP_K           = 20


# ============================================================
# HELPERS
# ============================================================

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_base_model():
    """Load the base model (no adapter)."""
    if USE_QUANT_VERSION:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID_FP8, dtype="auto", device_map={"": 0}
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID_FP8)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID_BF16,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map={"": 0}, dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID_BF16)

    tokenizer.padding_side = "right"
    return model, tokenizer


def generate_response(model, tokenizer, messages):
    """Generate a single assistant response given conversation history."""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # No thinking mode for Instruct
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )
    return response.strip()


def run_skeleton(model, tokenizer, skeleton):
    """
    Run a single conversation skeleton: feed user turns one at a time,
    generate assistant responses in between.

    Returns the full messages list (alternating user/assistant).
    """
    messages = []
    user_turns = skeleton["user_turns"]

    for i, user_msg in enumerate(user_turns):
        # Add user turn
        messages.append({"role": "user", "content": user_msg})

        # Generate assistant response
        response = generate_response(model, tokenizer, messages)
        messages.append({"role": "assistant", "content": response})

        is_last = (i == len(user_turns) - 1)
        status = "FINAL (should contain recommendation)" if is_last else "intermediate"
        print(f"    Turn {i+1}/{len(user_turns)} [{status}]: {len(response)} chars")

    return messages


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-turn eval generation")
    parser.add_argument("--n_top", type=int, default=5, help="Number of top trials to evaluate")
    parser.add_argument("--experiment", type=str, default=EXPERIMENT_NAME, help="Experiment name")
    parser.add_argument("--include_base", action="store_true", help="Also generate with the base model (no adapter) as a baseline")
    args = parser.parse_args()

    experiment_dir = OUTPUT_BASE / args.experiment

    # --- Load skeletons ---
    if not SKELETONS_PATH.exists():
        print(f"{Fore.RED}Skeletons file not found: {SKELETONS_PATH}{Style.RESET_ALL}")
        return

    with open(SKELETONS_PATH) as f:
        skeletons = json.load(f)

    print(f"\n{Fore.GREEN}{'='*60}")
    print(f" Multi-Turn Eval Generation")
    print(f" Experiment: {args.experiment}")
    print(f" Skeletons: {len(skeletons)}")
    print(f" Top trials: {args.n_top}")
    print(f"{'='*60}{Style.RESET_ALL}\n")

    # --- Load trial summary to find top trials ---
    summary_path = experiment_dir / "trial_summary.json"
    if not summary_path.exists():
        print(f"{Fore.RED}Trial summary not found: {summary_path}{Style.RESET_ALL}")
        print(f"Run train_sft.py first to complete the Optuna study.")
        return

    with open(summary_path) as f:
        trial_summary = json.load(f)

    # Already sorted by eval_loss in train_sft.py
    top_trials = trial_summary[:args.n_top]
    print(f"Top {len(top_trials)} trials by eval_loss:")
    for t in top_trials:
        print(f"  trial-{t['trial']}: eval_loss={t['eval_loss']:.4f}")

    # --- Build list of models to evaluate ---
    models_to_eval = []

    if args.include_base:
        models_to_eval.append({
            "name": "base-model",
            "adapter_path": None,
            "eval_loss": None,
            "params": {},
        })

    for t in top_trials:
        adapter_path = experiment_dir / f"trial-{t['trial']}" / "best_adapter"
        if adapter_path.exists():
            models_to_eval.append({
                "name": f"trial-{t['trial']}",
                "adapter_path": str(adapter_path),
                "eval_loss": t["eval_loss"],
                "params": t["params"],
            })
        else:
            print(f"{Fore.YELLOW}Adapter not found for trial-{t['trial']}, skipping.{Style.RESET_ALL}")

    # --- Generate conversations for each model ---
    all_results = {}

    for model_info in models_to_eval:
        model_name = model_info["name"]
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f" Loading: {model_name}")
        if model_info["eval_loss"]:
            print(f" eval_loss: {model_info['eval_loss']:.4f}")
        print(f"{'='*60}{Style.RESET_ALL}")

        clear_gpu()

        # Load model
        base_model, tokenizer = load_base_model()

        if model_info["adapter_path"]:
            model = PeftModel.from_pretrained(base_model, model_info["adapter_path"])
        else:
            model = base_model

        model.eval()

        # Run all skeletons
        model_results = {
            "model": model_name,
            "eval_loss": model_info["eval_loss"],
            "params": model_info["params"],
            "conversations": [],
        }

        for skeleton in skeletons:
            print(f"\n  {Fore.YELLOW}Skeleton: {skeleton['id']} "
                  f"(type={skeleton['opening_type']}, "
                  f"target_turns={skeleton['target_turns']}){Style.RESET_ALL}")

            messages = run_skeleton(model, tokenizer, skeleton)

            model_results["conversations"].append({
                "skeleton_id": skeleton["id"],
                "opening_type": skeleton["opening_type"],
                "target_turns": skeleton["target_turns"],
                "actual_turns": len(messages),
                "messages": messages,
            })

        all_results[model_name] = model_results

        del model, base_model
        clear_gpu()

    # --- Save results ---
    output_file = experiment_dir / "multiturn_eval_results_v2.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{Fore.GREEN}{'='*60}")
    print(f" GENERATION COMPLETE")
    print(f" Results: {output_file}")
    print(f" Models evaluated: {len(all_results)}")
    print(f" Conversations per model: {len(skeletons)}")
    print(f" Total conversations: {len(all_results) * len(skeletons)}")
    print(f"{'='*60}{Style.RESET_ALL}")

    # --- Print quick summary ---
    print(f"\n{Fore.CYAN}Quick preview of final assistant turns:{Style.RESET_ALL}")
    for model_name, results in all_results.items():
        print(f"\n  {Fore.GREEN}{model_name}{Style.RESET_ALL}")
        for conv in results["conversations"][:3]:  # Show first 3
            last_assistant = [m for m in conv["messages"] if m["role"] == "assistant"][-1]
            preview = last_assistant["content"][:120] + "..." if len(last_assistant["content"]) > 120 else last_assistant["content"]
            print(f"    {conv['skeleton_id']}: {preview}")
        if len(results["conversations"]) > 3:
            print(f"    ... and {len(results['conversations']) - 3} more")

    print(f"\nSend {output_file} to an LLM judge for scoring.")


if __name__ == "__main__":
    main()
