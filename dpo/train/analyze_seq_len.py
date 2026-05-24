"""
DPO dataset token length analysis.

Tokenizer: Qwen3-4B-Instruct-2507 (same base as steering-sft-v1.1 trial-17).
Data: DPO_522_prompt_a_and_prompt_b_V4_repaired.jsonl (522 rows).
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoTokenizer

from dpo.train.paths import DATA_PATH, MODEL_ID_FP8, SFT_ADAPTER


def token_len(tokenizer, messages: list[dict]) -> int:
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return len(tokenizer(text, add_special_tokens=True)["input_ids"])


def dpo_conversation(prompt: list[dict], completion: list[dict]) -> list[dict]:
    return list(prompt) + list(completion)


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_ID_FP8))
    print(f"Tokenizer: {MODEL_ID_FP8}")
    print(f"DPO base SFT (trial-17): {SFT_ADAPTER}")
    print(f"Dataset: {DATA_PATH}\n")

    chosen_lengths = []
    rejected_lengths = []
    pair_max_lengths = []

    with DATA_PATH.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row["prompt"]
            chosen_msgs = dpo_conversation(prompt, row["chosen"])
            rejected_msgs = dpo_conversation(prompt, row["rejected"])

            clen = token_len(tokenizer, chosen_msgs)
            rlen = token_len(tokenizer, rejected_msgs)
            chosen_lengths.append(clen)
            rejected_lengths.append(rlen)
            pair_max_lengths.append(max(clen, rlen))

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1} rows...")

    overall_max = max(pair_max_lengths)

    print(f"\n{'=' * 50}")
    print("DPO token length statistics (Qwen3-4B chat template)")
    print(f"{'=' * 50}")
    print(f"Pairs: {len(pair_max_lengths)}")
    print(f"Max (chosen branch):   {max(chosen_lengths)}")
    print(f"Max (rejected branch): {max(rejected_lengths)}")
    print(f"Max per pair (max of chosen/rejected): {overall_max}")
    print(f"Min per pair:  {min(pair_max_lengths)}")
    print(f"Avg per pair:  {sum(pair_max_lengths) / len(pair_max_lengths):.1f}")

    sorted_lengths = sorted(pair_max_lengths)
    for p in (50, 75, 90, 95, 99):
        idx = min(int(len(sorted_lengths) * p / 100), len(sorted_lengths) - 1)
        print(f"P{p} per pair:  {sorted_lengths[idx]}")

    print(f"\nRecommended MAX_SEQ_LEN (pair max + 50 headroom): {overall_max + 50}")


if __name__ == "__main__":
    main()
