"""
Qwen3 non-thinking decode defaults (Qwen3-4B-Instruct-2507 README).

Greedy decoding is discouraged: Qwen documents it can cause endless repetitions.
HF Transformers has no ``presence_penalty``; use ``top_k`` + sampling, optional
``repetition_penalty`` (e.g. 1.05) if loops persist.
"""

from __future__ import annotations

import warnings

DECODE_SAMPLE = "sample"
DECODE_GREEDY = "greedy"

MAX_NEW_TOKENS_DEFAULT = 350
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
REPETITION_PENALTY_DEFAULT = 1.0  # 1.0 = off; try 1.05 on HF if loops remain


def build_generate_kwargs(
    tokenizer,
    *,
    decode: str = DECODE_SAMPLE,
    max_new_tokens: int = MAX_NEW_TOKENS_DEFAULT,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    top_k: int = TOP_K,
    repetition_penalty: float = REPETITION_PENALTY_DEFAULT,
) -> dict:
    kwargs: dict = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if decode == DECODE_GREEDY:
        warnings.warn(
            "Greedy decode is discouraged for Qwen3 (risk of repetition loops). "
            "Use sample with temperature=0.7, top_p=0.8, top_k=20.",
            UserWarning,
            stacklevel=3,
        )
        kwargs["do_sample"] = False
        return kwargs
    if decode != DECODE_SAMPLE:
        raise ValueError(f"unknown decode mode: {decode!r}")
    kwargs.update(
        {
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }
    )
    if repetition_penalty != 1.0:
        kwargs["repetition_penalty"] = repetition_penalty
    return kwargs


def add_decode_argparse(parser, *, default_decode: str = DECODE_SAMPLE) -> None:
    """Register standard Qwen3 HF decode flags on an argparse parser."""
    parser.add_argument(
        "--decode",
        choices=(DECODE_SAMPLE, DECODE_GREEDY),
        default=default_decode,
        help="Default sample per Qwen3-4B-Instruct-2507. Greedy discouraged.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS_DEFAULT)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--repetition-penalty",
        "--repetition_penalty",
        type=float,
        default=REPETITION_PENALTY_DEFAULT,
        dest="repetition_penalty",
        help="HF only; 1.0=off. Try 1.05 if loops persist.",
    )


def generation_metadata(
    *,
    decode: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> dict:
    meta = {"decode": decode, "max_new_tokens": max_new_tokens}
    if decode == DECODE_SAMPLE:
        meta.update(
            {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "qwen3_non_thinking": True,
            }
        )
    return meta
