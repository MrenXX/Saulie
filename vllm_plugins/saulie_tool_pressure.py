"""First-token-only tool-call pressure for Qwen3 Hermes (<tool_call> = token 151657).

Registered at vLLM serve startup:
  --logits-processors vllm_plugins.saulie_tool_pressure:SaulieToolPressureWrapper

Per-request config via OpenAI extra_body.vllm_xargs (v0.11: scalar values only):
  tool_pressure_enabled: 0 | 1
  tool_pressure_mode: off | nudge | force
  tool_pressure_bias: int (nudge only, capped at 100)
  tool_pressure_opener_ids: comma-separated token ids (default "151657")
"""

from __future__ import annotations

from typing import Callable

import torch

from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

RequestLogitsProcessor = Callable[[list[int], torch.Tensor], torch.Tensor]

OPENER_TOKEN_ID = 151657


def _parse_opener_ids(raw: str | int | float | None) -> list[int]:
    if raw is None:
        return [OPENER_TOKEN_ID]
    text = str(raw).strip()
    if not text:
        return [OPENER_TOKEN_ID]
    return [int(part) for part in text.split(",") if part.strip()]


class _SaulieToolPressurePerReq:
    __slots__ = ("_bias", "_mode", "_opener_ids")

    def __init__(self, mode: str, bias: float, opener_ids: list[int]) -> None:
        self._mode = mode
        self._bias = min(100.0, max(0.0, float(bias)))
        self._opener_ids = opener_ids

    def __call__(self, output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        if self._mode == "off":
            return logits

        if any(token_id in output_ids for token_id in self._opener_ids):
            return logits

        if len(output_ids) > 0:
            return logits

        valid_ids = [token_id for token_id in self._opener_ids if 0 <= token_id < logits.shape[-1]]
        if not valid_ids:
            return logits

        if self._mode == "nudge":
            for token_id in valid_ids:
                logits[token_id] += self._bias
            return logits

        if self._mode == "force":
            kept = logits[valid_ids].clone()
            logits[:] = float("-inf")
            logits[valid_ids] = kept
            return logits

        return logits


class SaulieToolPressureWrapper(AdapterLogitsProcessor):
    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params) -> RequestLogitsProcessor | None:
        extra = params.extra_args or {}
        if int(extra.get("tool_pressure_enabled", 0)) == 0:
            return None

        mode = str(extra.get("tool_pressure_mode", "off")).strip().lower()
        if mode not in {"nudge", "force"}:
            return None

        opener_ids = _parse_opener_ids(extra.get("tool_pressure_opener_ids"))
        bias = float(extra.get("tool_pressure_bias", 0))
        return _SaulieToolPressurePerReq(mode=mode, bias=bias, opener_ids=opener_ids)
