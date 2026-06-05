"""Runtime LoRA load/unload for vLLM final eval (no container restart per trial)."""

from __future__ import annotations

import time
from typing import Any

import requests
from openai import OpenAI

from dpo.eval.v15_eval_config import (
    VLLM_API_KEY,
    VLLM_BASE_URL,
    VLLM_LOAD_LORA_URL,
    VLLM_UNLOAD_LORA_URL,
)


class VllmLoraError(RuntimeError):
    pass


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {VLLM_API_KEY}", "Content-Type": "application/json"}


def load_lora_adapter(lora_name: str, lora_path: str, *, timeout: float = 300.0) -> None:
    """POST /v1/load_lora_adapter — path must exist inside the container."""
    payload = {"lora_name": lora_name, "lora_path": lora_path}
    resp = requests.post(VLLM_LOAD_LORA_URL, json=payload, headers=_headers(), timeout=timeout)
    if resp.status_code == 200:
        print(f"  loaded LoRA {lora_name!r} from {lora_path}")
        return
    if resp.status_code == 400 and "already been loaded" in resp.text:
        print(f"  LoRA {lora_name!r} already loaded on server — continuing")
        return
    raise VllmLoraError(f"load_lora_adapter failed ({resp.status_code}): {resp.text}")


def unload_lora_adapter(lora_name: str, *, timeout: float = 120.0) -> None:
    payload = {"lora_name": lora_name}
    resp = requests.post(VLLM_UNLOAD_LORA_URL, json=payload, headers=_headers(), timeout=timeout)
    if resp.status_code != 200:
        raise VllmLoraError(f"unload_lora_adapter failed ({resp.status_code}): {resp.text}")
    print(f"  unloaded LoRA {lora_name!r}")


def wait_for_model(
    client: OpenAI,
    model_name: str,
    *,
    timeout_s: float = 120.0,
    poll_s: float = 2.0,
) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        names = [m.id for m in client.models.list().data]
        if model_name in names:
            return
        time.sleep(poll_s)
    raise VllmLoraError(f"model {model_name!r} not listed after {timeout_s}s")


def ensure_dpo_adapter_loaded(entry: dict[str, Any]) -> None:
    if not entry.get("runtime_load"):
        return
    # Clear stale adapter from a crashed prior run before load.
    try:
        unload_lora_adapter(entry["model_name"])
    except VllmLoraError:
        pass
    load_lora_adapter(entry["model_name"], entry["container_path"])
    client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    wait_for_model(client, entry["model_name"])


def ensure_dpo_adapter_unloaded(entry: dict[str, Any]) -> None:
    if not entry.get("runtime_load"):
        return
    try:
        unload_lora_adapter(entry["model_name"])
    except VllmLoraError as exc:
        print(f"  WARNING: unload {entry['model_name']}: {exc}")
