#!/usr/bin/env python3
"""Run scripted multi-turn conversations against agent_chat_api for matrix comparison."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEBUG_LOG = REPO / ".cursor" / "debug-049191.log"
AGENT_STDLOG = REPO / "agent_api.log"
OUTPUT_DIR = REPO / "dpo" / "eval" / "matrix_runs"

# Dependency endpoints for preflight (must match rag/query2.py defaults).
EMBED_URL = os.getenv("EMBED_URL", "http://localhost:8888/embed")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:1234")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "amazon_products_v2")

SCENARIOS: dict[str, list[str]] = {
    "camping": [
        "hello",
        "been camping with the boys, something feels missing",
        "tent is brand new, not the problem",
        "getting cold, sleeping isn't comfy",
        "huh that sounds cool you got anything specific?",
        "USE YOUR SEARCH_PRODUCTS TOOL TO GIVE ME A SPECIFIC PRODUCT",
    ],
    "camping_force_turn6": [
        "hello",
        "been camping with the boys, something feels missing",
        "tent is brand new, not the problem",
        "getting cold, sleeping isn't comfy",
        "huh that sounds cool you got anything specific?",
        "alright just show me what you've got",
    ],
    "kitchen": [
        "hello",
        "learning cooking lately, hate cleanup after prep not the cooking itself",
        "I just said cleanup is the problem not prep",
        "fine ok find me something",
        "still waiting for you to call the tool",
    ],
    "direct": [
        "find me a kids desk lamp with a focused beam for homework",
    ],
}

TOOL_EMIT_RE = re.compile(r'"tool_call_count"\s*:\s*([1-9]\d*)')
TOOL_EXEC_RE = re.compile(r"executing tools")
RESULT_COUNT_RE = re.compile(r'"result_count"\s*:\s*(\d+)')
HALLUCINATE_RE = re.compile(
    r"\$\d|R-value|closed-cell|2-inch|amazon\.com|https?://",
    re.I,
)


@dataclass
class TurnResult:
    turn: int
    user: str
    assistant: str
    tool_emitted: bool
    tool_executed: bool
    elapsed_s: float
    likely_hallucinated_product: bool
    results_count: int = 0


@dataclass
class ScenarioResult:
    scenario: str
    turns: list[TurnResult] = field(default_factory=list)

    @property
    def any_tool_emitted(self) -> bool:
        return any(t.tool_emitted for t in self.turns)

    @property
    def any_tool_executed(self) -> bool:
        return any(t.tool_executed for t in self.turns)


@dataclass
class CellResult:
    model_key: str
    model_name: str
    prompt: str
    scenarios: list[ScenarioResult] = field(default_factory=list)

    @property
    def tool_emit_turns(self) -> list[str]:
        out = []
        for sc in self.scenarios:
            for t in sc.turns:
                if t.tool_emitted:
                    out.append(f"{sc.scenario} turn {t.turn}")
        return out

    @property
    def tool_exec_turns(self) -> list[str]:
        out = []
        for sc in self.scenarios:
            for t in sc.turns:
                if t.tool_executed:
                    out.append(f"{sc.scenario} turn {t.turn}")
        return out


def _http_json(method: str, url: str, body: dict | None = None, timeout: float = 180) -> dict:
    data = None
    headers = {"Content-Type": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _log_byte_offset(path: Path) -> int:
    if not path.is_file():
        return 0
    return path.stat().st_size


def _parse_new_logs(debug_from: int, agent_from: int) -> tuple[bool, bool, int]:
    tool_emitted = False
    tool_executed = False
    results_count = 0

    if DEBUG_LOG.is_file():
        with DEBUG_LOG.open("rb") as f:
            f.seek(debug_from)
            chunk = f.read().decode("utf-8", errors="replace")
        if TOOL_EMIT_RE.search(chunk):
            tool_emitted = True
        if TOOL_EXEC_RE.search(chunk):
            tool_executed = True
        counts = [int(m) for m in RESULT_COUNT_RE.findall(chunk)]
        if counts:
            results_count = max(counts)

    if AGENT_STDLOG.is_file():
        with AGENT_STDLOG.open("rb") as f:
            f.seek(agent_from)
            chunk = f.read().decode("utf-8", errors="replace")
        if "[TOOL CALL]" in chunk:
            tool_executed = True

    return tool_emitted, tool_executed, results_count


def _preflight_deps() -> None:
    """Fail loudly if the embedding server or Qdrant is down/empty, so a run never silently
    produces 'no_results' everywhere (the failure mode that invalidated the first smoke)."""
    try:
        emb = _http_json("POST", EMBED_URL, {"text": ["preflight ping"]}, timeout=30)
        if not isinstance(emb, dict) or "dense" not in emb:
            raise RuntimeError("embed response missing 'dense'")
    except SystemExit:
        raise
    except Exception as e:
        raise SystemExit(
            f"PREFLIGHT FAIL: embedding server unreachable at {EMBED_URL} ({e}). "
            "Start tensorrt_bge-m3 (:8888) before running."
        )
    try:
        info = _http_json("GET", f"{QDRANT_URL.rstrip('/')}/collections/{QDRANT_COLLECTION}", timeout=15)
        count = (info.get("result") or {}).get("points_count")
        if not count:
            raise RuntimeError(f"collection '{QDRANT_COLLECTION}' empty/missing (points_count={count})")
    except SystemExit:
        raise
    except Exception as e:
        raise SystemExit(
            f"PREFLIGHT FAIL: Qdrant unreachable/empty at {QDRANT_URL} ({e}). "
            f"Start qdrant_index (:1234) and confirm '{QDRANT_COLLECTION}' is indexed."
        )
    print(f"Preflight OK: embed {EMBED_URL} reachable, Qdrant '{QDRANT_COLLECTION}' has {count} points.")


def run_cell(model_key: str, model_name: str, prompt: str, agent_url: str) -> CellResult:
    health = _http_json("GET", f"{agent_url.rstrip('/')}/health")
    if health.get("model") != model_name or health.get("prompt") != prompt:
        raise RuntimeError(
            f"Agent mismatch: expected model={model_name} prompt={prompt}, got {health}"
        )

    cell = CellResult(model_key=model_key, model_name=model_name, prompt=prompt)

    for scenario_name, user_lines in SCENARIOS.items():
        messages: list[dict] = []
        scenario = ScenarioResult(scenario=scenario_name)

        for i, user_text in enumerate(user_lines, start=1):
            messages.append({"role": "user", "content": user_text})
            debug_from = _log_byte_offset(DEBUG_LOG)
            agent_from = _log_byte_offset(AGENT_STDLOG)
            t0 = time.perf_counter()

            resp = _http_json(
                "POST",
                f"{agent_url.rstrip('/')}/v1/chat/completions",
                {
                    "model": model_name,
                    "messages": messages,
                    "stream": False,
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "max_tokens": 512,
                },
                timeout=300,
            )
            elapsed = time.perf_counter() - t0
            assistant = (
                resp.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                or ""
            )
            messages.append({"role": "assistant", "content": assistant})

            tool_emitted, tool_executed, results_count = _parse_new_logs(debug_from, agent_from)
            scenario.turns.append(
                TurnResult(
                    turn=i,
                    user=user_text,
                    assistant=assistant,
                    tool_emitted=tool_emitted,
                    tool_executed=tool_executed,
                    elapsed_s=round(elapsed, 2),
                    likely_hallucinated_product=bool(
                        HALLUCINATE_RE.search(assistant) and not tool_executed
                    ),
                    results_count=results_count,
                )
            )

        cell.scenarios.append(scenario)

    return cell


def main():
    parser = argparse.ArgumentParser(description="Model×prompt matrix eval cell")
    parser.add_argument("--model-key", required=True, help="sft | dpo_w05 | dpo_w10")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--prompt", required=True, choices=["legacy", "steering", "compressed"])
    parser.add_argument("--agent-url", default="http://127.0.0.1:9000")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--skip-preflight", action="store_true",
                        help="skip embed/Qdrant health checks (not recommended)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.output or OUTPUT_DIR / f"{args.model_key}_{args.prompt}.json"

    if not args.skip_preflight:
        _preflight_deps()

    cell = run_cell(args.model_key, args.model_name, args.prompt, args.agent_url)

    exec_turns = [t for sc in cell.scenarios for t in sc.turns if t.tool_executed]
    if exec_turns and all(t.results_count == 0 for t in exec_turns):
        raise SystemExit(
            "FAIL: every executed search returned 0 results across all scenarios. "
            "Embed/Qdrant is down, the index is stale, or the cosine gate is too strict "
            "(SAULIE_RAG_MIN_COSINE). Results are not trustworthy -- aborting."
        )
    payload = asdict(cell)
    payload["summary"] = {
        "any_tool_emitted": any(s.any_tool_emitted for s in cell.scenarios),
        "any_tool_executed": any(s.any_tool_executed for s in cell.scenarios),
        "tool_emit_turns": cell.tool_emit_turns,
        "tool_exec_turns": cell.tool_exec_turns,
        "hallucination_turns": [
            f"{sc.scenario} turn {t.turn}"
            for sc in cell.scenarios
            for t in sc.turns
            if t.likely_hallucinated_product
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
