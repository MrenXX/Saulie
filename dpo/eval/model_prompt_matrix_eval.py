#!/usr/bin/env python3
"""Run scripted multi-turn conversations against agent_chat_api for matrix comparison."""

from __future__ import annotations

import argparse
import json
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


def _parse_new_logs(debug_from: int, agent_from: int) -> tuple[bool, bool]:
    tool_emitted = False
    tool_executed = False

    if DEBUG_LOG.is_file():
        with DEBUG_LOG.open("rb") as f:
            f.seek(debug_from)
            chunk = f.read().decode("utf-8", errors="replace")
        if TOOL_EMIT_RE.search(chunk):
            tool_emitted = True
        if TOOL_EXEC_RE.search(chunk):
            tool_executed = True

    if AGENT_STDLOG.is_file():
        with AGENT_STDLOG.open("rb") as f:
            f.seek(agent_from)
            chunk = f.read().decode("utf-8", errors="replace")
        if "[TOOL CALL]" in chunk:
            tool_executed = True

    return tool_emitted, tool_executed


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
            t0 = time.time()

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
            elapsed = time.time() - t0
            assistant = (
                resp.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                or ""
            )
            messages.append({"role": "assistant", "content": assistant})

            tool_emitted, tool_executed = _parse_new_logs(debug_from, agent_from)
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
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.output or OUTPUT_DIR / f"{args.model_key}_{args.prompt}.json"

    cell = run_cell(args.model_key, args.model_name, args.prompt, args.agent_url)
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
