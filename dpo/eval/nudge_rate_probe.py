#!/usr/bin/env python3
"""Measure tool-emit RATE at a single turn to tell a nudge from a silent force.

A single scripted pass cannot distinguish a ~55%-trigger nudge from a 100% force. This sends
the SAME frozen conversation prefix N times and reports how often the model emits a tool call
on the target turn. A genuine nudge lands strictly between 0 and 100%; force is 100%.

Run on the machine with the agent up (deps up, run alone -- it reads the shared debug log by
byte offset, so concurrent traffic would corrupt the count):
    python dpo/eval/nudge_rate_probe.py --scenario camping --turn 3 --samples 20
    python dpo/eval/nudge_rate_probe.py --scenario camping_force_turn6 --turn 6 --samples 20
"""

from __future__ import annotations

import argparse

from model_prompt_matrix_eval import (
    AGENT_STDLOG,
    DEBUG_LOG,
    SCENARIOS,
    _http_json,
    _log_byte_offset,
    _parse_new_logs,
)


def _post(agent_url: str, model_name: str, messages: list[dict]) -> dict:
    return _http_json(
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


def main():
    ap = argparse.ArgumentParser(description="Nudge-rate probe (fixed-prefix sampling)")
    ap.add_argument("--scenario", required=True, choices=list(SCENARIOS))
    ap.add_argument("--turn", type=int, required=True, help="1-indexed target turn")
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--agent-url", default="http://127.0.0.1:9000")
    ap.add_argument("--model-name", default="dpo-v15-trial-4")
    args = ap.parse_args()

    lines = SCENARIOS[args.scenario]
    if not (1 <= args.turn <= len(lines)):
        raise SystemExit(f"--turn must be 1..{len(lines)} for scenario {args.scenario}")

    health = _http_json("GET", f"{args.agent_url.rstrip('/')}/health")
    print(f"agent: model={health.get('model')} prompt={health.get('prompt')}")

    # Freeze the prefix: run turns 1..T-1 once, capturing assistant replies.
    prefix: list[dict] = []
    for user_text in lines[: args.turn - 1]:
        prefix.append({"role": "user", "content": user_text})
        resp = _post(args.agent_url, args.model_name, prefix)
        assistant = resp.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        prefix.append({"role": "assistant", "content": assistant})

    target_user = lines[args.turn - 1]
    emits = 0
    for n in range(args.samples):
        debug_from = _log_byte_offset(DEBUG_LOG)
        agent_from = _log_byte_offset(AGENT_STDLOG)
        _post(args.agent_url, args.model_name, prefix + [{"role": "user", "content": target_user}])
        emitted, _executed, _count = _parse_new_logs(debug_from, agent_from)
        emits += int(emitted)
        print(f"  sample {n + 1:>2}/{args.samples}: tool_emitted={emitted}")

    rate = emits / args.samples if args.samples else 0.0
    print(f"\nscenario={args.scenario} turn={args.turn} samples={args.samples}")
    print(f"tool-emit rate = {emits}/{args.samples} = {rate:.0%}")
    if rate in (0.0, 1.0):
        print("  -> binary (0% or 100%): force or fully suppressed, NOT a probabilistic nudge.")
    else:
        print("  -> strictly between 0 and 100%: genuine nudge.")


if __name__ == "__main__":
    main()
