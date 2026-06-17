# Tool-Pressure & Retrieval Handoff

**Date:** 2026-06-17
**Base commit:** `0e4cbe0` on `main` (the patch in Appendix A applies cleanly on this).
**Scope:** smarter tool-call pressure (non-binary), tool-aware reset, dense-cosine retrieval
relevance gate, eval hardening, and the scripts to calibrate / A-B test on the GPU machine.

This document is the single source of truth for **what to change**, **why**, and **what to
run** on the machine that has the GPU + embed server + Qdrant. Nothing here has been executed
(the authoring environment had no model/RAG/Docker access); every change is lint-clean and
every runtime claim must be confirmed by the runbook below.

> **Why you (the dev-machine agent) must apply these changes yourself.** The work was authored
> and committed on a locked corporate laptop whose proxy/DLP blocks `git push` to GitHub (the
> code upload 403s after object negotiation, independent of credentials). So the changes could
> not be pushed. They are delivered as **exact code edits** (Appendix A) — plus, if you have the
> repo folder, a ready-made unified-diff patch file. Apply them, confirm lint, then run the
> staged runbook in Section 5. **You are implementing, not just running.**

---

## 0. How to apply these changes

The complete set of edits is in **Appendix A** (7 files: 4 modified as exact find -> replace
blocks, 3 new files in full). Pick whichever method fits how this reached you.

**Method 1 — git apply (only if you have the repo folder).** The repo root contains a
ready-made unified-diff file `saulie_tool_pressure_rag.patch`, verified to apply cleanly on base
commit `0e4cbe0`:

```bash
git fetch origin && git checkout -b fix/tool-pressure-reset-and-rag-gate origin/main
git apply --check saulie_tool_pressure_rag.patch   # dry run; must print nothing
git apply saulie_tool_pressure_rag.patch           # apply for real
git status                                         # 4 modified, 3 new files (see Section 6)
```

If `--check` complains the base differs, try `git apply --3way saulie_tool_pressure_rag.patch`
(3-way merge using blob context). If that still fails, fall back to Method 2.

**Method 2 — apply Appendix A directly (works from this doc alone).** This is the primary path
if only the Markdown reached you. For each modified file, do the exact find -> replace shown in
Appendix A; for each new file, create it verbatim from the full listing. Section 3 explains the
rationale behind every edit.

**After applying (either method):**

```bash
python -c "import ast; [ast.parse(open(f, encoding='utf-8').read()) for f in ['agent_chat_api.py','rag/query2.py','dpo/eval/model_prompt_matrix_eval.py','rag/calibrate_cosine_threshold.py','rag/retrieval_ab_eval.py','dpo/eval/nudge_rate_probe.py']]" && echo "syntax OK"
```

Then update `.env` from `.env.example` (the new `SAULIE_RAG_*` and changed
`SAULIE_TOOL_BIAS_PER_TURN` keys) and proceed to Section 5.

> The original commit author was set to `Amine-Marnewi-EY`. When you commit on the dev machine,
> use whatever identity is correct there — re-author freely.

---

## 1. Problem statement

Production model `dpo-v15-trial-4` + `compressed` prompt under-searches: it probes fine but
then **hallucinates product pitches instead of calling `search_products`** unless the user
explicitly screams "USE THE TOOL". We wanted something smarter than the binary
`tool_choice="auto"` vs `"required"` — graduated pressure that nudges first and only forces
late. A custom vLLM logits processor (`vllm_plugins/saulie_tool_pressure.py`) biases the
first reply token toward the `<tool_call>` opener (Qwen3 id **151657**). That part shipped
earlier; this handoff fixes the bugs found reviewing it, plus a retrieval-quality issue.

Out of scope (deliberately not addressed): occasional broken-English from DPO. Per manual
testing, tool-call *formation* is reliable whenever it fires, so forcing the opener is safe.

---

## 2. Root-cause findings (verified by reading code/data)

### 2.1 The cross-turn pressure reset was dead
- `_pressure_turn` reset by scanning history for `role="tool"`, but `sanitize_messages`
  **strips all tool messages** from incoming client history, and the API never echoes tool
  messages back to clients. At the decision point (`current_cycle == 0`) history has zero
  tool messages, so the counter equalled **total user turns in the whole conversation**.
- `SAULIE_TOOL_PRESSURE_MOD` was a blind modulo on that total — **not** tool-aware.
- Net bug: say a search fired on turn 5; on turn 6 the user says "thanks man" — the counter
  reads 6, hits force, and the model is compelled to open with `<tool_call>`. **Over-trigger.**

### 2.2 The "nudge" was a silent force
- `bias = min(turn * PER_TURN, 100)` with `PER_TURN=8` => +24/+32/+40 at turns 3/4/5. For a
  4B model those absolute logit adds **saturate the softmax** — the opener becomes argmax
  deterministically. The graduated ladder was cosmetic; every rung forced. Evidence: the
  prior smoke had camping/kitchen searching on turn 3 and never stopping.

### 2.3 RRF score is rank-based, not relevance-based
- Qdrant's server-side `FusionQuery(RRF)` hardcodes **k=2** (no k parameter exists in the
  API). Proven by `rag/fusion_benchmark_results.json`: scores `0.5` and `0.4167` **exceed**
  the k=5 maximum of 0.4, and are exact under k=2 (`1/(2+rank)` summed over dense+sparse).
- Because any non-empty prefetch always has a rank-0 item contributing 0.5, **the top fused
  hit is ~always >= 0.5 regardless of relevance.** A sneaker tops "wireless earbuds" at 0.5;
  a cooking apron tops "bluetooth speaker waterproof" at 0.5. So an RRF threshold cannot gate.
- The dense vectors are a **COSINE** collection (`rag/index2.py`, `rag/index_mccauley.py`),
  so dense cosine in [0,1] is the true semantic relevance signal. **RRF still ranks; cosine
  only gates.** These are complementary, not competing — RRF's strength on short product
  titles is preserved.

### 2.4 The earlier smoke's empty RAG was a dead dependency, not Qdrant
- `get_server_embeddings` catches **all** exceptions and returns `None`; `search_hybrid` then
  returns `[]`; `execute_search` reports a clean `no_results`. With no score threshold a
  populated index basically always returns hits, so uniform empties across every query meant
  the **BGE-M3 embed server (:8888)** (or Qdrant :1234) was down. `deploy_finalist_pick.sh`
  only starts vLLM; embed + Qdrant are manual prerequisites. Hence the new preflight (3.3).

---

## 3. Changes implemented

### 3.1 `agent_chat_api.py` — tool-aware pressure reset + bias default
- New server-side **`_PRESSURE_REGISTRY`** (bounded LRU `OrderedDict`, cap
  `SAULIE_TOOL_PRESSURE_REGISTRY_MAX=512`) mapping a **user-message fingerprint** ->
  user-turn count at which a search last fired.
- New `_user_fingerprint(user_contents)` (SHA1 over user message contents, order-sensitive)
  and `_note_search_fired(history)` (records the fingerprint of all user msgs when a search
  fires). Wired at **both** `current_cycle += 1` sites (non-stream `agent_loop`, streaming
  `_stream_agent_sse_inner`).
- `_pressure_turn` rewritten: looks up the registry by the fingerprint of all user msgs
  **except the latest** (== the key stored when the previous turn's search fired) and returns
  `total_user_turns - last_search_turn`; falls back to full count for a new conversation.
  - Trace: search fires at user-turn 5 -> store `fp(u1..u5)=5`. Next request adds u6 ->
    `_pressure_turn` looks up `fp(u1..u5)` -> 5 -> returns `6-5=1` -> **off**. "thanks man"
    can no longer force. A genuine dry spell climbs 1,2,3... and nudges again at 3.
  - Concurrency-safe: distinct conversations have distinct fingerprints (no shared global
    counter). Worst case under fingerprint collision is a mistimed nudge, never cross-talk.
- **Bias default lowered `SAULIE_TOOL_BIAS_PER_TURN` 8 -> 2** (+6/+8/+10). This is a real
  probabilistic nudge. Force at turn 6 remains the hard guarantee.
- `execute_search` now counts **total hits across sub-queries** (`total_hits`) and returns
  `no_results` when zero. This also fixes a latent gap: previously a per-sub-query empty
  retrieval still returned status `ok` with an empty inner list.

### 3.2 `rag/query2.py` — dense-cosine relevance gate (toggleable)
- New env: `SAULIE_RAG_MIN_COSINE` (default `0.5`), `SAULIE_RAG_MAX_RESULTS` (default `5`).
- `search_hybrid` per sub-query:
  - **Gate ON** (`MIN_COSINE > 0`): keep the hybrid RRF query for ranking, add **one
    dense-only `query_points`** (same dense vector, same category filter, `with_payload=False`)
    to get `id -> cosine`; return RRF-ordered hits with `cosine >= MIN_COSINE`, capped at
    `MAX_RESULTS`, each annotated with a `relevance` field. Zero above threshold -> empty ->
    `execute_search` emits `no_results`.
  - **Gate OFF** (`MIN_COSINE <= 0`): **pure RRF baseline** — top `MAX_RESULTS` by fused rank,
    no dense pass (no extra round-trip). This is the A/B baseline config.
- RRF `k` is fixed at 2 by Qdrant (the `FusionQuery` API exposes no k parameter); RRF ranking
  is used as-is and preserved identically in both configs. Only the relevance gate differs.

### 3.3 `dpo/eval/model_prompt_matrix_eval.py` — eval hardening
- `_preflight_deps()`: probes embed `:8888` (POST) and Qdrant collection `points_count`;
  **aborts with a clear error** if either is down/empty, so a run never silently produces
  all-`no_results` again. Skippable with `--skip-preflight`.
- Timing switched `time.time()` -> `time.perf_counter()` (the old negative `elapsed_s` bug).
- New per-turn `results_count` parsed from the debug log; **hard-fail if every executed
  search returned 0 results** across all scenarios (dependency / stale-index / over-strict
  gate signal).

### 3.4 New scripts (machine-run)
- `rag/calibrate_cosine_threshold.py` — runs the 18 labeled benchmark queries dense-only,
  labels hits with a keyword **RUBRIC** (reproduces the manual method behind
  `fusion_comparison_report.md`), sweeps cosine for best F1, prints a recommended
  `SAULIE_RAG_MIN_COSINE` + a top-hit table + an **overlap warning** that doubles as a
  stale/corrupt dense-index check (see README `SERVER_BATCH_SIZE` history).
- `rag/retrieval_ab_eval.py` — runs **both** configs (A: pure RRF, B: RRF+gate) over the 18
  queries and reports precision, top-1 accuracy, coverage, avg returned, and no-result count
  side by side, with a verdict. This is how you settle "is the gate actually better than RRF".
- `dpo/eval/nudge_rate_probe.py` — fixed-prefix **N-sample** tool-emit RATE at one turn, to
  prove a nudge is probabilistic (0 < rate < 1) and force is 100% (a single scripted pass
  cannot tell those apart).

### 3.5 `.env.example`
- Documented `SAULIE_RAG_MIN_COSINE` / `SAULIE_RAG_MAX_RESULTS` (incl. the `<=0` toggle),
  `SAULIE_TOOL_BIAS_PER_TURN=2`, `SAULIE_TOOL_PRESSURE_REGISTRY_MAX`.

---

## 4. New / changed environment variables

| Var | Default | Meaning |
|-----|---------|---------|
| `SAULIE_TOOL_BIAS_PER_TURN` | `2` | Absolute logit add per effective turn (was 8). Try 1 if turn 3 still always searches. |
| `SAULIE_TOOL_PRESSURE_REGISTRY_MAX` | `512` | LRU size for the tool-aware reset registry. |
| `SAULIE_RAG_MIN_COSINE` | `0.5` | Dense-cosine gate. `>0` gate on; `<=0` gate off (pure RRF). |
| `SAULIE_RAG_MAX_RESULTS` | `5` | Max products returned (both configs). |

Unchanged but relevant: `SAULIE_TOOL_BIAS_START_TURN=3`, `SAULIE_TOOL_FORCE_TURN=6`,
`SAULIE_TOOL_PRESSURE_MOD=0`, `SAULIE_TOOL_FORCE_MODE=off`, `SAULIE_TOOL_PRESSURE_ENABLED=1`,
`SAULIE_TOOL_PRESSURE_BACKEND=processor`.

---

## 5. Machine runbook (staged for clean attribution)

Run on the GPU machine. **Stage measurements** so you can attribute any change to the right
cause (do not change pressure + retrieval and measure once).

```bash
# 0. dependencies UP (the thing that was down in the first smoke)
docker start qdrant_index tensorrt_bge-m3
curl -s http://localhost:1234/collections/amazon_products_v2 | grep points_count   # > 0
curl -s -X POST http://localhost:8888/embed -d '{"text":["ping"]}' -H 'Content-Type: application/json' | head -c 80

# 1. CALIBRATE the cosine threshold on the labeled benchmark
python rag/calibrate_cosine_threshold.py
#   -> note recommended SAULIE_RAG_MIN_COSINE.
#   -> if it prints an OVERLAP WARNING, the dense index is likely stale/corrupted
#      (README SERVER_BATCH_SIZE bug). Rebuild before trusting cosine, or keep gate OFF.

# 2. A/B the retrieval: pure RRF vs RRF + dense gate (answers "is the gate better than RRF?")
python rag/retrieval_ab_eval.py --threshold <calibrated>
#   -> compare precision / coverage / no-result counts. Decide gate ON vs OFF from data.
#      Higher precision with acceptable coverage loss favors the gate; otherwise keep RRF.

# 3. deploy vLLM WITH the logits processor
bash dpo/eval/vllm_scripts/deploy_finalist_pick.sh
docker logs eval_deploy_qwenie 2>&1 | grep -i "logits\|SaulieToolPressure\|error" | tail -20

# 4. STAGE A — pressure only, retrieval gate OFF (isolate the pressure fix)
#    set in .env:  SAULIE_RAG_MIN_COSINE=0   SAULIE_TOOL_BIAS_PER_TURN=2
MODEL_NAME=dpo-v15-trial-4 SAULIE_PROMPT=compressed python agent_chat_api.py api &
python dpo/eval/model_prompt_matrix_eval.py --model-key dpo_w10 \
  --model-name dpo-v15-trial-4 --prompt compressed \
  --output dpo/eval/matrix_runs/dpo_w10_compressed_pressure_only.json

# 4b. prove nudge != silent force
python dpo/eval/nudge_rate_probe.py --scenario camping --turn 3 --samples 20            # expect 0<rate<1
python dpo/eval/nudge_rate_probe.py --scenario camping_force_turn6 --turn 6 --samples 20 # expect 100%

# 5. STAGE B — turn the calibrated retrieval gate ON, rerun the matrix
#    set in .env:  SAULIE_RAG_MIN_COSINE=<calibrated>
#    (restart the agent so it reloads .env)
python dpo/eval/model_prompt_matrix_eval.py --model-key dpo_w10 \
  --model-name dpo-v15-trial-4 --prompt compressed \
  --output dpo/eval/matrix_runs/dpo_w10_compressed_tool_pressure.json
```

Compare **baseline -> Stage A -> Stage B**. If turn 3 still always searches at `PER_TURN=2`,
drop to `1`. Regenerate `dpo/eval/matrix_runs/tool_pressure_smoke_report.md` from the runs.

### Verification checklist
- [ ] Preflight passes (embed + Qdrant reachable, collection non-empty).
- [ ] Calibration shows separation (no overlap warning); threshold chosen.
- [ ] A/B table reviewed; gate ON vs OFF decision recorded.
- [ ] Pressure trace: a conversation where a search fires at turn N has the **next** turn read
      `pressure_mode: off` in the debug log (registry reset working), not force.
- [ ] `nudge_rate_probe` camping turn 3 strictly between 0% and 100%; force turn = 100%.
- [ ] Matrix run has non-empty RAG results (no all-`no_results` hard-fail).

---

## 6. Files changed / added

```
 M agent_chat_api.py                     pressure registry + reset, bias default, no_results count
 M rag/query2.py                          dense-cosine gate (toggleable), RRF preserved
 M dpo/eval/model_prompt_matrix_eval.py   preflight, perf_counter, results_count + hard-fail
 M .env.example                           new SAULIE_RAG_* / pressure vars
?? rag/calibrate_cosine_threshold.py      cosine threshold calibration (18-query rubric)
?? rag/retrieval_ab_eval.py               A/B: pure RRF vs RRF+gate
?? dpo/eval/nudge_rate_probe.py           N-sample nudge-vs-force rate
?? RETRIEVAL_AND_TOOL_PRESSURE_HANDOFF.md  this doc
```

A ready-made `git apply` patch of all 7 code files also exists at the repo root as
`saulie_tool_pressure_rag.patch` (use it if you have the repo folder; otherwise reconstruct
from Appendix A).

---

## Appendix A — exact code changes

Apply these to a checkout at base `0e4cbe0`. Each modified-file block is an **exact
find → replace**; new files are given in full. (These are whitespace-tolerant for an LLM
editor; for a deterministic apply use `git apply` on the patch named above.)

### A.1 `.env.example` (2 edits)

**Edit 1** — in the `# --- RAG ---` block, after the `PREFETCH_LIMIT=50` line, insert:

```bash
# Relevance gate: RRF still RANKS (good for short product titles); dense COSINE gates relevance.
# Calibrate with rag/calibrate_cosine_threshold.py, A/B with rag/retrieval_ab_eval.py.
#   > 0  -> gate ON: RRF order, cosine >= MIN, capped at MAX; 0 above => "no relevant products"
#   <= 0 -> gate OFF: pure RRF baseline, top MAX by fused rank (use 0 for the A/B baseline)
SAULIE_RAG_MIN_COSINE=0.5
SAULIE_RAG_MAX_RESULTS=5
```

**Edit 2** — in the tool-pressure block, replace `SAULIE_TOOL_BIAS_PER_TURN=8` and the lines
after `SAULIE_TOOL_FORCE_MODE=off`:

```bash
# (replace this single line)
SAULIE_TOOL_BIAS_PER_TURN=8
```

with:

```bash
# Absolute logit add per effective turn. Keep SMALL: 8 -> +24/+32/+40 saturated and silently
# forced. 2 -> +6/+8/+10 is a real nudge. Try 2, drop to 1 if turn 3 still always searches.
SAULIE_TOOL_BIAS_PER_TURN=2
```

and after the `SAULIE_TOOL_FORCE_MODE=off` line, insert:

```bash
# Bounded LRU for the server-side tool-aware pressure reset (fingerprint -> last search turn).
SAULIE_TOOL_PRESSURE_REGISTRY_MAX=512
```

### A.2 `agent_chat_api.py` (6 edits)

**Edit 1 — imports.** Replace:

```python
import logging
import threading
from pathlib import Path
from queue import Empty, Queue
```

with:

```python
import logging
import threading
import hashlib
from pathlib import Path
from collections import OrderedDict
from queue import Empty, Queue
```

**Edit 2 — bias default + registry.** Replace:

```python
TOOL_BIAS_PER_TURN = int(os.getenv("SAULIE_TOOL_BIAS_PER_TURN", "8"))
TOOL_PRESSURE_MOD = int(os.getenv("SAULIE_TOOL_PRESSURE_MOD", "0"))
TOOL_FORCE_MODE = os.getenv("SAULIE_TOOL_FORCE_MODE", "off").strip().lower()
TOOL_PRESSURE_ENABLED = os.getenv("SAULIE_TOOL_PRESSURE_ENABLED", "1") == "1"
TOOL_PRESSURE_BACKEND = os.getenv("SAULIE_TOOL_PRESSURE_BACKEND", "processor").strip().lower()

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=LLM_TIMEOUT)
```

with:

```python
# Absolute logit add per effective turn. Keep SMALL: +24/+32/+40 (PER_TURN=8) saturated the
# softmax and silently forced. PER_TURN=2 -> +6/+8/+10 is a genuine probabilistic nudge.
TOOL_BIAS_PER_TURN = int(os.getenv("SAULIE_TOOL_BIAS_PER_TURN", "2"))
TOOL_PRESSURE_MOD = int(os.getenv("SAULIE_TOOL_PRESSURE_MOD", "0"))
TOOL_FORCE_MODE = os.getenv("SAULIE_TOOL_FORCE_MODE", "off").strip().lower()
TOOL_PRESSURE_ENABLED = os.getenv("SAULIE_TOOL_PRESSURE_ENABLED", "1") == "1"
TOOL_PRESSURE_BACKEND = os.getenv("SAULIE_TOOL_PRESSURE_BACKEND", "processor").strip().lower()

# Server-side tool-aware pressure reset. The stateless API strips tool messages from client
# history (see sanitize_messages), so _pressure_turn cannot see the last search position
# there. We map a conversation fingerprint (hash of user messages) -> the user-turn count at
# which a search last fired, and reset the pressure counter on the next request. Bounded LRU
# keeps it concurrency-safe without leaking memory.
_PRESSURE_REGISTRY: "OrderedDict[str, int]" = OrderedDict()
_PRESSURE_REGISTRY_MAX = int(os.getenv("SAULIE_TOOL_PRESSURE_REGISTRY_MAX", "512"))

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=LLM_TIMEOUT)
```

**Edit 3 — `execute_search` total-hit count.** Replace:

```python
        print(f"\033[93m [RESULTS]\033[0m Found {len(results)} products\n")
        # #region agent log
        _debug_log("D", "agent_chat_api.py:execute_search", "search results", {"result_count": len(results)})
        # #endregion
        if verbose:
            print(f"\033[93m [RESULTS]\033[0m Results {results} \n")
        
        if not results:
            result_json = json.dumps({
                "status": "no_results",
                "message": f"No products found for '{query_input}'. Recommend a general alternative."
            })
            return result_json, "no_results"
```

with:

```python
        total_hits = (
            sum(len(b.get("results", [])) for b in results)
            if isinstance(results, list) else 0
        )
        print(f"\033[93m [RESULTS]\033[0m Found {total_hits} products\n")
        # #region agent log
        _debug_log("D", "agent_chat_api.py:execute_search", "search results", {"result_count": total_hits})
        # #endregion
        if verbose:
            print(f"\033[93m [RESULTS]\033[0m Results {results} \n")
        
        if not results or total_hits == 0:
            result_json = json.dumps({
                "status": "no_results",
                "message": f"No relevant products found for '{query_input}'. Recommend a general alternative."
            })
            return result_json, "no_results"
```

**Edit 4 — fingerprint helpers + rewrite `_pressure_turn`.** Replace the whole old function:

```python
def _pressure_turn(history) -> int:
    """User messages since the last tool result (1-indexed episode counter)."""
    last_tool_idx = -1
    for i, m in enumerate(history):
        if m.get("role") == "tool":
            last_tool_idx = i
    return sum(1 for m in history[last_tool_idx + 1 :] if m.get("role") == "user")
```

with:

```python
def _user_fingerprint(user_contents) -> str:
    """Stable, order-sensitive hash of the user-message contents."""
    h = hashlib.sha1()
    for c in user_contents:
        h.update((c or "").encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()


def _note_search_fired(history) -> None:
    """Record that a search fired at the current user-turn count, keyed by the full
    user-message fingerprint. The next request (one more user message) looks this up via its
    prefix fingerprint and resets the pressure counter. Bounded LRU eviction."""
    user_contents = [m.get("content") or "" for m in history if m.get("role") == "user"]
    if not user_contents:
        return
    fp = _user_fingerprint(user_contents)
    _PRESSURE_REGISTRY[fp] = len(user_contents)
    _PRESSURE_REGISTRY.move_to_end(fp)
    while len(_PRESSURE_REGISTRY) > _PRESSURE_REGISTRY_MAX:
        _PRESSURE_REGISTRY.popitem(last=False)


def _pressure_turn(history) -> int:
    """User turns since the last search (episode counter).

    The stateless API strips tool messages from client history, so we cannot read the last
    search position from `history`. Instead we look up a server-side registry keyed by the
    fingerprint of all user messages EXCEPT the latest -- which equals the fingerprint stored
    when the previous request's search fired. Falls back to the full user count when no search
    has fired yet (new conversation)."""
    user_contents = [m.get("content") or "" for m in history if m.get("role") == "user"]
    total = len(user_contents)
    if total == 0:
        return 0
    prefix_fp = _user_fingerprint(user_contents[:-1])
    last_search_turn = _PRESSURE_REGISTRY.get(prefix_fp)
    if last_search_turn is None:
        return total
    return max(1, total - last_search_turn)
```

**Edit 5 — wire reset in `agent_loop`.** This is the non-streaming tool path. Replace:

```python
        # 3) Tool execution path
        if final_tool_calls and current_cycle < max_tool_cycles:
            current_cycle += 1
            # #region agent log
            _debug_log("C", "agent_chat_api.py:agent_loop", "executing tools", {"current_cycle": current_cycle, "tool_count": len(final_tool_calls), "max_tool_cycles": max_tool_cycles})
```

with (add the `_note_search_fired(history)` line):

```python
        # 3) Tool execution path
        if final_tool_calls and current_cycle < max_tool_cycles:
            current_cycle += 1
            _note_search_fired(history)
            # #region agent log
            _debug_log("C", "agent_chat_api.py:agent_loop", "executing tools", {"current_cycle": current_cycle, "tool_count": len(final_tool_calls), "max_tool_cycles": max_tool_cycles})
```

**Edit 6 — wire reset in `_stream_agent_sse_inner`.** This is the streaming tool path. Replace:

```python
        # Tool execution path
        if final_tool_calls and current_cycle < max_tool_cycles:
            current_cycle += 1
            # #region agent log
            _debug_log("E", "agent_chat_api.py:stream_agent_sse", "executing tools", {"current_cycle": current_cycle, "tool_count": len(final_tool_calls)})
```

with:

```python
        # Tool execution path
        if final_tool_calls and current_cycle < max_tool_cycles:
            current_cycle += 1
            _note_search_fired(history)
            # #region agent log
            _debug_log("E", "agent_chat_api.py:stream_agent_sse", "executing tools", {"current_cycle": current_cycle, "tool_count": len(final_tool_calls)})
```

### A.3 `rag/query2.py` (2 edits)

**Edit 1 — gate config.** After the `DEFAULT_PREFETCH = int(...)` line (just before
`client = QdrantClient(...)`), insert:

```python
# Relevance gating. RRF (Qdrant k=2) scores are RANK-based, not relevance-based: the top
# fused hit is ~always >= 0.5 even for garbage queries, so it cannot gate. RRF still RANKS
# (it is strong for short product titles); dense COSINE is added ONLY as a relevance gate.
# Calibrate SAULIE_RAG_MIN_COSINE on the labeled benchmark (calibrate_cosine_threshold.py).
# Behavior:
#   SAULIE_RAG_MIN_COSINE > 0  -> gate ON: return RRF-ordered hits with cosine >= threshold,
#                                 capped at MAX_RESULTS; zero above => "no relevant products".
#   SAULIE_RAG_MIN_COSINE <= 0 -> gate OFF: pure RRF baseline, top MAX_RESULTS by fused rank
#                                 (no extra dense pass). Use 0 for the A/B baseline config.
RAG_MIN_COSINE = float(os.getenv("SAULIE_RAG_MIN_COSINE", "0.5"))
RAG_MAX_RESULTS = int(os.getenv("SAULIE_RAG_MAX_RESULTS", "5"))
```

**Edit 2 — gate logic in `search_hybrid`.** Replace the per-sub-query block:

```python
        t_q = time.perf_counter()
        resp = client.query_points(
            collection_name=COLLECTION,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=_fusion_mode()),
            limit=top_k,
            with_payload=[
                "name", "parent_asin", "main_category",
                "ratings", "no_of_ratings", "discount_price", "actual_price",
            ],
        )
        qdrant_ms += (time.perf_counter() - t_q) * 1000
        out.append({
            "query": q,
            "results": [_format_hit(hit) for hit in resp.points],
        })
```

with:

```python
        gate_on = RAG_MIN_COSINE > 0.0
        t_q = time.perf_counter()
        fused_limit = max(top_k, RAG_MAX_RESULTS)
        resp = client.query_points(
            collection_name=COLLECTION,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=_fusion_mode()),
            limit=fused_limit,
            with_payload=[
                "name", "parent_asin", "main_category",
                "ratings", "no_of_ratings", "discount_price", "actual_price",
            ],
        )
        cosine_by_id: dict = {}
        if gate_on:
            # Dense-only pass for the relevance signal. Same dense vector (already embedded)
            # and same category filter; cosine score per point id is the true relevance
            # measure RRF cannot provide. with_payload=False -> only id + score.
            dense_resp = client.query_points(
                collection_name=COLLECTION,
                query=dense_vec,
                using="dense",
                limit=prefetch_limit,
                query_filter=query_filter,
                with_payload=False,
            )
            cosine_by_id = {p.id: p.score for p in dense_resp.points}
        qdrant_ms += (time.perf_counter() - t_q) * 1000

        if gate_on:
            # Gate by cosine, keep RRF order, cap at RAG_MAX_RESULTS. Hits absent from the
            # dense top-N (sparse-only keyword matches) have no cosine -> treated irrelevant.
            results = []
            for hit in resp.points:
                cos = cosine_by_id.get(hit.id)
                if cos is None or cos < RAG_MIN_COSINE:
                    continue
                formatted = _format_hit(hit)
                formatted["relevance"] = round(float(cos), 4)
                results.append(formatted)
                if len(results) >= RAG_MAX_RESULTS:
                    break
        else:
            # Pure RRF baseline: top hits by fused rank, no relevance gate.
            results = [_format_hit(hit) for hit in resp.points[:RAG_MAX_RESULTS]]
        out.append({
            "query": q,
            "results": results,
        })
```

### A.4 `dpo/eval/model_prompt_matrix_eval.py` (8 edits)

**Edit 1 — `import os`.** Replace:

```python
import argparse
import json
import re
import time
```

with:

```python
import argparse
import json
import os
import re
import time
```

**Edit 2 — dep config.** After the `OUTPUT_DIR = REPO / "dpo" / "eval" / "matrix_runs"` line, insert:

```python
# Dependency endpoints for preflight (must match rag/query2.py defaults).
EMBED_URL = os.getenv("EMBED_URL", "http://localhost:8888/embed")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:1234")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "amazon_products_v2")
```

**Edit 3 — result-count regex.** After the `TOOL_EXEC_RE = re.compile(r"executing tools")` line, insert:

```python
RESULT_COUNT_RE = re.compile(r'"result_count"\s*:\s*(\d+)')
```

**Edit 4 — `TurnResult` field.** Replace:

```python
    elapsed_s: float
    likely_hallucinated_product: bool


@dataclass
```

with:

```python
    elapsed_s: float
    likely_hallucinated_product: bool
    results_count: int = 0


@dataclass
```

**Edit 5 — `_parse_new_logs` + new `_preflight_deps`.** Replace the whole function:

```python
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
```

with:

```python
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
```

**Edit 6 — monotonic timing in `run_cell`.** Replace the two `time.time()` calls (only inside
`run_cell`) with `time.perf_counter()`:

```python
            t0 = time.time()
```
→
```python
            t0 = time.perf_counter()
```
and
```python
            elapsed = time.time() - t0
```
→
```python
            elapsed = time.perf_counter() - t0
```

**Edit 7 — capture `results_count`.** Replace:

```python
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
```

with:

```python
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
```

**Edit 8 — preflight + all-empty hard-fail in `main`.** Replace:

```python
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.output or OUTPUT_DIR / f"{args.model_key}_{args.prompt}.json"

    cell = run_cell(args.model_key, args.model_name, args.prompt, args.agent_url)
```

with:

```python
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
```

### A.5 New file — `rag/calibrate_cosine_threshold.py`

```python
#!/usr/bin/env python3
"""Calibrate SAULIE_RAG_MIN_COSINE on the 18-query labeled benchmark.

Why: RRF (Qdrant k=2) scores are rank-based, not relevance-based -- the top fused hit is
~always >= 0.5 even for garbage queries, so it cannot gate. The dense vectors are a COSINE
collection, so dense cosine is the true 0-1 relevance signal. This script runs the same 18
benchmark queries through a dense-only search, labels each returned product with a keyword
rubric (reproducing fusion_comparison_report.md's method), and reports the cosine threshold
that best separates relevant from irrelevant hits.

It also doubles as a sanity check for the historical dense-vector corruption bug (README:
SERVER_BATCH_SIZE). If cosine cannot separate good from junk (overlap warning), the index is
likely stale/corrupted and gating on cosine will not work until it is rebuilt.

Run on the machine with the embed server (:8888) and Qdrant (:1234) up:
    python rag/calibrate_cosine_threshold.py
    python rag/calibrate_cosine_threshold.py --top-n 10 --collection amazon_products_v2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from qdrant_client.http import models  # noqa: E402

import query2  # noqa: E402
from query2 import client, get_server_embeddings  # noqa: E402
from benchmark_fusion import QUERIES, category_for_collection  # noqa: E402

# Relevant if the product name contains any of these tokens (lowercased substring match).
# Mirrors the manual keyword rubric used to label fusion_comparison_report.md.
RUBRIC: dict[str, list[str]] = {
    "wireless earbuds noise cancelling": ["earbud", "earphone", "headphone", "in-ear", "in ear"],
    "bluetooth speaker portable waterproof": ["speaker"],
    "gaming laptop RTX": ["laptop", "notebook"],
    "32 inch smart TV 4K": ["tv", "television"],
    "men's running shoes lightweight": ["shoe", "sneaker", "running", "trainer"],
    "women's winter coat warm": ["coat", "jacket", "parka", "puffer"],
    "cotton bed sheets king size": ["sheet", "bedsheet", "bedding", "duvet"],
    "stainless steel cookware set": ["cookware", "pot", "pan", "saucepan", "skillet", "dutch oven"],
    "yoga mat non slip thick": ["yoga mat", "mat"],
    "protein powder whey chocolate": ["protein", "whey"],
    "baby diaper pants large pack": ["diaper", "nappy"],
    "dog food dry adult": ["dog food", "dog"],
    "car phone mount dashboard": ["mount", "holder", "cradle"],
    "mechanical keyboard RGB gaming": ["keyboard"],
    "men's formal leather belt": ["belt"],
    "kids school backpack waterproof": ["backpack", "bag", "rucksack"],
    "air fryer large capacity": ["air fryer", "fryer"],
    "face moisturizer dry skin": ["moisturizer", "moisturising", "moisturiser", "cream", "lotion", "hydra"],
}


def _is_relevant(query: str, name: str) -> bool:
    tokens = RUBRIC.get(query, [])
    low = (name or "").lower()
    return any(tok in low for tok in tokens)


def _dense_search(dense_vec, category, top_n):
    query_filter = None
    if category:
        query_filter = models.Filter(
            must=[models.FieldCondition(key="main_category", match=models.MatchValue(value=category))]
        )
    resp = client.query_points(
        collection_name=query2.COLLECTION,
        query=dense_vec,
        using="dense",
        limit=top_n,
        query_filter=query_filter,
        with_payload=["name"],
    )
    return resp.points


def _sweep(pairs: list[tuple[float, bool]]) -> tuple[float, dict]:
    """Pick the cosine threshold maximizing F1 over (cosine, relevant) pairs."""
    if not pairs:
        return 0.0, {}
    candidates = sorted({round(c, 3) for c, _ in pairs})
    best_t, best = candidates[0], {"f1": -1.0}
    for t in candidates:
        tp = sum(1 for c, r in pairs if c >= t and r)
        fp = sum(1 for c, r in pairs if c >= t and not r)
        fn = sum(1 for c, r in pairs if c < t and r)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best["f1"]:
            best_t, best = t, {"f1": round(f1, 3), "precision": round(prec, 3), "recall": round(rec, 3)}
    return best_t, best


def main():
    ap = argparse.ArgumentParser(description="Calibrate dense-cosine relevance threshold")
    ap.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "amazon_products_v2"))
    ap.add_argument("--top-n", type=int, default=10, help="dense hits to inspect per query")
    args = ap.parse_args()

    os.environ["QDRANT_COLLECTION"] = args.collection
    query2.COLLECTION = args.collection

    all_pairs: list[tuple[float, bool]] = []
    top_hit_rows: list[tuple[str, float, bool, str]] = []

    print(f"Collection: {args.collection}  top_n: {args.top_n}\n")
    for item in QUERIES:
        q = item["query"]
        cat = category_for_collection(args.collection, item)
        emb = get_server_embeddings([q])
        if not emb or not emb.get("dense"):
            raise SystemExit(f"Embed server returned nothing for {q!r} -- is :8888 up?")
        points = _dense_search(emb["dense"][0], cat, args.top_n)
        if not points:
            print(f"  [WARN] no dense hits for {q!r} (filter={cat!r})")
            continue
        for j, p in enumerate(points):
            name = (p.payload or {}).get("name", "")
            rel = _is_relevant(q, name)
            all_pairs.append((float(p.score), rel))
            if j == 0:
                top_hit_rows.append((q, float(p.score), rel, name[:60]))

    print("Top hit per query (cosine | relevant | name):")
    for q, cos, rel, name in top_hit_rows:
        flag = "REL " if rel else "junk"
        print(f"  {cos:6.4f}  {flag}  {q[:34]:34s} -> {name}")

    rel_cos = [c for c, r in all_pairs if r]
    junk_cos = [c for c, r in all_pairs if not r]
    best_t, metrics = _sweep(all_pairs)

    print("\n--- separation ---")
    if rel_cos:
        print(f"  relevant  cosine: min={min(rel_cos):.4f}  max={max(rel_cos):.4f}  n={len(rel_cos)}")
    if junk_cos:
        print(f"  irrelevant cosine: min={min(junk_cos):.4f}  max={max(junk_cos):.4f}  n={len(junk_cos)}")
    if rel_cos and junk_cos and max(junk_cos) >= min(rel_cos):
        print("  [WARN] relevant/irrelevant cosine ranges OVERLAP. Either the rubric needs"
              " tuning or the dense index is stale/corrupted (see README SERVER_BATCH_SIZE).")

    print("\n--- recommendation ---")
    print(f"  SAULIE_RAG_MIN_COSINE={best_t}   (F1={metrics.get('f1')}, "
          f"precision={metrics.get('precision')}, recall={metrics.get('recall')})")
    print("  Review the top-hit table above before committing the value.")


if __name__ == "__main__":
    main()
```

### A.6 New file — `rag/retrieval_ab_eval.py`

```python
#!/usr/bin/env python3
"""A/B retrieval eval: pure RRF baseline vs RRF + dense-cosine gate.

RRF (Qdrant k=2) is a strong RANKER for short product titles, but its score is rank-based and
cannot tell relevant from irrelevant. The dense-cosine gate drops sub-threshold hits. This
question -- does gating actually return better results, or does it throw away good RRF hits? --
is empirical, so this script runs BOTH configs over the 18 labeled benchmark queries and
reports relevance metrics side by side. RRF ranking is identical in both; only the gate differs.

Configs (RAG_MAX_RESULTS held equal so the only variable is the gate):
  A  baseline : SAULIE_RAG_MIN_COSINE = 0      -> top-N by fused RRF rank, no gate
  B  gated    : SAULIE_RAG_MIN_COSINE = <thr>  -> RRF order, cosine >= thr, dynamic count

Run on the machine with embed (:8888) + Qdrant (:1234) up, AFTER calibrating the threshold:
    python rag/calibrate_cosine_threshold.py        # -> pick THR
    python rag/retrieval_ab_eval.py --threshold THR  # default 0.5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import query2  # noqa: E402
from query2 import search_hybrid  # noqa: E402
from benchmark_fusion import QUERIES, category_for_collection  # noqa: E402
from calibrate_cosine_threshold import RUBRIC, _is_relevant  # noqa: E402  (reuse rubric)


def run_config(min_cosine: float, collection: str, top_k: int, max_results: int):
    query2.COLLECTION = collection
    query2.RAG_MIN_COSINE = min_cosine
    query2.RAG_MAX_RESULTS = max_results
    os.environ["QDRANT_COLLECTION"] = collection

    rows = []
    for item in QUERIES:
        q = item["query"]
        cat = category_for_collection(collection, item)
        blocks = search_hybrid(q, main_category=cat, top_k=top_k)
        hits = blocks[0]["results"] if blocks else []
        rels = [_is_relevant(q, h.get("name", "")) for h in hits]
        rows.append({
            "query": q,
            "returned": len(hits),
            "relevant": sum(rels),
            "top1_relevant": bool(rels and rels[0]),
            "any_relevant": any(rels),
            "no_results": len(hits) == 0,
            "top_name": (hits[0]["name"][:48] if hits else ""),
        })
    return rows


def summarize(rows):
    n = len(rows)
    returned = sum(r["returned"] for r in rows)
    relevant = sum(r["relevant"] for r in rows)
    return {
        "queries": n,
        "total_returned": returned,
        "total_relevant": relevant,
        "precision": round(relevant / returned, 3) if returned else 0.0,
        "top1_acc": round(sum(r["top1_relevant"] for r in rows) / n, 3) if n else 0.0,
        "coverage": round(sum(r["any_relevant"] for r in rows) / n, 3) if n else 0.0,
        "no_result_queries": sum(r["no_results"] for r in rows),
        "avg_returned": round(returned / n, 2) if n else 0.0,
    }


def _print_block(label, s):
    print(f"\n[{label}]")
    print(f"  precision (relevant/returned) : {s['precision']}")
    print(f"  top-1 accuracy                : {s['top1_acc']}")
    print(f"  coverage (>=1 relevant/query) : {s['coverage']}")
    print(f"  avg results returned          : {s['avg_returned']}")
    print(f"  queries returning nothing     : {s['no_result_queries']}/{s['queries']}")
    print(f"  totals: returned={s['total_returned']} relevant={s['total_relevant']}")


def main():
    ap = argparse.ArgumentParser(description="A/B retrieval: pure RRF vs RRF+dense gate")
    ap.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "amazon_products_v2"))
    ap.add_argument("--threshold", type=float, default=0.5, help="cosine gate for config B")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-results", type=int, default=5)
    args = ap.parse_args()

    print(f"Collection: {args.collection}  top_k={args.top_k}  max_results={args.max_results}")
    print(f"Config A (baseline): RRF only, no gate")
    print(f"Config B (gated):    RRF + cosine >= {args.threshold}")

    rows_a = run_config(0.0, args.collection, args.top_k, args.max_results)
    rows_b = run_config(args.threshold, args.collection, args.top_k, args.max_results)
    sa, sb = summarize(rows_a), summarize(rows_b)

    print("\nPer-query (A returned/relevant | B returned/relevant | query):")
    by_q_b = {r["query"]: r for r in rows_b}
    for ra in rows_a:
        rb = by_q_b[ra["query"]]
        print(f"  A {ra['returned']}/{ra['relevant']}  |  B {rb['returned']}/{rb['relevant']}  |  {ra['query']}")

    _print_block("A  pure RRF baseline", sa)
    _print_block("B  RRF + dense gate", sb)

    print("\n--- verdict ---")
    print(f"  precision delta (B - A): {round(sb['precision'] - sa['precision'], 3):+}")
    print(f"  coverage delta  (B - A): {round(sb['coverage'] - sa['coverage'], 3):+}")
    print(f"  B drops {sb['no_result_queries'] - sa['no_result_queries']:+} queries to 'no results' vs A")
    print("  Higher precision + acceptable coverage loss favors the gate; large coverage")
    print("  loss with little precision gain favors pure RRF. Decide from the table above.")
    if sb["no_result_queries"] == sb["queries"]:
        print("  [WARN] Config B returned nothing for EVERY query -- threshold too high or index stale.")


if __name__ == "__main__":
    main()
```

### A.7 New file — `dpo/eval/nudge_rate_probe.py`

```python
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
```
