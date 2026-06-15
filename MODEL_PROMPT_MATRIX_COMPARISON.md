# Model × Prompt Matrix — Handoff for New Agents

**Last updated:** June 2026  
**Status:** Matrix run complete. Production is **`dpo_w10` + `compressed`**. Several deployment issues remain open (see [Open problems](#open-problems-still-unfixed)).

This document is the single entry point for understanding **why** we ran a 3×3 eval, **what** we were trying to fix, **how** it was measured, and **what to do next**. Read this before touching `agent_chat_api.py`, prompts, or deploy scripts.

---

## TL;DR for a new conversation

| Question | Answer |
|----------|--------|
| What is Saulie? | A shopping agent persona (street-salesman voice) backed by Qwen3-4B + LoRA, RAG over Amazon products, OpenAI-compatible API. |
| Which model ships? | **DPO v1.5 trial-4** cat-merged at weight **1.0** (`dpo-v15-trial-4`). |
| Which prompt ships? | **`compressed`** (default via `SAULIE_PROMPT=compressed`). |
| What broke in prod? | Bad persona (catchphrase loops), wrong steering (instant search OR no search), hallucinated products, rigid templates. |
| Is the model the problem? | **Mostly no.** SFT/DPO training is healthy. **Deployment harness + system prompt** were overriding trained probe-first behavior. |
| What did the matrix prove? | **DPO w=0.5 never calls tools.** **DPO w=1.0 + SFT** call tools only after explicit user scream ("USE YOUR SEARCH_PRODUCTS TOOL"). **Legacy prompt** forces search earlier but worse persona. |
| Biggest remaining bug? | **Performative search** — model says "let me pull up what's selling" but does not emit `search_products` until user yells. Needs harness bridge (not more prompt-only iteration). |

---

## Background: what we're building

**Saulie** is a conversational product recommender:

```
User → agent_chat_api.py (:9000) → vLLM (:8000, LoRA) + search_products tool → RAG (BGE-M3 + Qdrant)
```

- **Agent:** `agent_chat_api.py` — injects system prompt, runs tool loop, streams SSE. With `stream: true`, vLLM tokens stream to the client; `delta.tool_calls` are buffered server-side (never sent to the client). See [`AGENT_STREAMING.md`](AGENT_STREAMING.md).
- **Model:** Qwen3-4B-Instruct FP8 + LoRA adapter (SFT trial-17 base, optional DPO overlay).
- **RAG:** `rag/query2.py` → `amazon_products_v2` (McAuley US catalog), fusion **RRF**.
- **Ops:** `start_saulie.sh` — stack startup; env `SAULIE_MODEL`, `SAULIE_PROMPT`, `QDRANT_COLLECTION`.

Training taught a **probe-first** skill: engage → ask sharp questions → search when ready → pitch real tool results → steer if miss. Deployment originally **broke** that skill.

Related docs (read if you need depth):

| Doc | Purpose |
|-----|---------|
| [`SAULIE_PERSONA_AND_STEERING_FIX_PLAN.md`](SAULIE_PERSONA_AND_STEERING_FIX_PLAN.md) | Patch plan: prompt rewrite, remove forced tool, sampling, retry rules |
| [`dpo/eval/README.md`](dpo/eval/README.md) | v1.5 merge gate, judge eval, deploy scripts |
| [`rag/README.md`](rag/README.md) | Indexing, benchmarks, collection defaults |
| `deployment` branch README | Production stack architecture |

---

## What we were trying to fix (user-reported prod bugs)

These came from live testing via `remote_chat.py` / ngrok, **before** the steering fix patches:

### 1. Persona authenticity

- Same 3–4 catchphrases every turn ("what the f*ck", "capisce", "forget about it", "pal").
- Felt like a cheap impression, not a real character.
- Rigid 5-field product template; markdown noise in terminal.
- Price bug: `$21.99 (Was: $21.99 (Was: $21.99))`.

**Root cause (code):** Old `SYSTEM_PROMPT` listed catchphrases to copy; low `repetition_penalty` (1.05); rigid "SALES PITCH FORMAT" block.

### 2. Steering / tool timing (two opposite failures)

| Failure mode | Symptom | Root cause |
|--------------|---------|------------|
| **Over-search** | Instant `searching for products...` on first product mention, zero probing | Harness forced `tool_choice=search_products` on turn 1 when keywords matched + prompt said "ALWAYS USE THE TOOL" |
| **Under-search** (after Patch 2) | Model says "let me check what's selling" but **never calls the tool** until user screams "USE YOUR SEARCH_PRODUCTS TOOL" | `_tool_choice_for_turn` now always `"auto"`; DPO w=1.0 + new prompts don't reliably emit function calls when "search-ready" |

**Author intent (fix plan):** The fine-tune is fine. Fix **deployment**, not retrain — unless matrix proves a model variant is unsalvageable.

### 3. Hallucinated products

- Model invents product names, brands (e.g. "Sennheiser"), specs, prices **without** a tool call or from weak RAG matches.
- Especially bad when RAG returns irrelevant items (wrong category / low RRF score) and model fills gaps from parametric knowledge.

**Mitigation added:** `COMPRESSED_SYSTEM_PROMPT` instructs **ALL CAPS + bold** for real tool-returned names only (e.g. **THERM-A-REST Z LITE SOL**) so manual testing can distinguish RAG vs hallucination.

### 4. RAG quality (separate but coupled)

- Switched from Indian catalog (~114k) to **McAuley US** (`amazon_products_v2`, ~443k).
- Old agent used wrong `VALID_CATEGORIES` (Indian taxonomy) — fixed to McAuley categories.
- Bad retrieval still happens on specific SKUs (e.g. "RTX 4070 laptop" → chargers). Model then hallucinates if it pitches anyway.

**Matrix scope:** We explicitly **did not** judge RAG relevance in this pass — only **did the model call the tool** and **is the English coherent**.

---

## Patches applied before / during matrix

From [`SAULIE_PERSONA_AND_STEERING_FIX_PLAN.md`](SAULIE_PERSONA_AND_STEERING_FIX_PLAN.md):

| Patch | Status | Effect |
|-------|--------|--------|
| **1 — System prompt rewrite** | Done | Three variants: `legacy`, `steering`, `compressed`. Default `compressed`. |
| **2 — Remove forced tool call** | Done | `_tool_choice_for_turn` always returns `"auto"`. Probing possible again; under-search emerged. |
| **3 — Sampling** (repetition_penalty ↑, frequency/presence penalty) | **Deferred** | User chose prompt/harness first. |
| **4A — Prompt-only retry rules** | Done | In compressed/steering prompts. |
| **4B — Harness keyword bridge** | **Not done** | Recommended next step for performative-search bug. |

Prompt selection:

```bash
SAULIE_PROMPT=legacy|steering|compressed   # default: compressed
curl http://127.0.0.1:9000/health          # returns active prompt + model
```

---

## Why we ran the 3×3 matrix

After v1.5 Optuna + merge gate, **trial-4 DPO (w=1.0 cat-merge)** won deployment. But prod still felt wrong after prompt/harness patches. We needed to separate:

1. **Model effect** — SFT-only vs DPO at w=0.5 vs DPO at w=1.0  
2. **Prompt effect** — old deployment prompt vs full steering prompt vs compressed production prompt  
3. **Tool calling vs English** — does softer DPO break function calling? Does new prompt fix persona without killing search?

**Decision question:** What is the best **model × prompt** combo for production given we want probe-first steering **and** reliable `search_products` **and** readable English?

---

## Methodology

### Grid (9 cells)

| Model key | vLLM deploy script | Served name | Adapter |
|-----------|-------------------|-------------|---------|
| `sft` | `deploy_sft_trial17_prod.sh` | `steering-sft-trial-17` | SFT trial-17 only |
| `dpo_w05` | `deploy_dpo_w05.sh` | `dpo-v15-trial-4-w05` | trial-4 cat-merge, DPO weight **0.5** |
| `dpo_w10` | `deploy_finalist_pick.sh` | `dpo-v15-trial-4` | trial-4 cat-merge, DPO weight **1.0** |

| Prompt | Description |
|--------|-------------|
| `legacy` | Original prod prompt: "ALWAYS USE THE TOOL", rigid template, catchphrase list |
| `steering` | Full ~1160-token probe-first prompt from fix plan |
| `compressed` | ~410-token production prompt with PRIORITY block + probe workflow |

### Test scenarios (scripted multi-turn)

Defined in `dpo/eval/model_prompt_matrix_eval.py`:

| Scenario | Intent |
|----------|--------|
| **camping** | Probe-heavy: user gradually reveals cold/uncomfortable sleeping; turn 5 asks for specifics; turn 6 **explicit tool command** |
| **kitchen** | Cleanup frustration; tests whether agent searches without explicit product ask |
| **direct** | Single turn: "kids desk lamp with focused beam" — tests immediate search on clear product ask |

### How tool calls were verified

**Not** from assistant text alone ("searching for products..." can appear without a real call).

Verified via `agent_api.log` debug lines:

- **Emitted:** LLM returned `tool_call_count > 0` in `_llm_once` debug log  
- **Executed:** `executing tools` in agent loop + RAG `[TOOL CALL]` entries

Raw JSON per cell: `dpo/eval/matrix_runs/{model}_{prompt}.json`

### Re-run

```bash
bash dpo/eval/run_model_prompt_matrix.sh
python dpo/eval/generate_matrix_report.py   # regenerates grid section from JSON
```

Requires BGE + Qdrant up; restarts vLLM per model cell (~45 min full grid).

---

## Results

**Focus:** tool call emission/execution and English clarity. Broken RAG matches were acceptable for this pass.

### Tool calling summary

- Cells with **any tool emitted** (6/9): all except **`dpo_w05` × all prompts**
- Cells with **tool executed** (6/9): same six — `dpo_w10` and `sft` × all three prompts
- **DPO w=0.5:** **zero tool calls** in every prompt variant — function calling effectively dead

### Full grid

| Model | Prompt | Tool | English | Emit turns | Exec turns |
|-------|--------|------|---------|------------|------------|
| dpo_w05 | compressed | none | poor | — | — |
| dpo_w05 | legacy | none | ok | — | — |
| dpo_w05 | steering | none | ok | — | — |
| dpo_w10 | compressed | **EXECUTED** | ok | camping turn 6 | camping turn 6 |
| dpo_w10 | legacy | **EXECUTED** | mixed | camping turn 6, direct turn 1 | camping turn 6, direct turn 1 |
| dpo_w10 | steering | **EXECUTED** | ok | camping turn 6 | camping turn 6 |
| sft | compressed | **EXECUTED** | ok | camping turn 6 | camping turn 6 |
| sft | legacy | **EXECUTED** | ok | camping turn 6 | camping turn 6 |
| sft | steering | **EXECUTED** | ok | camping turn 6 | camping turn 6 |

### Pattern across all tool-capable cells

- **Camping turn 5** ("you got anything specific?") → **no tool** in any cell  
- **Camping turn 6** (explicit "USE YOUR SEARCH_PRODUCTS TOOL") → **tool fires** in all six capable cells  
- **Kitchen scenario** → performative text, **no tool** in any cell  
- **Direct scenario** → tool only on **`dpo_w10` + `legacy`** (turn 1); other combos did not search on direct ask

### Cells that called the tool (detail)

#### `dpo_w10` + `compressed` (current production)

- **camping turn 6** [EXEC] user: *USE YOUR SEARCH_PRODUCTS TOOL TO GIVE ME A SPECIFIC PRODUCT*
  - assistant: Night Cat Sleeping Air Pad pitch with price/rating from tool

#### `dpo_w10` + `legacy`

- **camping turn 6** [EXEC] — KLYMIT Static V Sleeping Pad pitch  
- **direct turn 1** [EXEC] — searched on desk lamp ask; angry reply about irrelevant RAG results (honest but mixed English)

#### `dpo_w10` + `steering`

- **camping turn 6** [EXEC] — Sleepingo Sleeping Pad pitch

#### `sft` + `compressed` / `legacy` / `steering`

- **camping turn 6** [EXEC] only in each case  
- SFT pitches differ in tone; all required explicit tool command in camping flow

### English / hallucination notes

- **`dpo_w05/compressed`:** likely invented product/spec at camping turns 4–5 (no tool, parametric fill)  
- **Post-matrix manual testing (headphones):** after tool finally ran, follow-up "got a specific brand?" → **hallucinated Sennheiser** not in RAG JSON  
- **`dpo_w10/legacy`:** best tool coverage, worst persona/template behavior  
- **`dpo_w10/compressed`:** acceptable English when tool runs; still needs explicit user push to search

---

## Interpretation: what the matrix tells us

### Model conclusions

1. **Do not deploy DPO w=0.5** for tool-use — zero `search_products` across all prompts.  
2. **DPO w=1.0 vs SFT** — similar tool-call pattern in scripted tests; both need explicit scream in camping flow. DPO chosen for preference quality from earlier judge eval, not because matrix showed better tool calling.  
3. **DPO English** — user concern is **confusing/broken prose** on DPO more than SFT, especially when filling without tools (`dpo_w05`) or when RAG is bad.

### Prompt conclusions

1. **`legacy`** — only combo that searched on **direct turn-1** product ask (`dpo_w10/legacy`); trades away probe-first and persona quality.  
2. **`steering`** — good persona spec but long; same tool timing as compressed in matrix.  
3. **`compressed`** — best balance for prod: short, PRIORITY rules, probe workflow; **still doesn't fix under-search** without harness help.

### Harness conclusions

Removing forced `tool_choice` **fixed over-search** but **caused under-search**. Prompt text saying "call search_products in that same turn" is **not sufficient** for DPO w=1.0 — model prefers deferral language over function emission.

**Recommended harness fix (not yet implemented):** "search-ready gate" — after enough probing/user confirmation, force `tool_choice=search_products`; optionally retry when model emits performative "let me look" without a tool call. See fix plan Patch 4B discussion.

---

## Production config (as of matrix + follow-up)

```bash
# start_saulie.sh defaults
SAULIE_MODEL=dpo-v15-trial-4          # DPO w=1.0 trial-4
SAULIE_PROMPT=compressed
QDRANT_COLLECTION=amazon_products_v2
FUSION_METHOD=rrf
```

Verify: `curl http://127.0.0.1:9000/health` → `{"model":"dpo-v15-trial-4","prompt":"compressed",...}`

---

## Open problems (still unfixed)

Priority order for a new agent:

1. **Performative search / under-search** — User confirms budget/specs; model says it will look; no tool until explicit command. **Fix:** deferred harness bridge (search-ready gate + optional performative-text retry), not endless prompt edits.  
2. **Hallucination after tool** — Brand/spec invention on follow-ups ("Sennheiser"). **Fix:** stricter "only state tool JSON fields"; re-search on brand ask; CAPS marker helps manual QA only.  
3. **Turn-5 gap** — "got anything specific?" never triggers search in any matrix cell. **Fix:** harness should treat this as search-ready.  
4. **RAG relevance** — Specific queries return wrong category items. Separate from matrix; tune queries/categories/embeddings.  
5. **Patch 3 sampling** — Catchphrase repetition if still observed; deferred.  
6. **DPO w=0.5** — Dead for tools; only useful if you accept no-RAG conversational mode.

---

## Recommended next steps

| If your goal is… | Do this |
|------------------|---------|
| Restore reliable search without killing probes | Implement **search-ready gate** in `_tool_choice_for_turn` / `agent_loop` ([fix plan](SAULIE_PERSONA_AND_STEERING_FIX_PLAN.md)) |
| Quick demo with aggressive search | `SAULIE_PROMPT=legacy` (accept persona regression) |
| Compare a change objectively | Re-run matrix: `bash dpo/eval/run_model_prompt_matrix.sh` |
| Debug one conversation | Tail `agent_api.log` for `[TOOL CALL]`, `tool_call_count`, RAG results |
| Test hallucination vs RAG | Look for **ALL CAPS** product names (only valid after real tool result) |

**Do not assume** "searching for products..." in the client means a tool ran — check logs. Status lines are intentional user-facing SSE during RAG; raw tool JSON is never streamed.

**Streaming note:** With `stream: true`, final answers and probe text arrive token-by-token. Tool-call assembly stays internal; see [`AGENT_STREAMING.md`](AGENT_STREAMING.md).

---

## File index

| Path | Role |
|------|------|
| `agent_chat_api.py` | Prompt variants, tool loop, `_tool_choice_for_turn` |
| `start_saulie.sh` | `SAULIE_MODEL`, `SAULIE_PROMPT`, RAG env |
| `dpo/eval/run_model_prompt_matrix.sh` | Full 9-cell orchestrator |
| `dpo/eval/model_prompt_matrix_eval.py` | Scenarios + HTTP eval against agent |
| `dpo/eval/generate_matrix_report.py` | Regenerates grid from JSON (does not rewrite this handoff) |
| `dpo/eval/matrix_runs/*.json` | Raw per-cell transcripts + tool flags |
| `SAULIE_PERSONA_AND_STEERING_FIX_PLAN.md` | Original patch spec |
| `agent_api.log` | Ground truth for tool emit/exec |

---

## Decision record

**Shipped after matrix:** `dpo_w10` + `compressed` — best compromise of DPO preference training, acceptable English when tools run, and modern prompt rules; acknowledged that **tool reliability still requires harness work**.

**Rejected for prod:** `dpo_w05` (no tools), `legacy` as default (persona/template regression despite better search on direct asks).

**Explicit non-goals of this matrix:** RAG precision, judge-panel re-score, catchphrase rate automation, ngrok/SSE infra.
