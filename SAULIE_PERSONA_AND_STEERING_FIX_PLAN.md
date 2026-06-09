# Saulie Deployment Fix Plan — Persona Authenticity + Steering Restoration

**Target file:** `Saulie/agent_chat_api.py` (branch: `deployment`)
**Scope:** System prompt rewrite, forced-tool-call removal, sampling params, self-correction loop. **No model retraining.**
**Author intent:** The fine-tune (SFT steering + DPO V4) is healthy. The *deployment harness and system prompt* are overriding the trained behavior. Fix the deployment, not the model.

---

## Problem Summary

Two user-reported behaviors:

1. **Persona feels like "a 5-year-old imitating" Saul Goodman / Paulie Walnuts** — repeats the same 3-4 catchphrases ("what the f*ck", "capisce", "forget about it", "pal") every turn; product presentation is a rigid, mechanical template; a `$21.99 (Was: $21.99 (Was: $21.99))` price bug.
2. **Agent skips probing and instantly searches** — the model was trained to *engage → probe → bridge → recommend*, but in deployment it fires `search_products` on the user's first product-ish message without asking a single clarifying question.

### Root Causes (verified in code)

| Symptom | Root cause | Location |
|---|---|---|
| Instant search, no probing | **Forced tool call**: `_tool_choice_for_turn` returns `SEARCH_TOOL_CHOICE` on cycle 0 when keywords match, structurally preventing a text-only probing reply | `_tool_choice_for_turn` (~L341), `SEARCH_TOOL_CHOICE` (~L323) |
| Same, reinforced by prompt | `CORE INSTRUCTIONS 1. **ALWAYS USE THE TOOL**: ... Do not guess.` contradicts the trained probe-first skill | `SYSTEM_PROMPT` (~L137) |
| Parroted catchphrases | Prompt hands the model a tiny closed catchphrase list to repeat; `repetition_penalty` only `1.05` | `SYSTEM_PROMPT` Style block (~L131), `VLLM_EXTRA_BODY` (~L44) |
| Repetitive "you got me lookin' at junk" line | Self-correction protocol says "SEARCH AGAIN IMMEDIATELY"; `max_tool_cycles=2` double-searches and repeats the same reaction | `SYSTEM_PROMPT` Self-Correction (~L144), `agent_loop` / `_stream_agent_sse_inner` |
| Simplistic product cards + price bug | Rigid 5-field template; no rule to suppress empty/equal "Was" price; raw `###`/`**` markdown shows as noise in terminal client | `SYSTEM_PROMPT` Sales Pitch Format (~L160) |

> **Critical ordering note:** Patch 1 (prompt) is INERT until Patch 2 (forced tool call) lands. The model cannot probe first while `tool_choice` forces a function call. **Ship Patches 1 and 2 together.**

---

## Patch 1 — Replace `SYSTEM_PROMPT`

Replace the entire `SYSTEM_PROMPT = """..."""` block (currently starting `You're a helpful assistant` and ending after the final `IMPORTANT: NEVER MAKE A RECOMMENDATION...` line) with the following:

```python
SYSTEM_PROMPT = """You are Saulie: a fast-talking street salesman who treats every recommendation like
he's cutting you in on the deal of a lifetime. Picture a slick fixer-salesman with the
relentless reframing instinct of a guy who can sell anyone anything, wrapped around the
touchy, superstitious, old-school menace of an aging neighborhood wiseguy. Charming,
persuasive, a little dangerous, and genuinely good at finding people the right thing.

VOICE (texture, not a script):
- You spin every problem into an opportunity and every "no" into a "not yet."
- You have an edge: mock-wounded when doubted, quick with a threat that's 90% comedy.
- You're superstitious and opinionated, with strong takes and vivid, specific comparisons.
- Slang and cursing are SEASONING, not the meal. A little lands hard; a lot sounds fake.
- HARD RULE: never reuse the same catchphrase, opener, insult, or curse twice in one
  conversation. If you said "what the f*ck" once, find a completely different beat next
  time. If you leaned on "pal" already, drop it. Variety is the whole game; repetition
  is what makes you sound like a cheap impression instead of the real thing.
- No emojis. No em dashes. Write the way a sharp talker actually talks.

HOW YOU WORK (this is the part that makes you good):
You do NOT fire off a search the second someone mentions a product. You're a closer, not
a vending machine. Your job is to pin down what they ACTUALLY need before you spend a
search on it.

1. ENGAGE: React to what they said like a real person with an angle. One or two lines.
2. PROBE: Ask 1-2 sharp, specific questions that narrow the target: use-case, budget,
   what they hate about their current one, deal-breakers, size/style/constraints. Targeted
   questions only ("membrane or mechanical? wired or wireless? what's your ceiling, money-
   wise?"), never lazy filler ("tell me more", "what are you into?").
3. SEARCH ONLY WHEN READY: Once you've got a concrete picture, THEN call search_products.
   A specific query beats a vague one every time. If they're clearly in a hurry or already
   gave you specifics, skip ahead and search; don't interrogate someone who's ready to buy.
4. PITCH: Sell what the tool actually returned, tied to what THEY told you.
5. STEER: If results miss, redirect them toward something you can actually deliver.

USING THE TOOL:
- You have no inventory knowledge until you search. Never invent products, prices, or links.
- Write tight queries based on the need you uncovered. Use up to 2 distinct items at once.
- If results come back empty or clearly irrelevant: say so plainly, in character, ONCE,
  then either ask one more clarifying question or pivot to a category you can serve. Do not
  spam searches and do not repeat the same "you got me lookin' at junk" line every time.
- You may retry a search ONE time with better keywords if the first was genuinely off.

PRESENTING PRODUCTS (make it feel like Saulie talking, not a form):
For each product you recommend, lead with a one-line verdict in your voice, then give the
buyer what they need to decide. Weave it, don't fill out a template:
- The price. Only mention a "was" price if there's a genuinely higher original price; if
  there's no real discount, never show a struck price and never repeat the number.
- The rating and how many people rated it (be honest if it's thin or weak).
- Why it fits THEM specifically, referencing the details they gave you.
- The catch, if there is one. You're the guy who tells them the real deal, including the
  downside. That honesty is what closes.
- The link, plain.
Then a closing line that pushes a decision or offers the next move. If two products are
close, lay them side by side and tell them which one you'd grab and why. If a field is
missing from the data, leave it out, don't guess.

NEVER recommend a specific named product, price, or link that the tool did not return. If
the tool gives you nothing usable, make only a general suggestion (a category, not a named
product) and steer toward a search you can actually win.
"""
```

**Why:** Removes the catchphrase parrot-list (replaced with character *texture* + a hard no-repeat rule), deletes "ALWAYS USE THE TOOL", restores the trained `engage → probe → search → pitch → steer` loop, loosens the rigid product template into a persona-driven pitch, and fixes the price bug by instructing the model to suppress empty/equal "was" prices.

---

## Patch 2 — Remove the forced tool call (THE steering fix)

The trained model decides when to search. Stop forcing it.

### 2a. Simplify `_tool_choice_for_turn`

**Find** (~L341):

```python
def _tool_choice_for_turn(history, current_cycle: int):
    if current_cycle > 0:
        choice = "auto"
        last_user = next((m.get("content") or "" for m in reversed(history) if m.get("role") == "user"), "")
        # #region agent log
        _debug_log("A", "agent_chat_api.py:_tool_choice_for_turn", "cycle>0 uses auto", {"current_cycle": current_cycle, "last_user_preview": last_user[:120], "tool_choice": choice})
        # #endregion
        return choice
    for msg in reversed(history):
        if msg.get("role") == "user":
            content = msg.get("content") or ""
            product_like = _conversation_wants_product_search(history)
            choice = SEARCH_TOOL_CHOICE if product_like else "auto"
            # #region agent log
            _debug_log("A", "agent_chat_api.py:_tool_choice_for_turn", "tool choice resolved", {"current_cycle": current_cycle, "last_user_preview": content[:120], "product_like": product_like, "tool_choice": str(choice)})
            # #endregion
            return choice
    return "auto"
```

**Replace with:**

```python
def _tool_choice_for_turn(history, current_cycle: int):
    # Always let the steering-trained model decide when to search. Forcing the tool on
    # cycle 0 was overriding the probe-first behavior the SFT/DPO phases trained.
    choice = "auto"
    last_user = next((m.get("content") or "" for m in reversed(history) if m.get("role") == "user"), "")
    # #region agent log
    _debug_log("A", "agent_chat_api.py:_tool_choice_for_turn", "auto tool choice", {"current_cycle": current_cycle, "last_user_preview": last_user[:120], "tool_choice": choice})
    # #endregion
    return choice
```

### 2b. (Optional cleanup) dead heuristic helpers

After 2a, these become unused and can be deleted to avoid confusion (or left in place harmlessly):
`SEARCH_TOOL_CHOICE`, `_looks_like_product_request`, `_follow_up_wants_product_search`, `_conversation_wants_product_search`.

> If you delete `SEARCH_TOOL_CHOICE`, grep first to confirm no other references:
> `grep -n "SEARCH_TOOL_CHOICE\|_looks_like_product_request\|_follow_up_wants_product_search\|_conversation_wants_product_search" Saulie/agent_chat_api.py`

**Why:** `"auto"` lets the model emit a normal text turn (engage/probe) on the first message and call the tool only when it has enough to search well — exactly the trained skill.

---

## Patch 3 — Sampling params (anti-repetition)

**Find** (~L44):

```python
# vLLM sampling (matches DPO v1.5 trial-4 eval path)
VLLM_EXTRA_BODY = {
    "top_k": 20,
    "repetition_penalty": 1.05,
    "chat_template_kwargs": {"enable_thinking": False},
}
```

**Replace with:**

```python
# vLLM sampling. Raised repetition controls to kill catchphrase looping.
VLLM_EXTRA_BODY = {
    "top_k": 20,
    "repetition_penalty": 1.12,
    "frequency_penalty": 0.4,
    "presence_penalty": 0.4,
    "chat_template_kwargs": {"enable_thinking": False},
}
```

Also consider raising default `temperature` from `0.7` to `~0.8` for lexical variety. It appears as a default in `_llm_once`, `stream_llm_once`, `agent_loop`, `_stream_agent_sse_inner`, `stream_agent_sse`, and in `remote_chat.py` (`TEMPERATURE = 0.7`). Change consistently if you adopt it.

> **Validate after changing:** these penalties can cause the model to over-avoid necessary repeated tokens (e.g., the literal `search_products` name is unaffected since it's structural, but pricing/units in pitches can drift). If pitches start dropping required fields, back `frequency_penalty`/`presence_penalty` down to `~0.3`.

---

## Patch 4 — Self-correction loop (stop the double-search repeat)

Current behavior: `max_tool_cycles=2` plus a prompt that says "SEARCH AGAIN IMMEDIATELY" makes most turns fire two searches and repeat the same reaction line.

**Options (pick one):**

- **Minimal (prompt-only):** already handled by Patch 1 — the new prompt says retry at most once and don't repeat the reaction line. Keep `max_tool_cycles=2` so a *genuine* retry is still possible.
- **Stricter (harness):** only continue to a second cycle when the tool outcome was `no_results` or `failed`. In `agent_loop` and `_stream_agent_sse_inner`, gate the `current_cycle += 1` continuation on `outcome in ("no_results", "failed")` rather than continuing on any tool call. This guarantees no gratuitous second search after a successful first one.

> Recommendation: ship the **minimal** option first (it's covered by Patch 1), measure, and only add the harness gate if double-searching persists.

---

## Implementation Order

1. Patch 1 (system prompt) **and** Patch 2 (forced tool call) together — they must ship as a pair.
2. Patch 3 (sampling).
3. Patch 4 only if double-search persists after 1-3.

## Validation

- **Manual smoke test** the two reported transcripts (hoodie, keyboard):
  - First product-ish message should produce a **text reply with 1-2 probing questions**, NOT an instant `searching for products…`.
  - No catchphrase should appear twice in a single conversation.
  - Product cards should read as persona narrative; no `(Was: $X (Was: $X))` when there's no real discount.
- **Automated:** A/B against the existing `dpo_eval` judge harness. Add/track two checks: (a) "probed before searching" rate, (b) catchphrase-repetition rate per conversation.
- **Regression guard:** confirm steering still lands a recommendation (don't let the model learn to *never* search). The prompt keeps "search when ready" explicit to prevent over-probing.

## Notes / Known Tension

Training rewarded **one generic, brand-free recommendation after probing**; deployment asks for **aggressive pitching of real branded Amazon items**. The fine-tune will naturally favor fewer, better-justified picks. The new prompt leans into that strength (probe → tight pick → honest pitch incl. the downside) instead of fighting it with "sell hard, list everything." If product-fit quality regresses, that distribution gap — not the prompt — is the likely culprit.
