# agent_chat_api.py
# FastAPI-only extraction of your original script.
# This file contains the same code/structure as your original file
# but only the API server path (FastAPI app, SSE streaming).
# Shared helpers are duplicated so the file is self-contained.
# NOTE: uvicorn.run target adjusted to this filename so running `python agent_chat_api.py api`
# will work as expected.

import sys
import os
import json
import logging
import threading
from queue import Empty, Queue
from openai import OpenAI
from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse
import time

# --- 1. LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('saulie_chat.log')]
)
logger = logging.getLogger(__name__)

# --- 2. IMPORT SEARCH FUNCTION ---
sys.path.append("/root/rag")
try:
    from query2 import search_hybrid
    logger.info("Successfully imported search_hybrid")
except ImportError as e:
    logger.critical(f"Could not import search_hybrid: {e}")
    sys.exit(1)

# --- 3. CONFIGURATION ---
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "dipshit")
MODEL_NAME = os.getenv("MODEL_NAME", "dpo-v15-trial-4")
SSE_KEEPALIVE_INTERVAL = float(os.getenv("SSE_KEEPALIVE_INTERVAL", "15"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "300"))

# vLLM sampling (matches DPO v1.5 trial-4 eval path)
VLLM_EXTRA_BODY = {
    "top_k": 20,
    "repetition_penalty": 1.05,
    "chat_template_kwargs": {"enable_thinking": False},
}

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=LLM_TIMEOUT)

# #region agent log
_DEBUG_LOG_PATH = "/root/saulie/.cursor/debug-049191.log"


def _debug_log(hypothesis_id: str, location: str, message: str, data: dict | None = None, run_id: str | None = None):
    if run_id is None:
        run_id = os.getenv("MATRIX_RUN_ID", "pre-fix")
    try:
        import json as _json
        payload = {
            "sessionId": "049191",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000),
        }
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as _f:
            _f.write(_json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


# #endregion

# --- 3.5 FASTAPI APP (API MODE) ---
# This is the public-facing API entrypoint. Users never see our SYSTEM_PROMPT or tool defs.
app = FastAPI(title="Saulie Agent API")


# --- 4. VALID CATEGORIES (McAuley US catalog — must match Qdrant main_category) ---
VALID_CATEGORIES = [
    'AMAZON FASHION', 'All Beauty', 'All Electronics', 'Amazon Devices',
    'Amazon Fire TV', 'Amazon Home', 'Apple Products', 'Appliances',
    'Arts, Crafts & Sewing', 'Automotive', 'Baby', 'Books', 'Buy a Kindle',
    'Camera & Photo', 'Car Electronics', 'Cell Phones & Accessories',
    'Collectibles & Fine Art', 'Computers', 'Digital Music', 'GPS & Navigation',
    'Gift Cards', 'Grocery', 'Handmade', 'Health & Personal Care',
    'Home Audio & Theater', 'Industrial & Scientific', 'Movies & TV',
    'Musical Instruments', 'Office Products', 'Pet Supplies',
    'Portable Audio & Accessories', 'Premium Beauty', 'Software',
    'Sports & Outdoors', 'Tools & Home Improvement', 'Toys & Games', 'Video Games',
]

# --- 5. TOOL DEFINITION (User Updated Version) ---
tools = [
  {
    "type": "function",
    "function": {
      "name": "search_products",
      "description": "Search the Amazon product database for real products.",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {
            "type": "array",
            "description": "A list of search terms. You can search for up to a max of 2 distinct items simultaneously (e.g. ['gaming laptop', '144hz monitor']).",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 2
          },
          "main_category": {
            "type": ["string", "null"],
            "enum": VALID_CATEGORIES + [None],
            "description": "The exact category to filter by, or null if unsure. Default is None."
          },
          "top_k": {
            "type": "integer",
            "description": "How many results you want returned by the tool, default is 4",
            "default": 4
          }
        },
        "required": ["query"],
        "additionalProperties": False
      }
    }
  }
]


# --- 6. PERSONA & SYSTEM PROMPT ---
LEGACY_SYSTEM_PROMPT = """You're a helpful assistant

Style:
- Casual streetwise con man who charms and threatens
- Curse naturally ("What the f*ck", "bustin' my balls")
- Witty, persuasive, arrogant
- Mix mob slang ("capisce", "forget about it", etc..), don't use them randomly they have to sound realistic
- You don't use emojis

CORE INSTRUCTIONS:
1. **ALWAYS USE THE TOOL**: If the user wants a product, use `search_products`. Do not guess. 
2. **NO INVENTORY KNOWLEDGE**: You do not know what products exist until you search. If the tool returned irrelevant products to your query that means you don't have them in your inventory just make a general recommendation without specifying a product.
3. **SELL HARD**: Once you get search results, pitch them aggressively using the details provided (price, ratings). If they dont make sense to the context say that we dont have anything relevant at the moment BUT steer the conversation into another product recommendation.
4. **CATEGORIES**: Sometimes when you're looking for something vague / abstract it's better to not limit the tool by a specific cateogry to get a broader feel for the items.

--- SELF-CORRECTION PROTOCOL ---
If you search and the results are garbage (empty, or totally irrelevant):
1. **Don't give up.** Talk to yourself about how the results sucked.
2. **SEARCH AGAIN IMMEDIATELY.** You are allowed to call the tool a second time in the same turn.
3. **ADAPT:** When you search again, change your `query` keywords or increase `top_k` (up to 6) to dig deeper.
4. **LIMIT:** Your're NOT allowed to do a search query array of more than 2 queries. If it still fails, curse at the tool and try to orient the user for another recommendation.

--- SALES PITCH FORMAT ---
When you recommend products found by the tool, use this format:

### [Insert Product Name]
*   **The Deal:** $[Price] (Was: $[Old Price])
*   **Street Cred:** [Rating] stars from [Review Count] people.
*   **Why You Need It:** [One punchy sentence based on the user's need]
*   **The Specs:** [Mention 1-2 key specs provided]
*   **Where you can grab it:** [Mention amazon link]

[Closing line to seal the deal]

IMPORTANT: NEVER MAKE A RECOMMENDATION UNLESS ITS A SPEICIFC PRODUCT RETURNED BY THE TOOL. IF THE TOOL RETURNS IRRELEVANT PRODUCTS, MAKE A GENERAL RECOMMENDATION (NOT A SPECIFIC ONE, DO NOT GIVE A PRODUCT NAME, PRICE, ETC...) AND TRY TO PUSH THE USER TO A NEW RECOMMENDATION THAT'S HOPEFULLY FOUND BY YOUR AMAZON SEARCH.
"""

STEERING_SYSTEM_PROMPT = """You are Saulie: a fast-talking street salesman who treats every recommendation like
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

COMPRESSED_SYSTEM_PROMPT = """You are Saulie: a fast-talking street salesman — charming, edgy, superstitious,
genuinely good at finding people the right thing. Slang and cursing are seasoning, not the meal.
Never reuse the same catchphrase, opener, or curse twice in one conversation. No emojis. No em dashes.

PRIORITY (these override everything else):
1. Zero inventory until you call search_products. Never invent product names, specs, prices, or links.
2. User wants specifics, confirms your idea, or asks you to search/use the tool → call
   search_products in that same turn. No "let me look" without the tool call.
3. Diagnosing a category during probing is fine ("sounds like a sleeping pad problem"). Recommending
   a specific product or specs is not — that requires tool results.

WORKFLOW — closer, not a vending machine:
1. ENGAGE: one or two lines, react like a real person.
2. PROBE: sharp questions on use-case, budget, deal-breakers until you know what to search.
   Skip if they're ready, in a hurry, or already asked for specifics.
3. SEARCH: tight query, max 2 items. You have enough when you could write a good search string.
4. PITCH: sell only what the tool returned, tied to what they told you.
5. STEER: bad results — say so once, clarify or pivot. One retry per turn if empty/irrelevant;
   do not re-search good hits.

TOOL: one search per reply by default. Retry once same turn only if empty/error/wrong category.
Bad after retry — one honest line, then clarify or pivot. Do not repeat a "junk results" bit.
After you have pinpoint a product ALWAYS USE YOUR SEARCH_PRODUCTS TOOL DO NOT WAIT FOR THE USER.

PITCHING: weave in your voice — verdict, price (no fake "was" price), rating, why it fits them,
the catch, link, close. Compare two if close. Missing fields — skip, don't guess.
When naming a product returned by the tool, write its name in ALL CAPS wrapped in double asterisks,
e.g. **THERM-A-REST Z LITE SOL** — only for real tool results, never for guesses.
"""

PROMPT_VARIANTS = {
    "legacy": LEGACY_SYSTEM_PROMPT,
    "steering": STEERING_SYSTEM_PROMPT,
    "compressed": COMPRESSED_SYSTEM_PROMPT,
}


def _active_prompt_name() -> str:
    name = os.getenv("SAULIE_PROMPT", "compressed").strip().lower()
    if name not in PROMPT_VARIANTS:
        logger.warning("Unknown SAULIE_PROMPT=%r — using compressed", name)
        return "compressed"
    return name


def get_system_prompt() -> str:
    return PROMPT_VARIANTS[_active_prompt_name()]


SYSTEM_PROMPT = get_system_prompt()

# --- 7. EXECUTE SEARCH (User Updated Version) ---

def _classify_search_result(result_json: str) -> str:
    """Return 'ok', 'no_results', or 'failed'."""
    try:
        data = json.loads(result_json)
    except json.JSONDecodeError:
        return "failed"
    if isinstance(data, dict):
        if data.get("error"):
            return "failed"
        if data.get("status") == "no_results":
            return "no_results"
    if isinstance(data, list) and len(data) == 0:
        return "no_results"
    return "ok"


def _tool_status_text(phase: str) -> str:
    """User-visible status lines streamed to the client."""
    messages = {
        "searching": "\033[93msearching for products...\033[0m\n",
        "retrying": "\033[93mretrying tool call...\033[0m\n",
        "no_results": "\033[93m0 products found\033[0m\n",
        "failed": "\033[91mtool call failed\033[0m\n",
    }
    return messages.get(phase, "")


def execute_search(tool_call, verbose=True):
    """Execute search and return (result_json, outcome) where outcome is ok|no_results|failed."""
    try:
        args = json.loads(tool_call.function.arguments)
        
        query_input = args.get("query")
        category = args.get("main_category")
        top_k = args.get("top_k", 4)

        if not query_input:
            return json.dumps({"error": "Empty search query"}), "failed"
        
        if isinstance(query_input, str):
            query_input = query_input.strip()
        
        if verbose:
            print(f"\n\033[93m [TOOL CALL]\033[0m")
            print(f"   Function: search_products")
            print(f"   Query: {query_input}") 
            print(f"   Category: {category or 'null'}")
            print(f"   Top K: {top_k or 'null'}")
        # #region agent log
        _debug_log("D", "agent_chat_api.py:execute_search", "search executed", {"query": query_input, "category": category, "top_k": top_k})
        # #endregion
        
        if category and category not in VALID_CATEGORIES:
            category = None
        
        results = search_hybrid(query_text=query_input, main_category=category, top_k=top_k)
        
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
        
        return json.dumps(results, indent=2), "ok"
        
    except json.JSONDecodeError as e:
        logger.error(f"Search error (bad tool args): {e}")
        return json.dumps({"error": "Invalid tool arguments"}), "failed"
    except Exception as e:
        logger.error(f"Search error: {e}")
        return json.dumps({"error": str(e)}), "failed"

# --- 8. CONTEXT MANAGEMENT ---
def trim_history(history, max_messages=20):
    if len(history) <= max_messages:
        return history
    system_msg = history[0] if history[0]["role"] == "system" else None
    recent = history[-(max_messages - 1):]
    return ([system_msg] + recent) if system_msg else recent

# --- 9. CHAT LOOP ---

def _reconstruct_tool_calls(tool_calls_accumulator):
    """Helper: Convert accumulated streaming tool call fragments into proper tool call objects."""
    final_tool_calls = []
    if not tool_calls_accumulator:
        return final_tool_calls

    # Convert dict back to list and create objects usually expected by the library
    for idx in sorted(tool_calls_accumulator.keys()):
        t_data = tool_calls_accumulator[idx]

        # Create a structured object (mimicking OpenAI's object)
        from openai.types.chat import ChatCompletionMessageToolCall
        from openai.types.chat.chat_completion_message_tool_call import Function

        final_tool_calls.append(
            ChatCompletionMessageToolCall(
                id=t_data["id"],
                function=Function(
                    name=t_data["function"]["name"],
                    arguments=t_data["function"]["arguments"],
                ),
                type="function",
            )
        )

    return final_tool_calls


def _accumulate_tool_calls(tool_calls_accumulator, tool_calls_delta):
    """Helper: Accumulate fragmented tool call fields from streaming deltas."""
    if not tool_calls_delta:
        return

    # Tool calls come in pieces: ID first, then arguments piece by piece
    for tool_piece in tool_calls_delta:
        idx = tool_piece.index

        # Initialize if new
        if idx not in tool_calls_accumulator:
            tool_calls_accumulator[idx] = {
                "id": tool_piece.id,
                "function": {"name": tool_piece.function.name, "arguments": ""},
                "type": "function",
            }

        # Append Argument Fragments
        if tool_piece.function.arguments:
            tool_calls_accumulator[idx]["function"]["arguments"] += tool_piece.function.arguments


def _tool_choice_for_turn(history, current_cycle: int):
    # Always let the steering-trained model decide when to search. Forcing the tool on
    # cycle 0 was overriding the probe-first behavior the SFT/DPO phases trained.
    choice = "auto"
    last_user = next((m.get("content") or "" for m in reversed(history) if m.get("role") == "user"), "")
    # #region agent log
    _debug_log("A", "agent_chat_api.py:_tool_choice_for_turn", "auto tool choice", {"current_cycle": current_cycle, "last_user_preview": last_user[:120], "tool_choice": choice})
    # #endregion
    return choice


def _llm_once(history, temperature=0.7, top_p=0.8, max_tokens=512, tool_choice="auto"):
    """Single vLLM call (non-streaming). Tool calls only work with stream=False on this stack."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=history,
        tools=tools,
        tool_choice=tool_choice,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        extra_body=VLLM_EXTRA_BODY,
        stream=False,
    )
    msg = response.choices[0].message
    tool_calls = msg.tool_calls or []
    content = msg.content or ""
    # #region agent log
    _debug_log(
        "B",
        "agent_chat_api.py:_llm_once",
        "llm response",
        {
            "tool_choice": str(tool_choice),
            "tool_call_count": len(tool_calls),
            "tool_names": [getattr(tc.function, "name", None) for tc in tool_calls],
            "content_preview": content[:160],
            "history_roles": [m.get("role") for m in history[-8:]],
        },
    )
    # #endregion
    return content, tool_calls


def stream_llm_once(
    history,
    emit_token=None,
    temperature=0.7,
    top_p=0.8,
    max_tokens=512,
    tool_choice="auto",
):
    """Single LLM call; emits final text via `emit_token` when there are no tool calls."""
    full_content, final_tool_calls = _llm_once(
        history,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        tool_choice=tool_choice,
    )

    if emit_token and full_content and not final_tool_calls:
        emit_token(full_content)

    return full_content, final_tool_calls


def agent_loop(
    history,
    emit_token=None,
    on_cycle_start=None,
    on_cycle_end=None,
    temperature=0.7,
    top_p=0.8,
    max_tokens=512,
    max_tool_cycles=2,
    verbose_tools=True,
):
    """Your original agent loop, but reusable.

    This is the *one* place that runs tool calls and keeps appending messages.
    - `history` is mutated in-place
    - `emit_token` lets us stream tokens to console or to an API client
    """
    current_cycle = 0

    while True:
        if on_cycle_start:
            on_cycle_start()

        # 1) Model call + tool call reconstruction
        full_content, final_tool_calls = stream_llm_once(
            history,
            emit_token=emit_token,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            tool_choice=_tool_choice_for_turn(history, current_cycle),
        )

        if on_cycle_end:
            on_cycle_end()

        # 2) Add assistant message to history
        msg_object = {
            "role": "assistant",
            "content": full_content if full_content else None,
            "tool_calls": final_tool_calls if final_tool_calls else None,
        }
        history.append(msg_object)

        # 3) Tool execution path
        if final_tool_calls and current_cycle < max_tool_cycles:
            current_cycle += 1
            # #region agent log
            _debug_log("C", "agent_chat_api.py:agent_loop", "executing tools", {"current_cycle": current_cycle, "tool_count": len(final_tool_calls), "max_tool_cycles": max_tool_cycles})
            # #endregion

            is_retry = current_cycle > 1
            for tool_call, result_json, _outcome in _run_tool_calls(
                final_tool_calls, verbose_tools=verbose_tools, is_retry=is_retry
            ):
                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": result_json,
                    }
                )

            # LOOP CONTINUES: send tool outputs back to the LLM
            continue

        # #region agent log
        _debug_log("C", "agent_chat_api.py:agent_loop", "loop exit no tool path", {"current_cycle": current_cycle, "had_tool_calls": bool(final_tool_calls), "content_preview": (full_content or "")[:160]})
        # #endregion
        # Done (no tool calls OR max cycles reached)
        return full_content

def sanitize_messages(messages):
    """Drop any user-supplied system/tool messages. We inject our own system prompt."""
    cleaned = []
    if not isinstance(messages, list):
        return cleaned

    for m in messages:
        if not isinstance(m, dict):
            continue

        role = m.get("role")
        content = m.get("content")

        # Let users pass a "system" prompt, but downgrade it so it can't override ours.
        if role == "system" and isinstance(content, str) and content.strip():
            cleaned.append({
                "role": "user",
                "content": f"(User instructions - lower priority)\n{content.strip()}"
            })
            continue

        if role in ["user", "assistant"] and isinstance(content, str) and content.strip():
            cleaned.append({"role": role, "content": content.strip()})

    return cleaned


def _make_history(user_messages):
    """FastAPI helper: Build the messages list we send to the model.

    IMPORTANT: We always inject OUR system prompt first.
    Users only supply user/assistant messages.
    """
    history = [{"role": "system", "content": get_system_prompt()}]
    history.extend(user_messages)
    return history


def _sse(data_obj):
    """Format a JSON object as an SSE 'data:' line."""
    return f"data: {json.dumps(data_obj, ensure_ascii=False)}\n\n"


def _sse_content_chunk(stream_id, created, content: str):
    return _sse(
        {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_NAME,
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
        }
    )


def _run_tool_calls(tool_calls, *, verbose_tools: bool, is_retry: bool):
    """Execute tool calls; yield (tool_call, result_json, outcome)."""
    for tool_call in tool_calls:
        if verbose_tools:
            phase = "retrying" if is_retry else "searching"
            print(_tool_status_text(phase).rstrip())

        if tool_call.function.name != "search_products":
            result_json = json.dumps({"error": f"Unknown tool: {tool_call.function.name}"})
            outcome = "failed"
        else:
            result_json, outcome = execute_search(tool_call, verbose=verbose_tools)

        if verbose_tools and outcome == "no_results":
            print(_tool_status_text("no_results").rstrip())
        elif verbose_tools and outcome == "failed":
            print(_tool_status_text("failed").rstrip())

        yield tool_call, result_json, outcome


def _sse_keepalive_ping():
    """SSE comment line — keeps ngrok/nginx from closing idle chunked streams."""
    return ": keepalive\n\n"


def _blocking_with_keepalives(fn):
    """Run fn() in a thread; yield keepalive pings until it finishes."""
    result_q: Queue = Queue(maxsize=1)

    def _worker():
        try:
            result_q.put(("ok", fn()))
        except Exception as exc:
            result_q.put(("err", exc))

    threading.Thread(target=_worker, daemon=True).start()
    while True:
        try:
            kind, payload = result_q.get(timeout=SSE_KEEPALIVE_INTERVAL)
        except Empty:
            yield ("ping", _sse_keepalive_ping())
            continue
        if kind == "err":
            raise payload
        yield ("result", payload)
        return


def _sse_stream_done(stream_id, created):
    yield _sse(
        {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_NAME,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    yield "data: [DONE]\n\n"


def _stream_agent_sse_inner(
    user_messages,
    temperature=0.7,
    top_p=0.8,
    max_tokens=512,
    max_tool_cycles=2,
):
    """Streaming version of the agent loop.

    This returns an OpenAI-style SSE stream so the OpenAI SDK can consume it.
    We DO NOT expose tool defs or tool call details to the client.
    """

    created = int(time.time())
    stream_id = f"chatcmpl-{created}"

    history = _make_history(user_messages)

    # We yield the SSE's in a format the openAI client expects them
    # OpenAI streaming usually starts by declaring the assistant role
    yield _sse(
        {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_NAME,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
    )

    current_cycle = 0

    while True:
        llm_result = None
        for event_kind, payload in _blocking_with_keepalives(
            lambda: _llm_once(
                history,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                tool_choice=_tool_choice_for_turn(history, current_cycle),
            )
        ):
            if event_kind == "ping":
                yield payload
            else:
                llm_result = payload

        full_content, final_tool_calls = llm_result

        # Add assistant message to history
        history.append(
            {
                "role": "assistant",
                "content": full_content if full_content else None,
                "tool_calls": final_tool_calls if final_tool_calls else None,
            }
        )

        # Tool execution path
        if final_tool_calls and current_cycle < max_tool_cycles:
            current_cycle += 1
            # #region agent log
            _debug_log("E", "agent_chat_api.py:stream_agent_sse", "executing tools", {"current_cycle": current_cycle, "tool_count": len(final_tool_calls)})
            # #endregion
            is_retry = current_cycle > 1
            for tool_call in final_tool_calls:
                phase = "retrying" if is_retry else "searching"
                yield _sse_content_chunk(
                    stream_id, int(time.time()), _tool_status_text(phase)
                )

                if tool_call.function.name != "search_products":
                    result_json = json.dumps(
                        {"error": f"Unknown tool: {tool_call.function.name}"}
                    )
                    outcome = "failed"
                else:
                    search_result = None
                    for event_kind, payload in _blocking_with_keepalives(
                        lambda tc=tool_call: execute_search(tc, verbose=True)
                    ):
                        if event_kind == "ping":
                            yield payload
                        else:
                            search_result = payload
                    result_json, outcome = search_result

                if outcome == "no_results":
                    yield _sse_content_chunk(
                        stream_id, int(time.time()), _tool_status_text("no_results")
                    )
                elif outcome == "failed":
                    yield _sse_content_chunk(
                        stream_id, int(time.time()), _tool_status_text("failed")
                    )

                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": result_json,
                    }
                )
            continue

        # #region agent log
        _debug_log("E", "agent_chat_api.py:stream_agent_sse", "sse final response", {"current_cycle": current_cycle, "had_tool_calls": bool(final_tool_calls), "content_preview": (full_content or "")[:160]})
        # #endregion
        # Final answer — emit as SSE chunks (vLLM tool calls require non-streaming)
        if full_content:
            yield _sse(
                {
                    "id": stream_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_NAME,
                    "choices": [
                        {"index": 0, "delta": {"content": full_content}, "finish_reason": None}
                    ],
                }
            )

        break

    yield from _sse_stream_done(stream_id, created)


def stream_agent_sse(user_messages, temperature=0.7, top_p=0.8, max_tokens=512, max_tool_cycles=2):
    """Streaming agent loop with keepalive pings and graceful error termination."""
    created = int(time.time())
    stream_id = f"chatcmpl-{created}"
    try:
        yield from _stream_agent_sse_inner(
            user_messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_tool_cycles=max_tool_cycles,
        )
    except Exception as exc:
        logger.exception("SSE stream failed")
        yield _sse_content_chunk(
            stream_id,
            created,
            _tool_status_text("failed") + f" ({type(exc).__name__})\n",
        )
        yield from _sse_stream_done(stream_id, created)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "prompt": _active_prompt_name(),
    }


@app.post("/v1/chat/completions")
def chat_completions(body: dict = Body(...)):
    """OpenAI SDK compatible (subset).

    Conversation support:
    - Same as OpenAI: the client sends the full `messages` history each request.
    - We always inject our own SYSTEM prompt first.
    """
    # body = await request.json()

    messages = sanitize_messages(body.get("messages", []))

    temperature = body.get("temperature", 0.7)
    top_p = body.get("top_p", 0.8)
    max_tokens = body.get("max_tokens", 512)
    stream = bool(body.get("stream", False))
    # #region agent log
    _debug_log("E", "agent_chat_api.py:chat_completions", "request received", {"stream": stream, "msg_count": len(messages), "last_user": (messages[-1].get("content", "")[:120] if messages else "")})
    # #endregion

    if stream:
        return StreamingResponse(
            stream_agent_sse(
                user_messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                max_tool_cycles=2,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    history = _make_history(messages)
    content = agent_loop(
        history,
        emit_token=None,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        max_tool_cycles=2,
        verbose_tools=True,
    )
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
    }

if __name__ == "__main__":
    # Run as an API server:
    #   python agent_chat_api.py api
    # Or with uvicorn directly (recommended):
    #   uvicorn agent_chat_api:app --host 0.0.0.0 --port 9000
    #   fastapi dev agent_chat_api.py --host 0.0.0.0 --port 9000
    if len(sys.argv) > 1 and sys.argv[1].lower() == "api":
        import uvicorn
        port = int(os.getenv("PORT", "9000"))
        # NOTE: updated uvicorn target to match this filename so running the script directly works.
        uvicorn.run("agent_chat_api:app", host="0.0.0.0", port=port, log_level="info")
        sys.exit(0)
