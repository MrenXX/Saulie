#!/usr/bin/env python3
"""Chat with Saulie from another PC via ngrok -> nginx -> agent_chat_api."""

from openai import OpenAI

# Public ngrok URL (changes if ngrok restarts; check: curl -s http://127.0.0.1:4040/api/tunnels)
PUBLIC_URL = "https://plastic-esophagus-oxidant.ngrok-free.dev"
API_KEY = "secret"  # nginx Bearer token

TEMPERATURE = 0.7
TOP_P = 0.8
MAX_TOKENS = 512

client = OpenAI(
    api_key=API_KEY,
    base_url=f"{PUBLIC_URL.rstrip('/')}/v1",
)

messages: list[dict] = []
print("Saulie remote chat (quit/exit to leave)\n")

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye.")
        break

    if not user_input or user_input.lower() in {"quit", "exit", "q"}:
        if user_input.lower() in {"quit", "exit", "q"}:
            break
        continue

    messages.append({"role": "user", "content": user_input})
    print("Assistant: ", end="", flush=True)

    assistant_text = ""
    try:
        stream = client.chat.completions.create(
            model="dpo-v15-trial-4",
            messages=messages,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            stream=True,
            timeout=600,
        )

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                assistant_text += delta.content
                print(delta.content, end="", flush=True)
    except Exception as exc:
        print(f"\n[connection error: {exc}]", flush=True)
        if not assistant_text:
            messages.pop()
            continue

    print("\n")
    if assistant_text:
        messages.append({"role": "assistant", "content": assistant_text})
