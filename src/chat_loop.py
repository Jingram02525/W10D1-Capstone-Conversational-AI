from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

def _choose_provider(name: str):
    name = (name or "").strip().lower()
    if name == "openai":
        from src.providers import openai_client as client
        return client
    if name == "anthropic":
        from src.providers import anthropic_client as client
        return client
    from src.providers import mock_client as client
    return client

def _trim_history(messages: list[dict], max_turns: int = 12) -> list[dict]:
    """
    Keep system message (if any) + last N user/assistant turns.
    """
    if not messages:
        return messages

    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    # Each turn is typically 2 messages (user+assistant). We just keep the last `max_turns` messages total.
    trimmed_non_system = non_system[-max_turns:]
    return (system_msgs[:1] + trimmed_non_system) if system_msgs else trimmed_non_system

def _save_jsonl(path: str, messages: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser(
        prog="chat_loop",
        description="Conversational CLI with memory (OpenAI / Anthropic / Mock)"
    )
    parser.add_argument("--provider", choices=["openai", "anthropic", "mock"], default="mock")
    parser.add_argument("--model", help="Override model name")
    parser.add_argument("--system", default="You are a helpful, concise assistant.")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--max-turns", type=int, default=12, help="Max non-system messages kept in memory")
    parser.add_argument("--transcript", help="Optional path to save conversation JSONL (append mode)")
    args = parser.parse_args()

    client = _choose_provider(args.provider)

    messages: list[dict] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    print(f"[{args.provider}] Conversational mode. Type '/reset', '/save', or '/exit'.")
    while True:
        try:
            user = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user:
            continue

        if user.lower() in ("/exit", "/quit"):
            print("Goodbye.")
            break

        if user.lower() == "/reset":
            messages = []
            if args.system:
                messages.append({"role": "system", "content": args.system})
            print("(context cleared)")
            continue

        if user.lower().startswith("/save"):
            path = args.transcript or f"transcripts/chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            _save_jsonl(path, messages)
            print(f"(saved transcript to {path})")
            continue

        # Normal turn
        messages.append({"role": "user", "content": user})
        messages = _trim_history(messages, max_turns=args.max_turns)

        try:
            reply = client.ask_chat(messages=messages, model=args.model, timeout=args.timeout)
        except Exception as e:
            print(f"LLM unavailable: {args.provider} failed: {e}")
            print("Falling back to mock.")
            from src.providers import mock_client
            reply = mock_client.ask_chat(messages)

        reply = (reply or "").strip()
        messages.append({"role": "assistant", "content": reply})
        print(reply)

        # Optional: autosave after each turn if path is provided
        if args.transcript:
            _save_jsonl(args.transcript, [{"role":"user","content":user},{"role":"assistant","content":reply}])

if __name__ == "__main__":
    main()
