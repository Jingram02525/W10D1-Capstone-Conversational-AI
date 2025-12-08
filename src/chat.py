from __future__ import annotations
import argparse
import os
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
    # default mock
    from src.providers import mock_client as client
    return client

def main():
    parser = argparse.ArgumentParser(
        prog="chat",
        description="Tiny multi-provider LLM demo (OpenAI / Anthropic / Mock)"
    )
    parser.add_argument("--provider", choices=["openai", "anthropic", "mock"], default="mock",
                        help="LLM provider to use")
    parser.add_argument("--model", help="Override the model name for selected provider")
    parser.add_argument("--prompt", required=True, help="User prompt to send")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    args = parser.parse_args()

    client = _choose_provider(args.provider)

    try:
        text = client.ask(prompt=args.prompt, model=args.model, timeout=args.timeout)
        print(text.strip() or "<no text>")
    except Exception as e:
        # Provider failed → graceful fallback to mock
        if args.provider != "mock":
            print(f"LLM unavailable: {args.provider.capitalize()} call failed: {e}")
            print("Falling back to --provider mock\n")
            from src.providers import mock_client
            print(mock_client.ask(args.prompt))
        else:
            # Even mock failed… print the error
            print(f"Mock failed unexpectedly: {e}")

if __name__ == "__main__":
    main()
