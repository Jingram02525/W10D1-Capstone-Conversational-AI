import argparse
from src.providers import call_openai, call_anthropic, ProviderError
from src.mock import mock_answer
from dotenv import load_dotenv

load_dotenv()

def main():
    p = argparse.ArgumentParser(
        prog="chat",
        description="Tiny LLM CLI demo (OpenAI/Anthropic) with safe fallback."
    )
    p.add_argument("--prompt", required=False, default="Explain Pomodoro in one sentence.")
    p.add_argument("--provider", choices=["openai", "anthropic", "mock"], default="openai",
                   help="Choose LLM provider. Use 'mock' if offline.")
    p.add_argument("--timeout", type=int, default=20)
    args = p.parse_args()

    try:
        if args.provider == "openai":
            print(call_openai(args.prompt, timeout=args.timeout))
        elif args.provider == "anthropic":
            print(call_anthropic(args.prompt, timeout=args.timeout))
        else:
            print(mock_answer(args.prompt))
    except ProviderError as e:
        print(f"LLM unavailable: {e}")
        print("Falling back to --provider mock\n")
        print(mock_answer(args.prompt))

if __name__ == "__main__":
    main()
