import os
import time

class ProviderError(Exception):
    pass

def env(name: str) -> str:
    val = os.getenv(name, "")
    if not val:
        raise ProviderError(f"Missing environment variable: {name}")
    return val

def call_openai(prompt: str, timeout: int = 20) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=env("OPENAI_API_KEY"))
        start = time.time()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            timeout=timeout  # supported by SDK’s HTTP client
        )
        took = time.time() - start
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        meta = f"[openai] tokens_in={getattr(usage, 'prompt_tokens', '?')} tokens_out={getattr(usage, 'completion_tokens', '?')} time={took:.2f}s"
        return f"{text}\n\n{meta}"
    except Exception as e:
        raise ProviderError(f"OpenAI call failed: {e}")

def call_anthropic(prompt: str, timeout: int = 20) -> str:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=env("ANTHROPIC_API_KEY"))
        start = time.time()
        resp = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout
        )
        took = time.time() - start
        text = "".join([b.get("text", "") for b in resp.content if b.get("type") == "text"])
        # Anthropics’ usage fields vary by SDK version
        meta = f"[anthropic] time={took:.2f}s"
        return f"{text}\n\n{meta}"
    except Exception as e:
        raise ProviderError(f"Anthropic call failed: {e}")
