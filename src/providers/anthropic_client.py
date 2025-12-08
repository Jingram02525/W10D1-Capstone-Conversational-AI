from __future__ import annotations
import os
from typing import Optional
from dotenv import load_dotenv
import anthropic
from anthropic import APIConnectionError, RateLimitError, BadRequestError, AuthenticationError

load_dotenv()

# Use a real model ID (avoid "-latest" aliases to prevent 404s)
_DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

def _anthropic_text(resp) -> str:
    """
    Normalize Anthropic SDK response to plain text.
    SDK returns objects (e.g., TextBlock) in resp.content.
    """
    parts: list[str] = []
    for block in getattr(resp, "content", []) or []:
        # Preferred: SDK object with .text attribute
        if hasattr(block, "text"):
            parts.append(getattr(block, "text") or "")
            continue
        # Fallback: dict shape
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(p for p in parts if p)

def ask(prompt: str, model: Optional[str] = None, timeout: int = 30) -> str:
    """
    Ask Anthropic Messages API.
    - Uses env var ANTHROPIC_API_KEY.
    - Default model overridable via --model or ANTHROPIC_MODEL.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Missing environment variable: ANTHROPIC_API_KEY")

    client = anthropic.Anthropic(api_key=api_key)
    model = model or _DEFAULT_MODEL

    try:
        msg = client.messages.create(
            model=model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
            # NOTE: The Python SDK handles timeouts at transport level; keep prompt simple.
        )
        return _anthropic_text(msg)
    except AuthenticationError as e:
        raise RuntimeError(f"Anthropic auth failed: {e}") from e
    except RateLimitError as e:
        raise RuntimeError(f"Anthropic rate limit: {e}") from e
    except BadRequestError as e:
        # This often includes model-not-found or invalid params
        raise RuntimeError(f"Anthropic bad request: {e}") from e
    except APIConnectionError as e:
        raise RuntimeError(f"Anthropic network error: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Anthropic call failed: {e}") from e
