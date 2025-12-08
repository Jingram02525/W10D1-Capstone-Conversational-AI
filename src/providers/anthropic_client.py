from __future__ import annotations

import os
from typing import Optional

import anthropic
from anthropic import (APIConnectionError, AuthenticationError,
                       BadRequestError, RateLimitError)
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

def _anthropic_text(resp) -> str:
    parts: list[str] = []
    for block in getattr(resp, "content", []) or []:
        if hasattr(block, "text"):
            parts.append(getattr(block, "text") or "")
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(p for p in parts if p)

def _client_and_model(model: Optional[str]):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Missing environment variable: ANTHROPIC_API_KEY")
    return anthropic.Anthropic(api_key=api_key), (model or _DEFAULT_MODEL)

def ask(prompt: str, model: Optional[str] = None, timeout: int = 30) -> str:
    client, model = _client_and_model(model)
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return _anthropic_text(msg)
    except AuthenticationError as e:
        raise RuntimeError(f"Anthropic auth failed: {e}") from e
    except RateLimitError as e:
        raise RuntimeError(f"Anthropic rate limit: {e}") from e
    except BadRequestError as e:
        raise RuntimeError(f"Anthropic bad request: {e}") from e
    except APIConnectionError as e:
        raise RuntimeError(f"Anthropic network error: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Anthropic call failed: {e}") from e

def ask_chat(messages: list[dict], model: Optional[str] = None, timeout: int = 30) -> str:
    """
    messages format for Anthropic is the same style:
    [{"role":"user","content":"..."},{"role":"assistant","content":"..."}...]
    Include a system prompt by passing system="..." separately.
    We will extract a system message (if present) from the first message with role 'system'.
    """
    client, model = _client_and_model(model)

    system_text = None
    user_assistant_turns: list[dict] = []
    for m in messages:
        r = m.get("role")
        c = m.get("content", "")
        if r == "system" and system_text is None:
            system_text = c
        elif r in ("user", "assistant"):
            user_assistant_turns.append({"role": r, "content": c})

    try:
        msg = client.messages.create(
            model=model,
            max_tokens=400,
            system=system_text or None,
            messages=user_assistant_turns,
        )
        return _anthropic_text(msg)
    except AuthenticationError as e:
        raise RuntimeError(f"Anthropic auth failed: {e}") from e
    except RateLimitError as e:
        raise RuntimeError(f"Anthropic rate limit: {e}") from e
    except BadRequestError as e:
        raise RuntimeError(f"Anthropic bad request: {e}") from e
    except APIConnectionError as e:
        raise RuntimeError(f"Anthropic network error: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Anthropic call failed: {e}") from e
