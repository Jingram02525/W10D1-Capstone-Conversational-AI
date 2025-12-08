from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from openai import (APIConnectionError, AuthenticationError, BadRequestError,
                    OpenAI, RateLimitError)

load_dotenv()

_DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def _extract_text(msg_content) -> str:
    if isinstance(msg_content, str):
        return msg_content
    if isinstance(msg_content, list):
        parts = []
        for part in msg_content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(p for p in parts if p)
    return str(msg_content or "")

def _client_and_model(model: Optional[str]):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing environment variable: OPENAI_API_KEY")
    return OpenAI(api_key=api_key), (model or _DEFAULT_MODEL)

def ask(prompt: str, model: Optional[str] = None, timeout: int = 30) -> str:
    client, model = _client_and_model(model)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout,
        )
        return _extract_text(resp.choices[0].message.content)
    except AuthenticationError as e:
        raise RuntimeError(f"OpenAI auth failed: {e}") from e
    except RateLimitError as e:
        raise RuntimeError(f"OpenAI rate limit: {e}") from e
    except BadRequestError as e:
        raise RuntimeError(f"OpenAI bad request: {e}") from e
    except APIConnectionError as e:
        raise RuntimeError(f"OpenAI network error: {e}") from e
    except Exception as e:
        raise RuntimeError(f"OpenAI call failed: {e}") from e

def ask_chat(messages: list[dict], model: Optional[str] = None, timeout: int = 30) -> str:
    """
    messages: e.g. [
        {"role":"system","content":"You are helpful."},
        {"role":"user","content":"hi"},
        {"role":"assistant","content":"hello!"}, ...
    ]
    """
    client, model = _client_and_model(model)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=timeout,
        )
        return _extract_text(resp.choices[0].message.content)
    except AuthenticationError as e:
        raise RuntimeError(f"OpenAI auth failed: {e}") from e
    except RateLimitError as e:
        raise RuntimeError(f"OpenAI rate limit: {e}") from e
    except BadRequestError as e:
        raise RuntimeError(f"OpenAI bad request: {e}") from e
    except APIConnectionError as e:
        raise RuntimeError(f"OpenAI network error: {e}") from e
    except Exception as e:
        raise RuntimeError(f"OpenAI call failed: {e}") from e
