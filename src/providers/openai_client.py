from __future__ import annotations
import os
from typing import Optional
from dotenv import load_dotenv

# OpenAI SDK v1+
from openai import OpenAI, APIConnectionError, RateLimitError, BadRequestError, AuthenticationError

load_dotenv()

_DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def _extract_text(msg_content) -> str:
    """
    OpenAI may return a string or a list of content parts.
    Normalize to a single string.
    """
    if isinstance(msg_content, str):
        return msg_content
    if isinstance(msg_content, list):
        parts = []
        for part in msg_content:
            # part may be {"type":"text","text":"..."} or already str in older paths
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(p for p in parts if p)
    return str(msg_content or "")

def ask(prompt: str, model: Optional[str] = None, timeout: int = 30) -> str:
    """
    Ask OpenAI Chat Completions API.
    - Uses env var OPENAI_API_KEY.
    - Default model overridable via --model or OPENAI_MODEL.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing environment variable: OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)
    model = model or _DEFAULT_MODEL

    try:
        # Using Chat Completions for broad compatibility
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout,
        )
        choice = resp.choices[0]
        return _extract_text(choice.message.content)
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
