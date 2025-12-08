def ask(prompt: str, model: str | None = None, timeout: int = 30) -> str:
    return "Pomodoro is 25 minutes of focus followed by a 5-minute break."

def ask_chat(messages: list[dict], model: str | None = None, timeout: int = 30) -> str:
    # Very simple: reply to the latest user message, if any.
    last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    if last_user:
        return f"(mock) I hear you said: {last_user[:120]}"
    return "(mock) How can I help?"
