def mock_answer(prompt: str) -> str:
    # Domain-aware stubs are fine for demo; adjust per team problem.
    if "pomodoro" in prompt.lower():
        return "Pomodoro is 25 minutes focus + 5 minutes break."
    return "This is a mock response. Replace with your domain prompt."
