## LLM Demo

Install:
pip install -r requirements.txt

Run (OpenAI):
python -m src.chat --provider openai --prompt "Explain Pomodoro in one sentence."

Run (Anthropic):
python -m src.chat --provider anthropic --prompt "Explain Pomodoro in one sentence."

Offline mock (no key, no network):
python -m src.chat --provider mock --prompt "Explain Pomodoro in one sentence."
