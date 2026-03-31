"""
transcript_handler.py — Manages conversation history for Gradio's Chatbot component.

Stores turns as Gradio messages-format dicts: {"role": "user"|"assistant", "content": str}
Phase 2 will extend this to sync with the backend session summary store.
"""

from dataclasses import dataclass, field


@dataclass
class TranscriptHandler:
    """Holds the conversation history for one Gradio session."""

    history: list[dict] = field(default_factory=list)

    def add_user(self, text: str) -> None:
        """Append a user turn. Falls back to placeholder if transcript is empty."""
        self.history.append({"role": "user", "content": text or "[audio]"})

    def add_assistant(self, text: str) -> None:
        """Append an assistant turn. Falls back to placeholder if transcript is empty."""
        self.history.append({"role": "assistant", "content": text or "[audio response]"})

    def get_history(self) -> list[dict]:
        return list(self.history)

    def clear(self) -> None:
        self.history.clear()
