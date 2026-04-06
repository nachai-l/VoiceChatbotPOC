"""
summary_store.py — Rolling conversation summary for context injection.

Maintains the last N completed turns as compact text injected into every
Flash-Lite orchestration call (SR-3, requirements §9).  When the Live API
applies context-window compression, older turns may disappear from the live
session; this store ensures critical context survives across reconnections.
"""

from dataclasses import dataclass, field

MAX_TURNS: int = 6  # Keep the last 6 turns (≈ 3 back-and-forth exchanges)


@dataclass
class SummaryStore:
    """Stores the last MAX_TURNS (user, assistant) text pairs."""

    _turns: list[tuple[str, str]] = field(default_factory=list)

    def update(self, user_text: str, assistant_text: str) -> None:
        """Append a completed turn and trim to MAX_TURNS."""
        self._turns.append((user_text.strip(), assistant_text.strip()))
        if len(self._turns) > MAX_TURNS:
            self._turns = self._turns[-MAX_TURNS:]

    def get_summary(self) -> str:
        """Return a compact text summary of recent turns for prompt injection."""
        if not self._turns:
            return "No prior conversation."
        lines: list[str] = []
        for user, assistant in self._turns:
            lines.append(f"User: {user}")
            lines.append(f"Assistant: {assistant}")
        return "\n".join(lines)

    def turn_count(self) -> int:
        return len(self._turns)

    def clear(self) -> None:
        self._turns.clear()
