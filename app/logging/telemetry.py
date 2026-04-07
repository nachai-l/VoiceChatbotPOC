"""
telemetry.py — Structured turn-level logging for the POC.

Logs per-turn events to Python's standard logger so they can be captured
by any log handler (stdout, file, cloud sink).

Fields logged per turn (requirements §11 NFR-3):
  session_id, turn_id, timestamps, intent, Flash-Lite confidence,
  selected tool, tool latency, total latency, error events.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class TurnLog:
    """Accumulates timing and outcome data for one voice turn.

    Usage:
        log = TurnLog(session_id="abc123", utterance="what is my balance")
        # ... run orchestration ...
        log.intent = "balance_inquiry"
        log.selected_tool = "get_balance"
        # ... run tool ...
        log.tool_latency_s = 0.123
        log.finish()   # emits structured log line
    """

    session_id: str
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    start_time: float = field(default_factory=time.monotonic)
    utterance: str = ""
    intent: str = ""
    flash_lite_confidence: float = 0.0
    selected_tool: Optional[str] = None
    tool_latency_s: Optional[float] = None
    total_latency_s: Optional[float] = None
    error: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def finish(self, error: Optional[str] = None) -> None:
        """Record end time and emit a structured log line."""
        self.total_latency_s = round(time.monotonic() - self.start_time, 3)
        if error:
            self.error = error
        logger.info(
            "TURN session=%s turn=%s intent=%s confidence=%.2f tool=%s "
            "tool_latency=%s total_latency=%.3fs error=%s",
            self.session_id,
            self.turn_id,
            self.intent or "—",
            self.flash_lite_confidence,
            self.selected_tool or "—",
            f"{self.tool_latency_s:.3f}s" if self.tool_latency_s is not None else "—",
            self.total_latency_s,
            self.error or "—",
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for the UI debug panel."""
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "utterance": self.utterance,
            "intent": self.intent,
            "confidence": round(self.flash_lite_confidence, 3),
            "selected_tool": self.selected_tool,
            "tool_latency_s": self.tool_latency_s,
            "total_latency_s": self.total_latency_s,
            "error": self.error,
            **self.extra,
        }
