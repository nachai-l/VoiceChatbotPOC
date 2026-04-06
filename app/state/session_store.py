"""
session_store.py — Per-session backend state.

Holds lightweight mutable state that must survive across turns within one
voice session.  This is in-process storage for the POC — not a database.

SR-2 (requirements §9): the POC must maintain:
  - User name or pseudonymous session ID
  - Important verified values
  - Last successful tool result summary
  - Current workflow step
  - Latest unresolved question
  - Short rolling conversation summary  ← handled by SummaryStore
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SessionState:
    """Mutable state for one voice session.

    Attributes:
        session_id:           Unique identifier for this session.
        customer_id:          Customer ID once established during the session.
        verified_values:      Values confirmed by the user (e.g. dob, last4_phone).
        last_tool_result:     The most recent successful tool response dict.
        workflow_step:        High-level workflow position.
                              Values: "idle" | "awaiting_clarification" | "executing_tool"
        unresolved_question:  The follow-up question we are waiting for the user to answer.
        pending_intent:       Intent saved while waiting for a clarification answer.
        pending_tool_args:    Partial tool arguments collected so far for pending_intent.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    customer_id: Optional[str] = None
    verified_values: dict[str, Any] = field(default_factory=dict)
    last_tool_result: Optional[dict] = None
    workflow_step: str = "idle"
    unresolved_question: Optional[str] = None
    pending_intent: Optional[str] = None
    pending_tool_args: dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        """Reset all conversation-level fields, preserving session_id."""
        self.customer_id = None
        self.verified_values = {}
        self.last_tool_result = None
        self.workflow_step = "idle"
        self.unresolved_question = None
        self.pending_intent = None
        self.pending_tool_args = {}
