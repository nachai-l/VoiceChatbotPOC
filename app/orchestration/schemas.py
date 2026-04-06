"""
schemas.py — Pydantic models for Flash-Lite orchestration I/O.

The OrchestrationDecision is the contract between our backend and Flash-Lite.
Every Flash-Lite call must return JSON that validates against this schema.

Schema reference (requirements §8.2):
  {
    "intent": "balance_inquiry|transaction_status|faq|clarify|unsupported",
    "requires_tool": true,
    "selected_tool": "get_balance",
    "confidence": 0.92,
    "missing_fields": [],
    "tool_arguments": {"customer_id": "C12345"},
    "user_message_before_tool": "Let me check that for you.",
    "user_message_after_tool": "Your available balance is {TOOL_RESULT}.",
    "reason": "User explicitly asked for account balance"
  }
"""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, field_validator, model_validator


class IntentType(str, Enum):
    BALANCE_INQUIRY = "balance_inquiry"
    TRANSACTION_STATUS = "transaction_status"
    FAQ = "faq"
    CLARIFY = "clarify"
    UNSUPPORTED = "unsupported"


class OrchestrationDecision(BaseModel):
    intent: IntentType
    requires_tool: bool
    selected_tool: Optional[str] = None
    confidence: float
    missing_fields: list[str] = []
    tool_arguments: dict[str, Any] = {}
    user_message_before_tool: str = ""
    user_message_after_tool: str = ""
    reason: str = ""

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {v}")
        return v

    @model_validator(mode="after")
    def tool_name_required_when_requires_tool(self) -> "OrchestrationDecision":
        """If requires_tool is True, selected_tool must not be empty."""
        if self.requires_tool and not self.selected_tool:
            raise ValueError(
                "selected_tool must be set when requires_tool is True"
            )
        return self
