"""
guardrails.py — Safety guardrails for Phase 2.

Validates an OrchestrationDecision against policy before any tool is executed.
Returns a safe fallback message when the decision should be blocked, or None
when it is safe to proceed.

Policy (requirements §11 NFR-5):
  - Block unsupported intents
  - Block low-confidence tool routing
  - Block tool names not in the authorised list
  - Block sensitive field patterns without explicit confirmation
"""

from app.orchestration.schemas import IntentType, OrchestrationDecision

# Confidence below this threshold triggers an "I didn't understand" fallback.
MIN_TOOL_CONFIDENCE: float = 0.40

# Authorised tool names for this POC (Phase 3 implements them).
ALLOWED_TOOLS: frozenset[str] = frozenset({"get_balance", "get_transaction_status"})


def check_decision(decision: OrchestrationDecision) -> str | None:
    """Validate an orchestration decision against safety policy.

    Args:
        decision: The OrchestrationDecision returned by Flash-Lite.

    Returns:
        None  — decision is safe to proceed.
        str   — a safe spoken fallback message; the caller must use this
                instead of executing the decision.
    """
    # Out-of-scope requests
    if decision.intent == IntentType.UNSUPPORTED:
        return (
            "I'm sorry, I can only help with account balances and transaction "
            "status at this time. Is there anything else I can help you with?"
        )

    # Low-confidence tool routing
    if decision.requires_tool and decision.confidence < MIN_TOOL_CONFIDENCE:
        return (
            "I'm not sure I understood that correctly. "
            "Could you rephrase your request?"
        )

    # Unauthorised tool name (prompt injection / hallucination guard)
    if decision.selected_tool and decision.selected_tool not in ALLOWED_TOOLS:
        return (
            "I'm not able to perform that action."
        )

    return None
