"""
tool_executor.py — Dispatcher that routes tool calls from the orchestrator
to the correct Phase 3 tool implementation.

The orchestrator calls tool_executor(tool_name, args_dict) synchronously.
Each registered tool must return a plain dict.

Registered tools:
  - "get_balance"           → balance_tool.get_balance
  - "get_transaction_status" → transaction_status_tool.get_transaction_status

Unknown tool_name values raise ValueError so the orchestrator can surface a
meaningful error rather than silently returning empty data.
"""

import logging
from typing import Any

from app.tools.balance_tool import get_balance
from app.tools.transaction_status_tool import get_transaction_status

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, Any] = {
    "get_balance": get_balance,
    "get_transaction_status": get_transaction_status,
}


def tool_executor(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Dispatch *tool_name* with *args* and return the result dict.

    Args:
        tool_name: The name of the tool as returned in OrchestrationDecision.selected_tool.
        args:      Keyword arguments to pass to the tool function.

    Returns:
        Result dict from the underlying tool.

    Raises:
        ValueError: If tool_name is not registered.
    """
    fn = _REGISTRY.get(tool_name)
    if fn is None:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown tool '{tool_name}'. Available tools: {available}"
        )

    logger.info("tool_executor: dispatching tool_name=%r args_keys=%s", tool_name, list(args.keys()))
    return fn(**args)
