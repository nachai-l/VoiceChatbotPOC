"""
transaction_status_tool.py — Mock implementation of the get_transaction_status tool.

Phase 3 design:
  - Accepts a reference_id and customer_id.
  - Returns a dict describing the transaction (amount, type, status, timestamp,
    description) or an error dict.
  - All data is mock/in-memory.

Return schema:
  success → {"status": "ok", "transaction_status": str, "amount": float,
              "currency": str, "type": str, "timestamp": str, "description": str,
              "reference_id": str}
  not found  → {"status": "not_found", "reason": str}
  wrong owner → {"status": "not_found", "reason": str}  (security: no info leak)
  missing args → {"status": "invalid_args", "reason": str}
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mock data store
# ---------------------------------------------------------------------------
# Format: {reference_id: {owner_customer_id, ...fields}}

_MOCK_TRANSACTIONS: dict[str, dict[str, Any]] = {
    "TXN-001": {
        "owner": "C001",
        "transaction_status": "completed",
        "amount": 150.00,
        "currency": "USD",
        "type": "debit",
        "timestamp": "2025-04-01T10:23:00Z",
        "description": "Grocery store payment",
    },
    "TXN-002": {
        "owner": "C001",
        "transaction_status": "pending",
        "amount": 500.00,
        "currency": "USD",
        "type": "transfer",
        "timestamp": "2025-04-06T08:45:00Z",
        "description": "Bank transfer to external account",
    },
    "TXN-003": {
        "owner": "C002",
        "transaction_status": "completed",
        "amount": 2_000.00,
        "currency": "USD",
        "type": "credit",
        "timestamp": "2025-04-05T14:00:00Z",
        "description": "Salary deposit",
    },
    "TXN-004": {
        "owner": "C002",
        "transaction_status": "failed",
        "amount": 75.50,
        "currency": "USD",
        "type": "debit",
        "timestamp": "2025-04-06T19:12:00Z",
        "description": "Online purchase — insufficient funds",
    },
    "TXN-005": {
        "owner": "C003",
        "transaction_status": "completed",
        "amount": 30.00,
        "currency": "USD",
        "type": "debit",
        "timestamp": "2025-04-07T09:00:00Z",
        "description": "Utility bill payment",
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_transaction_status(
    reference_id: str,
    customer_id: str,
) -> dict[str, Any]:
    """Return the mock status of a transaction identified by *reference_id*.

    *customer_id* is used to verify ownership — a customer can only query
    their own transactions.  If the reference_id does not exist or belongs to
    a different customer, "not_found" is returned (no information leak).

    Args:
        reference_id:  Transaction identifier (e.g. "TXN-001").
        customer_id:   The requesting customer's identifier.

    Returns:
        dict with "status" key; see module docstring for full schema.
    """
    if not reference_id:
        return {"status": "invalid_args", "reason": "reference_id is required"}
    if not customer_id:
        return {"status": "invalid_args", "reason": "customer_id is required"}

    txn = _MOCK_TRANSACTIONS.get(reference_id)
    if txn is None or txn["owner"] != customer_id:
        logger.info(
            "get_transaction_status: not found — reference_id=%r customer_id=%r",
            reference_id, customer_id,
        )
        return {
            "status": "not_found",
            "reason": f"No transaction found for reference_id '{reference_id}'",
        }

    logger.info(
        "get_transaction_status: reference_id=%r status=%r customer_id=%r",
        reference_id, txn["transaction_status"], customer_id,
    )
    return {
        "status": "ok",
        "reference_id": reference_id,
        "transaction_status": txn["transaction_status"],
        "amount": txn["amount"],
        "currency": txn["currency"],
        "type": txn["type"],
        "timestamp": txn["timestamp"],
        "description": txn["description"],
    }
