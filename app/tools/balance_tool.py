"""
balance_tool.py — Mock implementation of the get_balance tool.

Phase 3 design:
  - Accepts customer_id + one authentication factor (auth_token, last4_phone, or dob).
  - Returns a dict with status, available_balance, currency, account_type, and
    a masked account_number.
  - All customer data is mock/in-memory — no external system is called.
  - Authentication is deliberately simple for POC purposes.

Return schema:
  success → {"status": "ok", "available_balance": float, "currency": str,
              "account_type": str, "account_number_masked": str}
  auth failure → {"status": "auth_failed", "reason": str}
  not found   → {"status": "not_found", "reason": str}
  missing args → {"status": "invalid_args", "reason": str}
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mock data store
# ---------------------------------------------------------------------------

_MOCK_CUSTOMERS: dict[str, dict[str, Any]] = {
    "C001": {
        "name": "Alice Johnson",
        "auth_token": "token-alice",
        "last4_phone": "4321",
        "dob": "1985-03-12",
        "available_balance": 2_543.67,
        "currency": "USD",
        "account_type": "checking",
        "account_number_masked": "****-****-****-1001",
    },
    "C002": {
        "name": "Bob Smith",
        "auth_token": "token-bob",
        "last4_phone": "9876",
        "dob": "1990-07-25",
        "available_balance": 14_820.00,
        "currency": "USD",
        "account_type": "savings",
        "account_number_masked": "****-****-****-2002",
    },
    "C003": {
        "name": "Carol White",
        "auth_token": "token-carol",
        "last4_phone": "1111",
        "dob": "1978-11-30",
        "available_balance": 501.25,
        "currency": "USD",
        "account_type": "checking",
        "account_number_masked": "****-****-****-3003",
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_balance(
    customer_id: str,
    auth_token: str | None = None,
    last4_phone: str | None = None,
    dob: str | None = None,
) -> dict[str, Any]:
    """Return the mock account balance for *customer_id*.

    Exactly one of auth_token, last4_phone, or dob must be supplied for
    authentication.  If none are provided or the supplied value does not
    match the stored value, an auth_failed result is returned.

    Args:
        customer_id:  Customer identifier (e.g. "C001").
        auth_token:   Bearer token — highest-priority auth factor.
        last4_phone:  Last 4 digits of the registered phone number.
        dob:          Date of birth in YYYY-MM-DD format.

    Returns:
        dict with "status" key; see module docstring for full schema.
    """
    if not customer_id:
        return {"status": "invalid_args", "reason": "customer_id is required"}

    customer = _MOCK_CUSTOMERS.get(customer_id)
    if customer is None:
        logger.info("get_balance: unknown customer_id=%r", customer_id)
        return {"status": "not_found", "reason": f"No account found for customer_id '{customer_id}'"}

    # Authenticate
    authenticated = False
    if auth_token is not None:
        authenticated = (auth_token == customer["auth_token"])
    elif last4_phone is not None:
        authenticated = (str(last4_phone).strip() == customer["last4_phone"])
    elif dob is not None:
        authenticated = (str(dob).strip() == customer["dob"])
    else:
        return {
            "status": "invalid_args",
            "reason": "At least one authentication factor is required: auth_token, last4_phone, or dob",
        }

    if not authenticated:
        logger.warning("get_balance: auth failed for customer_id=%r", customer_id)
        return {"status": "auth_failed", "reason": "Authentication failed — please verify your details"}

    logger.info(
        "get_balance: returning balance=%.2f for customer_id=%r",
        customer["available_balance"], customer_id,
    )
    return {
        "status": "ok",
        "available_balance": customer["available_balance"],
        "currency": customer["currency"],
        "account_type": customer["account_type"],
        "account_number_masked": customer["account_number_masked"],
    }
