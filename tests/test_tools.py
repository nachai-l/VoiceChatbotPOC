"""
tests/test_tools.py — Unit tests for Phase 3 mock tools.

Covers:
  - balance_tool.get_balance: success paths, all auth factors, auth failure, not found, missing args
  - transaction_status_tool.get_transaction_status: success, not found, wrong owner, missing args
  - tool_executor.tool_executor: dispatch, unknown tool, kwarg forwarding
"""

import pytest

from app.tools.balance_tool import get_balance
from app.tools.transaction_status_tool import get_transaction_status
from app.tools.tool_executor import tool_executor


# ---------------------------------------------------------------------------
# balance_tool.get_balance
# ---------------------------------------------------------------------------

class TestGetBalanceSuccess:
    def test_auth_via_auth_token(self):
        result = get_balance("C001", auth_token="token-alice")
        assert result["status"] == "ok"
        assert result["available_balance"] == pytest.approx(2543.67)
        assert result["currency"] == "USD"
        assert result["account_type"] == "checking"
        assert "****" in result["account_number_masked"]

    def test_auth_via_last4_phone(self):
        result = get_balance("C001", last4_phone="4321")
        assert result["status"] == "ok"
        assert result["available_balance"] == pytest.approx(2543.67)

    def test_auth_via_dob(self):
        result = get_balance("C001", dob="1985-03-12")
        assert result["status"] == "ok"
        assert result["available_balance"] == pytest.approx(2543.67)

    def test_second_customer_token(self):
        result = get_balance("C002", auth_token="token-bob")
        assert result["status"] == "ok"
        assert result["available_balance"] == pytest.approx(14820.00)
        assert result["account_type"] == "savings"

    def test_third_customer_phone(self):
        result = get_balance("C003", last4_phone="1111")
        assert result["status"] == "ok"
        assert result["available_balance"] == pytest.approx(501.25)

    def test_returns_masked_account_number(self):
        result = get_balance("C002", auth_token="token-bob")
        assert result["account_number_masked"].startswith("****")

    def test_auth_token_takes_priority_over_phone(self):
        """When both auth_token and last4_phone are supplied, auth_token wins."""
        # Correct token but wrong phone — should still succeed
        result = get_balance("C001", auth_token="token-alice", last4_phone="9999")
        assert result["status"] == "ok"


class TestGetBalanceAuthFailure:
    def test_wrong_token(self):
        result = get_balance("C001", auth_token="wrong-token")
        assert result["status"] == "auth_failed"
        assert "reason" in result

    def test_wrong_phone(self):
        result = get_balance("C001", last4_phone="0000")
        assert result["status"] == "auth_failed"

    def test_wrong_dob(self):
        result = get_balance("C001", dob="2000-01-01")
        assert result["status"] == "auth_failed"

    def test_auth_failed_message_is_user_friendly(self):
        result = get_balance("C001", auth_token="bad")
        assert len(result["reason"]) > 10  # Not empty / not a stack trace


class TestGetBalanceNotFound:
    def test_unknown_customer_id(self):
        result = get_balance("C999", auth_token="any")
        assert result["status"] == "not_found"
        assert "reason" in result

    def test_not_found_includes_customer_id_in_reason(self):
        result = get_balance("C999", auth_token="any")
        assert "C999" in result["reason"]


class TestGetBalanceInvalidArgs:
    def test_empty_customer_id(self):
        result = get_balance("", auth_token="token-alice")
        assert result["status"] == "invalid_args"

    def test_no_auth_factor_provided(self):
        result = get_balance("C001")
        assert result["status"] == "invalid_args"
        assert "authentication" in result["reason"].lower()

    def test_all_none_auth_factors(self):
        result = get_balance("C001", auth_token=None, last4_phone=None, dob=None)
        assert result["status"] == "invalid_args"


class TestGetBalanceResultShape:
    def test_ok_result_has_all_required_keys(self):
        result = get_balance("C001", auth_token="token-alice")
        required = {"status", "available_balance", "currency", "account_type", "account_number_masked"}
        assert required.issubset(result.keys())

    def test_available_balance_is_float(self):
        result = get_balance("C001", auth_token="token-alice")
        assert isinstance(result["available_balance"], float)

    def test_currency_is_string(self):
        result = get_balance("C001", auth_token="token-alice")
        assert isinstance(result["currency"], str)
        assert len(result["currency"]) == 3  # e.g. "USD"


# ---------------------------------------------------------------------------
# transaction_status_tool.get_transaction_status
# ---------------------------------------------------------------------------

class TestGetTransactionStatusSuccess:
    def test_completed_transaction(self):
        result = get_transaction_status("TXN-001", "C001")
        assert result["status"] == "ok"
        assert result["transaction_status"] == "completed"
        assert result["amount"] == pytest.approx(150.00)
        assert result["currency"] == "USD"
        assert result["type"] == "debit"
        assert result["reference_id"] == "TXN-001"

    def test_pending_transaction(self):
        result = get_transaction_status("TXN-002", "C001")
        assert result["status"] == "ok"
        assert result["transaction_status"] == "pending"

    def test_failed_transaction(self):
        result = get_transaction_status("TXN-004", "C002")
        assert result["status"] == "ok"
        assert result["transaction_status"] == "failed"

    def test_credit_transaction(self):
        result = get_transaction_status("TXN-003", "C002")
        assert result["status"] == "ok"
        assert result["type"] == "credit"
        assert result["amount"] == pytest.approx(2000.00)

    def test_result_includes_description(self):
        result = get_transaction_status("TXN-001", "C001")
        assert "description" in result
        assert len(result["description"]) > 0

    def test_result_includes_timestamp(self):
        result = get_transaction_status("TXN-001", "C001")
        assert "timestamp" in result
        assert "2025" in result["timestamp"]

    def test_ok_result_has_all_required_keys(self):
        result = get_transaction_status("TXN-001", "C001")
        required = {"status", "reference_id", "transaction_status", "amount",
                    "currency", "type", "timestamp", "description"}
        assert required.issubset(result.keys())


class TestGetTransactionStatusNotFound:
    def test_unknown_reference_id(self):
        result = get_transaction_status("TXN-999", "C001")
        assert result["status"] == "not_found"
        assert "reason" in result

    def test_reason_mentions_reference_id(self):
        result = get_transaction_status("TXN-999", "C001")
        assert "TXN-999" in result["reason"]

    def test_wrong_owner_returns_not_found(self):
        """Security: querying another customer's transaction must return not_found,
        not auth_failed (no info leak about transaction existence)."""
        result = get_transaction_status("TXN-001", "C002")  # TXN-001 belongs to C001
        assert result["status"] == "not_found"

    def test_wrong_owner_does_not_reveal_existence(self):
        """The reason must not mention the real owner."""
        result = get_transaction_status("TXN-001", "C002")
        assert "C001" not in result.get("reason", "")


class TestGetTransactionStatusInvalidArgs:
    def test_empty_reference_id(self):
        result = get_transaction_status("", "C001")
        assert result["status"] == "invalid_args"

    def test_empty_customer_id(self):
        result = get_transaction_status("TXN-001", "")
        assert result["status"] == "invalid_args"

    def test_both_empty(self):
        result = get_transaction_status("", "")
        assert result["status"] == "invalid_args"


# ---------------------------------------------------------------------------
# tool_executor dispatcher
# ---------------------------------------------------------------------------

class TestToolExecutorDispatch:
    def test_dispatches_get_balance(self):
        result = tool_executor("get_balance", {"customer_id": "C001", "auth_token": "token-alice"})
        assert result["status"] == "ok"

    def test_dispatches_get_transaction_status(self):
        result = tool_executor("get_transaction_status", {"reference_id": "TXN-001", "customer_id": "C001"})
        assert result["status"] == "ok"

    def test_unknown_tool_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown tool"):
            tool_executor("nonexistent_tool", {})

    def test_error_message_lists_available_tools(self):
        try:
            tool_executor("bad_tool", {})
        except ValueError as exc:
            assert "get_balance" in str(exc)
            assert "get_transaction_status" in str(exc)

    def test_get_balance_auth_failure_propagates(self):
        result = tool_executor("get_balance", {"customer_id": "C001", "auth_token": "wrong"})
        assert result["status"] == "auth_failed"

    def test_get_transaction_status_not_found_propagates(self):
        result = tool_executor("get_transaction_status", {"reference_id": "TXN-999", "customer_id": "C001"})
        assert result["status"] == "not_found"


# ---------------------------------------------------------------------------
# Integration: orchestrator → tool_executor (no API call)
# ---------------------------------------------------------------------------

class TestOrchestratorToolIntegration:
    """Verify that the orchestrator's _format_tool_result correctly substitutes
    {TOOL_RESULT} in the after_tool message using real tool output."""

    def test_format_tool_result_balance(self):
        from app.orchestration.flash_lite_orchestrator import _format_tool_result

        tool_result = get_balance("C001", auth_token="token-alice")
        template = "Your available balance is {TOOL_RESULT}."
        output = _format_tool_result(template, tool_result)
        assert "{TOOL_RESULT}" not in output
        assert "2543.67" in output or "ok" in output or "USD" in output

    def test_format_tool_result_transaction(self):
        from app.orchestration.flash_lite_orchestrator import _format_tool_result

        tool_result = get_transaction_status("TXN-001", "C001")
        template = "The transaction status is {TOOL_RESULT}."
        output = _format_tool_result(template, tool_result)
        assert "{TOOL_RESULT}" not in output
        # At least some relevant data must appear
        assert len(output) > len("The transaction status is .")
