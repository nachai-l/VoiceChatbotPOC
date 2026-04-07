"""
Tests for Phase 2 orchestration layer.

Covers:
  - OrchestrationDecision schema: validation, field validators, cross-field rules
  - SessionState: defaults, reset, field types
  - SummaryStore: update/trim, get_summary, clear, turn_count
  - guardrails.check_decision: all blocking and pass-through cases
  - TurnLog: timing, finish, to_dict
  - flash_lite_orchestrator.orchestrate: all code paths mocked
      - missing API key
      - Flash-Lite call failure
      - parse / validation error
      - clarification path (missing_fields)
      - FAQ / no-tool path
      - tool path with tool_executor=None (Phase 3 pending)
      - tool path with tool_executor provided (success + failure)
      - guardrail block
  - _parse_decision: valid JSON, invalid JSON, schema violation
  - _format_tool_result: placeholder substitution
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.orchestration.schemas import IntentType, OrchestrationDecision
from app.state.session_store import SessionState
from app.state.summary_store import SummaryStore
from app.safety.guardrails import check_decision, ALLOWED_TOOLS, MIN_TOOL_CONFIDENCE
from app.logging.telemetry import TurnLog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decision(**kwargs) -> OrchestrationDecision:
    defaults = dict(
        intent=IntentType.FAQ,
        requires_tool=False,
        confidence=0.9,
        reason="test",
        user_message_after_tool="Here is your answer.",
    )
    defaults.update(kwargs)
    return OrchestrationDecision(**defaults)


def _balance_decision(**kwargs) -> OrchestrationDecision:
    defaults = dict(
        intent=IntentType.BALANCE_INQUIRY,
        requires_tool=True,
        selected_tool="get_balance",
        confidence=0.92,
        missing_fields=[],
        tool_arguments={"customer_id": "C123"},
        user_message_before_tool="Let me check that for you.",
        user_message_after_tool="Your balance is {TOOL_RESULT}.",
        reason="User asked for balance.",
    )
    defaults.update(kwargs)
    return OrchestrationDecision(**defaults)


def _flash_lite_json(**kwargs) -> str:
    data = dict(
        intent="faq",
        requires_tool=False,
        selected_tool=None,
        confidence=0.85,
        missing_fields=[],
        tool_arguments={},
        user_message_before_tool="",
        user_message_after_tool="Our hours are 9 to 5.",
        reason="General FAQ.",
    )
    data.update(kwargs)
    return json.dumps(data)


# ---------------------------------------------------------------------------
# OrchestrationDecision — schema validation
# ---------------------------------------------------------------------------

class TestOrchestrationDecisionSchema:
    def test_valid_faq_decision(self):
        d = _decision()
        assert d.intent == IntentType.FAQ
        assert d.requires_tool is False
        assert d.confidence == 0.9

    def test_valid_tool_decision(self):
        d = _balance_decision()
        assert d.requires_tool is True
        assert d.selected_tool == "get_balance"

    def test_confidence_too_high_raises(self):
        with pytest.raises(Exception):
            _decision(confidence=1.5)

    def test_confidence_negative_raises(self):
        with pytest.raises(Exception):
            _decision(confidence=-0.1)

    def test_confidence_boundary_values(self):
        assert _decision(confidence=0.0).confidence == 0.0
        assert _decision(confidence=1.0).confidence == 1.0

    def test_requires_tool_true_without_selected_tool_raises(self):
        with pytest.raises(Exception):
            OrchestrationDecision(
                intent=IntentType.BALANCE_INQUIRY,
                requires_tool=True,
                selected_tool=None,
                confidence=0.9,
                reason="test",
            )

    def test_requires_tool_false_without_tool_is_valid(self):
        d = _decision(requires_tool=False, selected_tool=None)
        assert d.selected_tool is None

    def test_missing_fields_defaults_to_empty(self):
        d = _decision()
        assert d.missing_fields == []

    def test_tool_arguments_defaults_to_empty(self):
        d = _decision()
        assert d.tool_arguments == {}

    def test_all_intent_types_accepted(self):
        for intent in IntentType:
            if intent in (IntentType.BALANCE_INQUIRY, IntentType.TRANSACTION_STATUS):
                d = _decision(intent=intent, requires_tool=True, selected_tool="get_balance")
            else:
                d = _decision(intent=intent)
            assert d.intent == intent

    def test_intent_from_string(self):
        d = OrchestrationDecision(
            intent="balance_inquiry",
            requires_tool=True,
            selected_tool="get_balance",
            confidence=0.9,
            reason="test",
        )
        assert d.intent == IntentType.BALANCE_INQUIRY

    def test_invalid_intent_raises(self):
        with pytest.raises(Exception):
            _decision(intent="invalid_intent")


# ---------------------------------------------------------------------------
# SessionState
# ---------------------------------------------------------------------------

class TestSessionState:
    def test_default_workflow_step_is_idle(self):
        s = SessionState()
        assert s.workflow_step == "idle"

    def test_default_pending_task_fields_are_empty(self):
        s = SessionState()
        assert s.pending_intent is None
        assert s.unresolved_question is None
        assert s.pending_tool_args == {}
        assert s.verified_values == {}
        assert s.last_tool_result is None
        assert s.customer_id is None

    def test_session_id_is_unique(self):
        a = SessionState()
        b = SessionState()
        assert a.session_id != b.session_id

    def test_reset_clears_all_fields(self):
        s = SessionState()
        s.customer_id = "C123"
        s.workflow_step = "awaiting_clarification"
        s.unresolved_question = "What is your DOB?"
        s.pending_intent = "balance_inquiry"
        s.pending_tool_args = {"customer_id": "C123"}
        s.last_tool_result = {"balance": 500}
        original_id = s.session_id

        s.reset()

        assert s.session_id == original_id  # preserved
        assert s.customer_id is None
        assert s.workflow_step == "idle"
        assert s.unresolved_question is None
        assert s.pending_intent is None
        assert s.pending_tool_args == {}
        assert s.last_tool_result is None
        assert s.verified_values == {}


# ---------------------------------------------------------------------------
# SummaryStore
# ---------------------------------------------------------------------------

class TestSummaryStore:
    def test_empty_summary(self):
        store = SummaryStore()
        assert "No prior conversation" in store.get_summary()

    def test_update_adds_turn(self):
        store = SummaryStore()
        store.update("hello", "hi there")
        summary = store.get_summary()
        assert "hello" in summary
        assert "hi there" in summary

    def test_user_and_assistant_labels_present(self):
        store = SummaryStore()
        store.update("what is my balance", "Let me check.")
        summary = store.get_summary()
        assert "User:" in summary
        assert "Assistant:" in summary

    def test_turn_count(self):
        store = SummaryStore()
        assert store.turn_count() == 0
        store.update("a", "b")
        assert store.turn_count() == 1
        store.update("c", "d")
        assert store.turn_count() == 2

    def test_trims_to_max_turns(self):
        from app.state.summary_store import MAX_TURNS
        store = SummaryStore()
        for i in range(MAX_TURNS + 3):
            store.update(f"user {i}", f"assistant {i}")
        assert store.turn_count() == MAX_TURNS
        # Most recent turns should be present
        summary = store.get_summary()
        assert f"user {MAX_TURNS + 2}" in summary

    def test_oldest_turn_dropped_when_exceeds_max(self):
        from app.state.summary_store import MAX_TURNS
        store = SummaryStore()
        for i in range(MAX_TURNS + 1):
            store.update(f"user {i}", f"assistant {i}")
        summary = store.get_summary()
        # First turn should be gone
        assert "user 0" not in summary

    def test_clear_resets_store(self):
        store = SummaryStore()
        store.update("a", "b")
        store.clear()
        assert store.turn_count() == 0
        assert "No prior conversation" in store.get_summary()

    def test_strips_whitespace(self):
        store = SummaryStore()
        store.update("  hello  ", "  world  ")
        summary = store.get_summary()
        assert "  hello  " not in summary
        assert "hello" in summary


# ---------------------------------------------------------------------------
# guardrails.check_decision
# ---------------------------------------------------------------------------

class TestCheckDecision:
    def test_unsupported_intent_is_blocked(self):
        d = _decision(intent=IntentType.UNSUPPORTED)
        result = check_decision(d)
        assert result is not None
        assert isinstance(result, str)

    def test_faq_passes_through(self):
        d = _decision(intent=IntentType.FAQ)
        assert check_decision(d) is None

    def test_clarify_passes_through(self):
        d = _decision(intent=IntentType.CLARIFY)
        assert check_decision(d) is None

    def test_low_confidence_tool_blocked(self):
        d = _balance_decision(confidence=MIN_TOOL_CONFIDENCE - 0.01)
        result = check_decision(d)
        assert result is not None

    def test_min_confidence_tool_passes(self):
        d = _balance_decision(confidence=MIN_TOOL_CONFIDENCE)
        assert check_decision(d) is None

    def test_high_confidence_tool_passes(self):
        d = _balance_decision(confidence=0.95)
        assert check_decision(d) is None

    def test_unauthorised_tool_blocked(self):
        d = OrchestrationDecision(
            intent=IntentType.BALANCE_INQUIRY,
            requires_tool=True,
            selected_tool="drop_database",
            confidence=0.99,
            reason="test",
        )
        result = check_decision(d)
        assert result is not None

    def test_all_allowed_tools_pass(self):
        for tool in ALLOWED_TOOLS:
            d = OrchestrationDecision(
                intent=IntentType.BALANCE_INQUIRY,
                requires_tool=True,
                selected_tool=tool,
                confidence=0.9,
                reason="test",
            )
            assert check_decision(d) is None, f"Tool {tool!r} should be allowed"


# ---------------------------------------------------------------------------
# TurnLog
# ---------------------------------------------------------------------------

class TestTurnLog:
    def test_total_latency_set_after_finish(self):
        log = TurnLog(session_id="test123")
        log.finish()
        assert log.total_latency_s is not None
        assert log.total_latency_s >= 0.0

    def test_error_set_in_finish(self):
        log = TurnLog(session_id="test123")
        log.finish(error="timeout")
        assert log.error == "timeout"

    def test_to_dict_has_required_fields(self):
        log = TurnLog(session_id="s1", utterance="hello")
        log.intent = "faq"
        log.flash_lite_confidence = 0.88
        log.finish()
        d = log.to_dict()
        assert d["session_id"] == "s1"
        assert d["utterance"] == "hello"
        assert d["intent"] == "faq"
        assert d["confidence"] == pytest.approx(0.88, abs=0.001)
        assert d["total_latency_s"] is not None

    def test_to_dict_no_error_by_default(self):
        log = TurnLog(session_id="s1")
        log.finish()
        assert log.to_dict()["error"] is None


# ---------------------------------------------------------------------------
# _parse_decision (internal helper)
# ---------------------------------------------------------------------------

class TestParseDecision:
    def test_valid_json_returns_decision(self):
        from app.orchestration.flash_lite_orchestrator import _parse_decision
        raw = _flash_lite_json()
        d = _parse_decision(raw)
        assert d.intent == IntentType.FAQ

    def test_invalid_json_raises_value_error(self):
        from app.orchestration.flash_lite_orchestrator import _parse_decision
        with pytest.raises(ValueError, match="non-JSON"):
            _parse_decision("not json at all {{{")

    def test_schema_violation_raises_value_error(self):
        from app.orchestration.flash_lite_orchestrator import _parse_decision
        raw = json.dumps({"intent": "faq", "confidence": 5.0, "requires_tool": False, "reason": "x"})
        with pytest.raises(ValueError, match="schema validation"):
            _parse_decision(raw)

    def test_tool_decision_parsed_correctly(self):
        from app.orchestration.flash_lite_orchestrator import _parse_decision
        raw = _flash_lite_json(
            intent="balance_inquiry",
            requires_tool=True,
            selected_tool="get_balance",
            confidence=0.92,
            tool_arguments={"customer_id": "C123"},
        )
        d = _parse_decision(raw)
        assert d.intent == IntentType.BALANCE_INQUIRY
        assert d.selected_tool == "get_balance"
        assert d.tool_arguments == {"customer_id": "C123"}


# ---------------------------------------------------------------------------
# _format_tool_result
# ---------------------------------------------------------------------------

class TestFormatToolResult:
    def test_placeholder_substituted(self):
        from app.orchestration.flash_lite_orchestrator import _format_tool_result
        result = _format_tool_result("Your balance is {TOOL_RESULT}.", {"balance": 1234.56})
        assert "{TOOL_RESULT}" not in result
        assert "1234.56" in result

    def test_no_placeholder_returns_template_unchanged(self):
        from app.orchestration.flash_lite_orchestrator import _format_tool_result
        template = "Here is your information."
        result = _format_tool_result(template, {"balance": 100})
        assert result == template

    def test_empty_tool_result(self):
        from app.orchestration.flash_lite_orchestrator import _format_tool_result
        result = _format_tool_result("Balance: {TOOL_RESULT}.", {})
        assert "{TOOL_RESULT}" not in result


# ---------------------------------------------------------------------------
# orchestrate() — all code paths
# ---------------------------------------------------------------------------

def _make_mock_client(response_text: str):
    """Return a mock genai.Client whose aio.models.generate_content returns response_text."""
    mock_response = MagicMock()
    mock_response.text = response_text
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    return mock_client


class TestOrchestrate:
    @pytest.mark.anyio
    async def test_missing_api_key_returns_error_decision(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        session = SessionState()
        summary = SummaryStore()

        from app.orchestration.flash_lite_orchestrator import orchestrate
        text, decision = await orchestrate("hello", session, summary)

        assert decision.intent == IntentType.UNSUPPORTED
        assert decision.confidence == 0.0
        assert isinstance(text, str)

    @pytest.mark.anyio
    async def test_flash_lite_call_failure_returns_error(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        session = SessionState()
        summary = SummaryStore()

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            side_effect=RuntimeError("network error")
        )

        from app.orchestration.flash_lite_orchestrator import orchestrate
        with patch("app.orchestration.flash_lite_orchestrator.genai.Client", return_value=mock_client):
            text, decision = await orchestrate("hello", session, summary)

        assert decision.intent == IntentType.UNSUPPORTED
        assert "error" in decision.reason.lower() or "Flash-Lite" in decision.reason

    @pytest.mark.anyio
    async def test_parse_error_returns_fallback(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        session = SessionState()
        summary = SummaryStore()

        mock_client = _make_mock_client("this is not json at all")

        from app.orchestration.flash_lite_orchestrator import orchestrate
        with patch("app.orchestration.flash_lite_orchestrator.genai.Client", return_value=mock_client):
            text, decision = await orchestrate("what", session, summary)

        assert decision.intent == IntentType.UNSUPPORTED
        assert isinstance(text, str)

    @pytest.mark.anyio
    async def test_faq_intent_returns_direct_response(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        session = SessionState()
        summary = SummaryStore()

        mock_client = _make_mock_client(_flash_lite_json(
            intent="faq",
            requires_tool=False,
            user_message_after_tool="Our hours are 9 to 5.",
        ))

        from app.orchestration.flash_lite_orchestrator import orchestrate
        with patch("app.orchestration.flash_lite_orchestrator.genai.Client", return_value=mock_client):
            text, decision = await orchestrate("what are your hours", session, summary)

        assert decision.intent == IntentType.FAQ
        assert text == "Our hours are 9 to 5."
        assert session.workflow_step == "idle"

    @pytest.mark.anyio
    async def test_clarification_path_sets_session_state(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        session = SessionState()
        summary = SummaryStore()

        mock_client = _make_mock_client(_flash_lite_json(
            intent="balance_inquiry",
            requires_tool=True,
            selected_tool="get_balance",
            confidence=0.88,
            missing_fields=["customer_id"],
            tool_arguments={},
            user_message_before_tool="Could you give me your customer ID?",
        ))

        from app.orchestration.flash_lite_orchestrator import orchestrate
        with patch("app.orchestration.flash_lite_orchestrator.genai.Client", return_value=mock_client):
            text, decision = await orchestrate("what is my balance", session, summary)

        assert decision.missing_fields == ["customer_id"]
        assert text == "Could you give me your customer ID?"
        assert session.workflow_step == "awaiting_clarification"
        assert session.pending_intent == "balance_inquiry"
        assert session.unresolved_question == "Could you give me your customer ID?"

    @pytest.mark.anyio
    async def test_clarification_merges_partial_args(self, monkeypatch):
        """Partial args from previous turn should be preserved."""
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        session = SessionState()
        session.pending_tool_args = {"customer_id": "C123"}
        summary = SummaryStore()

        mock_client = _make_mock_client(_flash_lite_json(
            intent="balance_inquiry",
            requires_tool=True,
            selected_tool="get_balance",
            confidence=0.88,
            missing_fields=["dob"],
            tool_arguments={"customer_id": "C123"},
            user_message_before_tool="What is your date of birth?",
        ))

        from app.orchestration.flash_lite_orchestrator import orchestrate
        with patch("app.orchestration.flash_lite_orchestrator.genai.Client", return_value=mock_client):
            text, decision = await orchestrate("C123", session, summary)

        assert "customer_id" in session.pending_tool_args

    @pytest.mark.anyio
    async def test_tool_path_no_executor_returns_placeholder(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        session = SessionState()
        summary = SummaryStore()

        mock_client = _make_mock_client(_flash_lite_json(
            intent="balance_inquiry",
            requires_tool=True,
            selected_tool="get_balance",
            confidence=0.9,
            tool_arguments={"customer_id": "C123"},
            user_message_before_tool="Let me check that.",
            user_message_after_tool="Your balance is {TOOL_RESULT}.",
        ))

        from app.orchestration.flash_lite_orchestrator import orchestrate
        with patch("app.orchestration.flash_lite_orchestrator.genai.Client", return_value=mock_client):
            text, decision = await orchestrate("what is my balance", session, summary, tool_executor=None)

        assert decision.selected_tool == "get_balance"
        assert isinstance(text, str)
        # Placeholder response since no executor
        assert "Phase 3" in text or "Let me check" in text
        assert session.workflow_step == "idle"

    @pytest.mark.anyio
    async def test_tool_path_with_executor_success(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        session = SessionState()
        summary = SummaryStore()

        mock_client = _make_mock_client(_flash_lite_json(
            intent="balance_inquiry",
            requires_tool=True,
            selected_tool="get_balance",
            confidence=0.95,
            tool_arguments={"customer_id": "C123"},
            user_message_before_tool="Let me check.",
            user_message_after_tool="Your balance is {TOOL_RESULT}.",
        ))

        def mock_executor(tool_name: str, args: dict) -> dict:
            assert tool_name == "get_balance"
            return {"current_balance": "$1,234.56", "available_balance": "$1,200.00"}

        from app.orchestration.flash_lite_orchestrator import orchestrate
        with patch("app.orchestration.flash_lite_orchestrator.genai.Client", return_value=mock_client):
            text, decision = await orchestrate("balance", session, summary, tool_executor=mock_executor)

        assert "1,234.56" in text or "1,200.00" in text
        assert session.workflow_step == "idle"
        assert session.last_tool_result is not None

    @pytest.mark.anyio
    async def test_tool_path_with_executor_failure(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        session = SessionState()
        summary = SummaryStore()

        mock_client = _make_mock_client(_flash_lite_json(
            intent="balance_inquiry",
            requires_tool=True,
            selected_tool="get_balance",
            confidence=0.9,
            tool_arguments={"customer_id": "C123"},
        ))

        def failing_executor(tool_name: str, args: dict) -> dict:
            raise ConnectionError("backend unavailable")

        from app.orchestration.flash_lite_orchestrator import orchestrate
        with patch("app.orchestration.flash_lite_orchestrator.genai.Client", return_value=mock_client):
            text, decision = await orchestrate("balance", session, summary, tool_executor=failing_executor)

        assert "sorry" in text.lower() or "try again" in text.lower()
        assert session.last_tool_result == {"error": "backend unavailable"}
        assert session.workflow_step == "idle"

    @pytest.mark.anyio
    async def test_guardrail_block_unsupported(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        session = SessionState()
        summary = SummaryStore()

        mock_client = _make_mock_client(_flash_lite_json(intent="unsupported"))

        from app.orchestration.flash_lite_orchestrator import orchestrate
        with patch("app.orchestration.flash_lite_orchestrator.genai.Client", return_value=mock_client):
            text, decision = await orchestrate("wire me $10000", session, summary)

        # Guardrail fires and returns safe fallback
        assert isinstance(text, str)
        assert decision.intent == IntentType.UNSUPPORTED

    @pytest.mark.anyio
    async def test_summary_store_not_modified_by_orchestrate(self, monkeypatch):
        """orchestrate() does not update summary_store — the UI layer does that."""
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        session = SessionState()
        summary = SummaryStore()

        mock_client = _make_mock_client(_flash_lite_json())

        from app.orchestration.flash_lite_orchestrator import orchestrate
        with patch("app.orchestration.flash_lite_orchestrator.genai.Client", return_value=mock_client):
            await orchestrate("hello", session, summary)

        # Summary store is updated by the UI layer, not by orchestrate
        assert summary.turn_count() == 0


# ---------------------------------------------------------------------------
# live_session_manager — Phase 2: orchestration_decision field
# ---------------------------------------------------------------------------

class TestLiveSessionResultOrchestrationDecision:
    def test_default_orchestration_decision_is_none(self):
        from app.live.live_session_manager import LiveSessionResult
        r = LiveSessionResult(b"", "", "")
        assert r.orchestration_decision is None

    def test_orchestration_decision_can_be_set(self):
        from app.live.live_session_manager import LiveSessionResult
        d = _balance_decision()
        r = LiveSessionResult(b"", "", "", orchestration_decision=d)
        assert r.orchestration_decision is d


# ---------------------------------------------------------------------------
# live_session_manager — _build_live_config Phase 2
# ---------------------------------------------------------------------------

class TestBuildLiveConfigPhase2:
    def test_orchestration_disabled_has_no_tools(self):
        from app.live.live_session_manager import _build_live_config
        config = _build_live_config(enable_orchestration=False)
        assert not getattr(config, "tools", None)

    def test_orchestration_enabled_has_run_orchestrator_tool(self):
        from app.live.live_session_manager import _build_live_config
        config = _build_live_config(enable_orchestration=True)
        tools = getattr(config, "tools", None)
        assert tools, "tools must be set when enable_orchestration=True"
        names = [
            fd.name
            for t in tools
            for fd in getattr(t, "function_declarations", [])
        ]
        assert "run_orchestrator" in names

    def test_orchestration_enabled_has_system_instruction(self):
        from app.live.live_session_manager import _build_live_config
        config = _build_live_config(enable_orchestration=True)
        assert getattr(config, "system_instruction", None) is not None

    def test_orchestration_disabled_has_no_system_instruction(self):
        from app.live.live_session_manager import _build_live_config
        config = _build_live_config(enable_orchestration=False)
        assert not getattr(config, "system_instruction", None)
