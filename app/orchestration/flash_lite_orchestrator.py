"""
flash_lite_orchestrator.py — Flash-Lite structured orchestration.

This is the decision/routing layer between Flash Live (voice conversation) and
the tool execution layer (Phase 3).  For each user utterance it:

  1. Builds a structured prompt with session context and conversation history.
  2. Calls Flash-Lite with response_mime_type="application/json".
  3. Validates the JSON response against OrchestrationDecision.
  4. Applies safety guardrails.
  5. If missing_fields: returns a clarification question (clarification loop).
  6. If requires_tool: calls tool_executor if available, else returns placeholder.
  7. Returns (response_text, decision) — response_text is what Flash Live speaks.

Architecture note (requirements §5.2):
  Flash Live calls run_orchestrator() as a single backend function.
  This module IS that backend function's implementation.
"""

import json
import logging
import os
import time
from typing import Any, Awaitable, Callable, Optional

from google import genai
from google.genai import types
from pydantic import ValidationError

from app.logging.telemetry import TurnLog
from app.orchestration.prompts import (
    FLASH_LITE_SYSTEM_PROMPT,
    FLASH_LITE_USER_PROMPT_TEMPLATE,
)
from app.orchestration.schemas import IntentType, OrchestrationDecision
from app.safety.guardrails import check_decision
from app.state.session_store import SessionState
from app.state.summary_store import SummaryStore

logger = logging.getLogger(__name__)

FLASH_LITE_MODEL = os.environ.get("FLASH_LITE_MODEL", "gemini-3.1-flash-lite-preview")

# Callable: (tool_name: str, args: dict) -> result_dict
# None means Phase 3 tools are not yet wired (placeholder response is returned).
ToolExecutor = Callable[[str, dict[str, Any]], dict[str, Any]]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_user_prompt(
    utterance: str,
    session_state: SessionState,
    summary: str,
) -> str:
    return FLASH_LITE_USER_PROMPT_TEMPLATE.format(
        utterance=utterance,
        summary=summary,
        workflow_step=session_state.workflow_step,
        pending_intent=session_state.pending_intent or "none",
        unresolved_question=session_state.unresolved_question or "none",
        known_values=json.dumps(session_state.verified_values) if session_state.verified_values else "{}",
    )


def _parse_decision(raw_json: str) -> OrchestrationDecision:
    """Parse and validate the Flash-Lite JSON output.

    Raises:
        ValueError: if raw_json is not valid JSON or fails schema validation.
    """
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Flash-Lite returned non-JSON output: {exc}\nRaw (first 300 chars): {raw_json[:300]}"
        ) from exc

    try:
        return OrchestrationDecision(**data)
    except (ValidationError, TypeError) as exc:
        raise ValueError(f"Flash-Lite response failed schema validation: {exc}") from exc


def _format_tool_result(template: str, tool_result: dict[str, Any]) -> str:
    """Substitute {TOOL_RESULT} in the after_tool message with a summary."""
    summary_parts = [
        f"{k}: {v}"
        for k, v in tool_result.items()
        if k not in ("status",) or v not in ("ok", "success")
    ]
    result_text = ", ".join(summary_parts) if summary_parts else str(tool_result)
    return template.replace("{TOOL_RESULT}", result_text)


def _error_decision(reason: str, message: str) -> OrchestrationDecision:
    return OrchestrationDecision(
        intent=IntentType.UNSUPPORTED,
        requires_tool=False,
        confidence=0.0,
        reason=reason,
        user_message_after_tool=message,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def orchestrate(
    utterance: str,
    session_state: SessionState,
    summary_store: SummaryStore,
    tool_executor: Optional[ToolExecutor] = None,
) -> tuple[str, OrchestrationDecision]:
    """Run Flash-Lite orchestration for one user utterance.

    Args:
        utterance:      User's spoken text (transcribed by Flash Live).
        session_state:  Mutable session state — modified in-place as the
                        conversation progresses (workflow_step, pending_intent,
                        unresolved_question, last_tool_result, etc.).
        summary_store:  Rolling conversation summary injected into the prompt.
        tool_executor:  Optional (tool_name, args) → result_dict callable.
                        If None, a placeholder response is returned for any
                        tool-required decision (Phase 3 pending).

    Returns:
        (response_text, decision)
        response_text — what Flash Live should speak to the user.
        decision      — the full OrchestrationDecision for logging/debug.
    """
    turn_log = TurnLog(session_id=session_state.session_id, utterance=utterance)

    # ------------------------------------------------------------------
    # Guard: GEMINI_API_KEY required
    # ------------------------------------------------------------------
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        decision = _error_decision(
            "GEMINI_API_KEY not set",
            "I'm sorry, I can't process requests right now due to a configuration issue.",
        )
        turn_log.finish(error="GEMINI_API_KEY not set")
        return decision.user_message_after_tool, decision

    # ------------------------------------------------------------------
    # Call Flash-Lite
    # ------------------------------------------------------------------
    summary = summary_store.get_summary()
    user_prompt = _build_user_prompt(utterance, session_state, summary)

    logger.info(
        "Flash-Lite call: model=%s utterance=%r step=%s",
        FLASH_LITE_MODEL, utterance[:80], session_state.workflow_step,
    )

    client = genai.Client(api_key=api_key)
    try:
        response = await client.aio.models.generate_content(
            model=FLASH_LITE_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=FLASH_LITE_SYSTEM_PROMPT,
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        raw_text = response.text
    except Exception as exc:
        logger.exception("Flash-Lite call failed")
        decision = _error_decision(
            f"Flash-Lite error: {exc}",
            "I'm having trouble processing your request. Please try again in a moment.",
        )
        turn_log.finish(error=str(exc))
        return decision.user_message_after_tool, decision

    logger.debug("Flash-Lite raw response: %s", raw_text[:500])

    # ------------------------------------------------------------------
    # Parse + validate response
    # ------------------------------------------------------------------
    try:
        decision = _parse_decision(raw_text)
    except ValueError as exc:
        logger.error("Parse error: %s", exc)
        decision = _error_decision(
            f"parse error: {exc}",
            "I didn't quite catch that. Could you rephrase your request?",
        )
        turn_log.finish(error=str(exc))
        return decision.user_message_after_tool, decision

    turn_log.intent = decision.intent.value
    turn_log.flash_lite_confidence = decision.confidence
    turn_log.selected_tool = decision.selected_tool

    logger.info(
        "Decision: intent=%s tool=%s confidence=%.2f missing=%s",
        decision.intent, decision.selected_tool, decision.confidence, decision.missing_fields,
    )

    # ------------------------------------------------------------------
    # Safety guardrails
    # ------------------------------------------------------------------
    blocked_message = check_decision(decision)
    if blocked_message:
        logger.warning("Decision blocked by guardrails: %s", blocked_message)
        turn_log.finish(error="blocked_by_guardrails")
        return blocked_message, decision

    # ------------------------------------------------------------------
    # Clarification path: missing required arguments  (FR-9)
    # ------------------------------------------------------------------
    if decision.missing_fields:
        session_state.workflow_step = "awaiting_clarification"
        session_state.pending_intent = decision.intent.value
        # Merge any tool_arguments collected so far with what we already know
        session_state.pending_tool_args = {
            **session_state.pending_tool_args,
            **decision.tool_arguments,
        }
        question = (
            decision.user_message_before_tool
            or f"Could you please tell me your {decision.missing_fields[0]}?"
        )
        session_state.unresolved_question = question
        turn_log.finish()
        return question, decision

    # ------------------------------------------------------------------
    # No tool needed: conversational / FAQ response
    # ------------------------------------------------------------------
    if not decision.requires_tool:
        session_state.workflow_step = "idle"
        session_state.unresolved_question = None
        turn_log.finish()
        return decision.user_message_after_tool, decision

    # ------------------------------------------------------------------
    # Tool execution path
    # ------------------------------------------------------------------
    session_state.workflow_step = "executing_tool"
    preamble = decision.user_message_before_tool or "Let me check that for you."
    turn_log.selected_tool = decision.selected_tool

    if tool_executor is None:
        # Phase 3 not yet wired — return placeholder
        logger.warning(
            "tool_executor is None; %s not executed (Phase 3 pending)",
            decision.selected_tool,
        )
        spoken = (
            f"{preamble} "
            "Tool integration is still being set up — please check back soon."
        )
        session_state.last_tool_result = {"status": "pending", "note": "Phase 3 pending"}
        session_state.workflow_step = "idle"
        turn_log.finish()
        return spoken, decision

    tool_start = time.monotonic()
    try:
        tool_result = tool_executor(decision.selected_tool, decision.tool_arguments)
        turn_log.tool_latency_s = round(time.monotonic() - tool_start, 3)
        spoken = _format_tool_result(decision.user_message_after_tool, tool_result)
        logger.info("Tool %s succeeded in %.3fs", decision.selected_tool, turn_log.tool_latency_s)
    except Exception as exc:
        turn_log.tool_latency_s = round(time.monotonic() - tool_start, 3)
        logger.exception("Tool %s raised: %s", decision.selected_tool, exc)
        tool_result = {"error": str(exc)}
        spoken = (
            "I'm sorry, I wasn't able to retrieve that information. "
            "Please try again or contact support."
        )
        turn_log.finish(error=str(exc))
        session_state.last_tool_result = tool_result
        session_state.workflow_step = "idle"
        return spoken, decision

    session_state.last_tool_result = tool_result
    session_state.workflow_step = "idle"
    session_state.unresolved_question = None
    session_state.pending_intent = None
    session_state.pending_tool_args = {}
    turn_log.finish()
    return spoken, decision
