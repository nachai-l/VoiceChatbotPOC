"""
live_session_manager.py — Manages a single push-to-talk turn with the Gemini Live API.

Phase 1 design: one WebSocket session per voice turn, then closed.
Phase 2 adds: Flash Live function calling via run_orchestrator.

When orchestrate_fn is provided to send_turn(), Flash Live is given one tool
declaration — run_orchestrator — and the Live session manager handles the
tool_call event by calling orchestrate_fn and sending the response back within
the same open session.  Flash Live then generates the spoken audio response.

Live API audio contracts:
  Input:  audio/pcm;rate=16000  (int16, mono, little-endian)
  Output: audio/pcm;rate=24000  (int16, mono, little-endian)
"""

import asyncio
import logging
import os
from typing import Any, Awaitable, Callable, Optional

from google import genai
from google.genai import types

from app.orchestration.prompts import FLASH_LIVE_SYSTEM_INSTRUCTION

logger = logging.getLogger(__name__)

LIVE_MODEL = os.environ.get("LIVE_MODEL", "gemini-3.1-flash-live-preview")
VOICE_NAME = os.environ.get("LIVE_VOICE_NAME", "Aoede")
INPUT_MIME = "audio/pcm;rate=16000"

TURN_TIMEOUT_SECONDS = float(os.environ.get("LIVE_TURN_TIMEOUT_SECONDS", "60"))
EMPTY_RESPONSE_RETRY_COUNT = int(os.environ.get("LIVE_EMPTY_RESPONSE_RETRY_COUNT", "1"))
RETRY_BACKOFF_SECONDS = float(os.environ.get("LIVE_RETRY_BACKOFF_SECONDS", "0.35"))

# Callable: utterance str -> (response_text str, decision Any)
OrchestrateFn = Callable[[str], Awaitable[tuple[str, Any]]]

# ---------------------------------------------------------------------------
# Live API tool declaration for run_orchestrator  (Phase 2)
# ---------------------------------------------------------------------------

_RUN_ORCHESTRATOR_DECL = types.FunctionDeclaration(
    name="run_orchestrator",
    description=(
        "Call this function when the user asks about account balances, transaction "
        "status, or any operation that requires live account data.  Do NOT call this "
        "for greetings, FAQs, or questions you can answer directly."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "utterance": types.Schema(
                type=types.Type.STRING,
                description="The user's request verbatim or lightly normalised.",
            )
        },
        required=["utterance"],
    ),
)


# ---------------------------------------------------------------------------
# LiveSessionResult
# ---------------------------------------------------------------------------

class LiveSessionResult:
    """Container for the result of a single push-to-talk turn."""

    def __init__(
        self,
        audio_bytes: bytes,
        user_transcript: str,
        assistant_transcript: str,
        error: str | None = None,
        orchestration_decision: Any = None,
    ):
        self.audio_bytes = audio_bytes
        self.user_transcript = user_transcript
        self.assistant_transcript = assistant_transcript
        self.error = error
        # Phase 2: the OrchestrationDecision returned by Flash-Lite (if any).
        self.orchestration_decision = orchestration_decision

    @property
    def ok(self) -> bool:
        return self.error is None


def _has_meaningful_output(result: LiveSessionResult) -> bool:
    return bool(result.audio_bytes or result.user_transcript or result.assistant_transcript)


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def _build_live_config(enable_orchestration: bool = False) -> types.LiveConnectConfig:
    """Build the LiveConnectConfig for a new session.

    Args:
        enable_orchestration: When True, injects the run_orchestrator tool
            declaration and the system instruction into the config.
    """
    kwargs: dict[str, Any] = dict(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE_NAME)
            )
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )

    if enable_orchestration:
        kwargs["tools"] = [types.Tool(function_declarations=[_RUN_ORCHESTRATOR_DECL])]
        kwargs["system_instruction"] = FLASH_LIVE_SYSTEM_INSTRUCTION

    return types.LiveConnectConfig(**kwargs)


# ---------------------------------------------------------------------------
# Core turn logic
# ---------------------------------------------------------------------------

async def _send_turn_once(
    client: genai.Client,
    config: types.LiveConnectConfig,
    pcm_bytes: bytes,
    attempt: int,
    orchestrate_fn: Optional[OrchestrateFn],
) -> LiveSessionResult:
    """Send one turn over one Live API session."""
    audio_chunks: list[bytes] = []
    user_transcript = ""
    assistant_transcript = ""
    last_decision: Any = None

    try:
        async with client.aio.live.connect(model=LIVE_MODEL, config=config) as session:
            logger.info(
                "Live session connected (attempt %d), sending %d bytes of audio",
                attempt,
                len(pcm_bytes),
            )

            await session.send_realtime_input(
                audio=types.Blob(data=pcm_bytes, mime_type=INPUT_MIME)
            )
            await session.send_realtime_input(audio_stream_end=True)
            logger.info("Audio sent + stream_end, waiting for response...")

            async def _collect() -> None:
                nonlocal user_transcript, assistant_transcript, last_decision
                msg_count = 0

                async for response in session.receive():
                    msg_count += 1

                    # ---- audio data ----
                    data = getattr(response, "data", None)
                    if data:
                        audio_chunks.append(data)

                    # ---- Phase 2: function call handling ----
                    tc = getattr(response, "tool_call", None)
                    if tc is not None and orchestrate_fn is not None:
                        for fc in getattr(tc, "function_calls", []):
                            if fc.name == "run_orchestrator":
                                utterance_arg = (fc.args or {}).get("utterance", "")
                                logger.info(
                                    "Flash Live called run_orchestrator: utterance=%r",
                                    utterance_arg[:80],
                                )
                                try:
                                    result_text, last_decision = await orchestrate_fn(utterance_arg)
                                except Exception as exc:
                                    logger.exception("orchestrate_fn raised: %s", exc)
                                    result_text = (
                                        "I'm having trouble processing that right now."
                                    )
                                await session.send_tool_response(
                                    function_responses=[
                                        types.FunctionResponse(
                                            name=fc.name,
                                            id=fc.id,
                                            response={"result": result_text},
                                        )
                                    ]
                                )

                    # ---- transcripts ----
                    sc = getattr(response, "server_content", None)
                    if sc is None:
                        continue

                    input_tx = getattr(sc, "input_transcription", None)
                    if input_tx is not None:
                        text = getattr(input_tx, "text", None)
                        if text:
                            user_transcript = (
                                user_transcript + text if user_transcript else text
                            )

                    output_tx = getattr(sc, "output_transcription", None)
                    if output_tx is not None:
                        text = getattr(output_tx, "text", None)
                        if text:
                            assistant_transcript = (
                                assistant_transcript + text
                                if assistant_transcript
                                else text
                            )

                    if getattr(sc, "turn_complete", False):
                        logger.info(
                            "Turn complete after %d messages, audio=%d bytes",
                            msg_count,
                            sum(len(c) for c in audio_chunks),
                        )
                        break

            await asyncio.wait_for(_collect(), timeout=TURN_TIMEOUT_SECONDS)

    except asyncio.TimeoutError:
        partial_audio = b"".join(audio_chunks)
        logger.error(
            "Live API timed out after %.1fs; partial_audio=%d bytes",
            TURN_TIMEOUT_SECONDS, len(partial_audio),
        )
        return LiveSessionResult(
            partial_audio,
            user_transcript,
            assistant_transcript,
            error=f"timeout: no turn_complete within {TURN_TIMEOUT_SECONDS:.0f}s",
            orchestration_decision=last_decision,
        )

    except Exception as exc:
        logger.exception("Live API error during send_turn")
        return LiveSessionResult(
            b"".join(audio_chunks),
            user_transcript,
            assistant_transcript,
            error=str(exc),
            orchestration_decision=last_decision,
        )

    return LiveSessionResult(
        audio_bytes=b"".join(audio_chunks),
        user_transcript=user_transcript,
        assistant_transcript=assistant_transcript,
        orchestration_decision=last_decision,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def send_turn(
    pcm_bytes: bytes,
    orchestrate_fn: Optional[OrchestrateFn] = None,
) -> LiveSessionResult:
    """Open a Live API session, send one complete audio turn, return the response.

    Args:
        pcm_bytes:      Raw PCM-16 audio bytes (16 kHz, mono, little-endian).
        orchestrate_fn: Optional async callable for Phase 2 orchestration.
                        Signature: (utterance: str) -> (response_text, decision).
                        When provided, the run_orchestrator tool is registered
                        with Flash Live and tool_call events are handled here.

    Retry policy:
        Retry once only when the result is completely empty.
        Do not retry if any partial output (audio or transcript) exists.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return LiveSessionResult(
            b"",
            "",
            "",
            error="GEMINI_API_KEY environment variable is not set.",
        )

    client = genai.Client(api_key=api_key)
    config = _build_live_config(enable_orchestration=orchestrate_fn is not None)

    max_attempts = 1 + max(0, EMPTY_RESPONSE_RETRY_COUNT)
    last_result: LiveSessionResult | None = None

    for attempt in range(1, max_attempts + 1):
        result = await _send_turn_once(client, config, pcm_bytes, attempt, orchestrate_fn)
        last_result = result

        if result.ok and _has_meaningful_output(result):
            return result

        if (not result.ok) and _has_meaningful_output(result):
            return result

        if attempt < max_attempts:
            logger.warning(
                "Live API returned empty response on attempt %d/%d; retrying after %.2fs",
                attempt, max_attempts, RETRY_BACKOFF_SECONDS,
            )
            await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempt)
            continue

        if result.error:
            return result

        return LiveSessionResult(
            b"",
            "",
            "",
            error="empty response: no audio or transcript returned from Live API",
        )

    return last_result or LiveSessionResult(
        b"",
        "",
        "",
        error="unexpected failure: send_turn exited without a result",
    )
