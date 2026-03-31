"""
live_session_manager.py — Manages a single push-to-talk turn with the Gemini Live API.

Phase 1 design: one WebSocket session is opened per voice turn, then closed.
Phase 2 will upgrade this to a persistent session with reconnection logic (SR-1).

Live API audio contracts:
  Input:  audio/pcm;rate=16000  (int16, mono, little-endian)
  Output: audio/pcm;rate=24000  (int16, mono, little-endian)

This version adds:
  - configurable timeout (default 60s)
  - one retry when the turn returns completely empty
  - preservation of partial audio / transcripts on timeout or exception
"""

import asyncio
import logging
import os

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

LIVE_MODEL = os.environ.get("LIVE_MODEL", "gemini-3.1-flash-live-preview")
VOICE_NAME = os.environ.get("LIVE_VOICE_NAME", "Aoede")
INPUT_MIME = "audio/pcm;rate=16000"

# Raised from 30s because later turns were timing out with zero content.
TURN_TIMEOUT_SECONDS = float(os.environ.get("LIVE_TURN_TIMEOUT_SECONDS", "60"))

# Retry only when a turn returns absolutely nothing:
# no audio, no user transcript, no assistant transcript.
EMPTY_RESPONSE_RETRY_COUNT = int(os.environ.get("LIVE_EMPTY_RESPONSE_RETRY_COUNT", "1"))
RETRY_BACKOFF_SECONDS = float(os.environ.get("LIVE_RETRY_BACKOFF_SECONDS", "0.35"))


def _build_live_config() -> types.LiveConnectConfig:
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE_NAME)
            )
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )


class LiveSessionResult:
    """Container for the result of a single push-to-talk turn."""

    def __init__(
        self,
        audio_bytes: bytes,
        user_transcript: str,
        assistant_transcript: str,
        error: str | None = None,
    ):
        self.audio_bytes = audio_bytes
        self.user_transcript = user_transcript
        self.assistant_transcript = assistant_transcript
        self.error = error

    @property
    def ok(self) -> bool:
        return self.error is None


def _has_meaningful_output(result: LiveSessionResult) -> bool:
    return bool(result.audio_bytes or result.user_transcript or result.assistant_transcript)


async def _send_turn_once(
    client: genai.Client,
    config: types.LiveConnectConfig,
    pcm_bytes: bytes,
    attempt: int,
) -> LiveSessionResult:
    """Send one turn over one Live API session."""
    audio_chunks: list[bytes] = []
    user_transcript = ""
    assistant_transcript = ""

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
                nonlocal user_transcript, assistant_transcript
                msg_count = 0

                async for response in session.receive():
                    msg_count += 1

                    data = getattr(response, "data", None)
                    if data:
                        audio_chunks.append(data)

                    sc = getattr(response, "server_content", None)
                    if sc is None:
                        continue

                    input_tx = getattr(sc, "input_transcription", None)
                    if input_tx is not None:
                        text = getattr(input_tx, "text", None)
                        if text:
                            user_transcript = user_transcript + text if user_transcript else text

                    output_tx = getattr(sc, "output_transcription", None)
                    if output_tx is not None:
                        text = getattr(output_tx, "text", None)
                        if text:
                            assistant_transcript = (
                                assistant_transcript + text if assistant_transcript else text
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
            "Live API timed out waiting for turn_complete after %.1fs; partial_audio=%d bytes, user_tx_len=%d, assistant_tx_len=%d",
            TURN_TIMEOUT_SECONDS,
            len(partial_audio),
            len(user_transcript),
            len(assistant_transcript),
        )
        return LiveSessionResult(
            partial_audio,
            user_transcript,
            assistant_transcript,
            error=f"timeout: no turn_complete within {TURN_TIMEOUT_SECONDS:.0f}s",
        )

    except Exception as exc:
        logger.exception("Live API error during send_turn")
        return LiveSessionResult(
            b"".join(audio_chunks),
            user_transcript,
            assistant_transcript,
            error=str(exc),
        )

    return LiveSessionResult(
        audio_bytes=b"".join(audio_chunks),
        user_transcript=user_transcript,
        assistant_transcript=assistant_transcript,
    )


async def send_turn(pcm_bytes: bytes) -> LiveSessionResult:
    """Open a Live API session, send one complete audio turn, and return the response.

    Retry policy:
      - retry once only when the result is completely empty
      - do not retry if any partial output exists
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
    config = _build_live_config()

    max_attempts = 1 + max(0, EMPTY_RESPONSE_RETRY_COUNT)
    last_result: LiveSessionResult | None = None

    for attempt in range(1, max_attempts + 1):
        result = await _send_turn_once(client, config, pcm_bytes, attempt)
        last_result = result

        # Success with any content: keep it.
        if result.ok and _has_meaningful_output(result):
            return result

        # Error/timeout with partial output: keep it, do not retry and lose it.
        if (not result.ok) and _has_meaningful_output(result):
            return result

        # Completely empty result: retry if we still have attempts left.
        if attempt < max_attempts:
            logger.warning(
                "Live API returned empty response on attempt %d/%d; retrying after %.2fs",
                attempt,
                max_attempts,
                RETRY_BACKOFF_SECONDS,
            )
            await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempt)
            continue

        # Last attempt and still empty.
        if result.error:
            return result

        return LiveSessionResult(
            b"",
            "",
            "",
            error="empty response: no audio or transcript returned from Live API",
        )

    # Defensive fallback
    return last_result or LiveSessionResult(
        b"",
        "",
        "",
        error="unexpected failure: send_turn exited without a result",
    )