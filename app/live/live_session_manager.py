"""
live_session_manager.py — Manages a single push-to-talk turn with the Gemini Live API.

Phase 1 design: one WebSocket session is opened per voice turn, then closed.
Phase 2 will upgrade this to a persistent session with reconnection logic (SR-1).

Live API audio contracts:
  Input:  audio/pcm;rate=16000  (int16, mono, little-endian)
  Output: audio/pcm;rate=24000  (int16, mono, little-endian)

NOTE: google-genai SDK API surface — verify method names against installed SDK version
      if a method raises AttributeError. The SDK was at v1.x when this was written.
"""

import os
import logging

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

LIVE_MODEL = "gemini-3.1-flash-live-preview"
INPUT_MIME = "audio/pcm;rate=16000"


def _build_live_config() -> types.LiveConnectConfig:
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
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


async def send_turn(pcm_bytes: bytes) -> LiveSessionResult:
    """Open a Live API session, send one complete audio turn, and return the response.

    Args:
        pcm_bytes: PCM audio bytes at 16 kHz int16 (from audio_codec.numpy_to_pcm16).

    Returns:
        LiveSessionResult with audio response, transcripts, and optional error.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return LiveSessionResult(b"", "", "", error="GEMINI_API_KEY environment variable is not set.")

    client = genai.Client(api_key=api_key)
    config = _build_live_config()

    audio_chunks: list[bytes] = []
    user_transcript = ""
    assistant_transcript = ""

    try:
        async with client.aio.live.connect(model=LIVE_MODEL, config=config) as session:
            # Send the complete audio buffer as a single user turn.
            # send_client_content is used for complete, turn-based input.
            await session.send_client_content(
                turns=types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            inline_data=types.Blob(data=pcm_bytes, mime_type=INPUT_MIME)
                        )
                    ],
                ),
                turn_complete=True,
            )

            # Collect model response until turn_complete signal.
            async for response in session.receive():
                # Audio data shortcut (bytes)
                if response.data:
                    audio_chunks.append(response.data)

                # Transcripts and turn_complete are in server_content
                sc = getattr(response, "server_content", None)
                if sc is None:
                    continue

                if sc.input_transcription:
                    user_transcript = sc.input_transcription.text or user_transcript

                if sc.output_transcription:
                    assistant_transcript = sc.output_transcription.text or assistant_transcript

                if sc.turn_complete:
                    break

    except Exception as exc:
        logger.exception("Live API error during send_turn")
        return LiveSessionResult(b"", user_transcript, assistant_transcript, error=str(exc))

    return LiveSessionResult(
        audio_bytes=b"".join(audio_chunks),
        user_transcript=user_transcript,
        assistant_transcript=assistant_transcript,
    )
