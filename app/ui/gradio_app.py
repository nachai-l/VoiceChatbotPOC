"""
gradio_app.py — Gradio Blocks UI for the Voice Agent POC.

Phase 1 interaction model: always-on microphone with energy-based VAD.
  1. Microphone streams audio continuously in 0.25 s chunks.
  2. VAD detects speech start and end.
  3. When silence follows speech, the buffered utterance is sent to the Live API.
  4. Response audio plays back and transcripts update automatically.

Fixes in this version:
  - Use a timer poller so finished Live API tasks are surfaced even when mic callbacks stop.
  - Convert assistant PCM output into a temporary WAV file path for Gradio playback.
  - Add a short playback cooldown to reduce accidental re-triggering while audio starts.
  - Keep the existing VAD streaming UX.
"""

import asyncio
import logging
import time

import numpy as np
import gradio as gr

from app.live.audio_codec import numpy_to_pcm16, pcm16_to_numpy
from app.live.live_session_manager import send_turn
from app.live.transcript_handler import TranscriptHandler
from app.live.vad import VADState, process_chunk, get_buffer_array, reset_vad, compute_rms

logger = logging.getLogger(__name__)

PLAYBACK_GAIN = 2.5
DEFAULT_OUTPUT_SAMPLE_RATE = 24000
PLAYBACK_COOLDOWN_SECONDS = 0.75


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _new_transcript_handler() -> TranscriptHandler:
    return TranscriptHandler()


def _new_vad_state() -> VADState:
    state = VADState()
    setattr(state, "ignore_until", 0.0)
    return state


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _build_audio_output(audio_bytes: bytes):
    """Convert raw PCM16 bytes into a Gradio-compatible (sample_rate, numpy) tuple."""
    if not audio_bytes:
        return gr.update()

    sample_rate, audio_np = pcm16_to_numpy(audio_bytes, sample_rate=DEFAULT_OUTPUT_SAMPLE_RATE)
    audio_np = np.asarray(audio_np, dtype=np.float32)

    # Boost playback gain and clamp.
    audio_np = np.clip(audio_np * PLAYBACK_GAIN, -1.0, 1.0)

    logger.info("_build_audio_output: %d samples, sr=%d, peak=%.4f", len(audio_np), sample_rate, float(np.max(np.abs(audio_np))) if audio_np.size else 0.0)
    return (int(sample_rate), audio_np)


# ---------------------------------------------------------------------------
# Shared result handling
# ---------------------------------------------------------------------------

def _consume_finished_task(
    transcript_handler: TranscriptHandler,
    vad_state: VADState,
):
    """Collect a finished background task and build UI outputs."""
    NO_AUDIO = gr.update()

    if vad_state.pending_task is None:
        return NO_AUDIO, transcript_handler.get_history(), _status("Listening..."), {}, transcript_handler, vad_state

    try:
        result = vad_state.pending_task.result()
    except Exception as exc:  # pragma: no cover - defensive only
        logger.exception("Pending API task raised unexpectedly")
        vad_state.pending_task = None
        transcript_handler.add_assistant(f"[Error] {exc}")
        return (
            NO_AUDIO,
            transcript_handler.get_history(),
            _status("Error", error=True),
            {"error": str(exc)},
            transcript_handler,
            vad_state,
        )

    vad_state.pending_task = None
    logger.info("API result received: ok=%s, audio=%d bytes", result.ok, len(result.audio_bytes))

    if result.user_transcript:
        transcript_handler.add_user(result.user_transcript)

    assistant_text = result.assistant_transcript
    if result.error:
        if assistant_text:
            assistant_text = f"{assistant_text}\n\n[Warning] {result.error}"
        else:
            assistant_text = f"[Error] {result.error}"

    if assistant_text:
        transcript_handler.add_assistant(assistant_text)

    audio_output = _build_audio_output(result.audio_bytes)

    # Short cooldown so the always-on mic does not instantly retrigger
    # while response playback begins or while residual noise is present.
    if result.audio_bytes:
        setattr(vad_state, "ignore_until", time.monotonic() + PLAYBACK_COOLDOWN_SECONDS)

    debug = {
        "turn": "direct_conversation",
        "user_transcript": result.user_transcript,
        "assistant_transcript": result.assistant_transcript,
        "audio_bytes": len(result.audio_bytes),
        "ok": result.ok,
        "error": result.error,
    }

    if result.ok and result.audio_bytes:
        status = _status("Playing response...")
    elif result.ok:
        status = _status("Listening...")
    else:
        status = _status("Response received with warning", error=True)

    return audio_output, transcript_handler.get_history(), status, debug, transcript_handler, vad_state


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

async def stream_audio_chunk(
    chunk: tuple[int, np.ndarray] | None,
    transcript_handler: TranscriptHandler,
    vad_state: VADState,
):
    """Process one streamed audio chunk through VAD.

    IMPORTANT: This handler does NOT output to audio_output.
    Audio playback is owned exclusively by poll_pending_result to avoid
    rapid re-renders that kill the audio player before it can play.
    """
    history = transcript_handler.get_history()

    # Do not process mic input while a request is in flight.
    if vad_state.pending_task is not None:
        yield history, _status("Thinking..."), {"waiting": True}, transcript_handler, vad_state
        return

    # Short cooldown right after assistant audio is returned.
    ignore_until = float(getattr(vad_state, "ignore_until", 0.0) or 0.0)
    now = time.monotonic()
    if now < ignore_until:
        yield (
            history,
            _status("Playing response..."),
            {"cooldown_s": round(ignore_until - now, 2)},
            transcript_handler,
            vad_state,
        )
        return

    if chunk is None:
        yield history, _status("Listening..."), {}, transcript_handler, vad_state
        return

    sample_rate, data = chunk
    if data is None or len(data) == 0:
        yield history, _status("Listening..."), {}, transcript_handler, vad_state
        return

    rms = float(compute_rms(data))
    vad_state, should_send = process_chunk(vad_state, sample_rate, data)

    if not should_send:
        status = _status("Speaking detected..." if vad_state.is_speaking else "Listening...")
        diag = {
            "rms": round(rms, 5),
            "speech": vad_state.speech_chunks,
            "silence": vad_state.silence_chunks,
        }
        yield history, status, diag, transcript_handler, vad_state
        return

    audio_array = get_buffer_array(vad_state)
    pcm_bytes = numpy_to_pcm16(sample_rate, audio_array)
    vad_state = reset_vad(vad_state)
    setattr(vad_state, "ignore_until", 0.0)
    vad_state.pending_task = asyncio.create_task(send_turn(pcm_bytes))
    logger.info("VAD triggered — launched API task (%d PCM bytes)", len(pcm_bytes))

    yield history, _status("Thinking..."), {"should_send": True, "pcm_bytes": len(pcm_bytes)}, transcript_handler, vad_state


async def poll_pending_result(
    transcript_handler: TranscriptHandler,
    vad_state: VADState,
):
    """Timer-driven poller for completed Live API tasks."""
    NO_AUDIO = gr.update()

    if vad_state.pending_task is None:
        ignore_until = float(getattr(vad_state, "ignore_until", 0.0) or 0.0)
        if time.monotonic() < ignore_until:
            return (
                NO_AUDIO,
                transcript_handler.get_history(),
                _status("Playing response..."),
                {"cooldown_s": round(ignore_until - time.monotonic(), 2)},
                transcript_handler,
                vad_state,
            )

        return NO_AUDIO, transcript_handler.get_history(), gr.update(), gr.update(), transcript_handler, vad_state

    if not vad_state.pending_task.done():
        return NO_AUDIO, transcript_handler.get_history(), _status("Thinking..."), {"waiting": True}, transcript_handler, vad_state

    return _consume_finished_task(transcript_handler, vad_state)


async def handle_voice_turn(
    audio_input: tuple[int, np.ndarray] | None,
    history: list[dict],
    transcript_handler: TranscriptHandler,
) -> tuple:
    """Legacy push-to-talk handler — kept for test compatibility."""
    if audio_input is None:
        yield None, history, _status("Ready"), {}, transcript_handler
        return

    sample_rate, data = audio_input
    if data is None or len(data) == 0:
        yield None, history, _status("No audio captured — please try again"), {}, transcript_handler
        return

    yield None, history, _status("Thinking..."), {}, transcript_handler

    pcm_bytes = numpy_to_pcm16(sample_rate, data)
    result = await send_turn(pcm_bytes)

    if result.user_transcript:
        transcript_handler.add_user(result.user_transcript)

    assistant_text = result.assistant_transcript
    if result.error:
        if assistant_text:
            assistant_text = f"{assistant_text}\n\n[Warning] {result.error}"
        else:
            assistant_text = f"[Error] {result.error}"

    if assistant_text:
        transcript_handler.add_assistant(assistant_text)

    audio_output = _build_audio_output(result.audio_bytes)

    debug = {
        "turn": "direct_conversation",
        "user_transcript": result.user_transcript,
        "assistant_transcript": result.assistant_transcript,
        "audio_bytes": len(result.audio_bytes),
        "error": result.error,
        "note": "Orchestration not active in Phase 1",
    }

    if result.ok and result.audio_bytes:
        status = _status("Playing response...")
    elif result.ok:
        status = _status("Ready")
    else:
        status = _status("Response received with warning", error=True)

    yield audio_output, transcript_handler.get_history(), status, debug, transcript_handler


def handle_clear(transcript_handler: TranscriptHandler, vad_state: VADState) -> tuple:
    """Reset the session."""
    transcript_handler.clear()
    vad_state = reset_vad(vad_state)
    setattr(vad_state, "ignore_until", 0.0)
    vad_state.pending_task = None
    return [], None, _status("Session cleared — ready"), {}, transcript_handler, vad_state


def _status(message: str, error: bool = False) -> str:
    icon = "🔴" if error else "🟢"
    return f"**Status:** {icon} {message}"


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def create_app() -> gr.Blocks:
    with gr.Blocks(title="Voice Agent POC") as demo:
        gr.Markdown("# Voice Agent POC\n*Gemini Live + Flash-Lite Orchestration — Phase 1*")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=420,
                    buttons=["copy"],
                )

                audio_input = gr.Audio(
                    sources=["microphone"],
                    streaming=True,
                    type="numpy",
                    label="Microphone — auto-detects speech",
                    recording=True,
                )

                audio_output = gr.Audio(
                    label="Assistant response",
                    autoplay=True,
                    type="numpy",
                )

            with gr.Column(scale=1):
                status_md = gr.Markdown(_status("Listening..."))
                debug_panel = gr.JSON(label="Trace", value={})
                clear_btn = gr.Button("Clear session", variant="secondary")

        transcript_state = gr.State(_new_transcript_handler())
        vad_state = gr.State(_new_vad_state())

        poller = gr.Timer(value=0.25, active=True)

        audio_input.stream(
            fn=stream_audio_chunk,
            inputs=[audio_input, transcript_state, vad_state],
            outputs=[chatbot, status_md, debug_panel, transcript_state, vad_state],
            stream_every=0.25,
            time_limit=None,
        )

        poller.tick(
            fn=poll_pending_result,
            inputs=[transcript_state, vad_state],
            outputs=[audio_output, chatbot, status_md, debug_panel, transcript_state, vad_state],
        )

        clear_btn.click(
            fn=handle_clear,
            inputs=[transcript_state, vad_state],
            outputs=[chatbot, audio_output, status_md, debug_panel, transcript_state, vad_state],
        )

    return demo