"""
gradio_app.py — Gradio Blocks UI for the Voice Agent POC.

Phase 1 interaction model: always-on microphone with energy-based VAD.
  1. Microphone streams audio continuously in 0.25 s chunks.
  2. VAD detects speech start and end.
  3. When silence follows speech, the buffered utterance is sent to the Live API.
  4. Response audio plays back and transcripts update automatically.
"""

import numpy as np
import gradio as gr

from app.live.audio_codec import numpy_to_pcm16, pcm16_to_numpy
from app.live.live_session_manager import send_turn
from app.live.transcript_handler import TranscriptHandler
from app.live.vad import VADState, process_chunk, get_buffer_array, reset_vad


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _new_transcript_handler() -> TranscriptHandler:
    return TranscriptHandler()


def _new_vad_state() -> VADState:
    return VADState()


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

async def stream_audio_chunk(
    chunk: tuple[int, np.ndarray] | None,
    history: list[dict],
    transcript_handler: TranscriptHandler,
    vad_state: VADState,
):
    """Process one streamed audio chunk through VAD.

    Yields one result tuple per state update so the UI stays responsive.
    Only calls the Live API when VAD detects end of utterance.

    Yields:
        (audio_output, history, status_md, debug_json, transcript_handler, vad_state)
    """
    if chunk is None:
        yield None, history, _status("Listening..."), {}, transcript_handler, vad_state
        return

    sample_rate, data = chunk
    if data is None or len(data) == 0:
        yield None, history, _status("Listening..."), {}, transcript_handler, vad_state
        return

    vad_state, should_send = process_chunk(vad_state, sample_rate, data)

    if not should_send:
        status = _status("Speaking detected..." if vad_state.is_speaking else "Listening...")
        yield None, history, status, {}, transcript_handler, vad_state
        return

    # --- End of utterance detected — send to Live API ---
    yield None, history, _status("Thinking..."), {}, transcript_handler, vad_state

    audio_array = get_buffer_array(vad_state)
    vad_state = reset_vad(vad_state)

    pcm_bytes = numpy_to_pcm16(sample_rate, audio_array)
    result = await send_turn(pcm_bytes)

    if not result.ok:
        transcript_handler.add_user(result.user_transcript)
        transcript_handler.add_assistant(f"[Error] {result.error}")
        debug = {"error": result.error}
        yield None, transcript_handler.get_history(), _status("Error", error=True), debug, transcript_handler, vad_state
        return

    transcript_handler.add_user(result.user_transcript)
    transcript_handler.add_assistant(result.assistant_transcript)

    audio_output = None
    if result.audio_bytes:
        audio_output = pcm16_to_numpy(result.audio_bytes, sample_rate=24000)

    debug = {
        "turn": "direct_conversation",
        "user_transcript": result.user_transcript,
        "assistant_transcript": result.assistant_transcript,
        "audio_bytes": len(result.audio_bytes),
        "note": "Orchestration not active in Phase 1",
    }

    yield audio_output, transcript_handler.get_history(), _status("Listening..."), debug, transcript_handler, vad_state


async def handle_voice_turn(
    audio_input: tuple[int, np.ndarray] | None,
    history: list[dict],
    transcript_handler: TranscriptHandler,
) -> tuple:
    """Legacy push-to-talk handler — kept for test compatibility.

    Not wired in the Gradio UI (stream_audio_chunk is used instead).
    """
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

    if not result.ok:
        transcript_handler.add_user(result.user_transcript)
        transcript_handler.add_assistant(f"[Error] {result.error}")
        yield None, transcript_handler.get_history(), _status("Error", error=True), {"error": result.error}, transcript_handler
        return

    transcript_handler.add_user(result.user_transcript)
    transcript_handler.add_assistant(result.assistant_transcript)

    audio_output = None
    if result.audio_bytes:
        audio_output = pcm16_to_numpy(result.audio_bytes, sample_rate=24000)

    debug = {
        "turn": "direct_conversation",
        "user_transcript": result.user_transcript,
        "assistant_transcript": result.assistant_transcript,
        "audio_bytes": len(result.audio_bytes),
        "note": "Orchestration not active in Phase 1",
    }

    yield audio_output, transcript_handler.get_history(), _status("Ready"), debug, transcript_handler


def handle_clear(transcript_handler: TranscriptHandler) -> tuple:
    """Reset the session."""
    transcript_handler.clear()
    return [], _status("Session cleared — ready"), {}, transcript_handler


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
            # Left column: conversation
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

            # Right column: status + debug
            with gr.Column(scale=1):
                status_md = gr.Markdown(_status("Listening..."))
                debug_panel = gr.JSON(label="Trace", value={})
                clear_btn = gr.Button("Clear session", variant="secondary")

        # Per-session state
        transcript_state = gr.State(_new_transcript_handler)
        vad_state = gr.State(_new_vad_state)

        # Wire streaming VAD handler
        audio_input.stream(
            fn=stream_audio_chunk,
            inputs=[audio_input, chatbot, transcript_state, vad_state],
            outputs=[audio_output, chatbot, status_md, debug_panel, transcript_state, vad_state],
            stream_every=0.25,
            time_limit=None,
        )

        clear_btn.click(
            fn=handle_clear,
            inputs=[transcript_state],
            outputs=[chatbot, status_md, debug_panel, transcript_state],
        )

    return demo
