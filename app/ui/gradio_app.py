"""
gradio_app.py — Gradio Blocks UI for the Voice Agent POC.

Phase 1 interaction model: always-on microphone with energy-based VAD.
Phase 2 additions:
  - Flash-Lite orchestration wired into every voice turn via run_orchestrator.
  - Session state (SessionState) and rolling summary (SummaryStore) persisted
    across turns as Gradio State.
  - Debug panel shows full orchestration trace: intent, confidence, tool, args.
  - Summary store updated after each completed turn.
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
from app.orchestration.flash_lite_orchestrator import orchestrate
from app.state.session_store import SessionState
from app.state.summary_store import SummaryStore
from app.tools.tool_executor import tool_executor

logger = logging.getLogger(__name__)

PLAYBACK_GAIN = 2.5
DEFAULT_OUTPUT_SAMPLE_RATE = 24000
PLAYBACK_COOLDOWN_SECONDS = 0.75
# Cap total cooldown so status never appears permanently stuck even for very
# long responses (audio_bytes > ~264 kB / ~5.5 s of audio at 24 kHz PCM16).
MAX_PLAYBACK_COOLDOWN_SECONDS = 6.0


# ---------------------------------------------------------------------------
# State factories
# ---------------------------------------------------------------------------

def _new_transcript_handler() -> TranscriptHandler:
    return TranscriptHandler()


def _new_vad_state() -> VADState:
    return VADState()


def _new_session_state() -> SessionState:
    return SessionState()


def _new_summary_store() -> SummaryStore:
    return SummaryStore()


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _build_audio_output(audio_bytes: bytes):
    """Convert raw PCM16 bytes into a Gradio-compatible (sample_rate, numpy) tuple."""
    if not audio_bytes:
        return gr.update()

    sample_rate, audio_np = pcm16_to_numpy(audio_bytes, sample_rate=DEFAULT_OUTPUT_SAMPLE_RATE)
    audio_np = np.asarray(audio_np, dtype=np.float32)
    audio_np = np.clip(audio_np * PLAYBACK_GAIN, -1.0, 1.0)

    logger.info(
        "_build_audio_output: %d samples, sr=%d, peak=%.4f",
        len(audio_np), sample_rate,
        float(np.max(np.abs(audio_np))) if audio_np.size else 0.0,
    )
    return (int(sample_rate), audio_np)


# ---------------------------------------------------------------------------
# Orchestration closure builder
# ---------------------------------------------------------------------------

def _make_orchestrate_fn(session_state: SessionState, summary_store: SummaryStore):
    """Return an async callable suitable for passing to send_turn.

    The closure captures the mutable session_state and summary_store by reference.
    Mutations made inside orchestrate() are immediately visible to the caller.
    """
    async def _orchestrate_fn(utterance: str):
        return await orchestrate(utterance, session_state, summary_store, tool_executor=tool_executor)
    return _orchestrate_fn


# ---------------------------------------------------------------------------
# Shared result handling
# ---------------------------------------------------------------------------

def _is_task_consumed(task) -> bool:
    """Return True if this task has already been consumed by _consume_finished_task.

    Because Gradio may deepcopy VADState between handler invocations, two
    concurrent handlers can hold references to the same asyncio.Task object
    inside *different* copies of vad_state.  The poll handler clears
    vad_state.pending_task on its copy, but the stream handler's copy still
    holds the old reference and may write it back to Gradio State, making the
    poll handler re-consume the finished task on every tick.

    Marking the flag directly on the shared Task object (not on vad_state)
    makes the consumed state visible to all copies simultaneously.
    """
    return getattr(task, "_voice_consumed", False)


def _consume_finished_task(
    transcript_handler: TranscriptHandler,
    vad_state: VADState,
    session_state: SessionState,
    summary_store: SummaryStore,
):
    """Collect a finished background task and build UI outputs."""
    NO_AUDIO = gr.update()

    # Stale-copy guard: task object already processed by a previous handler run.
    if vad_state.pending_task is not None and _is_task_consumed(vad_state.pending_task):
        vad_state.pending_task = None

    if vad_state.pending_task is None:
        return (
            NO_AUDIO,
            transcript_handler.get_history(),
            _status("Listening..."),
            {},
            transcript_handler,
            vad_state,
            session_state,
            summary_store,
        )

    # Mark as consumed on the Task object BEFORE reading result, so that any
    # stale vad_state copy still pointing to this task will skip it.
    vad_state.pending_task._voice_consumed = True

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
            session_state,
            summary_store,
        )

    vad_state.pending_task = None
    logger.info(
        "API result: ok=%s audio=%d bytes orchestrated=%s",
        result.ok, len(result.audio_bytes), result.orchestration_decision is not None,
    )

    if result.user_transcript:
        transcript_handler.add_user(result.user_transcript)

    assistant_text = result.assistant_transcript
    if result.error:
        assistant_text = (
            f"{assistant_text}\n\n[Warning] {result.error}" if assistant_text
            else f"[Error] {result.error}"
        )

    if assistant_text:
        transcript_handler.add_assistant(assistant_text)

    # Update rolling summary after each completed turn
    if result.user_transcript or assistant_text:
        summary_store.update(
            result.user_transcript or "",
            result.assistant_transcript or "",
        )

    audio_output = _build_audio_output(result.audio_bytes)

    # Duration-based cooldown: covers actual playback + trailing buffer.
    # Capped at MAX_PLAYBACK_COOLDOWN_SECONDS so the status never appears
    # permanently stuck for very long (verbose Phase-2) responses.
    if result.audio_bytes:
        audio_duration_s = len(result.audio_bytes) / (DEFAULT_OUTPUT_SAMPLE_RATE * 2)
        cooldown = min(audio_duration_s + PLAYBACK_COOLDOWN_SECONDS, MAX_PLAYBACK_COOLDOWN_SECONDS)
        vad_state.ignore_until = time.monotonic() + cooldown

    # Build debug trace for the panel
    decision = result.orchestration_decision
    debug: dict = {
        "user_transcript": result.user_transcript,
        "assistant_transcript": result.assistant_transcript,
        "audio_bytes": len(result.audio_bytes),
        "ok": result.ok,
        "error": result.error,
    }
    if decision is not None:
        debug.update({
            "intent": decision.intent.value,
            "requires_tool": decision.requires_tool,
            "selected_tool": decision.selected_tool,
            "confidence": round(decision.confidence, 3),
            "missing_fields": decision.missing_fields,
            "tool_arguments": decision.tool_arguments,
            "workflow_step": session_state.workflow_step,
        })

    if result.ok and result.audio_bytes:
        status = _status("Playing response...")
    elif result.ok:
        status = _status("Listening...")
    else:
        status = _status("Response received with warning", error=True)

    return (
        audio_output,
        transcript_handler.get_history(),
        status,
        debug,
        transcript_handler,
        vad_state,
        session_state,
        summary_store,
    )


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

async def stream_audio_chunk(
    chunk: tuple[int, np.ndarray] | None,
    transcript_handler: TranscriptHandler,
    vad_state: VADState,
    session_state: SessionState,
    summary_store: SummaryStore,
):
    """Process one streamed audio chunk through VAD.

    Does NOT output to audio_output — audio playback is owned exclusively
    by poll_pending_result to prevent rapid re-renders that kill the player.
    """
    history = transcript_handler.get_history()

    # Stale-copy guard: if this vad_state copy still holds a task that was
    # already consumed by the poll handler, clear the stale reference here so
    # stream_audio_chunk doesn't write it back to Gradio State.
    if vad_state.pending_task is not None and _is_task_consumed(vad_state.pending_task):
        vad_state.pending_task = None

    if vad_state.pending_task is not None:
        yield history, _status("Thinking..."), {"waiting": True}, transcript_handler, vad_state, session_state, summary_store
        return

    now = time.monotonic()
    if now < vad_state.ignore_until:
        yield (
            history,
            _status("Playing response..."),
            {"cooldown_s": round(vad_state.ignore_until - now, 2)},
            transcript_handler,
            vad_state,
            session_state,
            summary_store,
        )
        return

    if chunk is None:
        yield history, _status("Listening..."), {}, transcript_handler, vad_state, session_state, summary_store
        return

    sample_rate, data = chunk
    if data is None or len(data) == 0:
        yield history, _status("Listening..."), {}, transcript_handler, vad_state, session_state, summary_store
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
        yield history, status, diag, transcript_handler, vad_state, session_state, summary_store
        return

    audio_array = get_buffer_array(vad_state)
    pcm_bytes = numpy_to_pcm16(sample_rate, audio_array)
    vad_state = reset_vad(vad_state)

    # Build orchestration closure bound to current session/summary stores
    orchestrate_fn = _make_orchestrate_fn(session_state, summary_store)
    vad_state.pending_task = asyncio.create_task(
        send_turn(pcm_bytes, orchestrate_fn=orchestrate_fn)
    )
    logger.info("VAD triggered — launched API task (%d PCM bytes)", len(pcm_bytes))

    yield (
        history,
        _status("Thinking..."),
        {"should_send": True, "pcm_bytes": len(pcm_bytes)},
        transcript_handler,
        vad_state,
        session_state,
        summary_store,
    )


async def poll_pending_result(
    transcript_handler: TranscriptHandler,
    vad_state: VADState,
    session_state: SessionState,
    summary_store: SummaryStore,
):
    """Timer-driven poller for completed Live API tasks."""
    NO_AUDIO = gr.update()

    # Stale-copy guard: task already consumed, clear the reference.
    if vad_state.pending_task is not None and _is_task_consumed(vad_state.pending_task):
        vad_state.pending_task = None

    if vad_state.pending_task is None:
        if time.monotonic() < vad_state.ignore_until:
            return (
                NO_AUDIO,
                transcript_handler.get_history(),
                _status("Playing response..."),
                {"cooldown_s": round(vad_state.ignore_until - time.monotonic(), 2)},
                transcript_handler,
                vad_state,
                session_state,
                summary_store,
            )
        # Cooldown has expired — always return "Listening..." so the status
        # never gets stuck.  (The is_speaking check was removed: if ambient
        # noise set is_speaking=True during playback the poller would return
        # gr.update() indefinitely, freezing the UI on "Playing response...".)
        return (
            NO_AUDIO,
            transcript_handler.get_history(),
            _status("Listening..."),
            {},
            transcript_handler,
            vad_state,
            session_state,
            summary_store,
        )

    if not vad_state.pending_task.done():
        return (
            NO_AUDIO,
            transcript_handler.get_history(),
            _status("Thinking..."),
            {"waiting": True},
            transcript_handler,
            vad_state,
            session_state,
            summary_store,
        )

    return _consume_finished_task(transcript_handler, vad_state, session_state, summary_store)


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
        assistant_text = (
            f"{assistant_text}\n\n[Warning] {result.error}" if assistant_text
            else f"[Error] {result.error}"
        )

    if assistant_text:
        transcript_handler.add_assistant(assistant_text)

    audio_output = _build_audio_output(result.audio_bytes)

    debug = {
        "turn": "direct_conversation",
        "user_transcript": result.user_transcript,
        "assistant_transcript": result.assistant_transcript,
        "audio_bytes": len(result.audio_bytes),
        "error": result.error,
        "note": "Orchestration not active in push-to-talk mode",
    }

    if result.ok and result.audio_bytes:
        status = _status("Playing response...")
    elif result.ok:
        status = _status("Ready")
    else:
        status = _status("Response received with warning", error=True)

    yield audio_output, transcript_handler.get_history(), status, debug, transcript_handler


def handle_clear(
    transcript_handler: TranscriptHandler,
    vad_state: VADState,
    session_state: SessionState,
    summary_store: SummaryStore,
) -> tuple:
    """Reset the session."""
    transcript_handler.clear()
    vad_state = reset_vad(vad_state)
    session_state.reset()
    summary_store.clear()
    return (
        [],
        None,
        _status("Session cleared — ready"),
        {},
        transcript_handler,
        vad_state,
        session_state,
        summary_store,
    )


def _status(message: str, error: bool = False) -> str:
    icon = "🔴" if error else "🟢"
    return f"**Status:** {icon} {message}"


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def create_app() -> gr.Blocks:
    with gr.Blocks(title="Voice Agent POC") as demo:
        gr.Markdown("# Voice Agent POC\n*Gemini Live + Flash-Lite Orchestration — Phase 2*")

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
        session_state = gr.State(_new_session_state())
        summary_state = gr.State(_new_summary_store())

        poller = gr.Timer(value=0.25, active=True)

        audio_input.stream(
            fn=stream_audio_chunk,
            inputs=[audio_input, transcript_state, vad_state, session_state, summary_state],
            outputs=[chatbot, status_md, debug_panel, transcript_state, vad_state, session_state, summary_state],
            stream_every=0.25,
            time_limit=None,
        )

        poller.tick(
            fn=poll_pending_result,
            inputs=[transcript_state, vad_state, session_state, summary_state],
            outputs=[audio_output, chatbot, status_md, debug_panel, transcript_state, vad_state, session_state, summary_state],
        )

        clear_btn.click(
            fn=handle_clear,
            inputs=[transcript_state, vad_state, session_state, summary_state],
            outputs=[chatbot, audio_output, status_md, debug_panel, transcript_state, vad_state, session_state, summary_state],
        )

    return demo
