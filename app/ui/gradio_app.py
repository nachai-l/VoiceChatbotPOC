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
import base64
import io
import logging
import time
import wave

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
MAX_PLAYBACK_COOLDOWN_SECONDS = 5.0
# Maximum number of trace entries retained in the sliding-window history.
TRACE_LOG_MAX = 20
# If the API task has not completed within this many seconds, auto-cancel it
# and return the system to listening mode so the user can try again.
TASK_TIMEOUT_SECONDS = 30.0


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


def _build_audio_html(audio_bytes: bytes) -> object:
    """Encode PCM16 bytes as a WAV file and return a self-contained HTML audio element.

    Returns gr.update() (no-op) when audio_bytes is empty so the player is
    untouched between responses.  Otherwise returns an HTML <audio autoplay>
    element with an inline base64 data URI.

    Using gr.HTML instead of gr.Audio for playback decouples audio output from
    Gradio's component update machinery.  When gr.Audio is updated with new audio
    from the poll handler the browser resets its AudioContext, which triggers
    echo-cancellation logic that pauses the microphone MediaStream — killing the
    streaming input after the first turn.  A gr.HTML update is a plain DOM text
    swap with no AudioContext involvement.
    """
    if not audio_bytes:
        return gr.update()

    # Apply playback gain (match _build_audio_output scaling)
    _, audio_np = pcm16_to_numpy(audio_bytes, sample_rate=DEFAULT_OUTPUT_SAMPLE_RATE)
    audio_np = np.asarray(audio_np, dtype=np.float32)
    audio_np = np.clip(audio_np * PLAYBACK_GAIN, -1.0, 1.0)
    # Convert back to PCM16 for WAV container
    pcm16 = (audio_np * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)       # 16-bit
        wf.setframerate(DEFAULT_OUTPUT_SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())

    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    logger.info("_build_audio_html: %d samples, gain=%.1f", len(audio_np), PLAYBACK_GAIN)
    # Each response gets a unique element id so replacing it triggers autoplay.
    uid = int(time.monotonic() * 1000) % 1_000_000
    return (
        f'<audio id="r{uid}" autoplay style="width:100%">'
        f'<source src="data:audio/wav;base64,{b64}" type="audio/wav">'
        f"</audio>"
    )


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
            _append_trace(vad_state, {"src": "poll"}),
            transcript_handler,
            vad_state,
            session_state,
            summary_store,
        )

    # Capture the task reference BEFORE clearing pending_task.
    # We need it for idempotent cooldown calculation later.
    _task_ref = vad_state.pending_task

    # Mark as consumed on the Task object BEFORE reading result, so that any
    # stale vad_state copy still pointing to this task will skip it.
    _task_ref._voice_consumed = True
    # Record completion time ONCE — makes ignore_until idempotent on re-entry.
    if not hasattr(_task_ref, "_completed_at"):
        _task_ref._completed_at = time.monotonic()

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
            _append_trace(vad_state, {"src": "poll", "error": str(exc)}),
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

    audio_output = _build_audio_html(result.audio_bytes)

    # Duration-based cooldown — IDEMPOTENT via task._completed_at.
    #
    # Using time.monotonic() directly here caused the stuck-status bug:
    # if _consume_finished_task ran twice (due to any re-entry), each call
    # pushed ignore_until further into the future, creating an infinite cooldown.
    #
    # Fix: record the completion timestamp ONCE on the task object the first
    # time this function runs.  Subsequent calls read the same timestamp, so
    # ignore_until is always set to the *same* absolute value regardless of
    # how many times the function is invoked.  After MAX_PLAYBACK_COOLDOWN_SECONDS
    # the value is in the past and the status correctly transitions to "Listening...".
    if result.audio_bytes:
        audio_duration_s = len(result.audio_bytes) / (DEFAULT_OUTPUT_SAMPLE_RATE * 2)
        cooldown = min(audio_duration_s + PLAYBACK_COOLDOWN_SECONDS, MAX_PLAYBACK_COOLDOWN_SECONDS)
        # IDEMPOTENT: use _task_ref._completed_at (set once above) so that
        # ignore_until is always the same absolute value even if this code
        # runs multiple times for the same task.
        vad_state.ignore_until = _task_ref._completed_at + cooldown
        logger.info(
            "_consume: completed_at=%.3f ignore_until=%.3f cooldown=%.2fs audio=%.2fs now=%.3f",
            _task_ref._completed_at, vad_state.ignore_until, cooldown, audio_duration_s,
            time.monotonic(),
        )

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
        _append_trace(vad_state, {"src": "poll", **debug}),
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

    # Heartbeat: stamp last_stream_at on every call so poll can detect a dead stream.
    # This mutation is on the shared vad_state object and persists even when we
    # return gr.update() for the vad_state gr.State output.
    vad_state.last_stream_at = time.monotonic()

    # Stale-copy guard: if this vad_state copy still holds a task that was
    # already consumed by the poll handler, clear the stale reference here so
    # stream_audio_chunk doesn't write it back to Gradio State.
    if vad_state.pending_task is not None and _is_task_consumed(vad_state.pending_task):
        vad_state.pending_task = None

    if vad_state.pending_task is not None:
        # CRITICAL: return actual values for non-State UI outputs (chatbot,
        # status, debug_panel) but gr.update() (no-op) for all gr.State outputs.
        #
        # Race condition: stream_audio_chunk fires every 0.25 s and, if it
        # returns the actual vad_state here, it can OVERWRITE the State that
        # poll_pending_result just wrote (pending_task=None, ignore_until=now+5).
        # That reset causes _consume_finished_task to re-run on every tick,
        # keeping ignore_until perpetually in the future → stuck "Playing response...".
        #
        # gr.update() tells Gradio "keep the stored value unchanged", so the
        # poll handler's write always survives regardless of call ordering.
        #
        # IMPORTANT: chatbot must NOT be gr.update() here.  When gr.update() is
        # returned for a non-State component like gr.Chatbot, Gradio's
        # postprocess_data creates a new component instance with render=False,
        # which terminates the mic streaming connection → system stops after 1 input.
        yield (
            history,                                                                  # chatbot — actual value
            _status("Thinking..."),
            _append_trace(vad_state, {"src": "stream", "waiting": True}),            # debug_panel — trace log
            gr.update(),           # transcript_state ← poll owns this
            gr.update(),           # vad_state  ← poll owns this while task runs
            gr.update(),           # session_state
            gr.update(),           # summary_state
        )
        return

    now = time.monotonic()
    if now < vad_state.ignore_until:
        yield (
            history,
            _status("Playing response..."),
            _append_trace(vad_state, {"src": "stream"}),
            transcript_handler,
            vad_state,
            session_state,
            summary_store,
        )
        return

    if chunk is None:
        yield history, _status("Listening..."), _append_trace(vad_state, {"src": "stream", "chunk_none": True}), transcript_handler, vad_state, session_state, summary_store
        return

    sample_rate, data = chunk
    if data is None or len(data) == 0:
        yield history, _status("Listening..."), _append_trace(vad_state, {"src": "stream", "chunk_empty": True}), transcript_handler, vad_state, session_state, summary_store
        return

    rms = float(compute_rms(data))
    vad_state, should_send = process_chunk(vad_state, sample_rate, data)

    if not should_send:
        status = _status("Speaking detected..." if vad_state.is_speaking else "Listening...")
        diag = {"src": "stream", "rms": round(rms, 5)}
        yield history, status, _append_trace(vad_state, diag), transcript_handler, vad_state, session_state, summary_store
        return

    audio_array = get_buffer_array(vad_state)
    pcm_bytes = numpy_to_pcm16(sample_rate, audio_array)
    vad_state = reset_vad(vad_state)

    # Build orchestration closure bound to current session/summary stores
    orchestrate_fn = _make_orchestrate_fn(session_state, summary_store)
    vad_state.pending_task = asyncio.create_task(
        send_turn(pcm_bytes, orchestrate_fn=orchestrate_fn)
    )
    vad_state.pending_task._started_at = time.monotonic()
    logger.info("VAD triggered — launched API task (%d PCM bytes)", len(pcm_bytes))

    yield (
        history,
        _status("Thinking..."),
        _append_trace(vad_state, {"src": "stream", "should_send": True, "pcm_bytes": len(pcm_bytes)}),
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

    now = time.monotonic()
    # Detect a dead mic stream: last_stream_at hasn't advanced in >2s even
    # though we're past cooldown and not thinking.  Show a warning so the user
    # knows to click the microphone button to restart.
    _stream_dead = (
        vad_state.last_stream_at > 0
        and (now - vad_state.last_stream_at) > 2.0
    )

    if vad_state.pending_task is None:
        if now < vad_state.ignore_until:
            return (
                NO_AUDIO,
                transcript_handler.get_history(),
                _status("Playing response..."),
                _append_trace(vad_state, {"src": "poll"}),
                transcript_handler,
                vad_state,
                session_state,
                summary_store,
            )
        # Cooldown has expired — always return "Listening..." so the status
        # never gets stuck.  (The is_speaking check was removed: if ambient
        # noise set is_speaking=True during playback the poller would return
        # gr.update() indefinitely, freezing the UI on "Playing response...".)
        if _stream_dead:
            listening_status = _status("Mic stream stopped — click 🎤 to restart", error=True)
        else:
            listening_status = _status("Listening...")
        return (
            NO_AUDIO,
            transcript_handler.get_history(),
            listening_status,
            _append_trace(vad_state, {"src": "poll"}),
            transcript_handler,
            vad_state,
            session_state,
            summary_store,
        )

    if not vad_state.pending_task.done():
        task_age = now - getattr(vad_state.pending_task, "_started_at", now)
        if task_age > TASK_TIMEOUT_SECONDS:
            # API task has been running too long — cancel it and reset so the
            # user can try again without refreshing the page.
            logger.warning("API task timed out after %.1fs — cancelling", task_age)
            vad_state.pending_task.cancel()
            vad_state.pending_task = None
            return (
                NO_AUDIO,
                transcript_handler.get_history(),
                _status("Request timed out — please try again", error=True),
                _append_trace(vad_state, {"src": "poll", "timeout": True, "task_age_s": round(task_age, 1)}),
                transcript_handler,
                vad_state,
                session_state,
                summary_store,
            )
        return (
            NO_AUDIO,
            transcript_handler.get_history(),
            _status("Thinking..."),
            _append_trace(vad_state, {"src": "poll", "waiting": True}),
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
    vad_state.trace_log = []  # clear trace history on full session reset
    session_state.reset()
    summary_store.clear()
    return (
        [],
        "",   # clear audio_player HTML
        _status("Session cleared — ready"),
        [],
        transcript_handler,
        vad_state,
        session_state,
        summary_store,
    )


def _status(message: str, error: bool = False) -> str:
    icon = "🔴" if error else "🟢"
    return f"**Status:** {icon} {message}"


def _live_trace(vad_state: VADState, extra: dict | None = None) -> dict:
    """Return a diagnostic dict shown in the live-trace panel on every tick.

    Rendered in the UI as a JSON block so the user can see exactly what the
    server state is — useful for debugging stuck-status issues.
    """
    now = time.monotonic()
    remaining = round(vad_state.ignore_until - now, 2) if vad_state.ignore_until > now else 0.0
    task = vad_state.pending_task

    if task is None:
        task_status = "none"
    elif getattr(task, "_voice_consumed", False):
        task_status = "consumed"
    elif task.done():
        task_status = "done_pending_consume"
    else:
        task_status = "running"

    if task is None and vad_state.ignore_until > now:
        mode = "playing"
    elif task is not None:
        mode = "thinking"
    elif vad_state.is_speaking:
        mode = "speaking"
    else:
        mode = "listening"

    stream_age = round(now - vad_state.last_stream_at, 2) if vad_state.last_stream_at > 0 else None
    task_age = round(now - task._started_at, 1) if task is not None and hasattr(task, "_started_at") else None
    trace = {
        "mode": mode,
        "cooldown_remaining_s": remaining,
        "ignore_until_offset_s": round(vad_state.ignore_until - now, 3),
        "task_status": task_status,
        "task_age_s": task_age,   # seconds since API task was launched; None = no task
        "is_speaking": vad_state.is_speaking,
        "speech_chunks": vad_state.speech_chunks,
        "silence_chunks": vad_state.silence_chunks,
        "now_monotonic": round(now % 1000, 3),  # last 3 digits for readability
        "stream_age_s": stream_age,  # seconds since last stream chunk; None = never received
    }
    if extra:
        trace.update(extra)
    return trace


def _append_trace(vad_state: VADState, extra: dict | None = None) -> list:
    """Append the current trace snapshot to vad_state.trace_log and return the log.

    The log is a sliding window capped at TRACE_LOG_MAX entries.  Returning the
    list directly as the debug_panel value gives the user a scrollable history
    of the last N ticks instead of only the most-recent snapshot.
    """
    entry = _live_trace(vad_state, extra)
    vad_state.trace_log.append(entry)
    if len(vad_state.trace_log) > TRACE_LOG_MAX:
        del vad_state.trace_log[:-TRACE_LOG_MAX]
    return vad_state.trace_log


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

                # Use gr.HTML for audio playback instead of gr.Audio.
                # gr.Audio(autoplay=True) triggers a browser AudioContext reset
                # when updated with new audio, which stops the mic MediaStream
                # and kills the streaming input after every turn.
                # gr.HTML is a plain DOM swap with no AudioContext involvement.
                audio_player = gr.HTML(value="", label="Assistant response")

            with gr.Column(scale=1):
                status_md = gr.Markdown(_status("Listening..."))
                gr.Markdown(
                    "**Live Trace** — sliding window of last 20 ticks. "
                    "`mode`: listening / speaking / thinking / playing. "
                    "`cooldown_remaining_s`: seconds until mic re-opens. "
                    "`task_status`: none / running / done_pending_consume / consumed."
                )
                debug_panel = gr.JSON(label="Trace (newest last)", value=[])
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
            outputs=[audio_player, chatbot, status_md, debug_panel, transcript_state, vad_state, session_state, summary_state],
        )

        clear_btn.click(
            fn=handle_clear,
            inputs=[transcript_state, vad_state, session_state, summary_state],
            outputs=[chatbot, audio_player, status_md, debug_panel, transcript_state, vad_state, session_state, summary_state],
        )

    return demo
