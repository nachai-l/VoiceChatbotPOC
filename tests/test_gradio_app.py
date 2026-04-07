"""
Tests for gradio_app.py event handlers.

Covers:
  - handle_voice_turn: async generator contract + early exits (kept for regression)
  - stream_audio_chunk: VAD-driven streaming handler
  - poll_pending_result: timer-driven completion path
  - _build_audio_output: numpy tuple creation and gain
  - _consume_finished_task: background task result handling
  - handle_clear: session reset
  - create_app: Gradio 6.x API compatibility
"""

import asyncio
import time

import numpy as np
import pytest

from app.live.transcript_handler import TranscriptHandler
from app.live.vad import VADState, SILENCE_CHUNKS_TO_TRIGGER, MIN_SPEECH_CHUNKS
from app.state.session_store import SessionState
from app.state.summary_store import SummaryStore
from app.ui.gradio_app import (
    handle_voice_turn,
    handle_clear,
    stream_audio_chunk,
    poll_pending_result,
    _build_audio_output,
    _consume_finished_task,
    _new_vad_state,
    PLAYBACK_COOLDOWN_SECONDS,
    MAX_PLAYBACK_COOLDOWN_SECONDS,
)

SR = 16000
CHUNK = SR // 4  # 0.25 s


def _speech_chunk(amplitude: float = 0.5) -> tuple[int, np.ndarray]:
    t = np.linspace(0, 0.25, CHUNK, dtype=np.float32)
    return SR, np.sin(2 * np.pi * 440 * t) * amplitude


def _silence_chunk() -> tuple[int, np.ndarray]:
    return SR, np.zeros(CHUNK, dtype=np.float32)


async def _collect(async_gen):
    results = []
    async for item in async_gen:
        results.append(item)
    return results


@pytest.fixture
def handler():
    return TranscriptHandler()


@pytest.fixture
def vad():
    return VADState()


@pytest.fixture
def session():
    return SessionState()


@pytest.fixture
def summary():
    return SummaryStore()


# ---------------------------------------------------------------------------
# handle_voice_turn — async generator regression (push-to-talk legacy)
# ---------------------------------------------------------------------------

class TestHandleVoiceTurnIsAsyncGenerator:
    """Regression: handle_voice_turn must be an async generator (no return-with-value)."""

    @pytest.mark.anyio
    async def test_is_async_generator_not_coroutine(self, handler):
        import inspect
        audio = (SR, np.array([], dtype=np.float32))
        result = handle_voice_turn(audio, [], handler)
        assert inspect.isasyncgen(result)

    @pytest.mark.anyio
    async def test_none_input_path_yields_not_returns(self, handler):
        results = await _collect(handle_voice_turn(None, [], handler))
        assert len(results) >= 1

    @pytest.mark.anyio
    async def test_empty_array_path_yields_not_returns(self, handler):
        audio = (SR, np.array([], dtype=np.float32))
        results = await _collect(handle_voice_turn(audio, [], handler))
        assert len(results) >= 1


class TestHandleVoiceTurnEarlyExits:
    @pytest.mark.anyio
    async def test_none_audio_returns_ready_status(self, handler):
        results = await _collect(handle_voice_turn(None, [], handler))
        _, _, status, _, _ = results[0]
        assert "Ready" in status

    @pytest.mark.anyio
    async def test_none_audio_preserves_history(self, handler):
        history = [{"role": "user", "content": "hello"}]
        results = await _collect(handle_voice_turn(None, history, handler))
        _, returned_history, _, _, _ = results[0]
        assert returned_history == history

    @pytest.mark.anyio
    async def test_empty_audio_array_status_mentions_retry(self, handler):
        audio = (SR, np.array([], dtype=np.float32))
        results = await _collect(handle_voice_turn(audio, [], handler))
        _, _, status, _, _ = results[0]
        assert "try again" in status.lower()


# ---------------------------------------------------------------------------
# stream_audio_chunk — VAD streaming handler
# ---------------------------------------------------------------------------

class TestStreamDoesNotOutputAudio:
    """Regression: stream_audio_chunk must NOT output to audio_output.

    Audio playback is owned exclusively by poll_pending_result.
    The stream handler outputs 7 values (no audio) to avoid rapid
    re-renders that kill the audio player before it can play.
    """

    @pytest.mark.anyio
    async def test_stream_yields_7_values_not_8(self, handler, vad, session, summary):
        results = await _collect(stream_audio_chunk(None, handler, vad, session, summary))
        # history, status, debug, handler, vad, session, summary — no audio_output
        assert len(results[0]) == 7

    @pytest.mark.anyio
    async def test_silence_chunk_yields_7_values(self, handler, vad, session, summary):
        sr, data = _silence_chunk()
        results = await _collect(stream_audio_chunk((sr, data), handler, vad, session, summary))
        assert len(results[0]) == 7

    @pytest.mark.anyio
    async def test_speech_chunk_yields_7_values(self, handler, vad, session, summary):
        sr, data = _speech_chunk()
        results = await _collect(stream_audio_chunk((sr, data), handler, vad, session, summary))
        assert len(results[0]) == 7


class TestStreamAudioChunkIsAsyncGenerator:
    @pytest.mark.anyio
    async def test_is_async_generator(self, handler, vad, session, summary):
        import inspect
        result = stream_audio_chunk(None, handler, vad, session, summary)
        assert inspect.isasyncgen(result)

    @pytest.mark.anyio
    async def test_none_chunk_yields_listening_status(self, handler, vad, session, summary):
        results = await _collect(stream_audio_chunk(None, handler, vad, session, summary))
        assert len(results) == 1
        _, status, _, _, _, _, _ = results[0]
        assert "Listening" in status

    @pytest.mark.anyio
    async def test_empty_data_yields_listening_status(self, handler, vad, session, summary):
        audio = (SR, np.array([], dtype=np.float32))
        results = await _collect(stream_audio_chunk(audio, handler, vad, session, summary))
        _, status, _, _, _, _, _ = results[0]
        assert "Listening" in status


async def _trigger_vad(handler, vad, session, summary, monkeypatch, mock_send_turn):
    """Helper: feed speech + silence to fire the VAD, then poll until the task completes."""
    import app.ui.gradio_app as module
    monkeypatch.setattr(module, "send_turn", mock_send_turn)

    for _ in range(MIN_SPEECH_CHUNKS):
        sr, data = _speech_chunk()
        async for item in stream_audio_chunk((sr, data), handler, vad, session, summary):
            _, _, _, _, vad, session, summary = item

    for _ in range(SILENCE_CHUNKS_TO_TRIGGER):
        sr, data = _silence_chunk()
        async for item in stream_audio_chunk((sr, data), handler, vad, session, summary):
            _, _, _, _, vad, session, summary = item

    if vad.pending_task:
        await vad.pending_task

    final_result = await poll_pending_result(handler, vad, session, summary)
    _, _, _, _, _, vad, session, summary = final_result
    return [final_result], vad


class TestStreamAudioChunkVADIntegration:
    @pytest.mark.anyio
    async def test_speech_chunk_yields_speaking_status(self, handler, vad, session, summary):
        sr, data = _speech_chunk()
        results = await _collect(stream_audio_chunk((sr, data), handler, vad, session, summary))
        _, status, _, _, _, _, _ = results[-1]
        assert "Speaking" in status or "Thinking" in status

    @pytest.mark.anyio
    async def test_no_api_call_during_speech_only(self, handler, vad, session, summary, monkeypatch):
        """Live API must NOT be called until silence threshold is reached."""
        import app.ui.gradio_app as module
        call_count = 0

        async def mock_send_turn(pcm, orchestrate_fn=None):
            nonlocal call_count
            call_count += 1
            from app.live.live_session_manager import LiveSessionResult
            return LiveSessionResult(b"\x00\x00", "u", "a")

        monkeypatch.setattr(module, "send_turn", mock_send_turn)

        for _ in range(MIN_SPEECH_CHUNKS + 1):
            sr, data = _speech_chunk()
            async for _ in stream_audio_chunk((sr, data), handler, vad, session, summary):
                pass

        assert call_count == 0

    @pytest.mark.anyio
    async def test_api_called_after_speech_then_silence(self, handler, vad, session, summary, monkeypatch):
        """Live API must be called exactly once after enough silence follows speech."""
        call_count = 0

        async def mock_send_turn(pcm, orchestrate_fn=None):
            nonlocal call_count
            call_count += 1
            from app.live.live_session_manager import LiveSessionResult
            return LiveSessionResult(b"\x00\x01", "what is my balance", "Your balance is 100.")

        await _trigger_vad(handler, vad, session, summary, monkeypatch, mock_send_turn)
        assert call_count == 1

    @pytest.mark.anyio
    async def test_transcripts_added_after_successful_turn(self, handler, vad, session, summary, monkeypatch):
        from app.live.live_session_manager import LiveSessionResult

        async def mock_send_turn(pcm, orchestrate_fn=None):
            return LiveSessionResult(b"\x00\x01", "what is my balance", "Your balance is 200.")

        results, _ = await _trigger_vad(handler, vad, session, summary, monkeypatch, mock_send_turn)
        _, history, _, _, _, _, _, _ = results[-1]
        roles = [m["role"] for m in history]
        assert "user" in roles
        assert "assistant" in roles

    @pytest.mark.anyio
    async def test_error_result_shown_in_status(self, handler, vad, session, summary, monkeypatch):
        from app.live.live_session_manager import LiveSessionResult

        async def mock_send_turn(pcm, orchestrate_fn=None):
            return LiveSessionResult(b"", "", "", error="timeout")

        results, _ = await _trigger_vad(handler, vad, session, summary, monkeypatch, mock_send_turn)
        _, _, status, debug, _, _, _, _ = results[-1]
        assert "warning" in status.lower()
        assert "timeout" in debug.get("error", "")

    @pytest.mark.anyio
    async def test_vad_resets_after_send(self, handler, vad, session, summary, monkeypatch):
        """After an utterance is sent, VAD state resets so next utterance is independent."""
        from app.live.live_session_manager import LiveSessionResult

        async def mock_send_turn(pcm, orchestrate_fn=None):
            return LiveSessionResult(b"\x00\x00", "hi", "hello")

        _, vad = await _trigger_vad(handler, vad, session, summary, monkeypatch, mock_send_turn)
        assert not vad.is_speaking
        assert vad.buffer_chunks == []


# ---------------------------------------------------------------------------
# poll_pending_result
# ---------------------------------------------------------------------------

class TestPollPendingResult:
    @pytest.mark.anyio
    async def test_no_task_returns_listening_status(self, handler, vad, session, summary):
        audio_out, history, status, debug, _, _, _, _ = await poll_pending_result(handler, vad, session, summary)
        assert audio_out is not None
        assert history == []
        # When truly idle (no task, no cooldown, not speaking) the poller must
        # explicitly return "Listening..." to prevent the stuck-state bug.
        assert "Listening" in status
        assert debug == {}


# ---------------------------------------------------------------------------
# handle_clear
# ---------------------------------------------------------------------------

class TestHandleClear:
    def test_clear_returns_empty_history(self, handler, vad, session, summary):
        handler.add_user("hello")
        history, _, _, _, _, _, _, _ = handle_clear(handler, vad, session, summary)
        assert history == []

    def test_clear_returns_ready_status(self, handler, vad, session, summary):
        _, _, status, _, _, _, _, _ = handle_clear(handler, vad, session, summary)
        assert "Ready" in status or "cleared" in status.lower()

    def test_clear_resets_transcript_handler(self, handler, vad, session, summary):
        handler.add_user("hello")
        _, _, _, _, returned_handler, _, _, _ = handle_clear(handler, vad, session, summary)
        assert returned_handler.get_history() == []

    def test_clear_resets_vad_state(self, handler, vad, session, summary):
        vad.is_speaking = True
        vad.buffer_chunks = [np.ones(CHUNK, dtype=np.float32)]
        _, _, _, _, _, returned_vad, _, _ = handle_clear(handler, vad, session, summary)
        assert returned_vad.buffer_chunks == []
        assert returned_vad.is_speaking is False
        assert returned_vad.pending_task is None

    def test_clear_resets_session_state(self, handler, vad, session, summary):
        session.customer_id = "C123"
        session.workflow_step = "awaiting_clarification"
        _, _, _, _, _, _, returned_session, _ = handle_clear(handler, vad, session, summary)
        assert returned_session.customer_id is None
        assert returned_session.workflow_step == "idle"

    def test_clear_resets_summary_store(self, handler, vad, session, summary):
        summary.update("hello", "hi")
        _, _, _, _, _, _, _, returned_summary = handle_clear(handler, vad, session, summary)
        assert returned_summary.turn_count() == 0


# ---------------------------------------------------------------------------
# create_app — Gradio 6.x API compatibility
# ---------------------------------------------------------------------------

class TestCreateApp:
    def test_create_app_does_not_raise(self):
        from app.ui.gradio_app import create_app
        demo = create_app()
        assert demo is not None

    def test_returns_gradio_blocks(self):
        from app.ui.gradio_app import create_app
        import gradio as gr
        demo = create_app()
        assert isinstance(demo, gr.Blocks)

    def test_theme_not_in_blocks_constructor(self):
        import inspect
        import app.ui.gradio_app as module
        src = inspect.getsource(module.create_app)
        blocks_call_line = next(l for l in src.splitlines() if "gr.Blocks(" in l)
        assert "theme=" not in blocks_call_line

    def test_chatbot_no_type_kwarg(self):
        """Regression: gr.Chatbot no longer accepts type= in Gradio 6.x."""
        import inspect
        import app.ui.gradio_app as module
        src = inspect.getsource(module.create_app)
        chatbot_lines = [l for l in src.splitlines() if "gr.Chatbot(" in l]
        for line in chatbot_lines:
            assert "type=" not in line

    def test_audio_input_has_streaming_and_recording(self):
        """Regression: microphone must use streaming=True + recording=True for Gradio 6.x VAD UX."""
        import inspect
        import app.ui.gradio_app as module
        src = inspect.getsource(module.create_app)
        assert "streaming=True" in src, "audio_input must set streaming=True"
        assert "recording=True" in src, "audio_input must set recording=True"

    def test_timer_is_present_for_result_polling(self):
        import inspect
        import app.ui.gradio_app as module
        src = inspect.getsource(module.create_app)
        assert "gr.Timer(" in src
        assert ".tick(" in src

    def test_audio_output_type_is_numpy(self):
        """Regression: audio output must use type='numpy' to match tuple returns."""
        import inspect
        import app.ui.gradio_app as module
        src = inspect.getsource(module.create_app)
        assert 'type="numpy"' in src

    def test_stream_handler_does_not_output_to_audio(self):
        """Regression: stream_audio_chunk must NOT write to audio_output.

        Audio is owned by the poller to prevent rapid re-renders that
        kill audio playback before it starts.
        """
        import inspect
        import app.ui.gradio_app as module
        src = inspect.getsource(module.create_app)
        lines = src.split("\n")
        in_stream_block = False
        for line in lines:
            if "audio_input.stream(" in line:
                in_stream_block = True
            if in_stream_block and "outputs=" in line:
                assert "audio_output" not in line
                break

    def test_session_state_and_summary_wired(self):
        """Phase 2: create_app must include session_state and summary_state as gr.State."""
        import inspect
        import app.ui.gradio_app as module
        src = inspect.getsource(module.create_app)
        assert "session_state" in src
        assert "summary_state" in src


# ---------------------------------------------------------------------------
# _build_audio_output — numpy tuple creation
# ---------------------------------------------------------------------------

class TestBuildAudioOutput:
    def test_empty_bytes_returns_gr_update(self):
        result = _build_audio_output(b"")
        assert not isinstance(result, tuple)

    def test_valid_pcm_returns_numpy_tuple(self):
        pcm = np.zeros(2400, dtype=np.int16).tobytes()
        result = _build_audio_output(pcm)
        assert isinstance(result, tuple)
        assert len(result) == 2
        sr, data = result
        assert sr == 24000
        assert isinstance(data, np.ndarray)
        assert len(data) == 2400

    def test_playback_gain_is_applied(self):
        """Audio should be louder than raw PCM (gain > 1.0)."""
        from app.ui.gradio_app import PLAYBACK_GAIN
        assert PLAYBACK_GAIN > 1.0

        amplitude = 0.1
        float_samples = np.ones(2400, dtype=np.float32) * amplitude
        pcm = (float_samples * 32767).astype(np.int16).tobytes()
        sr, data = _build_audio_output(pcm)
        peak = float(np.max(np.abs(data)))
        assert peak > amplitude


# ---------------------------------------------------------------------------
# _consume_finished_task
# ---------------------------------------------------------------------------

class TestConsumeFinishedTask:
    @pytest.mark.anyio
    async def test_successful_task_returns_audio_and_transcripts(self, handler, session, summary):
        from app.live.live_session_manager import LiveSessionResult

        vad = _new_vad_state()
        future = asyncio.get_event_loop().create_future()
        future.set_result(LiveSessionResult(b"\x00\x02" * 1200, "hello", "hi there"))
        vad.pending_task = future

        audio_out, history, status, debug, _, returned_vad, _, _ = _consume_finished_task(handler, vad, session, summary)
        assert isinstance(audio_out, tuple)
        assert len(audio_out) == 2
        assert returned_vad.pending_task is None
        assert debug["ok"] is True
        assert "Playing" in status

    @pytest.mark.anyio
    async def test_error_task_shows_error_status(self, handler, session, summary):
        from app.live.live_session_manager import LiveSessionResult

        vad = _new_vad_state()
        future = asyncio.get_event_loop().create_future()
        future.set_result(LiveSessionResult(b"", "", "", error="API timeout"))
        vad.pending_task = future

        audio_out, history, status, debug, _, returned_vad, _, _ = _consume_finished_task(handler, vad, session, summary)
        assert "warning" in status.lower() or "Error" in status
        assert debug["error"] == "API timeout"
        assert returned_vad.pending_task is None

    @pytest.mark.anyio
    async def test_clears_pending_task_after_consume(self, handler, session, summary):
        from app.live.live_session_manager import LiveSessionResult

        vad = _new_vad_state()
        future = asyncio.get_event_loop().create_future()
        future.set_result(LiveSessionResult(b"\x00\x00", "hi", "hello"))
        vad.pending_task = future

        _, _, _, _, _, returned_vad, _, _ = _consume_finished_task(handler, vad, session, summary)
        assert returned_vad.pending_task is None

    @pytest.mark.anyio
    async def test_sets_cooldown_when_audio_present(self, handler, session, summary):
        from app.live.live_session_manager import LiveSessionResult

        vad = _new_vad_state()
        future = asyncio.get_event_loop().create_future()
        future.set_result(LiveSessionResult(b"\x00\x02" * 100, "hi", "hello"))
        vad.pending_task = future

        before = time.monotonic()
        _, _, _, _, _, returned_vad, _, _ = _consume_finished_task(handler, vad, session, summary)
        assert returned_vad.ignore_until > before

    def test_summary_store_updated_after_turn(self, handler, session, summary):
        """summary_store must be updated with transcripts from a completed turn."""
        from app.live.live_session_manager import LiveSessionResult

        vad = _new_vad_state()
        future = asyncio.get_event_loop().create_future()
        future.set_result(LiveSessionResult(b"\x00\x02" * 100, "hello", "hi there"))
        vad.pending_task = future

        _, _, _, _, _, _, _, returned_summary = _consume_finished_task(handler, vad, session, summary)
        assert returned_summary.turn_count() == 1

    def test_orchestration_debug_included_when_decision_present(self, handler, session, summary):
        """When result carries an orchestration_decision, the debug panel must include intent/tool."""
        from app.live.live_session_manager import LiveSessionResult
        from app.orchestration.schemas import IntentType, OrchestrationDecision

        decision = OrchestrationDecision(
            intent=IntentType.BALANCE_INQUIRY,
            requires_tool=True,
            selected_tool="get_balance",
            confidence=0.9,
            reason="test",
        )
        vad = _new_vad_state()
        future = asyncio.get_event_loop().create_future()
        future.set_result(LiveSessionResult(b"\x00\x02" * 100, "balance", "Your balance is X.", orchestration_decision=decision))
        vad.pending_task = future

        _, _, _, debug, _, _, _, _ = _consume_finished_task(handler, vad, session, summary)
        assert debug.get("intent") == "balance_inquiry"
        assert debug.get("selected_tool") == "get_balance"
        assert debug.get("confidence") == pytest.approx(0.9, abs=0.001)

    @pytest.mark.anyio
    async def test_marks_task_as_consumed(self, handler, session, summary):
        """Regression (Gradio State race): _consume_finished_task must mark the
        asyncio Task as _voice_consumed=True so that stale vad_state copies
        held by stream_audio_chunk don't re-process the same result."""
        from app.live.live_session_manager import LiveSessionResult

        vad = _new_vad_state()
        future = asyncio.get_event_loop().create_future()
        future.set_result(LiveSessionResult(b"\x00\x02" * 100, "hi", "hello"))
        vad.pending_task = future

        _consume_finished_task(handler, vad, session, summary)
        assert getattr(future, "_voice_consumed", False) is True

    @pytest.mark.anyio
    async def test_second_call_with_same_stale_task_skips_re_consume(self, handler, session, summary):
        """Regression (Gradio State race): if stream_audio_chunk writes stale
        vad_state back to Gradio State, a second poll call on the same task
        must not re-process the result (no duplicate transcripts, no cooldown reset)."""
        from app.live.live_session_manager import LiveSessionResult

        vad = _new_vad_state()
        future = asyncio.get_event_loop().create_future()
        future.set_result(LiveSessionResult(b"\x00\x02" * 100, "hi", "hello"))
        vad.pending_task = future

        # First consumption
        _, _, _, _, _, vad1, _, summary1 = _consume_finished_task(handler, vad, session, summary)
        assert vad1.pending_task is None
        turn_count_after_first = summary1.turn_count()

        # Simulate Gradio State race: stale copy of vad_state with same task
        stale_vad = _new_vad_state()
        stale_vad.pending_task = future  # same task object, already _voice_consumed

        _, _, status2, _, _, vad2, _, summary2 = _consume_finished_task(
            TranscriptHandler(), stale_vad, session, summary1
        )
        # Must clear the stale reference
        assert vad2.pending_task is None
        # Must NOT re-process the result (summary should not gain another turn)
        assert summary2.turn_count() == turn_count_after_first
        # Status must not be "Playing response..." (no re-consume means no cooldown reset)
        assert "Listening" in status2


# ---------------------------------------------------------------------------
# _new_vad_state
# ---------------------------------------------------------------------------

class TestNewVadState:
    def test_has_ignore_until_attribute(self):
        state = _new_vad_state()
        assert hasattr(state, "ignore_until")
        assert getattr(state, "ignore_until") == 0.0

    def test_pending_task_is_none(self):
        state = _new_vad_state()
        assert state.pending_task is None


# ---------------------------------------------------------------------------
# poll_pending_result — expanded coverage
# ---------------------------------------------------------------------------

class TestPollPendingResultExpanded:
    @pytest.mark.anyio
    async def test_returns_thinking_when_task_pending(self, handler, session, summary):
        vad = _new_vad_state()
        vad.pending_task = asyncio.get_event_loop().create_future()

        audio_out, history, status, debug, _, _, _, _ = await poll_pending_result(handler, vad, session, summary)
        assert "Thinking" in status
        assert debug.get("waiting") is True

        vad.pending_task.cancel()

    @pytest.mark.anyio
    async def test_returns_result_when_task_done(self, handler, session, summary):
        from app.live.live_session_manager import LiveSessionResult

        vad = _new_vad_state()
        future = asyncio.get_event_loop().create_future()
        future.set_result(LiveSessionResult(b"\x00\x02" * 100, "test", "response"))
        vad.pending_task = future

        audio_out, history, status, debug, _, returned_vad, _, _ = await poll_pending_result(handler, vad, session, summary)
        assert returned_vad.pending_task is None
        assert debug["ok"] is True

    @pytest.mark.anyio
    async def test_shows_cooldown_during_playback(self, handler, session, summary):
        vad = _new_vad_state()
        vad.ignore_until = time.monotonic() + 5.0

        audio_out, _, status, debug, _, _, _, _ = await poll_pending_result(handler, vad, session, summary)
        assert "Playing" in status
        assert "cooldown_s" in debug


# ---------------------------------------------------------------------------
# poll_pending_result — idle transition (Bug 1 regression)
# ---------------------------------------------------------------------------

class TestPollIdleTransition:
    """Regression: after cooldown expires with no pending task, the poller must
    explicitly return 'Listening...' so the UI never gets stuck on
    'Playing response...' when the browser pauses the mic stream during playback."""

    @pytest.mark.anyio
    async def test_returns_listening_when_truly_idle(self, handler, session, summary):
        vad = _new_vad_state()
        _, _, status, debug, _, _, _, _ = await poll_pending_result(handler, vad, session, summary)
        assert isinstance(status, str), "status must be a string, not gr.update() no-op"
        assert "Listening" in status

    @pytest.mark.anyio
    async def test_returns_listening_after_cooldown_expires(self, handler, session, summary):
        """Regression: stuck 'Playing response...' state."""
        vad = _new_vad_state()
        vad.ignore_until = time.monotonic() - 1.0

        _, _, status, debug, _, _, _, _ = await poll_pending_result(handler, vad, session, summary)
        assert isinstance(status, str)
        assert "Listening" in status
        assert debug == {}

    @pytest.mark.anyio
    async def test_returns_listening_even_when_is_speaking_true(self, handler, session, summary):
        """Regression: old code returned gr.update() when is_speaking=True after cooldown
        expired, causing the UI to freeze permanently on 'Playing response...'.
        The poller must always return 'Listening...' once the cooldown is done,
        regardless of VAD speech state — the stream handler will override it on
        the next 0.25s tick if the user is actually speaking."""
        vad = _new_vad_state()
        vad.is_speaking = True
        vad.ignore_until = time.monotonic() - 1.0  # cooldown already expired

        _, _, status, _, _, _, _, _ = await poll_pending_result(handler, vad, session, summary)
        assert isinstance(status, str), (
            "poller must return 'Listening...' string, not gr.update(), when cooldown has expired"
        )
        assert "Listening" in status


# ---------------------------------------------------------------------------
# _consume_finished_task — duration-based cooldown (Bug 2 regression)
# ---------------------------------------------------------------------------

class TestDurationBasedCooldown:
    @pytest.mark.anyio
    async def test_long_audio_gets_longer_cooldown_than_short(self, handler, session, summary):
        from app.live.live_session_manager import LiveSessionResult

        sr = 24000
        bps = 2

        short_pcm = bytes(sr * bps // 10)
        long_pcm = bytes(sr * bps * 3)

        def _cooldown_for(pcm: bytes) -> float:
            vad2 = _new_vad_state()
            fut2 = asyncio.get_event_loop().create_future()
            fut2.set_result(LiveSessionResult(pcm, "hi", "hello"))
            vad2.pending_task = fut2
            t0 = time.monotonic()
            _, _, _, _, _, returned_vad, _, _ = _consume_finished_task(TranscriptHandler(), vad2, SessionState(), SummaryStore())
            return returned_vad.ignore_until - t0

        short_cooldown = _cooldown_for(short_pcm)
        long_cooldown = _cooldown_for(long_pcm)

        assert long_cooldown > short_cooldown

    @pytest.mark.anyio
    async def test_cooldown_covers_audio_duration(self, handler, session, summary):
        from app.live.live_session_manager import LiveSessionResult
        from app.ui.gradio_app import PLAYBACK_COOLDOWN_SECONDS, DEFAULT_OUTPUT_SAMPLE_RATE

        sr = DEFAULT_OUTPUT_SAMPLE_RATE
        bps = 2
        duration_s = 2.0
        pcm = bytes(int(sr * bps * duration_s))

        vad = _new_vad_state()
        fut = asyncio.get_event_loop().create_future()
        fut.set_result(LiveSessionResult(pcm, "hi", "hello"))
        vad.pending_task = fut

        t0 = time.monotonic()
        _, _, _, _, _, returned_vad, _, _ = _consume_finished_task(handler, vad, session, summary)
        cooldown_applied = returned_vad.ignore_until - t0

        assert cooldown_applied >= duration_s
        assert cooldown_applied >= duration_s + PLAYBACK_COOLDOWN_SECONDS - 0.05

    @pytest.mark.anyio
    async def test_cooldown_capped_at_five_seconds(self, handler, session, summary):
        """Regression: very long audio must not block the mic indefinitely.
        Cooldown must be capped at 5 s + PLAYBACK_COOLDOWN_SECONDS."""
        from app.live.live_session_manager import LiveSessionResult
        from app.ui.gradio_app import DEFAULT_OUTPUT_SAMPLE_RATE

        sr = DEFAULT_OUTPUT_SAMPLE_RATE
        bps = 2
        # 30 seconds of audio — without the cap this would lock the mic for 30s
        pcm = bytes(sr * bps * 30)

        vad = _new_vad_state()
        fut = asyncio.get_event_loop().create_future()
        fut.set_result(LiveSessionResult(pcm, "hi", "hello"))
        vad.pending_task = fut

        t0 = time.monotonic()
        _, _, _, _, _, returned_vad, _, _ = _consume_finished_task(handler, vad, session, summary)
        cooldown_applied = returned_vad.ignore_until - t0

        assert cooldown_applied <= 5.0 + PLAYBACK_COOLDOWN_SECONDS + 0.05, (
            "cooldown must be capped at 5 s to avoid permanently blocking the mic"
        )

    @pytest.mark.anyio
    async def test_empty_audio_no_cooldown_set(self, handler, session, summary):
        from app.live.live_session_manager import LiveSessionResult

        vad = _new_vad_state()
        fut = asyncio.get_event_loop().create_future()
        fut.set_result(LiveSessionResult(b"", "hi", ""))
        vad.pending_task = fut

        before = time.monotonic()
        _, _, _, _, _, returned_vad, _, _ = _consume_finished_task(handler, vad, session, summary)
        assert returned_vad.ignore_until <= before + 0.01

    @pytest.mark.anyio
    async def test_cooldown_capped_for_very_long_audio(self, handler, session, summary):
        """Regression (stuck 'Playing response...'): verbose Phase-2 responses with
        audio_bytes≥264 kB (~5.5 s) must not create a cooldown longer than
        MAX_PLAYBACK_COOLDOWN_SECONDS, preventing the UI from appearing permanently stuck."""
        from app.live.live_session_manager import LiveSessionResult
        from app.ui.gradio_app import MAX_PLAYBACK_COOLDOWN_SECONDS, DEFAULT_OUTPUT_SAMPLE_RATE

        # Simulate a very long response: 12 seconds of audio
        sr = DEFAULT_OUTPUT_SAMPLE_RATE
        bps = 2
        very_long_pcm = bytes(sr * bps * 12)  # 12 s → would give 12.75 s uncapped

        vad = _new_vad_state()
        fut = asyncio.get_event_loop().create_future()
        fut.set_result(LiveSessionResult(very_long_pcm, "hi", "hello"))
        vad.pending_task = fut

        t0 = time.monotonic()
        _, _, _, _, _, returned_vad, _, _ = _consume_finished_task(handler, vad, session, summary)
        cooldown_applied = returned_vad.ignore_until - t0

        # Must not exceed the cap
        assert cooldown_applied <= MAX_PLAYBACK_COOLDOWN_SECONDS + 0.05

    @pytest.mark.anyio
    async def test_cooldown_screenshot_case(self, handler, session, summary):
        """Exact case from the bug report: audio_bytes=240004 (≈5 s) must produce
        a cooldown ≤ MAX_PLAYBACK_COOLDOWN_SECONDS even though naive arithmetic
        would give 5.75 s which should be under the 6 s cap."""
        from app.live.live_session_manager import LiveSessionResult
        from app.ui.gradio_app import MAX_PLAYBACK_COOLDOWN_SECONDS

        vad = _new_vad_state()
        fut = asyncio.get_event_loop().create_future()
        fut.set_result(LiveSessionResult(b"\x00\x01" * 120002, "hi", "hello"))
        vad.pending_task = fut

        t0 = time.monotonic()
        _, _, _, _, _, returned_vad, _, _ = _consume_finished_task(handler, vad, session, summary)
        cooldown_applied = returned_vad.ignore_until - t0

        assert cooldown_applied <= MAX_PLAYBACK_COOLDOWN_SECONDS + 0.05


# ---------------------------------------------------------------------------
# stream_audio_chunk — cooldown behavior
# ---------------------------------------------------------------------------

class TestStreamAudioChunkCooldown:
    @pytest.mark.anyio
    async def test_ignores_input_during_cooldown(self, handler, session, summary):
        vad = _new_vad_state()
        vad.ignore_until = time.monotonic() + 5.0

        sr, data = _speech_chunk()
        results = await _collect(stream_audio_chunk((sr, data), handler, vad, session, summary))
        _, status, debug, _, _, _, _ = results[0]
        assert "Playing" in status
        assert "cooldown_s" in debug

    @pytest.mark.anyio
    async def test_resumes_listening_after_cooldown(self, handler, session, summary):
        vad = _new_vad_state()
        vad.ignore_until = time.monotonic() - 1.0

        sr, data = _silence_chunk()
        results = await _collect(stream_audio_chunk((sr, data), handler, vad, session, summary))
        _, status, _, _, _, _, _ = results[0]
        assert "Listening" in status

    @pytest.mark.anyio
    async def test_stream_clears_stale_consumed_task(self, handler, session, summary):
        """Regression (Gradio State race): when stream_audio_chunk receives a
        vad_state copy that still holds a _voice_consumed Task, it must clear
        the stale reference so it doesn't write it back to Gradio State and
        cause the poll handler to re-consume the result on the next tick."""
        from app.live.live_session_manager import LiveSessionResult

        # Simulate a task that was already consumed by the poll handler
        future = asyncio.get_event_loop().create_future()
        future.set_result(LiveSessionResult(b"\x00\x02" * 100, "hi", "hello"))
        future._voice_consumed = True  # mark as consumed

        vad = _new_vad_state()
        vad.pending_task = future  # stale copy still has it

        sr, data = _silence_chunk()
        results = await _collect(stream_audio_chunk((sr, data), handler, vad, session, summary))
        # stream_audio_chunk yields 7 values (no audio_output)
        _, _, _, _, returned_vad, _, _ = results[0]

        # The stale reference must be cleared — NOT written back
        assert returned_vad.pending_task is None
