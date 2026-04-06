"""
Tests for gradio_app.py event handlers.

Covers:
  - handle_voice_turn: async generator contract + early exits (kept for regression)
  - stream_audio_chunk: VAD-driven streaming handler
  - poll_pending_result: timer-driven completion path
  - _build_audio_output: WAV file creation and gain/normalization
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
from app.ui.gradio_app import (
    handle_voice_turn,
    handle_clear,
    stream_audio_chunk,
    poll_pending_result,
    _build_audio_output,
    _consume_finished_task,
    _new_vad_state,
    PLAYBACK_COOLDOWN_SECONDS,
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
    The stream handler outputs 5 values (no audio) to avoid rapid
    re-renders that kill the audio player before it can play.
    """

    @pytest.mark.anyio
    async def test_stream_yields_5_values_not_6(self, handler, vad):
        results = await _collect(stream_audio_chunk(None, handler, vad))
        assert len(results[0]) == 5  # no audio_output in the tuple

    @pytest.mark.anyio
    async def test_silence_chunk_yields_5_values(self, handler, vad):
        sr, data = _silence_chunk()
        results = await _collect(stream_audio_chunk((sr, data), handler, vad))
        assert len(results[0]) == 5

    @pytest.mark.anyio
    async def test_speech_chunk_yields_5_values(self, handler, vad):
        sr, data = _speech_chunk()
        results = await _collect(stream_audio_chunk((sr, data), handler, vad))
        assert len(results[0]) == 5


class TestStreamAudioChunkIsAsyncGenerator:
    @pytest.mark.anyio
    async def test_is_async_generator(self, handler, vad):
        import inspect
        result = stream_audio_chunk(None, handler, vad)
        assert inspect.isasyncgen(result)

    @pytest.mark.anyio
    async def test_none_chunk_yields_listening_status(self, handler, vad):
        results = await _collect(stream_audio_chunk(None, handler, vad))
        assert len(results) == 1
        _, status, _, _, _ = results[0]
        assert "Listening" in status

    @pytest.mark.anyio
    async def test_empty_data_yields_listening_status(self, handler, vad):
        audio = (SR, np.array([], dtype=np.float32))
        results = await _collect(stream_audio_chunk(audio, handler, vad))
        _, status, _, _, _ = results[0]
        assert "Listening" in status


async def _trigger_vad(handler, vad, monkeypatch, mock_send_turn):
    """Helper: feed speech + silence to fire the VAD, then poll until the task completes."""
    import app.ui.gradio_app as module
    monkeypatch.setattr(module, "send_turn", mock_send_turn)

    # Feed speech chunks (stream yields 5 values: history, status, debug, handler, vad)
    for _ in range(MIN_SPEECH_CHUNKS):
        sr, data = _speech_chunk()
        async for item in stream_audio_chunk((sr, data), handler, vad):
            _, _, _, _, vad = item

    # Feed silence to trigger VAD → launches background task
    for _ in range(SILENCE_CHUNKS_TO_TRIGGER):
        sr, data = _silence_chunk()
        async for item in stream_audio_chunk((sr, data), handler, vad):
            _, _, _, _, vad = item

    # Let the background task finish
    if vad.pending_task:
        await vad.pending_task

    # Collect the result using the timer-based poller (returns 6 values including audio)
    final_result = await poll_pending_result(handler, vad)
    _, _, _, _, _, vad = final_result
    return [final_result], vad


class TestStreamAudioChunkVADIntegration:
    @pytest.mark.anyio
    async def test_speech_chunk_yields_speaking_status(self, handler, vad):
        sr, data = _speech_chunk()
        results = await _collect(stream_audio_chunk((sr, data), handler, vad))
        _, status, _, _, _ = results[-1]
        assert "Speaking" in status or "Thinking" in status

    @pytest.mark.anyio
    async def test_no_api_call_during_speech_only(self, handler, vad, monkeypatch):
        """Live API must NOT be called until silence threshold is reached."""
        import app.ui.gradio_app as module
        call_count = 0

        async def mock_send_turn(pcm):
            nonlocal call_count
            call_count += 1
            from app.live.live_session_manager import LiveSessionResult
            return LiveSessionResult(b"\x00\x00", "u", "a")

        monkeypatch.setattr(module, "send_turn", mock_send_turn)

        for _ in range(MIN_SPEECH_CHUNKS + 1):
            sr, data = _speech_chunk()
            async for _ in stream_audio_chunk((sr, data), handler, vad):
                pass

        assert call_count == 0

    @pytest.mark.anyio
    async def test_api_called_after_speech_then_silence(self, handler, vad, monkeypatch):
        """Live API must be called exactly once after enough silence follows speech."""
        call_count = 0

        async def mock_send_turn(pcm):
            nonlocal call_count
            call_count += 1
            from app.live.live_session_manager import LiveSessionResult
            return LiveSessionResult(b"\x00\x01", "what is my balance", "Your balance is 100.")

        await _trigger_vad(handler, vad, monkeypatch, mock_send_turn)
        assert call_count == 1

    @pytest.mark.anyio
    async def test_transcripts_added_after_successful_turn(self, handler, vad, monkeypatch):
        from app.live.live_session_manager import LiveSessionResult

        async def mock_send_turn(pcm):
            return LiveSessionResult(b"\x00\x01", "what is my balance", "Your balance is 200.")

        results, _ = await _trigger_vad(handler, vad, monkeypatch, mock_send_turn)
        _, history, _, _, _, _ = results[-1]
        roles = [m["role"] for m in history]
        assert "user" in roles
        assert "assistant" in roles

    @pytest.mark.anyio
    async def test_error_result_shown_in_status(self, handler, vad, monkeypatch):
        from app.live.live_session_manager import LiveSessionResult

        async def mock_send_turn(pcm):
            return LiveSessionResult(b"", "", "", error="timeout")

        results, _ = await _trigger_vad(handler, vad, monkeypatch, mock_send_turn)
        _, _, status, debug, _, _ = results[-1]
        assert "warning" in status.lower()
        assert "timeout" in debug.get("error", "")

    @pytest.mark.anyio
    async def test_vad_resets_after_send(self, handler, vad, monkeypatch):
        """After an utterance is sent, VAD state resets so next utterance is independent."""
        from app.live.live_session_manager import LiveSessionResult

        async def mock_send_turn(pcm):
            return LiveSessionResult(b"\x00\x00", "hi", "hello")

        _, vad = await _trigger_vad(handler, vad, monkeypatch, mock_send_turn)
        assert not vad.is_speaking
        assert vad.buffer_chunks == []


# ---------------------------------------------------------------------------
# poll_pending_result
# ---------------------------------------------------------------------------

class TestPollPendingResult:
    @pytest.mark.anyio
    async def test_no_task_returns_listening_status(self, handler, vad):
        audio_out, history, status, debug, _, _ = await poll_pending_result(handler, vad)
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
    def test_clear_returns_empty_history(self, handler, vad):
        handler.add_user("hello")
        history, _, _, _, _, _ = handle_clear(handler, vad)
        assert history == []

    def test_clear_returns_ready_status(self, handler, vad):
        _, _, status, _, _, _ = handle_clear(handler, vad)
        assert "Ready" in status or "cleared" in status.lower()

    def test_clear_resets_transcript_handler(self, handler, vad):
        handler.add_user("hello")
        _, _, _, _, returned_handler, _ = handle_clear(handler, vad)
        assert returned_handler.get_history() == []

    def test_clear_resets_vad_state(self, handler, vad):
        vad.is_speaking = True
        vad.buffer_chunks = [np.ones(CHUNK, dtype=np.float32)]
        _, _, _, _, _, returned_vad = handle_clear(handler, vad)
        assert returned_vad.buffer_chunks == []
        assert returned_vad.is_speaking is False
        assert returned_vad.pending_task is None


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
        # Find the audio_input.stream() call and check audio_output is not in its outputs
        lines = src.split("\n")
        in_stream_block = False
        for line in lines:
            if "audio_input.stream(" in line:
                in_stream_block = True
            if in_stream_block and "outputs=" in line:
                assert "audio_output" not in line
                break


# ---------------------------------------------------------------------------
# _build_audio_output — numpy tuple creation
# ---------------------------------------------------------------------------

class TestBuildAudioOutput:
    def test_empty_bytes_returns_gr_update(self):
        result = _build_audio_output(b"")
        # gr.update() is not a tuple
        assert not isinstance(result, tuple)

    def test_valid_pcm_returns_numpy_tuple(self):
        # 0.1s of 24kHz silence = 2400 samples × 2 bytes = 4800 bytes
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

        # Quiet audio: 0.1 amplitude → int16 ~3277
        amplitude = 0.1
        float_samples = np.ones(2400, dtype=np.float32) * amplitude
        pcm = (float_samples * 32767).astype(np.int16).tobytes()
        sr, data = _build_audio_output(pcm)
        # After gain of 2.5, peak should be ~0.25
        peak = float(np.max(np.abs(data)))
        assert peak > amplitude  # gain was applied


# ---------------------------------------------------------------------------
# _consume_finished_task
# ---------------------------------------------------------------------------

class TestConsumeFinishedTask:
    @pytest.mark.anyio
    async def test_successful_task_returns_audio_and_transcripts(self, handler):
        from app.live.live_session_manager import LiveSessionResult

        vad = _new_vad_state()
        future = asyncio.get_event_loop().create_future()
        future.set_result(LiveSessionResult(b"\x00\x02" * 1200, "hello", "hi there"))
        vad.pending_task = future

        audio_out, history, status, debug, _, returned_vad = _consume_finished_task(handler, vad)
        assert isinstance(audio_out, tuple)  # (sample_rate, numpy_array)
        assert len(audio_out) == 2
        assert returned_vad.pending_task is None
        assert debug["ok"] is True
        assert "Playing" in status

    @pytest.mark.anyio
    async def test_error_task_shows_error_status(self, handler):
        from app.live.live_session_manager import LiveSessionResult

        vad = _new_vad_state()
        future = asyncio.get_event_loop().create_future()
        future.set_result(LiveSessionResult(b"", "", "", error="API timeout"))
        vad.pending_task = future

        audio_out, history, status, debug, _, returned_vad = _consume_finished_task(handler, vad)
        assert "warning" in status.lower() or "Error" in status
        assert debug["error"] == "API timeout"
        assert returned_vad.pending_task is None

    @pytest.mark.anyio
    async def test_clears_pending_task_after_consume(self, handler):
        from app.live.live_session_manager import LiveSessionResult

        vad = _new_vad_state()
        future = asyncio.get_event_loop().create_future()
        future.set_result(LiveSessionResult(b"\x00\x00", "hi", "hello"))
        vad.pending_task = future

        _, _, _, _, _, returned_vad = _consume_finished_task(handler, vad)
        assert returned_vad.pending_task is None

    @pytest.mark.anyio
    async def test_sets_cooldown_when_audio_present(self, handler):
        from app.live.live_session_manager import LiveSessionResult

        vad = _new_vad_state()
        future = asyncio.get_event_loop().create_future()
        future.set_result(LiveSessionResult(b"\x00\x02" * 100, "hi", "hello"))
        vad.pending_task = future

        before = time.monotonic()
        _, _, _, _, _, returned_vad = _consume_finished_task(handler, vad)
        ignore_until = getattr(returned_vad, "ignore_until", 0.0)
        assert ignore_until > before


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
    async def test_returns_thinking_when_task_pending(self, handler):
        vad = _new_vad_state()
        vad.pending_task = asyncio.get_event_loop().create_future()  # not done yet

        audio_out, history, status, debug, _, _ = await poll_pending_result(handler, vad)
        assert "Thinking" in status
        assert debug.get("waiting") is True

        # Cancel the future to avoid warning
        vad.pending_task.cancel()

    @pytest.mark.anyio
    async def test_returns_result_when_task_done(self, handler):
        from app.live.live_session_manager import LiveSessionResult

        vad = _new_vad_state()
        future = asyncio.get_event_loop().create_future()
        future.set_result(LiveSessionResult(b"\x00\x02" * 100, "test", "response"))
        vad.pending_task = future

        audio_out, history, status, debug, _, returned_vad = await poll_pending_result(handler, vad)
        assert returned_vad.pending_task is None
        assert debug["ok"] is True

    @pytest.mark.anyio
    async def test_shows_cooldown_during_playback(self, handler):
        vad = _new_vad_state()
        setattr(vad, "ignore_until", time.monotonic() + 5.0)  # 5s in the future

        audio_out, _, status, debug, _, _ = await poll_pending_result(handler, vad)
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
    async def test_returns_listening_when_truly_idle(self, handler):
        """No task, no cooldown → status must be 'Listening...' not gr.update()."""
        vad = _new_vad_state()
        _, _, status, debug, _, _ = await poll_pending_result(handler, vad)
        assert isinstance(status, str), "status must be a string, not gr.update() no-op"
        assert "Listening" in status

    @pytest.mark.anyio
    async def test_returns_listening_after_cooldown_expires(self, handler):
        """Regression: stuck 'Playing response...' state.
        After the cooldown expires the poller must push the UI back to 'Listening...'
        even when the stream handler is not firing (mic paused by browser)."""
        vad = _new_vad_state()
        vad.ignore_until = time.monotonic() - 1.0  # already expired

        _, _, status, debug, _, _ = await poll_pending_result(handler, vad)
        assert isinstance(status, str), "status must be explicit string, not gr.update()"
        assert "Listening" in status
        assert debug == {}

    @pytest.mark.anyio
    async def test_does_not_override_speaking_status(self, handler):
        """When VAD is accumulating speech, the stream handler owns 'Speaking...' status.
        The poller must not override it with 'Listening...' and cause visible flicker."""
        import gradio as gr
        vad = _new_vad_state()
        vad.is_speaking = True  # stream handler is currently detecting speech

        _, _, status, _, _, _ = await poll_pending_result(handler, vad)
        # gr.update() is returned — it's not a plain string
        assert not isinstance(status, str), (
            "poller must return gr.update() (no-op) when stream is in speech mode "
            "to avoid overwriting 'Speaking detected...' with 'Listening...'"
        )


# ---------------------------------------------------------------------------
# _consume_finished_task — duration-based cooldown (Bug 2 regression)
# ---------------------------------------------------------------------------

class TestDurationBasedCooldown:
    """Regression: fixed 0.75 s cooldown caused mic re-trigger during long responses.
    Cooldown must now be: audio_duration + PLAYBACK_COOLDOWN_SECONDS."""

    @pytest.mark.anyio
    async def test_long_audio_gets_longer_cooldown_than_short(self, handler):
        """A 3-second response must produce a longer cooldown than a 0.1-second response."""
        from app.live.live_session_manager import LiveSessionResult

        sr = 24000  # DEFAULT_OUTPUT_SAMPLE_RATE
        bytes_per_sample = 2

        # Short response: ~0.1 s of audio
        short_pcm = bytes(sr * bytes_per_sample // 10)  # 0.1 s
        # Long response: ~3 s of audio
        long_pcm = bytes(sr * bytes_per_sample * 3)    # 3 s

        def _cooldown_for(pcm: bytes) -> float:
            vad = _new_vad_state()
            fut = asyncio.get_event_loop().create_future()
            fut.set_result(LiveSessionResult(pcm, "hi", "hello"))
            vad.pending_task = fut
            before = time.monotonic()
            _consume_finished_task(handler, vad)
            # ignore_until is set relative to before — return the delta
            # We can't easily inspect vad after consume (it's returned), so
            # re-run and capture the returned vad
            vad2 = _new_vad_state()
            fut2 = asyncio.get_event_loop().create_future()
            fut2.set_result(LiveSessionResult(pcm, "hi", "hello"))
            vad2.pending_task = fut2
            t0 = time.monotonic()
            _, _, _, _, _, returned_vad = _consume_finished_task(TranscriptHandler(), vad2)
            return returned_vad.ignore_until - t0

        short_cooldown = _cooldown_for(short_pcm)
        long_cooldown = _cooldown_for(long_pcm)

        assert long_cooldown > short_cooldown, (
            f"Long audio ({len(long_pcm)} bytes) cooldown {long_cooldown:.2f}s must be "
            f"greater than short audio ({len(short_pcm)} bytes) cooldown {short_cooldown:.2f}s"
        )

    @pytest.mark.anyio
    async def test_cooldown_covers_audio_duration(self, handler):
        """Cooldown must be at least as long as the audio duration."""
        from app.live.live_session_manager import LiveSessionResult
        from app.ui.gradio_app import PLAYBACK_COOLDOWN_SECONDS, DEFAULT_OUTPUT_SAMPLE_RATE

        sr = DEFAULT_OUTPUT_SAMPLE_RATE
        bps = 2  # int16
        duration_s = 2.0
        pcm = bytes(int(sr * bps * duration_s))

        vad = _new_vad_state()
        fut = asyncio.get_event_loop().create_future()
        fut.set_result(LiveSessionResult(pcm, "hi", "hello"))
        vad.pending_task = fut

        t0 = time.monotonic()
        _, _, _, _, _, returned_vad = _consume_finished_task(handler, vad)
        cooldown_applied = returned_vad.ignore_until - t0

        assert cooldown_applied >= duration_s, (
            f"Cooldown {cooldown_applied:.2f}s must cover audio duration {duration_s}s"
        )
        assert cooldown_applied >= duration_s + PLAYBACK_COOLDOWN_SECONDS - 0.05  # small tolerance

    @pytest.mark.anyio
    async def test_empty_audio_no_cooldown_set(self, handler):
        """Empty audio response (e.g. error/text-only) must not set a cooldown."""
        from app.live.live_session_manager import LiveSessionResult

        vad = _new_vad_state()
        fut = asyncio.get_event_loop().create_future()
        fut.set_result(LiveSessionResult(b"", "hi", ""))
        vad.pending_task = fut

        before = time.monotonic()
        _, _, _, _, _, returned_vad = _consume_finished_task(handler, vad)
        assert returned_vad.ignore_until <= before + 0.01, (
            "No cooldown should be set when there are no audio bytes"
        )


# ---------------------------------------------------------------------------
# stream_audio_chunk — cooldown behavior
# ---------------------------------------------------------------------------

class TestStreamAudioChunkCooldown:
    @pytest.mark.anyio
    async def test_ignores_input_during_cooldown(self, handler):
        vad = _new_vad_state()
        setattr(vad, "ignore_until", time.monotonic() + 5.0)

        sr, data = _speech_chunk()
        results = await _collect(stream_audio_chunk((sr, data), handler, vad))
        _, status, debug, _, _ = results[0]
        assert "Playing" in status
        assert "cooldown_s" in debug

    @pytest.mark.anyio
    async def test_resumes_listening_after_cooldown(self, handler):
        vad = _new_vad_state()
        setattr(vad, "ignore_until", time.monotonic() - 1.0)  # Already expired

        sr, data = _silence_chunk()
        results = await _collect(stream_audio_chunk((sr, data), handler, vad))
        _, status, _, _, _ = results[0]
        assert "Listening" in status
