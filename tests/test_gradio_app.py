"""
Tests for gradio_app.py event handlers.

Covers:
  - handle_voice_turn: async generator contract + early exits (kept for regression)
  - stream_audio_chunk: VAD-driven streaming handler
  - handle_clear: session reset
  - create_app: Gradio 6.x API compatibility
"""

import numpy as np
import pytest

from app.live.transcript_handler import TranscriptHandler
from app.live.vad import VADState, SILENCE_CHUNKS_TO_TRIGGER, MIN_SPEECH_CHUNKS
from app.ui.gradio_app import handle_voice_turn, handle_clear, stream_audio_chunk

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

class TestStreamAudioChunkIsAsyncGenerator:
    @pytest.mark.anyio
    async def test_is_async_generator(self, handler, vad):
        import inspect
        result = stream_audio_chunk(None, [], handler, vad)
        assert inspect.isasyncgen(result)

    @pytest.mark.anyio
    async def test_none_chunk_yields_listening_status(self, handler, vad):
        results = await _collect(stream_audio_chunk(None, [], handler, vad))
        assert len(results) == 1
        _, _, status, _, _, _ = results[0]
        assert "Listening" in status

    @pytest.mark.anyio
    async def test_empty_data_yields_listening_status(self, handler, vad):
        audio = (SR, np.array([], dtype=np.float32))
        results = await _collect(stream_audio_chunk(audio, [], handler, vad))
        _, _, status, _, _, _ = results[0]
        assert "Listening" in status


class TestStreamAudioChunkVADIntegration:
    @pytest.mark.anyio
    async def test_speech_chunk_yields_speaking_status(self, handler, vad):
        sr, data = _speech_chunk()
        results = await _collect(stream_audio_chunk((sr, data), [], handler, vad))
        _, _, status, _, _, _ = results[-1]
        assert "Speaking" in status or "Thinking" in status  # either is valid mid-stream

    @pytest.mark.anyio
    async def test_no_api_call_during_speech_only(self, handler, vad, monkeypatch):
        """Live API must NOT be called until silence threshold is reached."""
        import app.ui.gradio_app as module
        call_count = 0

        async def mock_send_turn(pcm):
            nonlocal call_count
            call_count += 1
            from app.live.live_session_manager import LiveSessionResult
            return LiveSessionResult(b"\x00", "u", "a")

        monkeypatch.setattr(module, "send_turn", mock_send_turn)

        for _ in range(MIN_SPEECH_CHUNKS + 1):
            sr, data = _speech_chunk()
            async for _ in stream_audio_chunk((sr, data), [], handler, vad):
                pass

        assert call_count == 0

    @pytest.mark.anyio
    async def test_api_called_after_speech_then_silence(self, handler, vad, monkeypatch):
        """Live API must be called exactly once after enough silence follows speech."""
        import app.ui.gradio_app as module
        call_count = 0

        async def mock_send_turn(pcm):
            nonlocal call_count
            call_count += 1
            from app.live.live_session_manager import LiveSessionResult
            return LiveSessionResult(b"\x00\x01", "what is my balance", "Your balance is 100.")

        monkeypatch.setattr(module, "send_turn", mock_send_turn)

        # Enough speech
        for _ in range(MIN_SPEECH_CHUNKS):
            sr, data = _speech_chunk()
            async for _ in stream_audio_chunk((sr, data), [], handler, vad):
                pass

        # Enough silence to trigger
        for _ in range(SILENCE_CHUNKS_TO_TRIGGER):
            sr, data = _silence_chunk()
            async for _ in stream_audio_chunk((sr, data), [], handler, vad):
                pass

        assert call_count == 1

    @pytest.mark.anyio
    async def test_transcripts_added_after_successful_turn(self, handler, vad, monkeypatch):
        import app.ui.gradio_app as module
        from app.live.live_session_manager import LiveSessionResult

        async def mock_send_turn(pcm):
            return LiveSessionResult(b"\x00\x01", "what is my balance", "Your balance is 200.")

        monkeypatch.setattr(module, "send_turn", mock_send_turn)

        for _ in range(MIN_SPEECH_CHUNKS):
            sr, data = _speech_chunk()
            async for _ in stream_audio_chunk((sr, data), [], handler, vad):
                pass

        final_results = []
        for _ in range(SILENCE_CHUNKS_TO_TRIGGER):
            sr, data = _silence_chunk()
            async for item in stream_audio_chunk((sr, data), [], handler, vad):
                final_results.append(item)

        _, history, _, _, _, _ = final_results[-1]
        roles = [m["role"] for m in history]
        assert "user" in roles
        assert "assistant" in roles

    @pytest.mark.anyio
    async def test_error_result_shown_in_status(self, handler, vad, monkeypatch):
        import app.ui.gradio_app as module
        from app.live.live_session_manager import LiveSessionResult

        async def mock_send_turn(pcm):
            return LiveSessionResult(b"", "", "", error="timeout")

        monkeypatch.setattr(module, "send_turn", mock_send_turn)

        for _ in range(MIN_SPEECH_CHUNKS):
            sr, data = _speech_chunk()
            async for _ in stream_audio_chunk((sr, data), [], handler, vad):
                pass

        final_results = []
        for _ in range(SILENCE_CHUNKS_TO_TRIGGER):
            sr, data = _silence_chunk()
            async for item in stream_audio_chunk((sr, data), [], handler, vad):
                final_results.append(item)

        _, _, status, debug, _, _ = final_results[-1]
        assert "Error" in status
        assert "timeout" in debug.get("error", "")

    @pytest.mark.anyio
    async def test_vad_resets_after_send(self, handler, vad, monkeypatch):
        """After an utterance is sent, VAD state resets so next utterance is independent."""
        import app.ui.gradio_app as module
        from app.live.live_session_manager import LiveSessionResult

        async def mock_send_turn(pcm):
            return LiveSessionResult(b"\x00\x00", "hi", "hello")  # 2 bytes = valid int16

        monkeypatch.setattr(module, "send_turn", mock_send_turn)

        for _ in range(MIN_SPEECH_CHUNKS):
            sr, data = _speech_chunk()
            async for _ in stream_audio_chunk((sr, data), [], handler, vad):
                pass

        for _ in range(SILENCE_CHUNKS_TO_TRIGGER):
            sr, data = _silence_chunk()
            async for item in stream_audio_chunk((sr, data), [], handler, vad):
                _, _, _, _, _, vad = item  # pick up updated vad state

        assert not vad.is_speaking
        assert vad.buffer_chunks == []


# ---------------------------------------------------------------------------
# handle_clear
# ---------------------------------------------------------------------------

class TestHandleClear:
    def test_clear_returns_empty_history(self, handler):
        handler.add_user("hello")
        history, _, _, _ = handle_clear(handler)
        assert history == []

    def test_clear_returns_ready_status(self, handler):
        _, status, _, _ = handle_clear(handler)
        assert "Ready" in status or "cleared" in status.lower()

    def test_clear_resets_transcript_handler(self, handler):
        handler.add_user("hello")
        _, _, _, returned_handler = handle_clear(handler)
        assert returned_handler.get_history() == []


# ---------------------------------------------------------------------------
# create_app — Gradio 6.x API compatibility
# ---------------------------------------------------------------------------

class TestCreateApp:
    def test_create_app_does_not_raise(self):
        from app.ui.gradio_app import create_app
        import gradio as gr
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
