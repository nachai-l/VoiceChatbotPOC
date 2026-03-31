"""
Tests for live_session_manager.py.

Strategy:
- LiveSessionResult contract tests: pure unit, no mocking needed.
- send_turn tests: mock the google-genai client so no real API calls are made.
  We verify that our code correctly:
    - returns an error result when GEMINI_API_KEY is missing
    - calls send_realtime_input with audio + audio_stream_end
    - collects audio chunks from response.data
    - collects transcripts from response.server_content
    - stops iterating on turn_complete
    - returns an error result when the SDK raises an exception
"""

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.live.live_session_manager import LiveSessionResult, send_turn


# ---------------------------------------------------------------------------
# LiveSessionResult contract
# ---------------------------------------------------------------------------

class TestLiveSessionResult:
    def test_ok_when_no_error(self):
        r = LiveSessionResult(b"audio", "user said", "assistant said")
        assert r.ok is True

    def test_not_ok_when_error_set(self):
        r = LiveSessionResult(b"", "", "", error="something went wrong")
        assert r.ok is False

    def test_fields_accessible(self):
        r = LiveSessionResult(b"\x00\x01", "u", "a")
        assert r.audio_bytes == b"\x00\x01"
        assert r.user_transcript == "u"
        assert r.assistant_transcript == "a"
        assert r.error is None


# ---------------------------------------------------------------------------
# send_turn — missing API key
# ---------------------------------------------------------------------------

class TestSendTurnMissingKey:
    @pytest.mark.anyio
    async def test_returns_error_result_when_key_missing(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        result = await send_turn(b"\x00" * 100)
        assert result.ok is False
        assert "GEMINI_API_KEY" in result.error


# ---------------------------------------------------------------------------
# send_turn — SDK mocked
# ---------------------------------------------------------------------------

def _make_response(data=None, user_text=None, assistant_text=None, turn_complete=False):
    """Build a fake Live API response object."""
    sc = SimpleNamespace(
        input_transcription=SimpleNamespace(text=user_text) if user_text is not None else None,
        output_transcription=SimpleNamespace(text=assistant_text) if assistant_text is not None else None,
        turn_complete=turn_complete,
    )
    return SimpleNamespace(data=data, server_content=sc)


class MockLiveSession:
    """Async context manager that yields itself as the session."""

    def __init__(self, responses: list):
        self.responses = responses
        self.realtime_calls: list[dict] = []  # records each send_realtime_input call

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def send_realtime_input(self, *, audio=None, audio_stream_end=None, **kwargs):
        self.realtime_calls.append({"audio": audio, "audio_stream_end": audio_stream_end})

    async def receive(self):
        for r in self.responses:
            yield r


class TestSendTurnWithMockedSDK:
    def _patch_client(self, responses: list):
        mock_session = MockLiveSession(responses)
        mock_live = MagicMock()
        mock_live.connect.return_value = mock_session
        mock_aio = MagicMock()
        mock_aio.live = mock_live
        mock_client = MagicMock()
        mock_client.aio = mock_aio
        return mock_client, mock_session

    @pytest.mark.anyio
    async def test_collects_audio_chunks(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        responses = [
            _make_response(data=b"\x01\x02"),
            _make_response(data=b"\x03\x04", turn_complete=True),
        ]
        mock_client, _ = self._patch_client(responses)

        with patch("app.live.live_session_manager.genai.Client", return_value=mock_client):
            result = await send_turn(b"\x00" * 100)

        assert result.ok is True
        assert result.audio_bytes == b"\x01\x02\x03\x04"

    @pytest.mark.anyio
    async def test_captures_user_transcript(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        responses = [
            _make_response(user_text="what is my balance"),
            _make_response(turn_complete=True),
        ]
        mock_client, _ = self._patch_client(responses)

        with patch("app.live.live_session_manager.genai.Client", return_value=mock_client):
            result = await send_turn(b"\x00" * 100)

        assert result.user_transcript == "what is my balance"

    @pytest.mark.anyio
    async def test_captures_assistant_transcript(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        responses = [
            _make_response(assistant_text="Your balance is 500."),
            _make_response(turn_complete=True),
        ]
        mock_client, _ = self._patch_client(responses)

        with patch("app.live.live_session_manager.genai.Client", return_value=mock_client):
            result = await send_turn(b"\x00" * 100)

        assert result.assistant_transcript == "Your balance is 500."

    @pytest.mark.anyio
    async def test_stops_on_turn_complete(self, monkeypatch):
        """Responses after turn_complete must not be collected."""
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        responses = [
            _make_response(data=b"\x01", turn_complete=True),
            _make_response(data=b"\xff"),  # should never be reached
        ]
        mock_client, _ = self._patch_client(responses)

        with patch("app.live.live_session_manager.genai.Client", return_value=mock_client):
            result = await send_turn(b"\x00" * 100)

        assert result.audio_bytes == b"\x01"

    @pytest.mark.anyio
    async def test_send_realtime_input_called_with_audio_and_stream_end(self, monkeypatch):
        """Regression: must use send_realtime_input (not send_client_content) for audio."""
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        responses = [_make_response(turn_complete=True)]
        mock_client, mock_session = self._patch_client(responses)

        with patch("app.live.live_session_manager.genai.Client", return_value=mock_client):
            await send_turn(b"\x00" * 100)

        calls = mock_session.realtime_calls
        # First call: send audio blob
        assert calls[0]["audio"] is not None
        assert calls[0]["audio_stream_end"] is None
        # Second call: signal end-of-stream
        assert calls[1]["audio_stream_end"] is True

    @pytest.mark.anyio
    async def test_sdk_exception_returns_error_result(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        mock_client = MagicMock()
        mock_client.aio.live.connect.side_effect = RuntimeError("network error")

        with patch("app.live.live_session_manager.genai.Client", return_value=mock_client):
            result = await send_turn(b"\x00" * 100)

        assert result.ok is False
        assert "network error" in result.error


# ---------------------------------------------------------------------------
# _has_meaningful_output
# ---------------------------------------------------------------------------

class TestHasMeaningfulOutput:
    def test_empty_result_has_no_output(self):
        from app.live.live_session_manager import _has_meaningful_output
        r = LiveSessionResult(b"", "", "")
        assert not _has_meaningful_output(r)

    def test_audio_only_is_meaningful(self):
        from app.live.live_session_manager import _has_meaningful_output
        r = LiveSessionResult(b"\x01\x02", "", "")
        assert _has_meaningful_output(r)

    def test_user_transcript_only_is_meaningful(self):
        from app.live.live_session_manager import _has_meaningful_output
        r = LiveSessionResult(b"", "hello", "")
        assert _has_meaningful_output(r)

    def test_assistant_transcript_only_is_meaningful(self):
        from app.live.live_session_manager import _has_meaningful_output
        r = LiveSessionResult(b"", "", "hi there")
        assert _has_meaningful_output(r)


# ---------------------------------------------------------------------------
# send_turn — retry logic
# ---------------------------------------------------------------------------

class TestSendTurnRetry:
    def _patch_client(self, responses: list):
        mock_session = MockLiveSession(responses)
        mock_live = MagicMock()
        mock_live.connect.return_value = mock_session
        mock_aio = MagicMock()
        mock_aio.live = mock_live
        mock_client = MagicMock()
        mock_client.aio = mock_aio
        return mock_client, mock_session

    @pytest.mark.anyio
    async def test_retries_on_empty_response(self, monkeypatch):
        """When first attempt returns empty, a retry should be attempted."""
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        monkeypatch.setenv("LIVE_EMPTY_RESPONSE_RETRY_COUNT", "1")
        monkeypatch.setenv("LIVE_RETRY_BACKOFF_SECONDS", "0.01")

        call_count = 0

        class RetryMockSession(MockLiveSession):
            async def receive(self_inner):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # First attempt: empty turn_complete
                    yield _make_response(turn_complete=True)
                else:
                    # Second attempt: has audio
                    yield _make_response(data=b"\x01\x02", turn_complete=True)

        mock_session = RetryMockSession([])
        mock_live = MagicMock()
        mock_live.connect.return_value = mock_session
        mock_aio = MagicMock()
        mock_aio.live = mock_live
        mock_client = MagicMock()
        mock_client.aio = mock_aio

        with patch("app.live.live_session_manager.genai.Client", return_value=mock_client):
            result = await send_turn(b"\x00" * 100)

        assert call_count == 2
        assert result.ok is True
        assert result.audio_bytes == b"\x01\x02"

    @pytest.mark.anyio
    async def test_no_retry_when_partial_output_exists(self, monkeypatch):
        """Do not retry if partial output was already collected."""
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        monkeypatch.setenv("LIVE_EMPTY_RESPONSE_RETRY_COUNT", "1")

        responses = [
            _make_response(user_text="hello"),
            _make_response(turn_complete=True),
        ]
        mock_client, _ = self._patch_client(responses)

        with patch("app.live.live_session_manager.genai.Client", return_value=mock_client):
            result = await send_turn(b"\x00" * 100)

        assert result.ok is True
        assert result.user_transcript == "hello"

    @pytest.mark.anyio
    async def test_empty_response_after_all_retries_returns_error(self, monkeypatch):
        """After exhausting retries with empty responses, return an error."""
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        monkeypatch.setenv("LIVE_EMPTY_RESPONSE_RETRY_COUNT", "1")
        monkeypatch.setenv("LIVE_RETRY_BACKOFF_SECONDS", "0.01")

        # Always return empty
        responses = [_make_response(turn_complete=True)]
        mock_client, _ = self._patch_client(responses)

        with patch("app.live.live_session_manager.genai.Client", return_value=mock_client):
            result = await send_turn(b"\x00" * 100)

        assert result.ok is False
        assert "empty response" in result.error


# ---------------------------------------------------------------------------
# _build_live_config
# ---------------------------------------------------------------------------

class TestBuildLiveConfig:
    def test_config_has_audio_modality(self):
        from app.live.live_session_manager import _build_live_config
        config = _build_live_config()
        assert "AUDIO" in config.response_modalities

    def test_config_has_voice_name(self):
        from app.live.live_session_manager import _build_live_config
        config = _build_live_config()
        assert config.speech_config.voice_config.prebuilt_voice_config.voice_name is not None

    def test_config_has_transcription_enabled(self):
        from app.live.live_session_manager import _build_live_config
        config = _build_live_config()
        assert config.input_audio_transcription is not None
        assert config.output_audio_transcription is not None
