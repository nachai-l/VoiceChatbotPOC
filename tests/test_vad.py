"""
Tests for app/live/vad.py — energy-based Voice Activity Detection.

VAD state machine:
  idle → speaking (energy > threshold)
  speaking → silence (energy drops)
  silence × N chunks → should_send = True → reset
"""

import numpy as np
import pytest

from app.live.vad import (
    VADState,
    SPEECH_ENERGY_THRESHOLD,
    SILENCE_CHUNKS_TO_TRIGGER,
    MIN_SPEECH_CHUNKS,
    compute_rms,
    process_chunk,
    get_buffer_array,
    reset_vad,
)

SR = 16000
CHUNK = SR // 4  # 0.25 s chunk (4 per second)


def _speech_chunk(amplitude: float = 0.5) -> np.ndarray:
    """Sine wave chunk guaranteed to be above threshold."""
    t = np.linspace(0, 0.25, CHUNK, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t) * amplitude


def _silence_chunk() -> np.ndarray:
    """Near-silent chunk guaranteed to be below threshold."""
    return np.zeros(CHUNK, dtype=np.float32)


# ---------------------------------------------------------------------------
# compute_rms
# ---------------------------------------------------------------------------

class TestComputeRms:
    def test_silence_is_zero(self):
        assert compute_rms(_silence_chunk()) == pytest.approx(0.0)

    def test_sine_wave_rms(self):
        # RMS of sin(x)*A = A / sqrt(2)
        rms = compute_rms(_speech_chunk(amplitude=1.0))
        assert rms == pytest.approx(1.0 / (2 ** 0.5), abs=0.01)

    def test_empty_array_returns_zero(self):
        assert compute_rms(np.array([], dtype=np.float32)) == 0.0


# ---------------------------------------------------------------------------
# process_chunk — state transitions
# ---------------------------------------------------------------------------

class TestProcessChunkTransitions:
    def test_silence_when_not_speaking_does_not_set_is_speaking(self):
        state = VADState()
        state, should_send = process_chunk(state, SR, _silence_chunk())
        assert not state.is_speaking
        assert not should_send

    def test_speech_chunk_sets_is_speaking(self):
        state = VADState()
        state, should_send = process_chunk(state, SR, _speech_chunk())
        assert state.is_speaking
        assert not should_send

    def test_speech_chunk_appended_to_buffer(self):
        state = VADState()
        state, _ = process_chunk(state, SR, _speech_chunk())
        assert len(state.buffer_chunks) == 1

    def test_silence_before_speech_not_buffered(self):
        state = VADState()
        state, _ = process_chunk(state, SR, _silence_chunk())
        assert len(state.buffer_chunks) == 0

    def test_silence_after_speech_is_buffered(self):
        """Trailing silence is included to preserve natural speech cadence."""
        state = VADState()
        state, _ = process_chunk(state, SR, _speech_chunk())
        state, _ = process_chunk(state, SR, _silence_chunk())
        assert len(state.buffer_chunks) == 2

    def test_should_send_after_enough_silence_chunks(self):
        state = VADState()
        # Enough speech
        for _ in range(MIN_SPEECH_CHUNKS):
            state, _ = process_chunk(state, SR, _speech_chunk())
        # Enough silence
        for _ in range(SILENCE_CHUNKS_TO_TRIGGER):
            state, should_send = process_chunk(state, SR, _silence_chunk())
        assert should_send

    def test_does_not_send_before_enough_silence(self):
        state = VADState()
        for _ in range(MIN_SPEECH_CHUNKS):
            state, _ = process_chunk(state, SR, _speech_chunk())
        # One fewer than needed
        for _ in range(SILENCE_CHUNKS_TO_TRIGGER - 1):
            state, should_send = process_chunk(state, SR, _silence_chunk())
        assert not should_send

    def test_does_not_send_with_too_little_speech(self):
        """Reject very short sounds (click, noise) below MIN_SPEECH_CHUNKS."""
        state = VADState()
        for _ in range(MIN_SPEECH_CHUNKS - 1):
            state, _ = process_chunk(state, SR, _speech_chunk())
        for _ in range(SILENCE_CHUNKS_TO_TRIGGER):
            state, should_send = process_chunk(state, SR, _silence_chunk())
        assert not should_send

    def test_speech_resets_silence_counter(self):
        """If user pauses briefly then continues speaking, silence counter resets."""
        state = VADState()
        for _ in range(MIN_SPEECH_CHUNKS):
            state, _ = process_chunk(state, SR, _speech_chunk())
        # Partial silence — not enough to trigger
        for _ in range(SILENCE_CHUNKS_TO_TRIGGER - 1):
            state, _ = process_chunk(state, SR, _silence_chunk())
        # Speech resumes — silence counter must reset
        state, should_send = process_chunk(state, SR, _speech_chunk())
        assert not should_send
        assert state.silence_chunks == 0


# ---------------------------------------------------------------------------
# get_buffer_array
# ---------------------------------------------------------------------------

class TestGetBufferArray:
    def test_empty_state_returns_none(self):
        state = VADState()
        assert get_buffer_array(state) is None

    def test_concatenates_chunks(self):
        state = VADState()
        for _ in range(3):
            state, _ = process_chunk(state, SR, _speech_chunk())
        arr = get_buffer_array(state)
        assert arr is not None
        assert len(arr) == CHUNK * 3


# ---------------------------------------------------------------------------
# reset_vad
# ---------------------------------------------------------------------------

class TestResetVad:
    def test_clears_buffer(self):
        state = VADState()
        state, _ = process_chunk(state, SR, _speech_chunk())
        state = reset_vad(state)
        assert state.buffer_chunks == []

    def test_clears_is_speaking(self):
        state = VADState()
        state, _ = process_chunk(state, SR, _speech_chunk())
        state = reset_vad(state)
        assert not state.is_speaking

    def test_preserves_sample_rate(self):
        state = VADState(buffer_sr=44100)
        state = reset_vad(state)
        assert state.buffer_sr == 44100
