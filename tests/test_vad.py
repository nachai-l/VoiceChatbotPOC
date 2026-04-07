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

    def test_int16_normalized_same_as_float32(self):
        """Regression: Gradio streaming sends int16 — RMS must normalize to [0,1]."""
        float_data = _speech_chunk(amplitude=0.5)
        int16_data = (float_data * 32767).astype(np.int16)
        rms_float = compute_rms(float_data)
        rms_int16 = compute_rms(int16_data)
        assert rms_int16 == pytest.approx(rms_float, abs=0.01)

    def test_int16_silence_below_threshold(self):
        """Regression: int16 silence must have RMS near 0 (not 250+)."""
        silence = np.zeros(4000, dtype=np.int16)
        assert compute_rms(silence) == pytest.approx(0.0)


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


# ---------------------------------------------------------------------------
# VAD constants sanity checks
# ---------------------------------------------------------------------------

class TestVADConstants:
    """Regression: constants must stay in sync with tuned values."""

    def test_speech_threshold_is_low_enough_for_real_mic(self):
        """Observed real browser mic RMS during speech is ~0.004–0.01."""
        assert SPEECH_ENERGY_THRESHOLD <= 0.01

    def test_silence_trigger_gives_responsive_turn_taking(self):
        """4 chunks × 0.25s = 1.0s — must be <= 1.5s for conversational feel."""
        assert SILENCE_CHUNKS_TO_TRIGGER * 0.25 <= 1.5

    def test_min_speech_chunks_allows_short_utterances(self):
        """MIN_SPEECH_CHUNKS=1 allows short follow-up replies like 'yes'."""
        assert MIN_SPEECH_CHUNKS >= 1
        assert MIN_SPEECH_CHUNKS <= 2


# ---------------------------------------------------------------------------
# VADState.pending_task field
# ---------------------------------------------------------------------------

class TestVADStatePendingTask:
    def test_default_pending_task_is_none(self):
        state = VADState()
        assert state.pending_task is None

    def test_reset_clears_pending_task(self):
        """reset_vad creates a fresh state; pending_task must be None."""
        state = VADState()
        state.pending_task = "some_task"
        state = reset_vad(state)
        assert state.pending_task is None


# ---------------------------------------------------------------------------
# VADState.ignore_until — regression: must be a proper dataclass field
# ---------------------------------------------------------------------------

class TestVADStateIgnoreUntil:
    def test_ignore_until_is_dataclass_field(self):
        """Regression: ignore_until was previously injected via setattr, making it
        absent from fresh VADState instances created by reset_vad.
        It must be a proper dataclass field so every instance has it from init."""
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(VADState)}
        assert "ignore_until" in field_names

    def test_default_ignore_until_is_zero(self):
        state = VADState()
        assert state.ignore_until == 0.0

    def test_reset_vad_clears_ignore_until(self):
        """Regression: reset_vad must return a state with ignore_until == 0.0.
        Previously, a setattr-based field was lost after reset_vad, leaving the
        new state without the attribute entirely (relying on getattr fallback)."""
        state = VADState()
        state.ignore_until = 999.0
        state = reset_vad(state)
        assert state.ignore_until == 0.0

    def test_fresh_reset_does_not_require_getattr_fallback(self):
        """Direct attribute access (not getattr) must work after reset_vad."""
        state = VADState()
        state.ignore_until = 5.0
        state = reset_vad(state)
        # This would AttributeError if ignore_until were not a dataclass field
        _ = state.ignore_until
