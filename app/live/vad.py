"""
vad.py — Energy-based Voice Activity Detection.

State machine:
  idle      — no speech yet, chunks ignored
  speaking  — energy above threshold, accumulating buffer
  silence   — energy below threshold after speech, counting down
  → send    — silence counter hits threshold, caller should send buffer

Tuning constants (adjust based on microphone gain and environment):
  SPEECH_ENERGY_THRESHOLD  — RMS level that counts as speech
  SILENCE_CHUNKS_TO_TRIGGER — consecutive silent chunks before sending (~1.5 s at 0.25 s/chunk)
  MIN_SPEECH_CHUNKS         — minimum speech chunks to reject noise clicks
"""

from dataclasses import dataclass, field

import numpy as np

SPEECH_ENERGY_THRESHOLD: float = 0.01
SILENCE_CHUNKS_TO_TRIGGER: int = 6   # 6 × 0.25 s = 1.5 s of silence
MIN_SPEECH_CHUNKS: int = 2           # ~0.5 s minimum utterance


@dataclass
class VADState:
    buffer_sr: int = 16000
    buffer_chunks: list = field(default_factory=list)
    silence_chunks: int = 0
    speech_chunks: int = 0
    is_speaking: bool = False


def compute_rms(data: np.ndarray) -> float:
    """Return the RMS energy of an audio chunk."""
    if len(data) == 0:
        return 0.0
    return float(np.sqrt(np.mean(data.astype(np.float32) ** 2)))


def process_chunk(
    state: VADState, sample_rate: int, chunk: np.ndarray
) -> tuple[VADState, bool]:
    """Feed one audio chunk into the VAD state machine.

    Returns:
        (updated_state, should_send)
        should_send=True means an utterance ended — caller should read the buffer,
        send it to the Live API, then call reset_vad().
    """
    state.buffer_sr = sample_rate
    rms = compute_rms(chunk)

    if rms >= SPEECH_ENERGY_THRESHOLD:
        state.is_speaking = True
        state.speech_chunks += 1
        state.silence_chunks = 0
        state.buffer_chunks.append(chunk)
        return state, False

    # Silent chunk
    if not state.is_speaking:
        return state, False

    state.silence_chunks += 1
    state.buffer_chunks.append(chunk)  # keep trailing silence for natural cadence

    if (
        state.silence_chunks >= SILENCE_CHUNKS_TO_TRIGGER
        and state.speech_chunks >= MIN_SPEECH_CHUNKS
    ):
        return state, True

    return state, False


def get_buffer_array(state: VADState) -> np.ndarray | None:
    """Concatenate all buffered chunks into a single numpy array, or None if empty."""
    if not state.buffer_chunks:
        return None
    return np.concatenate(state.buffer_chunks)


def reset_vad(state: VADState) -> VADState:
    """Return a fresh VADState, preserving only the sample rate."""
    return VADState(buffer_sr=state.buffer_sr)
