"""
vad.py — Energy-based Voice Activity Detection.

State machine:
  idle      — no speech yet, chunks ignored
  speaking  — energy above threshold, accumulating buffer
  silence   — energy below threshold after speech, counting down
  → send    — silence counter hits threshold, caller should send buffer

Tuning notes:
  - Lower threshold because real browser/mic RMS is much lower than expected.
  - Require only 1 speech chunk so short follow-up utterances are not dropped.
  - Keep a short silence window so turn-taking still feels responsive.
"""

from dataclasses import dataclass, field

import numpy as np

# Lower than before because observed real RMS was around 0.0045
SPEECH_ENERGY_THRESHOLD: float = 0.006

# 4 × 0.25 s = 1.0 s silence before sending
SILENCE_CHUNKS_TO_TRIGGER: int = 4

# Allow shorter follow-up turns
MIN_SPEECH_CHUNKS: int = 1


@dataclass
class VADState:
    buffer_sr: int = 16000
    buffer_chunks: list = field(default_factory=list)
    silence_chunks: int = 0
    speech_chunks: int = 0
    is_speaking: bool = False
    pending_task: object = None  # asyncio.Task for in-flight API call
    ignore_until: float = 0.0   # monotonic timestamp: ignore mic input until this time
    trace_log: list = field(default_factory=list)  # sliding window of trace dicts


def compute_rms(data: np.ndarray) -> float:
    """Return RMS energy normalized to [0.0, 1.0]."""
    if data is None or len(data) == 0:
        return 0.0

    arr = np.asarray(data, dtype=np.float32)

    # Normalize int16-style inputs
    if np.max(np.abs(arr)) > 1.0:
        arr = arr / 32768.0

    return float(np.sqrt(np.mean(arr ** 2)))


def process_chunk(
    state: VADState, sample_rate: int, chunk: np.ndarray
) -> tuple[VADState, bool]:
    """Feed one audio chunk into the VAD state machine.

    Returns:
        (updated_state, should_send)
    """
    state.buffer_sr = sample_rate
    rms = compute_rms(chunk)

    if rms >= SPEECH_ENERGY_THRESHOLD:
        state.is_speaking = True
        state.speech_chunks += 1
        state.silence_chunks = 0
        state.buffer_chunks.append(chunk)
        return state, False

    # Silent chunk before speech starts: ignore
    if not state.is_speaking:
        return state, False

    # Silent chunk after speech started: keep it for natural cadence
    state.silence_chunks += 1
    state.buffer_chunks.append(chunk)

    if (
        state.silence_chunks >= SILENCE_CHUNKS_TO_TRIGGER
        and state.speech_chunks >= MIN_SPEECH_CHUNKS
    ):
        return state, True

    return state, False


def get_buffer_array(state: VADState) -> np.ndarray | None:
    """Concatenate all buffered chunks into a single array, or None if empty."""
    if not state.buffer_chunks:
        return None
    return np.concatenate(state.buffer_chunks)


def reset_vad(state: VADState) -> VADState:
    """Return a fresh VADState, preserving sample rate and trace log across turns."""
    new_state = VADState(buffer_sr=state.buffer_sr)
    new_state.trace_log = state.trace_log  # preserve trace history across turns
    return new_state