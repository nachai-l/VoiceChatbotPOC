"""
audio_codec.py — Audio format conversion for the Live API.

Live API audio contracts (from requirements FR-1):
  Input  (browser → model):  16-bit PCM, 16 kHz, mono, little-endian
  Output (model → browser):  16-bit PCM, 24 kHz, mono, little-endian
"""

import numpy as np
from scipy import signal


def numpy_to_pcm16(sample_rate: int, data: np.ndarray) -> bytes:
    """Convert a Gradio audio capture to PCM bytes ready for the Live API.

    Args:
        sample_rate: Sample rate of the incoming numpy array (from gr.Audio).
        data: Audio samples as numpy array, float32 or int16, mono or stereo.

    Returns:
        Raw PCM bytes: int16, 16 kHz, mono, little-endian.
    """
    # Stereo → mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Normalise to float32 in [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype != np.float32:
        data = data.astype(np.float32)

    # Resample to 16 kHz if needed
    if sample_rate != 16000:
        target_samples = int(len(data) * 16000 / sample_rate)
        data = signal.resample(data, target_samples)

    # Clip and convert to int16
    data = np.clip(data, -1.0, 1.0)
    pcm = (data * 32767).astype(np.int16)
    return pcm.tobytes()


def pcm16_to_numpy(pcm_bytes: bytes, sample_rate: int = 24000) -> tuple[int, np.ndarray]:
    """Convert PCM bytes from the Live API to a Gradio-compatible numpy array.

    Args:
        pcm_bytes: Raw PCM bytes: int16, little-endian (from model output).
        sample_rate: Sample rate of the incoming PCM (default 24 kHz per Live API spec).

    Returns:
        Tuple of (sample_rate, float32 numpy array) as expected by gr.Audio.
    """
    if not pcm_bytes:
        return sample_rate, np.zeros(0, dtype=np.float32)

    data = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return sample_rate, data
