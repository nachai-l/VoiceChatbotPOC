import numpy as np
import pytest


@pytest.fixture
def mono_audio_16k():
    """1-second mono sine wave at 16 kHz (float32)."""
    sr = 16000
    t = np.linspace(0, 1, sr, dtype=np.float32)
    data = np.sin(2 * np.pi * 440 * t)
    return sr, data


@pytest.fixture
def stereo_audio_44k():
    """0.5-second stereo sine wave at 44100 Hz (float32)."""
    sr = 44100
    n = sr // 2
    t = np.linspace(0, 0.5, n, dtype=np.float32)
    left = np.sin(2 * np.pi * 440 * t)
    right = np.sin(2 * np.pi * 880 * t)
    return sr, np.stack([left, right], axis=1)


@pytest.fixture
def int16_audio_16k():
    """1-second mono sine wave at 16 kHz (int16)."""
    sr = 16000
    t = np.linspace(0, 1, sr, dtype=np.float32)
    data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    return sr, data
