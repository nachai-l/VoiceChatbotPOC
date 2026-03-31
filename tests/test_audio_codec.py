import numpy as np
import pytest

from app.live.audio_codec import numpy_to_pcm16, pcm16_to_numpy


class TestNumpyToPcm16:
    def test_returns_bytes(self, mono_audio_16k):
        sr, data = mono_audio_16k
        result = numpy_to_pcm16(sr, data)
        assert isinstance(result, bytes)

    def test_correct_byte_length_mono_16k(self, mono_audio_16k):
        """1 second at 16 kHz = 16000 samples × 2 bytes each."""
        sr, data = mono_audio_16k
        result = numpy_to_pcm16(sr, data)
        assert len(result) == len(data) * 2

    def test_stereo_collapsed_to_mono(self, stereo_audio_44k):
        """Stereo input must produce mono output (half the channels)."""
        sr, data = stereo_audio_44k
        result = numpy_to_pcm16(sr, data)
        # After stereo→mono + resample to 16k: sample count ≈ n_samples * 16000 / sr
        expected_samples = int(data.shape[0] * 16000 / sr)
        actual_samples = len(result) // 2
        assert abs(actual_samples - expected_samples) <= 2  # allow ±1 sample rounding

    def test_resamples_to_16k(self, stereo_audio_44k):
        """Input at 44100 Hz must be resampled to 16000 Hz."""
        sr, data = stereo_audio_44k
        result_44k = numpy_to_pcm16(sr, data)
        result_16k = numpy_to_pcm16(16000, data[:, 0])  # mono at 16k for comparison
        # 44k stereo resampled should be much shorter than original sample count
        assert len(result_44k) // 2 < data.shape[0]

    def test_int16_input_accepted(self, int16_audio_16k):
        """int16 numpy arrays must be handled without error."""
        sr, data = int16_audio_16k
        result = numpy_to_pcm16(sr, data)
        assert isinstance(result, bytes)
        assert len(result) == len(data) * 2

    def test_output_values_clamped(self):
        """Values outside [-1, 1] in float input must not overflow int16."""
        sr = 16000
        data = np.array([2.0, -3.0, 0.5], dtype=np.float32)
        result = numpy_to_pcm16(sr, data)
        decoded = np.frombuffer(result, dtype=np.int16)
        assert decoded[0] == 32767   # clipped at max
        assert decoded[1] == -32767  # clipped at min (np.clip gives -1.0 → int16 -32767)
        assert -32768 <= decoded[2] <= 32767

    def test_empty_array_returns_empty_bytes(self):
        data = np.array([], dtype=np.float32)
        result = numpy_to_pcm16(16000, data)
        assert result == b""


class TestPcm16ToNumpy:
    def test_returns_tuple(self):
        pcm = np.zeros(100, dtype=np.int16).tobytes()
        sr, data = pcm16_to_numpy(pcm)
        assert isinstance(sr, int)
        assert isinstance(data, np.ndarray)

    def test_default_sample_rate_is_24k(self):
        pcm = np.zeros(100, dtype=np.int16).tobytes()
        sr, _ = pcm16_to_numpy(pcm)
        assert sr == 24000

    def test_custom_sample_rate_respected(self):
        pcm = np.zeros(100, dtype=np.int16).tobytes()
        sr, _ = pcm16_to_numpy(pcm, sample_rate=16000)
        assert sr == 16000

    def test_output_dtype_is_float32(self):
        pcm = np.zeros(100, dtype=np.int16).tobytes()
        _, data = pcm16_to_numpy(pcm)
        assert data.dtype == np.float32

    def test_sample_count_matches(self):
        """100 int16 samples → 100 float32 samples."""
        samples = np.random.randint(-1000, 1000, 100, dtype=np.int16)
        _, data = pcm16_to_numpy(samples.tobytes())
        assert len(data) == 100

    def test_values_normalised_to_float_range(self):
        """int16 max (32767) should decode close to +1.0."""
        samples = np.array([32767, -32768, 0], dtype=np.int16)
        _, data = pcm16_to_numpy(samples.tobytes())
        assert data[0] == pytest.approx(1.0, abs=1e-4)
        assert data[1] == pytest.approx(-1.0, abs=1e-4)
        assert data[2] == pytest.approx(0.0, abs=1e-4)

    def test_empty_bytes_returns_empty_array(self):
        sr, data = pcm16_to_numpy(b"")
        assert len(data) == 0

    def test_roundtrip_mono_16k(self, mono_audio_16k):
        """Encode then decode should recover the signal within float32 precision."""
        sr, original = mono_audio_16k
        pcm = numpy_to_pcm16(sr, original)
        _, recovered = pcm16_to_numpy(pcm, sample_rate=sr)
        # int16 quantisation noise: max error is 1/32767 ≈ 3e-5
        assert np.max(np.abs(original - recovered)) < 1e-3
