"""Tests for acoustic feature extraction (Phase 4)."""

import numpy as np
import pytest

from src.features import AcousticFeatureExtractor


@pytest.fixture
def config():
    return {
        "audio": {"sample_rate": 16000},
        "features": {
            "frame_ms": 25,
            "hop_ms": 10,
            "n_mfcc": 13,
            "f0_min": 75,
            "f0_max": 600,
        },
    }


@pytest.fixture
def extractor(config):
    return AcousticFeatureExtractor(config)


def _sine_wave(freq: float = 440.0, duration: float = 1.0, sr: int = 16000) -> np.ndarray:
    """Generate a pure sine wave."""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * freq * t)


def _silence(duration: float = 1.0, sr: int = 16000) -> np.ndarray:
    """Generate silence."""
    return np.zeros(int(sr * duration), dtype=np.float32)


class TestMFCC:

    def test_mfcc_shape(self, extractor):
        """MFCCs should be 39-dim (13 + delta + delta-delta)."""
        audio = _sine_wave(duration=1.0)
        mfcc = extractor.extract_mfcc(audio)
        assert mfcc is not None
        assert mfcc.shape[0] == 39  # 13 * 3

    def test_mfcc_short_audio_returns_none(self, extractor):
        """Audio shorter than 50ms should return None."""
        short = np.zeros(100, dtype=np.float32)  # ~6ms at 16kHz
        assert extractor.extract_mfcc(short) is None


class TestF0:

    def test_f0_sine_440(self, extractor):
        """F0 of a 440Hz sine wave should be ~440Hz."""
        audio = _sine_wave(freq=440.0, duration=1.0)
        f0 = extractor.extract_f0(audio)
        assert abs(f0["f0_mean"] - 440.0) < 5.0, f"Expected ~440Hz, got {f0['f0_mean']}"

    def test_f0_sine_200(self, extractor):
        """F0 of a 200Hz sine wave should be ~200Hz."""
        audio = _sine_wave(freq=200.0, duration=1.0)
        f0 = extractor.extract_f0(audio)
        assert abs(f0["f0_mean"] - 200.0) < 5.0, f"Expected ~200Hz, got {f0['f0_mean']}"

    def test_f0_silence_returns_zero(self, extractor):
        """Silence should return F0 mean of 0."""
        audio = _silence(duration=0.5)
        f0 = extractor.extract_f0(audio)
        assert f0["f0_mean"] == 0.0

    def test_f0_short_audio(self, extractor):
        """Very short audio returns defaults."""
        short = np.zeros(100, dtype=np.float32)
        f0 = extractor.extract_f0(short)
        assert f0["f0_mean"] == 0.0


class TestEnergy:

    def test_energy_silence_is_low(self, extractor):
        """Silence should have very low energy."""
        audio = _silence(duration=0.5)
        energy = extractor.extract_energy(audio)
        assert energy["rms_mean_db"] < -60  # very low dB

    def test_energy_sine_is_higher(self, extractor):
        """Sine wave should have higher energy than silence."""
        sine = _sine_wave(duration=0.5)
        silence = _silence(duration=0.5)
        e_sine = extractor.extract_energy(sine)
        e_silence = extractor.extract_energy(silence)
        assert e_sine["rms_mean_db"] > e_silence["rms_mean_db"]


class TestSpectral:

    def test_spectral_returns_arrays(self, extractor):
        """Spectral features should return non-empty arrays."""
        audio = _sine_wave(duration=0.5)
        spec = extractor.extract_spectral(audio)
        assert len(spec["centroid"]) > 0
        assert len(spec["bandwidth"]) > 0
        assert len(spec["rolloff"]) > 0

    def test_spectral_short_audio(self, extractor):
        """Too-short audio returns empty arrays."""
        short = np.zeros(100, dtype=np.float32)
        spec = extractor.extract_spectral(short)
        assert len(spec["centroid"]) == 0


class TestVoiceQuality:

    def test_voice_quality_silence(self, extractor):
        """Silence should return default voice quality."""
        audio = _silence(duration=0.5)
        vq = extractor.extract_voice_quality(audio)
        assert vq["jitter_percent"] == 0.0
        assert vq["shimmer_percent"] == 0.0


class TestExtractAll:

    def test_extract_all_keys(self, extractor):
        """extract_all should return all feature groups."""
        audio = _sine_wave(duration=1.0)
        result = extractor.extract_all(audio)
        assert "mfcc" in result
        assert "f0" in result
        assert "energy" in result
        assert "spectral" in result
        assert "voice_quality" in result


class TestWindowed:

    def test_windowed_count(self, extractor):
        """Windowed extraction should produce correct number of windows."""
        audio = _sine_wave(duration=5.0)
        windows = extractor.extract_windowed(audio, window_sec=2.0, hop_sec=1.0)
        # 5s audio, 2s window, 1s hop → windows at 0,1,2,3 = 4 windows
        assert len(windows) >= 4
        assert "start_sec" in windows[0]
        assert "end_sec" in windows[0]

    def test_windowed_timestamps_increase(self, extractor):
        """Window start times should be monotonically increasing."""
        audio = _sine_wave(duration=4.0)
        windows = extractor.extract_windowed(audio, window_sec=1.0, hop_sec=0.5)
        starts = [w["start_sec"] for w in windows]
        for i in range(1, len(starts)):
            assert starts[i] > starts[i - 1]
