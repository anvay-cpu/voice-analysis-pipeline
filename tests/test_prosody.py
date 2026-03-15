"""Tests for prosody analysis (Phase 7)."""

import numpy as np
import pytest

from src.prosody import (
    estimate_syllable_rate,
    classify_rate,
    assess_pitch_dynamics,
    assess_volume_dynamics,
    analyze_prosody,
    ProsodyResult,
)


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
        "prosody": {
            "rate_window_sec": 10.0,
            "optimal_rate_range": [3.5, 5.0],
            "pitch_cv_optimal": [0.15, 0.25],
        },
    }


def _sine_wave(freq: float = 440.0, duration: float = 1.0,
               sr: int = 16000) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * freq * t)


def _silence(duration: float = 1.0, sr: int = 16000) -> np.ndarray:
    return np.zeros(int(sr * duration), dtype=np.float32)


def _clicks(rate: float = 4.0, duration: float = 5.0,
            sr: int = 16000) -> np.ndarray:
    """Generate periodic clicks at a given rate (clicks/sec).

    Simulates syllable nuclei for rate estimation testing.
    """
    audio = np.zeros(int(sr * duration), dtype=np.float32)
    interval = int(sr / rate)
    click_len = int(sr * 0.03)  # 30ms click

    for i in range(0, len(audio) - click_len, interval):
        # Each click is a short burst of energy
        t = np.linspace(0, 0.03, click_len, dtype=np.float32)
        click = 0.5 * np.sin(2 * np.pi * 300 * t) * np.hanning(click_len)
        audio[i:i + click_len] += click

    return audio


class TestEstimateSyllableRate:

    def test_clicks_at_4hz(self):
        """4 clicks/sec should give rate near 4.0 syl/sec."""
        audio = _clicks(rate=4.0, duration=10.0)
        rates = estimate_syllable_rate(audio, sr=16000, window_sec=10.0)
        assert len(rates) >= 1
        assert 2.5 <= rates[0] <= 6.0, f"Expected ~4.0, got {rates[0]}"

    def test_clicks_at_2hz(self):
        """2 clicks/sec should give lower rate than 4 clicks/sec."""
        slow = _clicks(rate=2.0, duration=10.0)
        fast = _clicks(rate=4.0, duration=10.0)
        slow_rates = estimate_syllable_rate(slow, sr=16000, window_sec=10.0)
        fast_rates = estimate_syllable_rate(fast, sr=16000, window_sec=10.0)
        assert slow_rates[0] < fast_rates[0]

    def test_silence_returns_zero(self):
        """Silence should produce 0 syllable rate."""
        audio = _silence(duration=10.0)
        rates = estimate_syllable_rate(audio, sr=16000, window_sec=10.0)
        assert rates[0] == 0.0

    def test_very_short_audio(self):
        """Very short audio should return a rate list."""
        audio = np.zeros(100, dtype=np.float32)
        rates = estimate_syllable_rate(audio, sr=16000, window_sec=10.0)
        assert len(rates) >= 1


class TestClassifyRate:

    def test_too_slow(self):
        assert classify_rate(1.5) == "Too slow"

    def test_slow(self):
        assert classify_rate(3.0) == "Slow"

    def test_optimal(self):
        assert classify_rate(4.0) == "Optimal"

    def test_fast(self):
        assert classify_rate(6.0) == "Fast"

    def test_too_fast(self):
        assert classify_rate(7.5) == "Too fast"

    def test_boundary_optimal_low(self):
        assert classify_rate(3.5) == "Optimal"

    def test_boundary_optimal_high(self):
        assert classify_rate(5.0) == "Optimal"


class TestAssessPitchDynamics:

    def test_monotone_low_cv(self):
        """Low CV should be classified as monotone."""
        f0_data = {"f0_cv": 0.05, "f0_range": 10.0, "f0_mean": 150.0}
        result = assess_pitch_dynamics(f0_data)
        assert result["assessment"] == "Very monotone"
        assert result["score"] < 50

    def test_good_variation(self):
        """CV in optimal range should be good."""
        f0_data = {"f0_cv": 0.20, "f0_range": 80.0, "f0_mean": 200.0}
        result = assess_pitch_dynamics(f0_data)
        assert result["assessment"] == "Good variation"
        assert result["score"] >= 80

    def test_erratic_high_cv(self):
        """Very high CV should be classified as erratic."""
        f0_data = {"f0_cv": 0.50, "f0_range": 200.0, "f0_mean": 180.0}
        result = assess_pitch_dynamics(f0_data)
        assert result["assessment"] == "Erratic"
        assert result["score"] < 50

    def test_no_speech(self):
        """Zero f0_mean should indicate no speech."""
        f0_data = {"f0_cv": 0.0, "f0_range": 0.0, "f0_mean": 0.0}
        result = assess_pitch_dynamics(f0_data)
        assert result["assessment"] == "No speech detected"
        assert result["score"] == 0

    def test_constant_pitch_sine(self):
        """A pure sine wave has near-zero CV — monotone."""
        f0_data = {"f0_cv": 0.001, "f0_range": 0.5, "f0_mean": 440.0}
        result = assess_pitch_dynamics(f0_data)
        assert "monotone" in result["assessment"].lower()


class TestAssessVolumeDynamics:

    def test_good_dynamics(self):
        """Dynamic range in optimal range should score well."""
        energy = {"dynamic_range_db": 15.0, "rms_mean_db": -20.0}
        result = assess_volume_dynamics(energy)
        assert result["assessment"] == "Good dynamics"
        assert result["score"] >= 80
        assert len(result["issues"]) == 0

    def test_flat_volume(self):
        """Low dynamic range should be flagged as flat."""
        energy = {"dynamic_range_db": 3.0, "rms_mean_db": -20.0}
        result = assess_volume_dynamics(energy)
        assert result["assessment"] == "Flat"
        assert result["score"] < 50

    def test_uneven_volume(self):
        """Extreme dynamic range should be flagged."""
        energy = {"dynamic_range_db": 50.0, "rms_mean_db": -15.0}
        result = assess_volume_dynamics(energy)
        assert result["assessment"] == "Uneven"
        assert result["score"] < 50

    def test_silent_input(self):
        """Very low mean dB should indicate no speech."""
        energy = {"dynamic_range_db": 0.0, "rms_mean_db": -80.0}
        result = assess_volume_dynamics(energy)
        assert result["assessment"] == "No speech detected"


class TestAnalyzeProsody:

    def test_returns_prosody_results(self, config):
        """analyze_prosody should return list of ProsodyResult."""
        audio = _sine_wave(freq=200.0, duration=10.0)
        results = analyze_prosody(audio, None, config)
        assert len(results) >= 1
        assert isinstance(results[0], ProsodyResult)

    def test_result_fields(self, config):
        """ProsodyResult should have all expected fields."""
        audio = _sine_wave(freq=200.0, duration=10.0)
        results = analyze_prosody(audio, None, config)
        r = results[0]
        assert r.start_sec >= 0.0
        assert r.end_sec > r.start_sec
        assert isinstance(r.syllable_rate, float)
        assert isinstance(r.rate_label, str)
        assert isinstance(r.pitch_cv, float)
        assert isinstance(r.pitch_score, int)
        assert isinstance(r.volume_score, int)

    def test_silence_produces_results(self, config):
        """Silence should still produce results (with low scores)."""
        audio = _silence(duration=10.0)
        results = analyze_prosody(audio, None, config)
        assert len(results) >= 1
        assert results[0].pitch_score == 0
