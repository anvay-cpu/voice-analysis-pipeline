"""Prosody analysis: speaking rate, pitch variation, volume dynamics."""

import numpy as np
import librosa
from dataclasses import dataclass

from src.features import AcousticFeatureExtractor


@dataclass
class ProsodyResult:
    """Prosody analysis result for a single window."""
    start_sec: float
    end_sec: float
    syllable_rate: float
    rate_label: str
    pitch_cv: float
    pitch_range: float
    pitch_mean: float
    pitch_assessment: str
    pitch_score: int
    volume_dynamic_range: float
    volume_mean_db: float
    volume_assessment: str
    volume_score: int


def estimate_syllable_rate(audio: np.ndarray, sr: int = 16000,
                           window_sec: float = 10.0) -> list[float]:
    """Estimate syllable rate per window using energy-peak counting.

    Uses the Mermelstein method: bandpass filter, compute energy envelope,
    find peaks corresponding to syllable nuclei.

    Args:
        audio: Audio signal as float32 numpy array.
        sr: Sample rate.
        window_sec: Window size in seconds.

    Returns:
        List of syllable rates (syl/sec) per window.
    """
    if len(audio) < int(sr * 0.1):
        return [0.0]

    # Bandpass filter 100-4000 Hz to focus on speech energy
    audio_filtered = _bandpass_filter(audio, sr, low=100, high=4000)

    # Compute smoothed energy envelope
    hop = int(sr * 0.01)  # 10ms hop
    frame_len = int(sr * 0.025)  # 25ms frame
    rms = librosa.feature.rms(y=audio_filtered, frame_length=frame_len,
                               hop_length=hop)[0]

    # Smooth the envelope to merge micro-peaks
    kernel_size = max(3, int(0.04 * sr / hop))  # ~40ms smoothing
    if kernel_size % 2 == 0:
        kernel_size += 1
    envelope = np.convolve(rms, np.ones(kernel_size) / kernel_size, mode='same')

    # Process in windows
    frames_per_window = int(window_sec * sr / hop)
    rates = []
    pos = 0

    while pos < len(envelope):
        chunk = envelope[pos:pos + frames_per_window]
        chunk_duration = len(chunk) * hop / sr

        if chunk_duration < 0.2:
            break

        # Dynamic threshold: fraction of mean energy in this window
        threshold = np.mean(chunk) * 0.4
        if threshold < 1e-6:
            rates.append(0.0)
            pos += frames_per_window
            continue

        # Find peaks above threshold with minimum distance (~100ms between syllables)
        min_dist = max(1, int(0.1 * sr / hop))
        peaks = _find_peaks(chunk, threshold, min_dist)

        rate = len(peaks) / chunk_duration if chunk_duration > 0 else 0.0
        rates.append(round(rate, 2))
        pos += frames_per_window

    return rates if rates else [0.0]


def _bandpass_filter(audio: np.ndarray, sr: int,
                     low: int = 100, high: int = 4000) -> np.ndarray:
    """Simple bandpass using librosa's STFT filtering."""
    stft = librosa.stft(audio)
    freqs = librosa.fft_frequencies(sr=sr)
    mask = np.ones(len(freqs))
    mask[freqs < low] = 0.0
    mask[freqs > high] = 0.0
    # Smooth edges to avoid ringing
    stft_filtered = stft * mask[:, np.newaxis]
    return librosa.istft(stft_filtered, length=len(audio))


def _find_peaks(signal: np.ndarray, threshold: float,
                min_distance: int) -> list[int]:
    """Find peaks above threshold with minimum distance between them."""
    peaks = []
    i = 1
    while i < len(signal) - 1:
        if (signal[i] > signal[i - 1] and
                signal[i] > signal[i + 1] and
                signal[i] > threshold):
            peaks.append(i)
            i += min_distance
        else:
            i += 1
    return peaks


def classify_rate(syl_per_sec: float) -> str:
    """Classify speaking rate into a descriptive label.

    Args:
        syl_per_sec: Syllables per second.

    Returns:
        One of: "Too slow", "Slow", "Optimal", "Fast", "Too fast".
    """
    if syl_per_sec < 2.0:
        return "Too slow"
    elif syl_per_sec < 3.5:
        return "Slow"
    elif syl_per_sec <= 5.0:
        return "Optimal"
    elif syl_per_sec <= 6.5:
        return "Fast"
    else:
        return "Too fast"


def assess_pitch_dynamics(f0_data: dict,
                          cv_optimal: tuple[float, float] = (0.15, 0.25)) -> dict:
    """Assess pitch variation quality.

    Args:
        f0_data: Dict from AcousticFeatureExtractor.extract_f0() with
                 f0_cv, f0_range, f0_mean.
        cv_optimal: Tuple of (min, max) optimal coefficient of variation.

    Returns:
        Dict with cv, range, mean, assessment, score (0-100).
    """
    cv = f0_data.get("f0_cv", 0.0)
    f0_range = f0_data.get("f0_range", 0.0)
    f0_mean = f0_data.get("f0_mean", 0.0)

    cv_min, cv_max = cv_optimal

    if f0_mean == 0.0:
        return {
            "cv": 0.0, "range": 0.0, "mean": 0.0,
            "assessment": "No speech detected",
            "score": 0,
        }

    if cv < cv_min * 0.5:
        assessment = "Very monotone"
        score = 20
    elif cv < cv_min:
        assessment = "Monotone"
        score = 45
    elif cv <= cv_max:
        assessment = "Good variation"
        score = 85
    elif cv <= cv_max * 1.5:
        assessment = "Slightly erratic"
        score = 55
    else:
        assessment = "Erratic"
        score = 25

    return {
        "cv": round(cv, 4),
        "range": round(f0_range, 2),
        "mean": round(f0_mean, 2),
        "assessment": assessment,
        "score": score,
    }


def assess_volume_dynamics(energy_data: dict,
                           optimal_range: tuple[float, float] = (6.0, 30.0)) -> dict:
    """Assess volume control quality.

    Args:
        energy_data: Dict from AcousticFeatureExtractor.extract_energy() with
                     dynamic_range_db, rms_mean_db.
        optimal_range: Tuple of (min, max) optimal dynamic range in dB.

    Returns:
        Dict with dynamic_range, mean_db, issues, assessment, score (0-100).
    """
    dyn_range = energy_data.get("dynamic_range_db", 0.0)
    mean_db = energy_data.get("rms_mean_db", -np.inf)

    dr_min, dr_max = optimal_range
    issues = []

    if mean_db < -50:
        return {
            "dynamic_range": 0.0, "mean_db": round(mean_db, 2),
            "issues": ["Very quiet or silent"],
            "assessment": "No speech detected",
            "score": 0,
        }

    if dyn_range < dr_min:
        issues.append("Too flat — vary your volume more")
        score = 40
        assessment = "Flat"
    elif dyn_range <= dr_max:
        score = 85
        assessment = "Good dynamics"
    elif dyn_range <= dr_max * 1.5:
        issues.append("Slightly too much volume variation")
        score = 55
        assessment = "Slightly uneven"
    else:
        issues.append("Volume swings too extreme — inconsistent mic distance?")
        score = 25
        assessment = "Uneven"

    return {
        "dynamic_range": round(dyn_range, 2),
        "mean_db": round(mean_db, 2),
        "issues": issues,
        "assessment": assessment,
        "score": score,
    }


def analyze_prosody(audio: np.ndarray, features: dict,
                    config: dict) -> list[ProsodyResult]:
    """Run complete prosody analysis on audio.

    Args:
        audio: Audio signal as float32 numpy array.
        features: Pre-extracted features from AcousticFeatureExtractor.extract_all()
                  or None (will extract internally).
        config: Pipeline config dict.

    Returns:
        List of ProsodyResult, one per analysis window.
    """
    prosody_cfg = config.get("prosody", {})
    sr = config.get("audio", {}).get("sample_rate", 16000)
    window_sec = prosody_cfg.get("rate_window_sec", 10.0)
    rate_range = prosody_cfg.get("optimal_rate_range", [3.5, 5.0])
    cv_optimal = tuple(prosody_cfg.get("pitch_cv_optimal", [0.15, 0.25]))

    extractor = AcousticFeatureExtractor(config)

    # Compute syllable rates per window
    syl_rates = estimate_syllable_rate(audio, sr, window_sec)

    # Extract features per window
    window_samples = int(window_sec * sr)
    results = []

    for i, rate in enumerate(syl_rates):
        start_sample = i * window_samples
        end_sample = min(start_sample + window_samples, len(audio))
        chunk = audio[start_sample:end_sample]

        start_sec = round(start_sample / sr, 3)
        end_sec = round(end_sample / sr, 3)

        # Extract F0 and energy for this window
        f0_data = extractor.extract_f0(chunk)
        energy_data = extractor.extract_energy(chunk)

        # Assess
        rate_label = classify_rate(rate)
        pitch = assess_pitch_dynamics(f0_data, cv_optimal)
        volume = assess_volume_dynamics(energy_data)

        results.append(ProsodyResult(
            start_sec=start_sec,
            end_sec=end_sec,
            syllable_rate=rate,
            rate_label=rate_label,
            pitch_cv=pitch["cv"],
            pitch_range=pitch["range"],
            pitch_mean=pitch["mean"],
            pitch_assessment=pitch["assessment"],
            pitch_score=pitch["score"],
            volume_dynamic_range=volume["dynamic_range"],
            volume_mean_db=volume["mean_db"],
            volume_assessment=volume["assessment"],
            volume_score=volume["score"],
        ))

    return results
