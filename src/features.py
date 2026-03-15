"""Acoustic feature extraction (MFCCs, pitch, energy, spectral, voice quality)."""

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call


class AcousticFeatureExtractor:
    """Extract handcrafted audio features using librosa and parselmouth."""

    def __init__(self, config: dict):
        feat_cfg = config.get("features", {})
        self.sr = config.get("audio", {}).get("sample_rate", 16000)
        self.frame_ms = feat_cfg.get("frame_ms", 25)
        self.hop_ms = feat_cfg.get("hop_ms", 10)
        self.n_mfcc = feat_cfg.get("n_mfcc", 13)
        self.f0_min = feat_cfg.get("f0_min", 75)
        self.f0_max = feat_cfg.get("f0_max", 600)

        self.n_fft = int(self.sr * self.frame_ms / 1000)
        self.hop_length = int(self.sr * self.hop_ms / 1000)

    def _is_too_short(self, audio: np.ndarray) -> bool:
        """Check if audio is too short for analysis (< 50ms)."""
        return len(audio) < int(self.sr * 0.05)

    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray | None:
        """Extract 39-dim MFCCs (13 static + 13 delta + 13 delta-delta).

        Args:
            audio: Audio signal as float32 numpy array.

        Returns:
            Array of shape (39, T) or None if audio too short.
        """
        if self._is_too_short(audio):
            return None

        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sr, n_mfcc=self.n_mfcc,
            n_fft=self.n_fft, hop_length=self.hop_length,
        )
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        return np.vstack([mfcc, delta, delta2])

    def extract_f0(self, audio: np.ndarray) -> dict:
        """Extract fundamental frequency (F0/pitch) features using Praat.

        Returns:
            Dict with f0_contour, f0_mean, f0_std, f0_range, f0_cv.
        """
        defaults = {
            "f0_contour": np.array([]),
            "f0_mean": 0.0,
            "f0_std": 0.0,
            "f0_range": 0.0,
            "f0_cv": 0.0,
        }
        if self._is_too_short(audio):
            return defaults

        snd = parselmouth.Sound(audio, sampling_frequency=self.sr)
        pitch = call(snd, "To Pitch", 0.0, self.f0_min, self.f0_max)

        f0_values = pitch.selected_array["frequency"]
        voiced = f0_values[f0_values > 0]

        if len(voiced) == 0:
            return defaults

        return {
            "f0_contour": f0_values,
            "f0_mean": round(float(np.mean(voiced)), 2),
            "f0_std": round(float(np.std(voiced)), 2),
            "f0_range": round(float(np.max(voiced) - np.min(voiced)), 2),
            "f0_cv": round(float(np.std(voiced) / np.mean(voiced)), 4) if np.mean(voiced) > 0 else 0.0,
        }

    def extract_energy(self, audio: np.ndarray) -> dict:
        """Extract RMS energy features.

        Returns:
            Dict with rms_contour_db, rms_mean_db, dynamic_range_db.
        """
        if self._is_too_short(audio):
            return {
                "rms_contour_db": np.array([]),
                "rms_mean_db": -np.inf,
                "dynamic_range_db": 0.0,
            }

        rms = librosa.feature.rms(
            y=audio, frame_length=self.n_fft, hop_length=self.hop_length
        )[0]

        # Convert to dB, avoiding log(0)
        rms_db = librosa.amplitude_to_db(rms, ref=1.0)

        return {
            "rms_contour_db": rms_db,
            "rms_mean_db": round(float(np.mean(rms_db)), 2),
            "dynamic_range_db": round(float(np.max(rms_db) - np.min(rms_db)), 2),
        }

    def extract_spectral(self, audio: np.ndarray) -> dict:
        """Extract spectral features: centroid, bandwidth, rolloff.

        Returns:
            Dict with centroid, bandwidth, rolloff arrays.
        """
        if self._is_too_short(audio):
            return {
                "centroid": np.array([]),
                "bandwidth": np.array([]),
                "rolloff": np.array([]),
            }

        centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]

        return {
            "centroid": centroid,
            "bandwidth": bandwidth,
            "rolloff": rolloff,
        }

    def extract_voice_quality(self, audio: np.ndarray) -> dict:
        """Extract voice quality metrics: jitter, shimmer, HNR.

        Returns:
            Dict with jitter_percent, shimmer_percent, hnr_db.
        """
        defaults = {
            "jitter_percent": 0.0,
            "shimmer_percent": 0.0,
            "hnr_db": 0.0,
        }
        if self._is_too_short(audio):
            return defaults

        snd = parselmouth.Sound(audio, sampling_frequency=self.sr)

        # Check if there's voiced content
        pitch = call(snd, "To Pitch", 0.0, self.f0_min, self.f0_max)
        voiced = pitch.selected_array["frequency"]
        if np.sum(voiced > 0) < 3:
            return defaults

        try:
            point_process = call(snd, "To PointProcess (periodic, cc)", self.f0_min, self.f0_max)

            jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer = call(
                [snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
            )

            harmonicity = call(snd, "To Harmonicity (cc)", 0.01, self.f0_min, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
        except Exception:
            return defaults

        return {
            "jitter_percent": round(float(jitter) * 100, 4) if not np.isnan(jitter) else 0.0,
            "shimmer_percent": round(float(shimmer) * 100, 4) if not np.isnan(shimmer) else 0.0,
            "hnr_db": round(float(hnr), 2) if not np.isnan(hnr) else 0.0,
        }

    def extract_all(self, audio: np.ndarray) -> dict:
        """Extract all acoustic features from an audio signal.

        Returns:
            Combined dict with keys: mfcc, f0, energy, spectral, voice_quality.
        """
        return {
            "mfcc": self.extract_mfcc(audio),
            "f0": self.extract_f0(audio),
            "energy": self.extract_energy(audio),
            "spectral": self.extract_spectral(audio),
            "voice_quality": self.extract_voice_quality(audio),
        }

    def extract_windowed(self, audio: np.ndarray,
                         window_sec: float = 2.0,
                         hop_sec: float = 1.0) -> list[dict]:
        """Extract features in sliding windows over the audio.

        Args:
            audio: Audio signal.
            window_sec: Window size in seconds.
            hop_sec: Hop size in seconds.

        Returns:
            List of dicts, one per window, each with start_sec, end_sec, and features.
        """
        window_samples = int(window_sec * self.sr)
        hop_samples = int(hop_sec * self.sr)
        total_samples = len(audio)

        results = []
        pos = 0
        while pos + window_samples <= total_samples:
            chunk = audio[pos:pos + window_samples]
            features = self.extract_all(chunk)
            features["start_sec"] = round(pos / self.sr, 3)
            features["end_sec"] = round((pos + window_samples) / self.sr, 3)
            results.append(features)
            pos += hop_samples

        # Handle trailing chunk if long enough
        if pos < total_samples and (total_samples - pos) >= int(self.sr * 0.1):
            chunk = audio[pos:]
            features = self.extract_all(chunk)
            features["start_sec"] = round(pos / self.sr, 3)
            features["end_sec"] = round(total_samples / self.sr, 3)
            results.append(features)

        return results
