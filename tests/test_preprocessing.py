"""Tests for audio preprocessing pipeline (Phase 2)."""

import os
import tempfile
import numpy as np
import soundfile as sf
import pytest

# ---------------------------------------------------------------------------
# Helpers — synthetic audio generation
# ---------------------------------------------------------------------------

def _generate_sine_wav(path: str, duration_sec: float = 10.0, sr: int = 16000,
                       freq: float = 440.0, silence_start: float = 4.0,
                       silence_end: float = 6.0):
    """Generate a WAV with a sine tone and a silence gap in the middle."""
    n_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, n_samples, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * freq * t)

    # Insert silence
    s_start = int(silence_start * sr)
    s_end = int(silence_end * sr)
    audio[s_start:s_end] = 0.0

    sf.write(path, audio, sr)
    return path


def _generate_noisy_wav(path: str, duration_sec: float = 5.0, sr: int = 16000):
    """Generate a sine tone with heavy additive noise (low SNR)."""
    n = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, n, dtype=np.float32)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = 0.3 * np.random.randn(n).astype(np.float32)
    audio = signal + noise
    sf.write(path, audio, sr)
    return path


def _make_test_video(path: str, duration_sec: float = 3.0, sr: int = 16000):
    """Create a minimal MP4 test video with a sine-tone audio track."""
    import subprocess
    # Generate raw audio
    n = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, n, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    wav_path = path.replace(".mp4", "_temp.wav")
    sf.write(wav_path, audio, sr)

    # Wrap in MP4 using FFmpeg (black video + audio)
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=black:s=320x240:d={duration_sec}:r=1",
        "-i", wav_path,
        "-c:v", "libx264", "-tune", "stillimage",
        "-c:a", "aac", "-b:a", "64k",
        "-shortest", "-pix_fmt", "yuv420p",
        path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    os.remove(wav_path)
    return path


# ---------------------------------------------------------------------------
# Pipeline config fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline_config():
    return {
        "audio": {"sample_rate": 16000, "channels": 1, "max_duration_sec": 3600},
        "vad": {
            "threshold": 0.5,
            "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 100,
        },
    }


# ---------------------------------------------------------------------------
# Test: audio extraction
# ---------------------------------------------------------------------------

class TestAudioExtractor:

    def test_extract_audio(self, tmp_path):
        """Extract audio from a synthetic test MP4."""
        from src.audio_extractor import extract_audio, validate_input

        video_path = str(tmp_path / "test.mp4")
        _make_test_video(video_path, duration_sec=3.0)

        info = validate_input(video_path)
        assert info["format"] == ".mp4"
        assert info["duration_sec"] > 0

        wav_path = extract_audio(video_path, str(tmp_path / "audio_out"))
        assert os.path.exists(wav_path)
        assert wav_path.endswith(".wav")

        # Verify WAV properties
        data, sr = sf.read(wav_path)
        assert sr == 16000
        assert len(data.shape) == 1  # mono

    def test_invalid_format(self, tmp_path):
        """Reject unsupported file formats."""
        from src.audio_extractor import validate_input

        bad_path = str(tmp_path / "test.txt")
        with open(bad_path, "w") as f:
            f.write("not a video")

        with pytest.raises(ValueError, match="Unsupported format"):
            validate_input(bad_path)

    def test_file_not_found(self):
        from src.audio_extractor import validate_input
        with pytest.raises(FileNotFoundError):
            validate_input("/nonexistent/video.mp4")


# ---------------------------------------------------------------------------
# Test: VAD
# ---------------------------------------------------------------------------

class TestVAD:

    def test_vad_detects_segments(self, tmp_path, pipeline_config):
        """VAD returns valid structure; sine waves are correctly ignored as non-speech."""
        from src.vad import detect_speech_segments

        wav_path = str(tmp_path / "sine_with_silence.wav")
        _generate_sine_wav(wav_path, duration_sec=10.0,
                           silence_start=4.0, silence_end=6.0)

        result = detect_speech_segments(wav_path, pipeline_config)

        assert "speech_segments" in result
        assert "pauses" in result
        assert isinstance(result["speech_segments"], list)
        assert isinstance(result["pauses"], list)
        assert result["speech_ratio"] >= 0.0
        assert result["speech_ratio"] <= 1.0
        assert result["total_speech_sec"] >= 0
        assert result["total_silence_sec"] >= 0

    def test_vad_returns_correct_keys(self, tmp_path, pipeline_config):
        from src.vad import detect_speech_segments

        wav_path = str(tmp_path / "tone.wav")
        _generate_sine_wav(wav_path, duration_sec=3.0,
                           silence_start=1.0, silence_end=2.0)

        result = detect_speech_segments(wav_path, pipeline_config)

        expected_keys = {"speech_segments", "pauses", "total_speech_sec",
                         "total_silence_sec", "speech_ratio"}
        assert expected_keys == set(result.keys())


# ---------------------------------------------------------------------------
# Test: Noise reduction
# ---------------------------------------------------------------------------

class TestNoiseReduction:

    def test_snr_estimation(self, tmp_path):
        """SNR should be lower for noisy audio."""
        from src.noise_reduction import estimate_snr
        import librosa

        # Clean signal
        clean_path = str(tmp_path / "clean.wav")
        _generate_sine_wav(clean_path, duration_sec=3.0,
                           silence_start=10.0, silence_end=10.0)  # no silence
        clean, sr = librosa.load(clean_path, sr=16000, mono=True)
        segments_clean = [{"start_sec": 0.0, "end_sec": 3.0}]

        # Noisy signal
        noisy_path = str(tmp_path / "noisy.wav")
        _generate_noisy_wav(noisy_path, duration_sec=3.0)
        noisy, _ = librosa.load(noisy_path, sr=16000, mono=True)
        segments_noisy = [{"start_sec": 0.0, "end_sec": 2.0}]

        snr_clean = estimate_snr(clean, segments_clean, sr)
        snr_noisy = estimate_snr(noisy, segments_noisy, sr)

        # Clean should have higher SNR
        assert snr_clean > snr_noisy

    def test_noise_reduction_improves_snr(self, tmp_path):
        """Noise reduction should improve SNR on noisy audio."""
        from src.noise_reduction import reduce_noise, estimate_snr
        import librosa

        noisy_path = str(tmp_path / "noisy.wav")
        _generate_noisy_wav(noisy_path, duration_sec=5.0)

        speech_segments = [{"start_sec": 0.0, "end_sec": 3.0}]
        output_path = str(tmp_path / "cleaned.wav")

        reduce_noise(noisy_path, speech_segments, output_path)

        assert os.path.exists(output_path)

        # Verify output is valid audio
        data, sr = sf.read(output_path)
        assert sr == 16000
        assert len(data) > 0


# ---------------------------------------------------------------------------
# Test: Full preprocessing chain
# ---------------------------------------------------------------------------

class TestFullPreprocess:

    def test_preprocess_chain(self, tmp_path, pipeline_config):
        """Full pipeline: video -> extract -> VAD -> denoise."""
        from src.noise_reduction import preprocess_audio

        video_path = str(tmp_path / "test_full.mp4")
        _make_test_video(video_path, duration_sec=5.0)

        output_dir = str(tmp_path / "output")
        result = preprocess_audio(video_path, output_dir, pipeline_config)

        assert os.path.exists(result["raw_audio_path"])
        assert os.path.exists(result["clean_audio_path"])
        assert isinstance(result["speech_segments"], list)
        assert isinstance(result["pauses"], list)
        assert result["duration_sec"] > 0
        assert isinstance(result["snr_db"], float)
        assert isinstance(result["noise_reduced"], bool)
