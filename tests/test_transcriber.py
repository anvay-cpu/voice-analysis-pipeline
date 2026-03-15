"""Tests for speech-to-text transcription (Phase 3)."""

import os
import pytest
import numpy as np
import soundfile as sf


def _make_test_audio(path: str, duration_sec: float = 3.0, sr: int = 16000):
    """Create a short sine wave audio file for testing."""
    n = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, n, dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, audio, sr)
    return path


class TestWhisperLoad:

    def test_load_whisper_returns_pipeline(self):
        """Whisper pipeline should load without error."""
        from src.transcriber import load_whisper
        pipe = load_whisper(device="cpu")  # CPU for test reliability
        assert pipe is not None
        assert pipe.task == "automatic-speech-recognition"

    def test_load_whisper_caches(self):
        """Second call should return cached pipeline."""
        from src.transcriber import load_whisper
        pipe1 = load_whisper(device="cpu")
        pipe2 = load_whisper(device="cpu")
        assert pipe1 is pipe2


class TestTranscribe:

    @pytest.fixture(autouse=True)
    def _load_pipe(self):
        from src.transcriber import load_whisper
        self.pipe = load_whisper(device="cpu")

    def test_transcribe_returns_structure(self, tmp_path):
        """Transcribe should return dict with 'text' and 'words' keys."""
        from src.transcriber import transcribe

        audio_path = str(tmp_path / "test.wav")
        _make_test_audio(audio_path, duration_sec=2.0)

        result = transcribe(self.pipe, audio_path)
        assert "text" in result
        assert "words" in result
        assert isinstance(result["text"], str)
        assert isinstance(result["words"], list)

    def test_word_timestamps_monotonic(self, tmp_path):
        """Word timestamps should be monotonically increasing."""
        from src.transcriber import transcribe

        audio_path = str(tmp_path / "test_mono.wav")
        _make_test_audio(audio_path, duration_sec=3.0)

        result = transcribe(self.pipe, audio_path)

        if len(result["words"]) > 1:
            starts = [w["start"] for w in result["words"]]
            for i in range(1, len(starts)):
                assert starts[i] >= starts[i - 1], \
                    f"Timestamps not monotonic: {starts[i-1]} > {starts[i]}"


class TestVADFiltering:

    def test_filter_removes_outside_words(self):
        """Words outside VAD speech regions should be removed."""
        from src.transcriber import filter_by_vad

        transcript = {
            "text": "hello world phantom",
            "words": [
                {"word": "hello", "start": 1.0, "end": 1.5},
                {"word": "world", "start": 1.6, "end": 2.0},
                {"word": "phantom", "start": 5.0, "end": 5.5},  # outside speech
            ],
        }
        speech_segments = [
            {"start_sec": 0.8, "end_sec": 2.2},
            # Gap from 2.2 to 8.0 — "phantom" at 5.0 is outside
        ]

        filtered = filter_by_vad(transcript, speech_segments)

        words = [w["word"] for w in filtered["words"]]
        assert "hello" in words
        assert "world" in words
        assert "phantom" not in words

    def test_filter_keeps_all_when_in_speech(self):
        """All words within speech regions should be preserved."""
        from src.transcriber import filter_by_vad

        transcript = {
            "text": "one two three",
            "words": [
                {"word": "one", "start": 0.5, "end": 0.8},
                {"word": "two", "start": 1.0, "end": 1.3},
                {"word": "three", "start": 1.5, "end": 1.9},
            ],
        }
        speech_segments = [{"start_sec": 0.0, "end_sec": 3.0}]

        filtered = filter_by_vad(transcript, speech_segments)
        assert len(filtered["words"]) == 3

    def test_filter_empty_segments(self):
        """Empty speech segments should return transcript unchanged."""
        from src.transcriber import filter_by_vad

        transcript = {"text": "hello", "words": [{"word": "hello", "start": 0.0, "end": 0.5}]}
        filtered = filter_by_vad(transcript, [])
        assert filtered == transcript
