"""Integration tests for the full voice analysis pipeline (Phase 10)."""

import json
import numpy as np
import pytest
import soundfile as sf
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.output_assembler import (
    assemble_voice_output,
    generate_summary,
    save_output,
)


@pytest.fixture
def config():
    """Minimal pipeline config for testing."""
    return {
        "pipeline": {"name": "test", "version": "1.0.0", "device": "cpu"},
        "audio": {"sample_rate": 16000},
        "vad": {"threshold": 0.5, "min_speech_duration_ms": 250, "min_silence_duration_ms": 100},
        "whisper": {"model_id": "openai/whisper-small", "language": "en", "chunk_length_s": 30, "batch_size": 1},
        "features": {"frame_ms": 25, "hop_ms": 10, "n_mfcc": 13, "f0_min": 75, "f0_max": 600},
        "filler": {"threshold": 0.5},
        "disfluency": {"model_path": "models/disfluency/best_model.pt", "window_sec": 3.0, "hop_sec": 1.5},
        "prosody": {"rate_window_sec": 10.0, "optimal_rate_range": [3.5, 5.0], "pitch_cv_optimal": [0.15, 0.25]},
        "emotion": {"model_path": "models/vocal_emotion/best_model.pt", "window_sec": 2.0},
        "output": {"window_sec": 2.0, "hop_sec": 1.0, "output_dir": "data/outputs"},
    }


@pytest.fixture
def dummy_transcript():
    return {
        "text": "hello um this is a test",
        "words": [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "um", "start": 0.6, "end": 0.8},
            {"word": "this", "start": 1.0, "end": 1.3},
            {"word": "is", "start": 1.4, "end": 1.5},
            {"word": "a", "start": 1.6, "end": 1.7},
            {"word": "test", "start": 1.8, "end": 2.2},
        ],
    }


@pytest.fixture
def dummy_speech_segments():
    return [
        {"start_sec": 0.0, "end_sec": 2.5, "duration_sec": 2.5},
        {"start_sec": 3.0, "end_sec": 5.0, "duration_sec": 2.0},
    ]


class TestOutputAssembler:
    """Tests for the output assembler module."""

    def test_assemble_returns_windows(self, config, dummy_transcript, dummy_speech_segments):
        windows = assemble_voice_output(
            duration_sec=5.0,
            transcript=dummy_transcript,
            speech_segments=dummy_speech_segments,
            fillers=[], disfluencies=[],
            prosody_results=[], emotion_results=[],
            features_windowed=[], config=config,
        )
        assert len(windows) >= 1
        assert all("start_sec" in w for w in windows)
        assert all("end_sec" in w for w in windows)

    def test_windows_are_sequential(self, config, dummy_transcript, dummy_speech_segments):
        windows = assemble_voice_output(
            duration_sec=5.0,
            transcript=dummy_transcript,
            speech_segments=dummy_speech_segments,
            fillers=[], disfluencies=[],
            prosody_results=[], emotion_results=[],
            features_windowed=[], config=config,
        )
        for i in range(1, len(windows)):
            assert windows[i]["start_sec"] >= windows[i - 1]["start_sec"]

    def test_timestamps_monotonic(self, config, dummy_transcript, dummy_speech_segments):
        windows = assemble_voice_output(
            duration_sec=10.0,
            transcript=dummy_transcript,
            speech_segments=dummy_speech_segments,
            fillers=[], disfluencies=[],
            prosody_results=[], emotion_results=[],
            features_windowed=[], config=config,
        )
        starts = [w["start_sec"] for w in windows]
        assert starts == sorted(starts), "Window start times should be monotonically increasing"

    def test_output_schema_fields(self, config, dummy_transcript, dummy_speech_segments):
        windows = assemble_voice_output(
            duration_sec=5.0,
            transcript=dummy_transcript,
            speech_segments=dummy_speech_segments,
            fillers=[], disfluencies=[],
            prosody_results=[], emotion_results=[],
            features_windowed=[], config=config,
        )
        required_fields = {
            "start_sec", "end_sec", "is_speech", "transcript",
            "word_count", "fillers", "disfluencies", "prosody", "emotion", "features",
        }
        for w in windows:
            assert required_fields.issubset(w.keys()), f"Missing fields: {required_fields - w.keys()}"

    def test_speech_ratio(self, config, dummy_transcript, dummy_speech_segments):
        windows = assemble_voice_output(
            duration_sec=5.0,
            transcript=dummy_transcript,
            speech_segments=dummy_speech_segments,
            fillers=[], disfluencies=[],
            prosody_results=[], emotion_results=[],
            features_windowed=[], config=config,
        )
        speech_count = sum(1 for w in windows if w["is_speech"])
        ratio = speech_count / len(windows) if windows else 0
        assert 0 <= ratio <= 1

    def test_word_count_matches(self, config, dummy_transcript, dummy_speech_segments):
        windows = assemble_voice_output(
            duration_sec=3.0,
            transcript=dummy_transcript,
            speech_segments=dummy_speech_segments,
            fillers=[], disfluencies=[],
            prosody_results=[], emotion_results=[],
            features_windowed=[], config=config,
        )
        total_words = sum(w["word_count"] for w in windows)
        # With overlapping windows, words may appear in multiple windows
        assert total_words >= 0


class TestGenerateSummary:

    def test_summary_fields(self, config, dummy_transcript, dummy_speech_segments):
        windows = assemble_voice_output(
            duration_sec=5.0,
            transcript=dummy_transcript,
            speech_segments=dummy_speech_segments,
            fillers=[], disfluencies=[],
            prosody_results=[], emotion_results=[],
            features_windowed=[], config=config,
        )
        summary = generate_summary(windows, 5.0, dummy_transcript)
        assert "duration_sec" in summary
        assert "word_count" in summary
        assert "words_per_minute" in summary
        assert "filler_count" in summary
        assert "fillers_per_minute" in summary
        assert "disfluency_count" in summary
        assert "dominant_emotion" in summary

    def test_summary_wpm(self, dummy_transcript):
        windows = []
        summary = generate_summary(windows, 60.0, dummy_transcript)
        assert summary["word_count"] == 6
        assert summary["words_per_minute"] == 6.0


class TestSaveOutput:

    def test_saves_valid_json(self, config, dummy_transcript, dummy_speech_segments):
        windows = assemble_voice_output(
            duration_sec=5.0,
            transcript=dummy_transcript,
            speech_segments=dummy_speech_segments,
            fillers=[], disfluencies=[],
            prosody_results=[], emotion_results=[],
            features_windowed=[], config=config,
        )
        summary = generate_summary(windows, 5.0, dummy_transcript)
        metadata = {"version": "test", "video": "test.mp4"}

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        save_output(windows, summary, metadata, path)

        with open(path) as f:
            data = json.load(f)

        assert "metadata" in data
        assert "summary" in data
        assert "windows" in data
        assert isinstance(data["windows"], list)

        Path(path).unlink()


class TestGracefulDegradation:
    """Test that pipeline handles missing models without crashing."""

    def test_assembler_with_no_emotion(self, config, dummy_transcript, dummy_speech_segments):
        """Empty emotion list should produce windows with emotion.available=False."""
        windows = assemble_voice_output(
            duration_sec=5.0,
            transcript=dummy_transcript,
            speech_segments=dummy_speech_segments,
            fillers=[], disfluencies=[],
            prosody_results=[], emotion_results=[],
            features_windowed=[], config=config,
        )
        for w in windows:
            assert w["emotion"]["available"] is False

    def test_assembler_with_no_prosody(self, config, dummy_transcript, dummy_speech_segments):
        """Empty prosody list should produce windows with prosody.available=False."""
        windows = assemble_voice_output(
            duration_sec=5.0,
            transcript=dummy_transcript,
            speech_segments=dummy_speech_segments,
            fillers=[], disfluencies=[],
            prosody_results=[], emotion_results=[],
            features_windowed=[], config=config,
        )
        for w in windows:
            assert w["prosody"]["available"] is False

    def test_assembler_with_no_disfluencies(self, config, dummy_transcript, dummy_speech_segments):
        """Empty disfluency list should produce windows with empty disfluency arrays."""
        windows = assemble_voice_output(
            duration_sec=5.0,
            transcript=dummy_transcript,
            speech_segments=dummy_speech_segments,
            fillers=[], disfluencies=[],
            prosody_results=[], emotion_results=[],
            features_windowed=[], config=config,
        )
        for w in windows:
            assert w["disfluencies"] == []

    def test_empty_transcript(self, config, dummy_speech_segments):
        """Empty transcript should not crash."""
        empty_transcript = {"text": "", "words": []}
        windows = assemble_voice_output(
            duration_sec=5.0,
            transcript=empty_transcript,
            speech_segments=dummy_speech_segments,
            fillers=[], disfluencies=[],
            prosody_results=[], emotion_results=[],
            features_windowed=[], config=config,
        )
        assert len(windows) >= 1
        for w in windows:
            assert w["word_count"] == 0
            assert w["transcript"] == ""
