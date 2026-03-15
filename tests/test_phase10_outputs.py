"""Phase 10 — Validate real pipeline outputs from 3 test videos."""

import json
import pytest
from pathlib import Path


OUTPUT_DIR = Path("voice_pipeline_outputs")

VIDEOS = [
    "test_good_speaker_5min",
    "test_nervous_speaker_5min",
    "test_monotone_speaker_5min",
]


@pytest.fixture(params=VIDEOS)
def output_data(request):
    path = OUTPUT_DIR / f"{request.param}.json"
    if not path.exists():
        pytest.skip(f"Output file not found: {path}")
    with open(path) as f:
        return json.load(f)


class TestOutputSchema:
    """Validate JSON structure of pipeline outputs."""

    def test_top_level_keys(self, output_data):
        assert "metadata" in output_data
        assert "summary" in output_data
        assert "windows" in output_data

    def test_metadata_fields(self, output_data):
        meta = output_data["metadata"]
        assert "duration_sec" in meta
        assert "timings" in meta
        assert meta["duration_sec"] > 0

    def test_summary_fields(self, output_data):
        s = output_data["summary"]
        required = [
            "duration_sec", "word_count", "words_per_minute",
            "filler_count", "fillers_per_minute",
            "disfluency_count", "disfluency_types",
            "avg_syllable_rate", "avg_pitch_cv",
            "avg_pitch_score", "avg_volume_score",
            "dominant_emotion", "emotion_variety", "emotion_distribution",
        ]
        for field in required:
            assert field in s, f"Missing summary field: {field}"

    def test_window_fields(self, output_data):
        required = {
            "start_sec", "end_sec", "is_speech", "transcript",
            "word_count", "fillers", "disfluencies", "prosody",
            "emotion", "features",
        }
        for w in output_data["windows"]:
            assert required.issubset(w.keys()), f"Missing: {required - w.keys()}"

    def test_window_count(self, output_data):
        """5-minute audio with 2s window, 1s hop → ~299 windows."""
        windows = output_data["windows"]
        assert 250 <= len(windows) <= 350

    def test_timestamps_monotonic(self, output_data):
        starts = [w["start_sec"] for w in output_data["windows"]]
        assert starts == sorted(starts)

    def test_no_gaps_in_windows(self, output_data):
        """Windows should have 1-second hop (no large gaps)."""
        windows = output_data["windows"]
        for i in range(1, len(windows)):
            gap = windows[i]["start_sec"] - windows[i - 1]["start_sec"]
            assert 0.9 <= gap <= 1.1, f"Unexpected gap at window {i}: {gap}s"


class TestSummaryMetrics:
    """Validate that summary metrics are within reasonable ranges."""

    def test_wpm_range(self, output_data):
        wpm = output_data["summary"]["words_per_minute"]
        assert 50 <= wpm <= 250, f"WPM {wpm} outside expected range"

    def test_filler_rate(self, output_data):
        fpm = output_data["summary"]["fillers_per_minute"]
        assert 0 <= fpm <= 30, f"Fillers/min {fpm} outside expected range"

    def test_syllable_rate(self, output_data):
        rate = output_data["summary"]["avg_syllable_rate"]
        assert 1.0 <= rate <= 8.0, f"Syllable rate {rate} outside expected range"

    def test_pitch_cv(self, output_data):
        cv = output_data["summary"]["avg_pitch_cv"]
        assert 0.0 <= cv <= 1.0, f"Pitch CV {cv} outside expected range"

    def test_emotion_variety(self, output_data):
        v = output_data["summary"]["emotion_variety"]
        assert 0.0 <= v <= 1.0

    def test_emotion_distribution_sums_to_one(self, output_data):
        dist = output_data["summary"]["emotion_distribution"]
        if dist:
            total = sum(dist.values())
            assert 0.95 <= total <= 1.05, f"Emotion distribution sums to {total}"

    def test_duration_matches(self, output_data):
        meta_dur = output_data["metadata"]["duration_sec"]
        sum_dur = output_data["summary"]["duration_sec"]
        assert abs(meta_dur - sum_dur) < 1.0


class TestSpeakerProfiles:
    """Validate that different speaker types show expected patterns."""

    @pytest.fixture
    def all_summaries(self):
        summaries = {}
        for v in VIDEOS:
            path = OUTPUT_DIR / f"{v}.json"
            if not path.exists():
                pytest.skip("Output files not found")
            with open(path) as f:
                summaries[v] = json.load(f)["summary"]
        return summaries

    def test_nervous_has_most_disfluencies(self, all_summaries):
        nervous = all_summaries["test_nervous_speaker_5min"]["disfluency_count"]
        good = all_summaries["test_good_speaker_5min"]["disfluency_count"]
        monotone = all_summaries["test_monotone_speaker_5min"]["disfluency_count"]
        assert nervous > good, f"Nervous ({nervous}) should have more disfluencies than good ({good})"
        assert nervous > monotone, f"Nervous ({nervous}) should have more disfluencies than monotone ({monotone})"
