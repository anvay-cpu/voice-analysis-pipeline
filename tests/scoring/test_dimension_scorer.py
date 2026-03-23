"""Tests for Phase 5.1 — 6-Dimension Scoring."""

import pytest

from src.scoring.dimension_scorer import DimensionScorer
from src.fusion.fusion_engine import FusionEngine
from tests.fusion.test_timeline import (
    make_voice_windows,
    make_body_output,
    make_content_output,
)


def make_fusion_output(duration=30):
    """Generate a full fusion output for testing scoring."""
    engine = FusionEngine()
    voice = make_voice_windows(duration)
    body = make_body_output(duration)
    content = make_content_output(duration)
    return engine.fuse(voice, body, content, duration)


class TestDimensionScorer:

    @pytest.fixture
    def scorer(self):
        return DimensionScorer()

    @pytest.fixture
    def fusion_output(self):
        return make_fusion_output(30)

    def test_score_all_returns_7_keys(self, scorer, fusion_output):
        """Should return 6 dimensions + overall."""
        scores = scorer.score_all(fusion_output)
        assert len(scores) == 7
        expected_keys = [
            "vocal_clarity", "body_language", "content_structure",
            "audience_engagement", "emotional_expressiveness",
            "regime_adaptability", "overall",
        ]
        for key in expected_keys:
            assert key in scores

    def test_all_scores_in_range(self, scorer, fusion_output):
        """All scores should be 0-100."""
        scores = scorer.score_all(fusion_output)
        for key, value in scores.items():
            assert 0.0 <= value <= 100.0, f"{key} = {value} out of range"

    def test_overall_is_mean(self, scorer, fusion_output):
        """Overall should be the mean of the 6 dimensions."""
        scores = scorer.score_all(fusion_output)
        dims = [v for k, v in scores.items() if k != "overall"]
        expected_overall = sum(dims) / len(dims)
        assert abs(scores["overall"] - expected_overall) < 0.2

    def test_vocal_clarity_reasonable(self, scorer, fusion_output):
        """Vocal clarity should be reasonable for normal speech."""
        scores = scorer.score_all(fusion_output)
        assert 30.0 <= scores["vocal_clarity"] <= 100.0

    def test_body_language_reasonable(self, scorer, fusion_output):
        """Body language should be reasonable for normal body data."""
        scores = scorer.score_all(fusion_output)
        assert 30.0 <= scores["body_language"] <= 100.0

    def test_content_structure_with_good_structure(self, scorer, fusion_output):
        """Content with intro/data/conclusion should score well."""
        scores = scorer.score_all(fusion_output)
        # Our mock has Introduction, Data Presentation, Conclusion
        assert scores["content_structure"] >= 40.0

    def test_empty_fusion_output(self, scorer):
        """Empty fusion output should return safe defaults."""
        empty = {
            "timeline": [],
            "transitions": {},
            "recovery": {},
            "emotion_coherence": {},
        }
        scores = scorer.score_all(empty)
        assert len(scores) == 7
        for value in scores.values():
            assert 0.0 <= value <= 100.0

    def test_missing_modality_data(self, scorer):
        """Timeline with None modalities should not crash."""
        timeline = [
            {"time_sec": i, "voice": None, "body": None, "content": None}
            for i in range(30)
        ]
        fusion = {
            "timeline": timeline,
            "transitions": {"transitions": []},
            "recovery": {"mean_composure": 1.0, "disruptions_per_minute": 0},
            "emotion_coherence": {"coherence_score_0_100": 50},
        }
        scores = scorer.score_all(fusion)
        assert len(scores) == 7

    def test_high_filler_lowers_vocal_clarity(self, scorer):
        """High filler count should lower vocal clarity."""
        fusion_clean = make_fusion_output(30)
        scores_clean = scorer.score_all(fusion_clean)

        # Inject fillers
        fusion_filler = make_fusion_output(30)
        for entry in fusion_filler["timeline"]:
            if entry.get("voice"):
                entry["voice"]["filler_count"] = 3
        scores_filler = scorer.score_all(fusion_filler)

        assert scores_filler["vocal_clarity"] < scores_clean["vocal_clarity"]

    def test_adaptability_with_no_transitions(self, scorer):
        """No transitions should give neutral adaptability."""
        fusion = make_fusion_output(30)
        fusion["transitions"] = {"transitions": []}
        fusion["recovery"] = {
            "mean_composure": 1.0,
            "disruptions_per_minute": 0,
        }
        scores = scorer.score_all(fusion)
        assert 40.0 <= scores["regime_adaptability"] <= 100.0

    def test_longer_speech(self, scorer):
        """60-second speech should also work."""
        fusion = make_fusion_output(60)
        scores = scorer.score_all(fusion)
        assert len(scores) == 7
        for value in scores.values():
            assert 0.0 <= value <= 100.0
