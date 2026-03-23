"""Tests for Phase 4.2 — Regime Transition Scoring."""

import pytest

from src.fusion.timeline_aligner import TimelineAligner
from src.fusion.regime_transition_scorer import (
    RegimeTransitionScorer,
    EXPECTED_DIRECTION,
    DEFAULT_EXPECTED,
)


# ── Helpers ─────────────────────────────────────────────────────

def make_timeline_entry(
    time_sec,
    syllable_rate=4.0,
    pitch_cv=0.2,
    f0_mean=150.0,
    rms_mean_db=-20.0,
    gesture="Rest",
    gaze_engagement=0.7,
    posture_score=70.0,
    regime_type="Introduction",
    emotion_label="Confident",
):
    """Create a single aligned timeline entry."""
    return {
        "time_sec": time_sec,
        "voice": {
            "is_speech": True,
            "transcript": "word",
            "word_count": 2,
            "filler_count": 0,
            "fillers": [],
            "disfluency_count": 0,
            "disfluencies": [],
            "syllable_rate": syllable_rate,
            "rate_label": "Normal",
            "pitch_cv": pitch_cv,
            "pitch_mean": f0_mean,
            "pitch_score": 70,
            "volume_score": 75,
            "emotion_label": emotion_label,
            "emotion_confidence": 0.8,
            "f0_mean": f0_mean,
            "f0_cv": pitch_cv,
            "rms_mean_db": rms_mean_db,
        },
        "body": {
            "posture_score": posture_score,
            "posture_std": 3.0,
            "shoulder_deg": 2.0,
            "spine_deg": 8.0,
            "gesture_dominant": gesture,
            "gesture_distribution": {},
            "hand_state": "Visible",
            "hand_distribution": {},
            "gaze_engagement": gaze_engagement,
            "gaze_distribution": {},
            "facial_emotion": "Happy",
            "facial_emotion_confidence": 0.8,
            "movement_pattern": "Anchored",
            "movement_velocity": 0.02,
        },
        "content": {
            "regime_type": regime_type,
            "regime_confidence": 0.85,
            "sentiment_polarity": "Neutral",
            "sentiment_intensity": 0.5,
            "sentiment_valence": 0.0,
            "tone": "Informative",
            "tone_confidence": 0.8,
            "readability_grade": 8.0,
            "grammar_error_count": 0,
            "argument_effectiveness": 70.0,
        },
    }


def build_transition_timeline(
    before_kwargs, after_kwargs, boundary_sec=20, window=10
):
    """Build a 40-second timeline with a transition at boundary_sec.

    before_kwargs/after_kwargs override defaults for entries before/after the boundary.
    """
    timeline = []
    for sec in range(40):
        if sec < boundary_sec:
            kwargs = {"time_sec": sec, "regime_type": "Introduction", **before_kwargs}
        else:
            kwargs = {"time_sec": sec, "regime_type": "Main Argument", **after_kwargs}
        timeline.append(make_timeline_entry(**kwargs))
    return timeline


# ── Tests ───────────────────────────────────────────────────────

class TestRegimeTransitionScorer:

    @pytest.fixture
    def scorer(self):
        return RegimeTransitionScorer(window_sec=10)

    def test_score_in_range(self, scorer):
        """Score should be between 0.0 and 1.0."""
        timeline = build_transition_timeline(
            before_kwargs={"f0_mean": 120.0, "syllable_rate": 3.0},
            after_kwargs={"f0_mean": 180.0, "syllable_rate": 5.0},
        )
        boundary = {"time_sec": 20, "from_regime": "Introduction", "to_regime": "Main Argument"}
        result = scorer.score_transition(timeline, boundary)
        assert 0.0 <= result["score"] <= 1.0

    def test_perfect_adaptation(self, scorer):
        """Speaker who adapts perfectly should score high."""
        # Intro→Main Argument expects: energy +1, rate +1, gesture +1
        timeline = build_transition_timeline(
            before_kwargs={
                "f0_mean": 100.0, "rms_mean_db": -30.0,
                "syllable_rate": 3.0, "gesture": "Rest",
            },
            after_kwargs={
                "f0_mean": 200.0, "rms_mean_db": -15.0,
                "syllable_rate": 5.5, "gesture": "Open Palm",
            },
        )
        boundary = {"time_sec": 20, "from_regime": "Introduction", "to_regime": "Main Argument"}
        result = scorer.score_transition(timeline, boundary)
        assert result["score"] >= 0.8
        assert len(result["channels_misaligned"]) == 0

    def test_no_adaptation(self, scorer):
        """Speaker who doesn't change at all should score around 0.5 (neutral)."""
        timeline = build_transition_timeline(
            before_kwargs={"f0_mean": 150.0, "syllable_rate": 4.0},
            after_kwargs={"f0_mean": 150.0, "syllable_rate": 4.0},
        )
        boundary = {"time_sec": 20, "from_regime": "Introduction", "to_regime": "Main Argument"}
        result = scorer.score_transition(timeline, boundary)
        # Near-zero deltas → 0 points → score = max_possible / (2 * max_possible) = 0.5
        assert abs(result["score"] - 0.5) < 0.15

    def test_wrong_direction(self, scorer):
        """Speaker who adapts in the wrong direction should score low."""
        # Intro→Main: energy should increase, but we decrease it
        timeline = build_transition_timeline(
            before_kwargs={
                "f0_mean": 200.0, "rms_mean_db": -10.0,
                "syllable_rate": 5.0, "gesture": "Open Palm",
            },
            after_kwargs={
                "f0_mean": 80.0, "rms_mean_db": -35.0,
                "syllable_rate": 2.0, "gesture": "Rest",
            },
        )
        boundary = {"time_sec": 20, "from_regime": "Introduction", "to_regime": "Main Argument"}
        result = scorer.score_transition(timeline, boundary)
        assert result["score"] < 0.4
        assert len(result["channels_misaligned"]) > 0

    def test_score_all_transitions(self, scorer):
        """score_all_transitions returns correct structure."""
        timeline = []
        for sec in range(60):
            if sec < 20:
                regime = "Introduction"
            elif sec < 40:
                regime = "Main Argument"
            else:
                regime = "Conclusion"
            timeline.append(make_timeline_entry(
                time_sec=sec,
                regime_type=regime,
                f0_mean=150.0 + sec,
                syllable_rate=4.0 + sec * 0.05,
            ))

        boundaries = [
            {"time_sec": 20, "from_regime": "Introduction", "to_regime": "Main Argument"},
            {"time_sec": 40, "from_regime": "Main Argument", "to_regime": "Conclusion"},
        ]

        result = scorer.score_all_transitions(timeline, boundaries)
        assert len(result["transitions"]) == 2
        assert 0.0 <= result["mean_score"] <= 1.0
        assert 0.0 <= result["adaptability_score"] <= 100.0
        assert result["best_transition"] is not None
        assert result["worst_transition"] is not None

    def test_no_boundaries(self, scorer):
        """No boundaries should return neutral scores."""
        timeline = [make_timeline_entry(i) for i in range(30)]
        result = scorer.score_all_transitions(timeline, [])
        assert result["mean_score"] == 0.5
        assert result["adaptability_score"] == 50.0
        assert len(result["transitions"]) == 0

    def test_unknown_transition_type(self, scorer):
        """Unknown regime pair should use default (all neutral) expectations."""
        timeline = build_transition_timeline(
            before_kwargs={"regime_type": "WeirdRegime"},
            after_kwargs={"regime_type": "OtherWeird"},
        )
        boundary = {"time_sec": 20, "from_regime": "WeirdRegime", "to_regime": "OtherWeird"}
        result = scorer.score_transition(timeline, boundary)
        # All expected=0 → score=0.5
        assert abs(result["score"] - 0.5) < 0.01

    def test_wildcard_to_conclusion(self, scorer):
        """Any regime → Conclusion should match the wildcard entry."""
        timeline = build_transition_timeline(
            before_kwargs={"regime_type": "Anecdote"},
            after_kwargs={"regime_type": "Conclusion"},
        )
        boundary = {"time_sec": 20, "from_regime": "Anecdote", "to_regime": "Conclusion"}
        result = scorer.score_transition(timeline, boundary)
        # Should use the specific ("Anecdote", "Conclusion") entry
        assert result["expected"] is not None

    def test_deltas_have_all_channels(self, scorer):
        """Deltas dict should contain all 6 channels."""
        timeline = build_transition_timeline(
            before_kwargs={}, after_kwargs={},
        )
        boundary = {"time_sec": 20, "from_regime": "Introduction", "to_regime": "Main Argument"}
        result = scorer.score_transition(timeline, boundary)
        for channel in ["energy", "rate", "variety", "gesture", "gaze", "posture"]:
            assert channel in result["deltas"]

    def test_channels_aligned_subset_of_expected(self, scorer):
        """Aligned channels should only include channels with non-zero expectations."""
        timeline = build_transition_timeline(
            before_kwargs={"f0_mean": 100.0, "syllable_rate": 2.0, "gesture": "Rest"},
            after_kwargs={"f0_mean": 200.0, "syllable_rate": 5.0, "gesture": "Open Palm"},
        )
        boundary = {"time_sec": 20, "from_regime": "Introduction", "to_regime": "Main Argument"}
        result = scorer.score_transition(timeline, boundary)
        expected = result["expected"]
        for ch in result["channels_aligned"]:
            assert expected.get(ch, 0) != 0

    def test_boundary_at_edge(self, scorer):
        """Boundary at very start should not crash."""
        timeline = [make_timeline_entry(i) for i in range(20)]
        boundary = {"time_sec": 1, "from_regime": "Introduction", "to_regime": "Main Argument"}
        result = scorer.score_transition(timeline, boundary)
        assert 0.0 <= result["score"] <= 1.0

    def test_expected_direction_matrix_complete(self):
        """All entries in the direction matrix should have 6 channels."""
        for key, directions in EXPECTED_DIRECTION.items():
            for ch in ["energy", "rate", "variety", "gesture", "gaze", "posture"]:
                assert ch in directions, f"Missing {ch} in {key}"
