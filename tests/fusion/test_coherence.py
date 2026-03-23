"""Tests for Phase 4.4 — Emotion Coherence Analysis."""

import pytest

from src.fusion.emotion_coherence import (
    EmotionCoherenceAnalyzer,
    VOICE_VALENCE,
    FACE_VALENCE,
)


# ── Helpers ─────────────────────────────────────────────────────

def make_entry(
    time_sec,
    voice_emotion="Neutral",
    face_emotion="Neutral",
    sentiment_polarity="Neutral",
    sentiment_valence=0.0,
):
    """Create a timeline entry with specified emotions."""
    return {
        "time_sec": time_sec,
        "voice": {
            "is_speech": True,
            "emotion_label": voice_emotion,
            "emotion_confidence": 0.8,
            "syllable_rate": 4.0,
            "pitch_cv": 0.2,
            "f0_mean": 150.0,
            "rms_mean_db": -20.0,
        },
        "body": {
            "facial_emotion": face_emotion,
            "facial_emotion_confidence": 0.8,
            "posture_score": 70.0,
            "gesture_dominant": "Rest",
            "gaze_engagement": 0.7,
        },
        "content": {
            "regime_type": "Main Argument",
            "sentiment_polarity": sentiment_polarity,
            "sentiment_valence": sentiment_valence,
            "tone": "Informative",
        },
    }


# ── Tests ───────────────────────────────────────────────────────

class TestEmotionCoherence:

    @pytest.fixture
    def analyzer(self):
        return EmotionCoherenceAnalyzer()

    def test_perfect_coherence_all_neutral(self, analyzer):
        """All neutral emotions should give perfect coherence."""
        score = analyzer.compute_coherence_at_time("Neutral", "Neutral", 0.0)
        assert score == 1.0

    def test_perfect_coherence_all_positive(self, analyzer):
        """All positive emotions should give high coherence."""
        score = analyzer.compute_coherence_at_time("Enthusiastic", "Happy", 1.0)
        assert score >= 0.9

    def test_low_coherence_mixed(self, analyzer):
        """Opposite emotions should give low coherence."""
        score = analyzer.compute_coherence_at_time("Enthusiastic", "Anger", -1.0)
        assert score < 0.5

    def test_coherence_range(self, analyzer):
        """Coherence should always be 0.0-1.0."""
        for v_emo in VOICE_VALENCE:
            for f_emo in FACE_VALENCE:
                for val in [-1.0, 0.0, 1.0]:
                    score = analyzer.compute_coherence_at_time(v_emo, f_emo, val)
                    assert 0.0 <= score <= 1.0, (
                        f"Out of range: {v_emo}/{f_emo}/{val} → {score}"
                    )

    def test_detect_mismatches_none_in_coherent(self, analyzer):
        """Coherent timeline should have no mismatches."""
        timeline = [make_entry(i) for i in range(30)]
        mismatches = analyzer.detect_mismatches(timeline)
        assert len(mismatches) == 0

    def test_detect_mismatches_found(self, analyzer):
        """Incoherent emotions should be detected."""
        timeline = [make_entry(i) for i in range(30)]
        # Create mismatch: enthusiastic voice + angry face + negative content
        for i in range(10, 15):
            timeline[i] = make_entry(
                i,
                voice_emotion="Enthusiastic",
                face_emotion="Anger",
                sentiment_valence=-1.0,
            )

        mismatches = analyzer.detect_mismatches(timeline)
        assert len(mismatches) >= 1

    def test_mismatches_clustered(self, analyzer):
        """Consecutive mismatches should be clustered into single events."""
        timeline = [make_entry(i) for i in range(30)]
        for i in range(10, 15):
            timeline[i] = make_entry(
                i,
                voice_emotion="Enthusiastic",
                face_emotion="Anger",
                sentiment_valence=-1.0,
            )

        mismatches = analyzer.detect_mismatches(timeline)
        assert len(mismatches) == 1  # All within 5s → single cluster
        assert mismatches[0]["duration_sec"] == 5

    def test_two_separate_mismatch_events(self, analyzer):
        """Mismatches > gap apart should be separate events."""
        analyzer_wide = EmotionCoherenceAnalyzer(cluster_gap_sec=2)
        timeline = [make_entry(i) for i in range(40)]
        # Mismatch 1: seconds 5-7
        for i in range(5, 8):
            timeline[i] = make_entry(i, voice_emotion="Angry", face_emotion="Happy")
        # Mismatch 2: seconds 20-22 (>2s gap)
        for i in range(20, 23):
            timeline[i] = make_entry(i, voice_emotion="Angry", face_emotion="Happy")

        mismatches = analyzer_wide.detect_mismatches(timeline)
        assert len(mismatches) == 2

    def test_analyze_full_structure(self, analyzer):
        """analyze_full should return complete structure."""
        timeline = [make_entry(i) for i in range(30)]
        result = analyzer.analyze_full(timeline)

        assert "mean_coherence" in result
        assert "coherence_timeline" in result
        assert "mismatch_events" in result
        assert "total_mismatch_seconds" in result
        assert "coherence_score_0_100" in result
        assert len(result["coherence_timeline"]) == 30

    def test_analyze_full_score_range(self, analyzer):
        """Coherence score should be 0-100."""
        timeline = [make_entry(i) for i in range(30)]
        result = analyzer.analyze_full(timeline)
        assert 0 <= result["coherence_score_0_100"] <= 100

    def test_empty_timeline(self, analyzer):
        """Empty timeline should not crash."""
        result = analyzer.analyze_full([])
        assert result["mean_coherence"] == 1.0
        assert result["coherence_score_0_100"] == 100

    def test_missing_modality_data(self, analyzer):
        """Missing voice/body/content should default to neutral (no crash)."""
        timeline = [
            {"time_sec": 0, "voice": None, "body": None, "content": None},
            {"time_sec": 1, "voice": {}, "body": {}, "content": {}},
        ]
        result = analyzer.analyze_full(timeline)
        assert len(result["coherence_timeline"]) == 2
        # Should default to all-neutral → coherence = 1.0
        assert result["coherence_timeline"][0] == 1.0

    def test_describe_mismatch_nervous_smiling(self, analyzer):
        """Nervous voice + happy face should generate specific description."""
        desc = analyzer._describe_mismatch("Nervous", "Happy", 0.0)
        assert "nervousness" in desc.lower()

    def test_describe_mismatch_monotone_positive(self, analyzer):
        """Neutral voice + positive content should flag monotone delivery."""
        desc = analyzer._describe_mismatch("Neutral", "Neutral", 0.8)
        assert "monotone" in desc.lower()

    def test_sentiment_valence_used_over_polarity(self, analyzer):
        """Should use sentiment_valence when available."""
        # valence = 0.8 but polarity = Negative → valence should win
        entry = make_entry(0, sentiment_polarity="Negative", sentiment_valence=0.8)
        result = analyzer.analyze_full([entry])
        # With Neutral voice, Neutral face, and 0.8 valence:
        # voice_content = 1 - |0 - 0.8|/2 = 0.6
        assert result["coherence_timeline"][0] < 1.0
