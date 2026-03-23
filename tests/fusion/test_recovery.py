"""Tests for Phase 4.3 — Recovery Analysis."""

import pytest

from src.fusion.recovery_analyzer import RecoveryAnalyzer


# ── Helpers ─────────────────────────────────────────────────────

def make_entry(
    time_sec,
    is_speech=True,
    filler_count=0,
    disfluency_count=0,
    syllable_rate=4.0,
    f0_mean=150.0,
    pitch_cv=0.2,
    posture_score=70.0,
    gesture="Rest",
    regime_type="Main Argument",
):
    """Create a single timeline entry for testing."""
    return {
        "time_sec": time_sec,
        "voice": {
            "is_speech": is_speech,
            "transcript": "word",
            "word_count": 2,
            "filler_count": filler_count,
            "fillers": [{"word": "um"}] * filler_count,
            "disfluency_count": disfluency_count,
            "disfluencies": [{"type": "repetition"}] * disfluency_count,
            "syllable_rate": syllable_rate,
            "rate_label": "Normal",
            "pitch_cv": pitch_cv,
            "pitch_mean": f0_mean,
            "pitch_score": 70,
            "volume_score": 75,
            "emotion_label": "Confident",
            "emotion_confidence": 0.8,
            "f0_mean": f0_mean,
            "f0_cv": pitch_cv,
            "rms_mean_db": -20.0,
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
            "gaze_engagement": 0.7,
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


def make_clean_timeline(duration: int = 60):
    """Create a clean timeline with no disruptions."""
    return [make_entry(i) for i in range(duration)]


# ── Tests ───────────────────────────────────────────────────────

class TestRecoveryAnalyzer:

    @pytest.fixture
    def analyzer(self):
        return RecoveryAnalyzer()

    def test_no_disruptions_in_clean_speech(self, analyzer):
        """Clean speech should have 0 disruptions."""
        timeline = make_clean_timeline(60)
        result = analyzer.analyze_all(timeline)
        assert result["total_disruptions"] == 0
        assert result["mean_composure"] == 1.0

    def test_detect_filler_burst(self, analyzer):
        """3+ fillers in 10 seconds should be detected."""
        timeline = make_clean_timeline(60)
        # Place 3 fillers at seconds 20-22
        for i in range(20, 23):
            timeline[i] = make_entry(i, filler_count=1)

        disruptions = analyzer.detect_disruptions(timeline)
        filler_disruptions = [d for d in disruptions if "filler_burst" in d["type"]]
        assert len(filler_disruptions) >= 1

    def test_detect_disfluency(self, analyzer):
        """Disfluency events should be detected."""
        timeline = make_clean_timeline(60)
        timeline[30] = make_entry(30, disfluency_count=2)

        disruptions = analyzer.detect_disruptions(timeline)
        dis_events = [d for d in disruptions if "disfluency" in d["type"]]
        assert len(dis_events) >= 1

    def test_detect_long_pause(self, analyzer):
        """Silence > 3 seconds should be detected."""
        timeline = make_clean_timeline(60)
        # 4 seconds of silence
        for i in range(25, 29):
            timeline[i] = make_entry(i, is_speech=False)

        disruptions = analyzer.detect_disruptions(timeline)
        pauses = [d for d in disruptions if "long_pause" in d["type"]]
        assert len(pauses) >= 1

    def test_regime_boundary_pause_not_detected(self, analyzer):
        """Silence at a regime boundary should NOT be a disruption."""
        timeline = make_clean_timeline(60)
        # Regime change at second 30
        for i in range(30, 60):
            timeline[i] = make_entry(i, regime_type="Conclusion")
        # Silence at the boundary
        for i in range(29, 32):
            timeline[i] = make_entry(
                i, is_speech=False,
                regime_type="Main Argument" if i < 30 else "Conclusion"
            )

        disruptions = analyzer.detect_disruptions(timeline)
        pauses = [d for d in disruptions if "long_pause" in d["type"]]
        assert len(pauses) == 0

    def test_detect_pace_spike(self, analyzer):
        """Speaking rate jump > 40% should be detected."""
        timeline = make_clean_timeline(60)
        # Sudden rate spike at second 30
        timeline[30] = make_entry(30, syllable_rate=7.0)  # 4.0 → 7.0 = 75% increase

        disruptions = analyzer.detect_disruptions(timeline)
        spikes = [d for d in disruptions if "pace_spike" in d["type"]]
        assert len(spikes) >= 1

    def test_detect_posture_collapse(self, analyzer):
        """Posture drop > 15 points in 3 seconds should be detected."""
        timeline = make_clean_timeline(60)
        # Posture collapses at second 25
        timeline[25] = make_entry(25, posture_score=50.0)  # 70 → 50 = 20 point drop

        disruptions = analyzer.detect_disruptions(timeline)
        collapses = [d for d in disruptions if "posture_collapse" in d["type"]]
        assert len(collapses) >= 1

    def test_fast_recovery_high_composure(self, analyzer):
        """Quick recovery should yield high composure score."""
        timeline = make_clean_timeline(60)
        # Single disfluency at second 30, all other seconds are normal
        timeline[30] = make_entry(30, disfluency_count=1)

        result = analyzer.analyze_all(timeline)
        if result["total_disruptions"] > 0:
            for r in result["recoveries"]:
                # Should recover within 1-2 seconds (next second is normal)
                assert r["composure_score"] >= 0.6

    def test_analyze_all_structure(self, analyzer):
        """analyze_all should return complete structure."""
        timeline = make_clean_timeline(60)
        timeline[20] = make_entry(20, disfluency_count=2)

        result = analyzer.analyze_all(timeline)
        assert "total_disruptions" in result
        assert "disruptions_per_minute" in result
        assert "recoveries" in result
        assert "mean_composure" in result

    def test_recovery_with_topic_change(self, analyzer):
        """Topic change after disruption should lower composure."""
        timeline = make_clean_timeline(60)
        # Disfluency at second 30
        timeline[30] = make_entry(30, disfluency_count=2)
        # Topic changes right after
        for i in range(32, 60):
            timeline[i] = make_entry(i, regime_type="Conclusion")

        result = analyzer.analyze_all(timeline)
        if result["total_disruptions"] > 0:
            for r in result["recoveries"]:
                assert r["content_maintained"] is False

    def test_merge_overlapping_disruptions(self, analyzer):
        """Overlapping disruptions should be merged."""
        timeline = make_clean_timeline(60)
        # Disfluency at 30 and 31 (within 3s merge window)
        timeline[30] = make_entry(30, disfluency_count=2)
        timeline[31] = make_entry(31, disfluency_count=1)

        disruptions = analyzer.detect_disruptions(timeline)
        # Should be merged into 1 disruption
        times = [d["time_sec"] for d in disruptions]
        # No two disruptions within 3s of each other
        for i in range(1, len(times)):
            assert times[i] - times[i - 1] > 3

    def test_empty_timeline(self, analyzer):
        """Empty timeline should not crash."""
        result = analyzer.analyze_all([])
        assert result["total_disruptions"] == 0
        assert result["mean_composure"] == 1.0

    def test_composure_score_range(self, analyzer):
        """Composure scores should always be 0.0-1.0."""
        timeline = make_clean_timeline(60)
        timeline[20] = make_entry(20, filler_count=3)
        timeline[21] = make_entry(21, filler_count=2)
        timeline[40] = make_entry(40, disfluency_count=2, posture_score=40.0)

        result = analyzer.analyze_all(timeline)
        for r in result["recoveries"]:
            assert 0.0 <= r["composure_score"] <= 1.0
