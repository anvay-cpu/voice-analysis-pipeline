"""Tests for Phase 4.5 — Fusion Engine integration."""

import pytest

from src.fusion.fusion_engine import FusionEngine
from tests.fusion.test_timeline import (
    make_voice_windows,
    make_body_output,
    make_content_output,
)


class TestFusionEngine:

    @pytest.fixture
    def engine(self):
        return FusionEngine()

    @pytest.fixture
    def engine_with_config(self):
        config = {
            "fusion": {
                "transition_context_sec": 5,
                "recovery": {
                    "filler_burst_threshold": 3,
                    "pause_threshold_sec": 3,
                },
                "coherence": {
                    "mismatch_threshold": 0.4,
                },
            }
        }
        return FusionEngine(config)

    @pytest.fixture
    def duration(self):
        return 30

    @pytest.fixture
    def voice(self, duration):
        return make_voice_windows(duration)

    @pytest.fixture
    def body(self, duration):
        return make_body_output(duration)

    @pytest.fixture
    def content(self, duration):
        return make_content_output(duration)

    def test_fuse_returns_all_keys(self, engine, voice, body, content, duration):
        """Fusion output should contain all expected top-level keys."""
        result = engine.fuse(voice, body, content, duration)
        assert "timeline" in result
        assert "duration_sec" in result
        assert "regime_boundaries" in result
        assert "transitions" in result
        assert "recovery" in result
        assert "emotion_coherence" in result

    def test_timeline_length(self, engine, voice, body, content, duration):
        """Timeline should match duration."""
        result = engine.fuse(voice, body, content, duration)
        assert len(result["timeline"]) == duration

    def test_regime_boundaries_detected(self, engine, voice, body, content, duration):
        """Should detect regime boundaries from content."""
        result = engine.fuse(voice, body, content, duration)
        assert len(result["regime_boundaries"]) == 2

    def test_transitions_scored(self, engine, voice, body, content, duration):
        """Transitions should be scored."""
        result = engine.fuse(voice, body, content, duration)
        assert len(result["transitions"]["transitions"]) == 2
        assert 0.0 <= result["transitions"]["mean_score"] <= 1.0

    def test_recovery_analysis_present(self, engine, voice, body, content, duration):
        """Recovery analysis should be present."""
        result = engine.fuse(voice, body, content, duration)
        assert "total_disruptions" in result["recovery"]
        assert "mean_composure" in result["recovery"]

    def test_coherence_present(self, engine, voice, body, content, duration):
        """Emotion coherence should be present."""
        result = engine.fuse(voice, body, content, duration)
        assert "mean_coherence" in result["emotion_coherence"]
        assert "coherence_score_0_100" in result["emotion_coherence"]

    def test_config_passed_through(self, engine_with_config, voice, body, content, duration):
        """Config should affect component behavior."""
        result = engine_with_config.fuse(voice, body, content, duration)
        # Should still produce valid output
        assert len(result["timeline"]) == duration

    def test_empty_inputs(self, engine):
        """All empty inputs should not crash."""
        result = engine.fuse([], None, None, 10)
        assert len(result["timeline"]) == 10
        assert result["recovery"]["total_disruptions"] == 0

    def test_duration_matches(self, engine, voice, body, content, duration):
        """Output duration should match input."""
        result = engine.fuse(voice, body, content, duration)
        assert result["duration_sec"] == duration

    def test_adaptability_score_present(self, engine, voice, body, content, duration):
        """Adaptability score (0-100) should be in transitions."""
        result = engine.fuse(voice, body, content, duration)
        assert 0.0 <= result["transitions"]["adaptability_score"] <= 100.0
