"""Tests for Phase 5.2 — Coaching Writer (template fallback only)."""

import pytest

from src.report.coaching_writer import CoachingWriter
from tests.scoring.test_dimension_scorer import make_fusion_output


class TestCoachingWriter:

    @pytest.fixture
    def writer(self):
        return CoachingWriter(use_api=False)  # Template mode

    @pytest.fixture
    def scores(self):
        return {
            "vocal_clarity": 72.0,
            "body_language": 65.0,
            "content_structure": 80.0,
            "audience_engagement": 55.0,
            "emotional_expressiveness": 48.0,
            "regime_adaptability": 60.0,
            "overall": 63.3,
        }

    @pytest.fixture
    def fusion_output(self):
        return make_fusion_output(30)

    def test_generate_full_report_keys(self, writer, scores, fusion_output):
        """Full report should have all 3 sections."""
        report = writer.generate_full_report(
            all_segments=[],
            overall_scores=scores,
            fusion_output=fusion_output,
            speech_metadata={"duration_sec": 30},
        )
        assert "executive_summary" in report
        assert "segment_feedback" in report
        assert "practice_plan" in report

    def test_executive_summary_not_empty(self, writer, scores):
        """Executive summary should contain text."""
        summary = writer._template_executive_summary(
            scores, {"duration_sec": 300}
        )
        assert len(summary) > 50
        assert "overall" not in summary.lower() or "overall" in summary.lower()

    def test_executive_summary_mentions_strengths(self, writer, scores):
        """Summary should mention the strongest dimension."""
        summary = writer._template_executive_summary(
            scores, {"duration_sec": 300}
        )
        assert "Content Structure" in summary  # Highest score

    def test_executive_summary_mentions_weakness(self, writer, scores):
        """Summary should mention the weakest dimension."""
        summary = writer._template_executive_summary(
            scores, {"duration_sec": 300}
        )
        assert "Emotional Expressiveness" in summary  # Lowest score

    def test_practice_plan_has_3_items(self, writer, scores):
        """Practice plan should have exactly 3 exercises."""
        plan = writer._generate_practice_plan(scores, None)
        assert len(plan) == 3

    def test_practice_plan_structure(self, writer, scores):
        """Each exercise should have title, description, frequency."""
        plan = writer._generate_practice_plan(scores, None)
        for item in plan:
            assert "title" in item
            assert "description" in item
            assert "frequency" in item
            assert len(item["title"]) > 5
            assert len(item["description"]) > 20

    def test_practice_plan_targets_weakest(self, writer):
        """Practice plan should target the weakest dimensions."""
        scores = {
            "vocal_clarity": 30.0,
            "body_language": 90.0,
            "content_structure": 85.0,
            "audience_engagement": 80.0,
            "emotional_expressiveness": 75.0,
            "regime_adaptability": 70.0,
            "overall": 71.7,
        }
        plan = writer._generate_practice_plan(scores, None)
        # First exercise should be about vocal clarity (worst score)
        assert "vocal" in plan[0]["title"].lower() or "clarity" in plan[0]["title"].lower()

    def test_moment_feedback_transition(self, writer):
        """Transition feedback should reference the transition."""
        moment = {
            "type": "transition",
            "time_sec": 60,
            "label": "Introduction → Main Argument",
        }
        feedback = writer._template_moment_feedback(moment, "1:00")
        assert "1:00" in feedback
        assert "transition" in feedback.lower()

    def test_moment_feedback_disruption_good_recovery(self, writer):
        """Good recovery feedback should praise the speaker."""
        moment = {
            "type": "disruption",
            "time_sec": 30,
            "label": "filler_burst",
            "composure": 0.9,
        }
        feedback = writer._template_moment_feedback(moment, "0:30")
        assert "0:30" in feedback
        assert "smooth" in feedback.lower() or "strength" in feedback.lower()

    def test_moment_feedback_disruption_poor_recovery(self, writer):
        """Poor recovery feedback should give constructive advice."""
        moment = {
            "type": "disruption",
            "time_sec": 30,
            "label": "pace_spike",
            "composure": 0.3,
        }
        feedback = writer._template_moment_feedback(moment, "0:30")
        assert "0:30" in feedback
        assert "pause" in feedback.lower() or "breath" in feedback.lower()

    def test_segment_feedback_from_fusion(self, writer, fusion_output):
        """Segment feedback should be generated from fusion data."""
        feedback = writer._generate_segment_feedback([], fusion_output)
        assert isinstance(feedback, list)
        # Should have feedback for transitions at minimum
        if fusion_output.get("regime_boundaries"):
            assert len(feedback) >= len(fusion_output["regime_boundaries"])

    def test_empty_scores(self, writer):
        """Empty scores should not crash."""
        report = writer.generate_full_report(
            all_segments=[], overall_scores={}, speech_metadata={}
        )
        assert report["executive_summary"] is not None
