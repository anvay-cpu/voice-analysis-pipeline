"""Tests for Phase 5.4 — Report Builder."""

import os
import pytest

from src.report.report_builder import ReportBuilder
from src.report.chart_generator import ChartGenerator
from src.report.coaching_writer import CoachingWriter
from src.scoring.dimension_scorer import DimensionScorer
from tests.scoring.test_dimension_scorer import make_fusion_output


@pytest.fixture
def fusion_output():
    return make_fusion_output(30)


@pytest.fixture
def scores(fusion_output):
    return DimensionScorer().score_all(fusion_output)


@pytest.fixture
def charts(tmp_path, scores, fusion_output):
    gen = ChartGenerator(output_dir=str(tmp_path / "charts"), dpi=72)
    return gen.generate_all(scores, fusion_output["timeline"], fusion_output)


@pytest.fixture
def coaching(scores, fusion_output):
    writer = CoachingWriter(use_api=False)
    return writer.generate_full_report(
        all_segments=[], overall_scores=scores,
        fusion_output=fusion_output,
        speech_metadata={"duration_sec": 30},
    )


@pytest.fixture
def builder():
    return ReportBuilder()


class TestReportBuilder:

    def test_build_html_returns_string(self, builder, coaching, scores, charts):
        """build_html should return a non-empty HTML string."""
        html = builder.build_html(
            coaching, scores, charts, {"duration_sec": 30, "processing_time_sec": 5}
        )
        assert isinstance(html, str)
        assert len(html) > 500

    def test_html_contains_structure(self, builder, coaching, scores, charts):
        """HTML should contain key sections."""
        html = builder.build_html(
            coaching, scores, charts, {"duration_sec": 30, "processing_time_sec": 5}
        )
        assert "<!DOCTYPE html>" in html
        assert "Executive Summary" in html
        assert "Dimension Scores" in html
        assert "Practice Plan" in html
        assert "Speech Quality Timeline" in html

    def test_html_contains_scores(self, builder, coaching, scores, charts):
        """HTML should contain score values."""
        html = builder.build_html(
            coaching, scores, charts, {"duration_sec": 30, "processing_time_sec": 5}
        )
        assert "Vocal Clarity" in html
        assert "Body Language" in html
        assert "score-badge" in html

    def test_html_contains_chart_images(self, builder, coaching, scores, charts):
        """HTML should contain embedded chart images."""
        html = builder.build_html(
            coaching, scores, charts, {"duration_sec": 30, "processing_time_sec": 5}
        )
        assert "data:image/png;base64," in html

    def test_save_html(self, builder, coaching, scores, charts, tmp_path):
        """save_html should write a file."""
        html = builder.build_html(
            coaching, scores, charts, {"duration_sec": 30, "processing_time_sec": 5}
        )
        path = str(tmp_path / "report.html")
        result = builder.save_html(html, path)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 500

    def test_build_and_save(self, builder, coaching, scores, charts, tmp_path):
        """build_and_save should create HTML file."""
        output_dir = str(tmp_path / "output")
        result = builder.build_and_save(
            coaching, scores, charts,
            {"duration_sec": 30, "processing_time_sec": 5},
            output_dir,
        )
        assert "html" in result
        assert os.path.exists(result["html"])

    def test_empty_coaching_data(self, builder, scores, charts):
        """Empty coaching data should not crash."""
        html = builder.build_html(
            {}, scores, charts, {"duration_sec": 0, "processing_time_sec": 0}
        )
        assert "<!DOCTYPE html>" in html

    def test_no_charts(self, builder, coaching, scores):
        """Missing charts should produce valid HTML."""
        html = builder.build_html(
            coaching, scores, {}, {"duration_sec": 30, "processing_time_sec": 5}
        )
        assert "<!DOCTYPE html>" in html
        # Should not have chart images
        assert "data:image/png;base64," not in html

    def test_html_valid_for_browser(self, builder, coaching, scores, charts, tmp_path):
        """HTML should be openable (well-formed)."""
        html = builder.build_html(
            coaching, scores, charts, {"duration_sec": 30, "processing_time_sec": 5}
        )
        assert html.count("<html") == 1
        assert html.count("</html>") == 1
        assert html.count("<body") == 1
        assert html.count("</body>") == 1
