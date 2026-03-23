"""Tests for Phase 5.5 — Master Pipeline (from pre-computed outputs)."""

import os
import pytest

from src.master_pipeline import SpeechCoachPipeline
from tests.fusion.test_timeline import (
    make_voice_windows,
    make_body_output,
    make_content_output,
)


class TestSpeechCoachPipeline:

    @pytest.fixture
    def pipeline(self):
        return SpeechCoachPipeline()

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

    def test_process_from_outputs(self, pipeline, voice, body, content, duration, tmp_path):
        """process_from_outputs should produce report files."""
        output_dir = str(tmp_path / "report")
        result = pipeline.process_from_outputs(
            voice_output=voice,
            body_output=body,
            content_output=content,
            duration_sec=duration,
            output_dir=output_dir,
        )

        assert "report" in result
        assert "scores" in result
        assert "charts" in result
        assert os.path.exists(result["report"]["html"])

    def test_scores_present(self, pipeline, voice, body, content, duration, tmp_path):
        """Scores should have all 7 keys."""
        result = pipeline.process_from_outputs(
            voice, body, content, duration, str(tmp_path / "out")
        )
        scores = result["scores"]
        assert len(scores) == 7
        assert "overall" in scores

    def test_charts_generated(self, pipeline, voice, body, content, duration, tmp_path):
        """All 5 charts should be generated."""
        result = pipeline.process_from_outputs(
            voice, body, content, duration, str(tmp_path / "out")
        )
        assert len(result["charts"]) == 5

    def test_html_report_contains_content(self, pipeline, voice, body, content, duration, tmp_path):
        """HTML report should contain key sections."""
        output_dir = str(tmp_path / "out")
        result = pipeline.process_from_outputs(
            voice, body, content, duration, output_dir
        )
        html_path = result["report"]["html"]
        with open(html_path, "r") as f:
            html = f.read()

        assert "Speech Coaching Report" in html
        assert "Executive Summary" in html
        assert "Practice Plan" in html
        assert "data:image/png;base64," in html

    def test_empty_inputs(self, pipeline, tmp_path):
        """Empty inputs should not crash."""
        result = pipeline.process_from_outputs(
            voice_output=[],
            body_output={"segments": []},
            content_output={"segments": []},
            duration_sec=10,
            output_dir=str(tmp_path / "empty"),
        )
        assert os.path.exists(result["report"]["html"])

    def test_voice_dict_format(self, pipeline, voice, body, content, duration, tmp_path):
        """Voice output as dict with 'windows' key should work."""
        voice_dict = {"windows": voice, "metadata": {"duration_sec": duration}}
        result = pipeline.process_from_outputs(
            voice_dict, body, content, duration, str(tmp_path / "out")
        )
        assert os.path.exists(result["report"]["html"])

    def test_config_loading(self):
        """Pipeline should load config without crashing."""
        pipeline = SpeechCoachPipeline(config_path="configs/fusion_config.yaml")
        assert pipeline.config is not None

    def test_config_missing_file(self):
        """Missing config should use defaults."""
        pipeline = SpeechCoachPipeline(config_path="nonexistent.yaml")
        assert pipeline.config == {}
