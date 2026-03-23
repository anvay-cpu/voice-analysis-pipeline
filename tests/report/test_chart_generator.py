"""Tests for Phase 5.3 — Chart Generation."""

import os
import pytest

from src.report.chart_generator import ChartGenerator
from src.scoring.dimension_scorer import DimensionScorer
from tests.scoring.test_dimension_scorer import make_fusion_output


@pytest.fixture
def tmp_chart_dir(tmp_path):
    return str(tmp_path / "charts")


@pytest.fixture
def generator(tmp_chart_dir):
    return ChartGenerator(output_dir=tmp_chart_dir, dpi=72)


@pytest.fixture
def fusion_output():
    return make_fusion_output(30)


@pytest.fixture
def scores(fusion_output):
    return DimensionScorer().score_all(fusion_output)


class TestChartGenerator:

    def test_radar_chart_created(self, generator, scores):
        """Radar chart should be saved as PNG."""
        path = generator.radar_chart(scores)
        assert os.path.exists(path)
        assert path.endswith(".png")
        assert os.path.getsize(path) > 1000  # Not empty

    def test_timeline_heatmap_created(self, generator, fusion_output):
        """Timeline heatmap should be saved as PNG."""
        path = generator.timeline_heatmap(fusion_output["timeline"])
        assert os.path.exists(path)
        assert os.path.getsize(path) > 1000

    def test_emotion_arc_created(self, generator, fusion_output):
        """Emotion arc should be saved as PNG."""
        path = generator.emotion_arc(fusion_output["timeline"], fusion_output)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 1000

    def test_regime_flow_created(self, generator, fusion_output):
        """Regime flow should be saved as PNG."""
        path = generator.regime_flow(fusion_output)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 1000

    def test_filler_map_created(self, generator, fusion_output):
        """Filler map should be saved as PNG."""
        path = generator.filler_map(fusion_output["timeline"])
        assert os.path.exists(path)
        assert os.path.getsize(path) > 1000

    def test_generate_all(self, generator, scores, fusion_output):
        """generate_all should return all 5 chart paths."""
        paths = generator.generate_all(scores, fusion_output["timeline"], fusion_output)
        assert len(paths) == 5
        for name, path in paths.items():
            assert os.path.exists(path), f"{name} chart not found at {path}"

    def test_empty_timeline_no_crash(self, generator):
        """Empty timeline should produce placeholder charts."""
        path = generator.timeline_heatmap([])
        assert os.path.exists(path)

    def test_empty_fusion_no_crash(self, generator):
        """Empty fusion output should not crash."""
        empty = {"timeline": [], "regime_boundaries": [], "transitions": {"transitions": []}}
        path = generator.regime_flow(empty)
        assert os.path.exists(path)

    def test_radar_with_empty_scores(self, generator):
        """Radar with empty scores should use defaults."""
        path = generator.radar_chart({})
        assert os.path.exists(path)
