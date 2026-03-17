"""Tests for stage movement analysis (Phase 9)."""

import math

import numpy as np
import pytest

from src.body.stage_movement import (
    ANCHORED,
    PACING,
    PURPOSEFUL,
    ROAMING,
    StageMovementAnalyzer,
    classify_movement,
    compute_autocorrelation,
    compute_centroids,
    compute_convex_hull_area,
    compute_directional_entropy,
    compute_displacement,
    compute_mean_velocity,
    score_stage_usage,
)


# ─── Helpers ───


def _make_pose(cx, cy, vis=0.9):
    """Create a (33, 4) keypoint array with hips at given centroid."""
    kp = np.zeros((33, 4), dtype=np.float32)
    # Left hip (idx 23) slightly left of center
    kp[23] = [cx - 0.02, cy, 0, vis]
    # Right hip (idx 24) slightly right of center
    kp[24] = [cx + 0.02, cy, 0, vis]
    return kp


def _make_poses_stationary(n=100, cx=0.5, cy=0.5):
    """Stationary speaker — all frames at same position."""
    return {i: _make_pose(cx, cy) for i in range(n)}


def _make_poses_pacing(n=100, y=0.5):
    """Pacing speaker — oscillates left-right."""
    poses = {}
    for i in range(n):
        # Sinusoidal x oscillation: period ~30 frames
        cx = 0.5 + 0.15 * math.sin(2 * math.pi * i / 30)
        poses[i] = _make_pose(cx, y)
    return poses


def _make_poses_roaming(n=100):
    """Roaming speaker — moves around the stage randomly."""
    rng = np.random.RandomState(42)
    poses = {}
    cx, cy = 0.5, 0.5
    for i in range(n):
        cx += rng.uniform(-0.02, 0.02)
        cy += rng.uniform(-0.02, 0.02)
        cx = np.clip(cx, 0.1, 0.9)
        cy = np.clip(cy, 0.1, 0.9)
        poses[i] = _make_pose(cx, cy)
    return poses


def _make_poses_purposeful(n=100):
    """Purposeful speaker — moves to spots then pauses."""
    poses = {}
    positions = [(0.3, 0.5), (0.5, 0.5), (0.7, 0.5)]
    segment = n // len(positions)
    for i in range(n):
        pos_idx = min(i // segment, len(positions) - 1)
        cx, cy = positions[pos_idx]
        # Add tiny jitter during pauses
        cx += np.random.RandomState(i).uniform(-0.005, 0.005)
        poses[i] = _make_pose(cx, cy)
    return poses


# ─── Centroid Tests ───


def test_centroids_from_poses():
    """Centroids computed as hip midpoints."""
    poses = _make_poses_stationary(10, cx=0.5, cy=0.6)
    centroids = compute_centroids(poses)
    assert len(centroids) == 10
    for idx, cx, cy in centroids:
        assert abs(cx - 0.5) < 0.03
        assert abs(cy - 0.6) < 0.01


def test_centroids_skip_none():
    """None poses are skipped."""
    poses = {0: _make_pose(0.5, 0.5), 1: None, 2: _make_pose(0.5, 0.5)}
    centroids = compute_centroids(poses)
    assert len(centroids) == 2


def test_centroids_skip_low_visibility():
    """Low visibility hips are skipped."""
    poses = {0: _make_pose(0.5, 0.5, vis=0.1)}
    centroids = compute_centroids(poses)
    assert len(centroids) == 0


def test_centroids_empty():
    """Empty poses → empty centroids."""
    assert compute_centroids({}) == []


# ─── Displacement Tests ───


def test_displacement_stationary():
    """Stationary speaker → near-zero displacement."""
    centroids = [(i, 0.5, 0.5) for i in range(50)]
    assert compute_displacement(centroids) == 0.0


def test_displacement_moving():
    """Linear movement → positive displacement."""
    centroids = [(i, 0.1 + i * 0.01, 0.5) for i in range(10)]
    disp = compute_displacement(centroids)
    assert disp > 0.08


def test_displacement_single_point():
    """Single point → zero."""
    assert compute_displacement([(0, 0.5, 0.5)]) == 0.0


# ─── Velocity Tests ───


def test_velocity_stationary():
    """Stationary → zero velocity."""
    centroids = [(i, 0.5, 0.5) for i in range(50)]
    assert compute_mean_velocity(centroids) == 0.0


def test_velocity_constant_speed():
    """Constant movement → positive velocity."""
    centroids = [(i, 0.1 + i * 0.01, 0.5) for i in range(20)]
    vel = compute_mean_velocity(centroids)
    assert vel > 0


# ─── Convex Hull Tests ───


def test_hull_stationary():
    """All points at same location → zero area."""
    centroids = [(i, 0.5, 0.5) for i in range(10)]
    assert compute_convex_hull_area(centroids) == 0.0


def test_hull_spread():
    """Spread points → positive area."""
    centroids = [
        (0, 0.2, 0.2), (1, 0.8, 0.2),
        (2, 0.8, 0.8), (3, 0.2, 0.8),
    ]
    area = compute_convex_hull_area(centroids)
    assert area > 0.3  # ~0.36 for unit square corners


def test_hull_collinear():
    """Collinear points → zero area."""
    centroids = [(i, 0.1 * i, 0.5) for i in range(5)]
    assert compute_convex_hull_area(centroids) == 0.0


def test_hull_too_few_points():
    """Fewer than 3 points → zero."""
    assert compute_convex_hull_area([(0, 0.5, 0.5), (1, 0.6, 0.5)]) == 0.0


# ─── Directional Entropy Tests ───


def test_entropy_single_direction():
    """Constant direction → low entropy."""
    centroids = [(i, 0.1 + i * 0.01, 0.5) for i in range(20)]
    entropy = compute_directional_entropy(centroids)
    assert entropy < 1.0


def test_entropy_varied_directions():
    """Random movement → higher entropy."""
    rng = np.random.RandomState(42)
    centroids = [(i, rng.uniform(0, 1), rng.uniform(0, 1)) for i in range(100)]
    entropy = compute_directional_entropy(centroids)
    assert entropy > 2.0  # near-uniform over 8 bins → max 3.0


def test_entropy_too_few():
    """Too few points → zero."""
    assert compute_directional_entropy([(0, 0.5, 0.5), (1, 0.6, 0.5)]) == 0.0


# ─── Autocorrelation Tests ───


def test_autocorr_pacing():
    """Sinusoidal movement → high autocorrelation."""
    centroids = [
        (i, 0.5 + 0.15 * math.sin(2 * math.pi * i / 30), 0.5)
        for i in range(200)
    ]
    ac = compute_autocorrelation(centroids)
    assert ac > 0.5


def test_autocorr_stationary():
    """Stationary → zero autocorrelation."""
    centroids = [(i, 0.5, 0.5) for i in range(100)]
    assert compute_autocorrelation(centroids) == 0.0


def test_autocorr_too_few():
    """Too few frames → zero."""
    centroids = [(i, 0.5, 0.5) for i in range(10)]
    assert compute_autocorrelation(centroids) == 0.0


# ─── Classification Tests ───


def test_classify_anchored():
    """Low hull + low velocity → Anchored."""
    assert classify_movement(0.01, 0.001, 0.0, 0.0) == ANCHORED


def test_classify_pacing():
    """High autocorrelation → Pacing."""
    assert classify_movement(0.10, 0.01, 0.8, 1.0) == PACING


def test_classify_roaming():
    """Large hull + high entropy → Roaming."""
    assert classify_movement(0.15, 0.01, 0.3, 2.5) == ROAMING


def test_classify_purposeful():
    """Moderate movement, low autocorr → Purposeful."""
    assert classify_movement(0.08, 0.01, 0.2, 1.5) == PURPOSEFUL


# ─── Score Tests ───


def test_score_values():
    """Each pattern has expected score."""
    assert score_stage_usage(ANCHORED) == 5.0
    assert score_stage_usage(PACING) == 3.0
    assert score_stage_usage(PURPOSEFUL) == 8.0
    assert score_stage_usage(ROAMING) == 7.0


def test_score_unknown():
    """Unknown pattern → default 5.0."""
    assert score_stage_usage("Unknown") == 5.0


# ─── Analyzer Integration Tests ───


def test_analyzer_stationary():
    """Stationary speaker → Anchored pattern."""
    analyzer = StageMovementAnalyzer()
    result = analyzer.analyze(_make_poses_stationary(100))
    assert result["movement_pattern"] == ANCHORED
    assert result["stage_usage_score"] == 5.0
    assert result["num_frames"] == 100
    assert result["total_displacement"] < 0.1


def test_analyzer_pacing():
    """Pacing speaker → Pacing pattern."""
    analyzer = StageMovementAnalyzer()
    result = analyzer.analyze(_make_poses_pacing(200))
    assert result["movement_pattern"] == PACING
    assert result["stage_usage_score"] == 3.0


def test_analyzer_empty():
    """No poses → Anchored fallback."""
    analyzer = StageMovementAnalyzer()
    result = analyzer.analyze({})
    assert result["movement_pattern"] == ANCHORED
    assert result["num_frames"] == 0


def test_analyzer_output_structure():
    """Analyzer returns all expected keys."""
    analyzer = StageMovementAnalyzer()
    result = analyzer.analyze(_make_poses_stationary(50))
    expected_keys = {
        "centroids", "total_displacement", "mean_velocity",
        "convex_hull_area", "directional_entropy", "autocorrelation",
        "movement_pattern", "stage_usage_score", "num_frames",
    }
    assert set(result.keys()) == expected_keys


def test_analyzer_centroids_structure():
    """Centroid entries have frame_idx, x, y."""
    analyzer = StageMovementAnalyzer()
    result = analyzer.analyze(_make_poses_stationary(10))
    for c in result["centroids"]:
        assert "frame_idx" in c
        assert "x" in c
        assert "y" in c
