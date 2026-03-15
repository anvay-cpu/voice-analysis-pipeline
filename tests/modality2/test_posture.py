"""Tests for posture scoring module."""

import numpy as np
import pytest
import torch

from src.body.posture_scorer import (
    PostureMLP,
    PostureScorer,
    compute_rule_score,
    shoulder_alignment_score,
    spine_angle_score,
    weight_distribution_score,
    head_position_score,
    shoulder_tension_score,
    _linear_decay,
    MLP_INPUT_DIM,
)


# ── Helpers ───────────────────────────────────────────────────


def _make_ideal_keypoints() -> np.ndarray:
    """Create perfectly upright, symmetric pose keypoints."""
    kp = np.zeros((33, 4), dtype=np.float32)
    # Set high visibility for all
    kp[:, 3] = 0.95
    # Nose centered
    kp[0] = [0.50, 0.20, 0, 0.95]
    # Ears
    kp[7] = [0.45, 0.22, 0, 0.95]
    kp[8] = [0.55, 0.22, 0, 0.95]
    # Shoulders — level
    kp[11] = [0.40, 0.35, 0, 0.95]
    kp[12] = [0.60, 0.35, 0, 0.95]
    # Hips — symmetric
    kp[23] = [0.43, 0.60, 0, 0.95]
    kp[24] = [0.57, 0.60, 0, 0.95]
    return kp


def _make_slouched_keypoints() -> np.ndarray:
    """Create obviously slouched/tilted pose."""
    kp = _make_ideal_keypoints()
    # Tilt shoulders heavily
    kp[11, 1] = 0.30  # left shoulder up
    kp[12, 1] = 0.45  # right shoulder down
    # Lean spine
    kp[11, 0] = 0.30
    kp[12, 0] = 0.50
    # Head off center
    kp[0, 0] = 0.35
    return kp


# ── Linear Decay ──────────────────────────────────────────────


class TestLinearDecay:
    def test_below_good(self):
        assert _linear_decay(1.0, 3.0, 15.0) == 100.0

    def test_above_bad(self):
        assert _linear_decay(20.0, 3.0, 15.0) == 0.0

    def test_midpoint(self):
        score = _linear_decay(9.0, 3.0, 15.0)
        assert 40 < score < 60


# ── Individual Rule Scores ────────────────────────────────────


class TestRuleScores:
    def test_ideal_shoulder_alignment(self):
        kp = _make_ideal_keypoints()
        score = shoulder_alignment_score(kp)
        assert score >= 90

    def test_bad_shoulder_alignment(self):
        kp = _make_slouched_keypoints()
        score = shoulder_alignment_score(kp)
        assert score < 60

    def test_ideal_spine(self):
        kp = _make_ideal_keypoints()
        score = spine_angle_score(kp)
        assert score >= 80

    def test_ideal_weight_distribution(self):
        kp = _make_ideal_keypoints()
        score = weight_distribution_score(kp)
        assert score >= 80

    def test_ideal_head_position(self):
        kp = _make_ideal_keypoints()
        score = head_position_score(kp)
        assert score >= 80

    def test_low_visibility_returns_neutral(self):
        kp = _make_ideal_keypoints()
        kp[:, 3] = 0.1  # low visibility
        score = shoulder_alignment_score(kp)
        assert score == 50.0  # neutral fallback


# ── Combined Rule Score ───────────────────────────────────────


class TestCombinedRuleScore:
    def test_ideal_pose_high_score(self):
        kp = _make_ideal_keypoints()
        result = compute_rule_score(kp)
        assert result["combined"] >= 70
        assert all(k in result for k in [
            "shoulder_alignment", "spine_angle",
            "weight_distribution", "head_position",
            "shoulder_tension", "combined",
        ])

    def test_slouched_lower_score(self):
        ideal = compute_rule_score(_make_ideal_keypoints())["combined"]
        slouched = compute_rule_score(_make_slouched_keypoints())["combined"]
        assert slouched < ideal

    def test_score_range(self):
        kp = _make_ideal_keypoints()
        result = compute_rule_score(kp)
        for key, val in result.items():
            assert 0 <= val <= 100, f"{key}={val} out of range"


# ── PostureMLP ────────────────────────────────────────────────


class TestPostureMLP:
    def test_forward_shape(self):
        model = PostureMLP()
        x = torch.randn(4, MLP_INPUT_DIM)
        out = model(x)
        assert out.shape == (4,)

    def test_single_sample(self):
        model = PostureMLP()
        model.eval()  # BatchNorm needs eval mode for single samples
        x = torch.randn(1, MLP_INPUT_DIM)
        out = model(x)
        assert out.shape == (1,)


# ── PostureScorer (hybrid) ────────────────────────────────────


class TestPostureScorer:
    def test_rule_only_mode(self):
        """Without MLP, should use rule-based only."""
        scorer = PostureScorer(model_path=None)
        kp = _make_ideal_keypoints()
        result = scorer.score_frame(kp)
        assert result["mlp_score"] is None
        assert result["combined_score"] == result["rule_score"]
        assert 0 <= result["combined_score"] <= 100

    def test_score_batch(self):
        scorer = PostureScorer(model_path=None)
        poses = {
            0: _make_ideal_keypoints(),
            1: None,
            2: _make_slouched_keypoints(),
        }
        results = scorer.score_batch(poses)
        assert results[0] is not None
        assert results[1] is None
        assert results[2] is not None
        assert results[0]["combined_score"] > results[2]["combined_score"]

    def test_missing_model_path_falls_back(self):
        scorer = PostureScorer(model_path="/nonexistent/model.pt")
        assert scorer.mlp is None
