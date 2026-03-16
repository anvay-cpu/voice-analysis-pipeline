"""Tests for gesture classification (Phase 5)."""

import numpy as np
import pytest
import torch

from src.body.gesture_classifier import (
    GESTURE_CLASSES,
    GestureClassifier,
    GestureTransformer,
    KEYPOINT_DIM,
    NUM_CLASSES,
    WINDOW_FRAMES,
)


# ─── Model Architecture Tests ───


def test_transformer_output_shape():
    """Transformer produces correct output shape."""
    model = GestureTransformer()
    x = torch.randn(4, 15, 132)
    out = model(x)
    assert out.shape == (4, NUM_CLASSES)


def test_transformer_single_sample():
    """Transformer works with batch size 1."""
    model = GestureTransformer()
    x = torch.randn(1, 15, 132)
    out = model(x)
    assert out.shape == (1, NUM_CLASSES)


def test_transformer_param_count():
    """Model has roughly expected parameter count (~3.8M)."""
    model = GestureTransformer()
    params = sum(p.numel() for p in model.parameters())
    assert 2_000_000 < params < 6_000_000, f"Unexpected param count: {params}"


def test_transformer_gradient_flow():
    """Gradients flow through the full model."""
    model = GestureTransformer()
    x = torch.randn(2, 15, 132, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


# ─── Window Creation Tests ───


def _make_poses(n_frames=100):
    """Create dummy pose data."""
    poses = {}
    for i in range(n_frames):
        kp = np.random.rand(33, 4).astype(np.float32)
        kp[:, :2] = kp[:, :2] * 0.5 + 0.25  # keep in center
        poses[i] = kp
    return poses


def test_create_windows_count():
    """Correct number of windows created."""
    gc = GestureClassifier(model_path="nonexistent", device="cpu")
    poses = _make_poses(100)
    windows = gc.create_windows(poses)
    expected = (100 - WINDOW_FRAMES) // gc.hop_frames + 1
    assert abs(len(windows) - expected) <= 1


def test_create_windows_shape():
    """Each window has correct keypoint shape."""
    gc = GestureClassifier(model_path="nonexistent", device="cpu")
    poses = _make_poses(50)
    windows = gc.create_windows(poses)
    assert len(windows) > 0
    for w in windows:
        assert w["keypoints"].shape == (WINDOW_FRAMES, KEYPOINT_DIM)


def test_create_windows_skips_sparse():
    """Windows with too many missing frames are skipped."""
    gc = GestureClassifier(model_path="nonexistent", device="cpu")
    poses = {}
    # Only every 5th frame has data — too sparse for 80% threshold
    for i in range(0, 100, 5):
        poses[i] = np.random.rand(33, 4).astype(np.float32)
    # Fill intervening indices as None
    for i in range(100):
        if i not in poses:
            poses[i] = None
    windows = gc.create_windows(poses)
    assert len(windows) == 0  # 3/15 valid < 80%


def test_create_windows_empty():
    """Empty poses produce no windows."""
    gc = GestureClassifier(model_path="nonexistent", device="cpu")
    assert gc.create_windows({}) == []


# ─── Heuristic Classification Tests ───


def test_heuristic_rest():
    """Stationary hands below waist → Rest."""
    gc = GestureClassifier(model_path="nonexistent", device="cpu")
    poses = {}
    for i in range(50):
        kp = np.zeros((33, 4), dtype=np.float32)
        # Shoulders at y=0.35, hips at y=0.55
        kp[11, :2] = [0.4, 0.35]  # L shoulder
        kp[12, :2] = [0.6, 0.35]  # R shoulder
        kp[23, :2] = [0.4, 0.55]  # L hip
        kp[24, :2] = [0.6, 0.55]  # R hip
        kp[0, :2] = [0.5, 0.2]    # nose
        # Wrists below hips, stationary
        kp[15, :2] = [0.4, 0.7]   # L wrist
        kp[16, :2] = [0.6, 0.7]   # R wrist
        kp[:, 3] = 1.0
        poses[i] = kp

    windows = gc.create_windows(poses)
    results = gc.classify_windows(windows)
    assert len(results) > 0
    rest_count = sum(1 for r in results if r["gesture_label"] == "Rest")
    assert rest_count / len(results) > 0.5, "Expected mostly Rest gestures"


def test_heuristic_adaptor():
    """Hands near face → Adaptor."""
    gc = GestureClassifier(model_path="nonexistent", device="cpu")
    poses = {}
    for i in range(50):
        kp = np.zeros((33, 4), dtype=np.float32)
        kp[0, :2] = [0.5, 0.2]    # nose
        kp[11, :2] = [0.4, 0.35]  # L shoulder
        kp[12, :2] = [0.6, 0.35]  # R shoulder
        kp[23, :2] = [0.4, 0.55]  # L hip
        kp[24, :2] = [0.6, 0.55]  # R hip
        kp[7, :2] = [0.4, 0.2]    # L ear
        kp[8, :2] = [0.6, 0.2]    # R ear
        # Wrist very close to nose with slight movement
        kp[15, :2] = [0.5 + 0.01 * np.sin(i), 0.22]
        kp[16, :2] = [0.6, 0.6]   # other hand away
        kp[:, 3] = 1.0
        poses[i] = kp

    windows = gc.create_windows(poses)
    results = gc.classify_windows(windows)
    assert len(results) > 0
    adaptor_count = sum(1 for r in results if r["gesture_label"] == "Adaptor")
    assert adaptor_count > 0, "Expected at least some Adaptor gestures"


def test_classify_video_adds_timestamps():
    """classify_video adds start_sec and end_sec."""
    gc = GestureClassifier(model_path="nonexistent", device="cpu")
    poses = _make_poses(50)
    results = gc.classify_video(poses, fps=5)
    assert len(results) > 0
    for r in results:
        assert "start_sec" in r
        assert "end_sec" in r
        assert "gesture_label" in r
        assert r["gesture_label"] in GESTURE_CLASSES
        assert r["start_sec"] >= 0
        assert r["end_sec"] > r["start_sec"]


def test_all_results_have_confidence():
    """Every result includes a confidence score."""
    gc = GestureClassifier(model_path="nonexistent", device="cpu")
    poses = _make_poses(50)
    results = gc.classify_video(poses, fps=5)
    for r in results:
        assert "confidence" in r
        assert 0 <= r["confidence"] <= 1.0


# ─── Constants Tests ───


def test_gesture_classes():
    """5 gesture classes defined correctly."""
    assert len(GESTURE_CLASSES) == 5
    assert "Illustrator" in GESTURE_CLASSES
    assert "Rest" in GESTURE_CLASSES
    assert "Adaptor" in GESTURE_CLASSES
