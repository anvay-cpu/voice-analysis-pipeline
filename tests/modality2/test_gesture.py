"""Tests for gesture classification (Phase 5) — 3-class version."""

import numpy as np
import pytest
import torch

from src.body.gesture_classifier import (
    GESTURE_CLASSES,
    GestureClassifier,
    GestureBiLSTM,
    GestureTransformer,
    TinyTransformer,
    NUM_CLASSES,
    WINDOW_FRAMES,
    process_window,
)


# ─── Model Architecture Tests ───


def test_transformer_output_shape():
    """Transformer produces correct output shape."""
    model = GestureTransformer(input_dim=99, num_classes=3)
    x = torch.randn(4, 15, 99)
    out = model(x)
    assert out.shape == (4, NUM_CLASSES)


def test_tiny_transformer_output_shape():
    """TinyTransformer produces correct output shape."""
    model = TinyTransformer(input_dim=99, num_classes=3)
    x = torch.randn(4, 15, 99)
    out = model(x)
    assert out.shape == (4, NUM_CLASSES)


def test_bilstm_output_shape():
    """BiLSTM produces correct output shape."""
    model = GestureBiLSTM(input_dim=99, num_classes=3)
    x = torch.randn(4, 15, 99)
    out = model(x)
    assert out.shape == (4, NUM_CLASSES)


def test_tiny_transformer_param_count():
    """TinyTransformer is much smaller than original."""
    tiny = TinyTransformer(input_dim=99, num_classes=3)
    big = GestureTransformer(input_dim=99, num_classes=3)
    p_tiny = sum(p.numel() for p in tiny.parameters())
    p_big = sum(p.numel() for p in big.parameters())
    assert p_tiny < p_big / 3, "TinyTransformer should be at least 3x smaller"
    assert 100_000 < p_tiny < 1_000_000


def test_bilstm_param_count():
    """BiLSTM is small."""
    model = GestureBiLSTM(input_dim=99, num_classes=3)
    params = sum(p.numel() for p in model.parameters())
    assert 50_000 < params < 500_000


def test_transformer_gradient_flow():
    """Gradients flow through the full model."""
    model = GestureTransformer(input_dim=99, num_classes=3)
    x = torch.randn(2, 15, 99, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


# ─── Feature Processing Tests ───


def test_process_window_shape():
    """process_window converts (15, 33, 4) → (15, 99)."""
    raw = np.random.rand(15, 33, 4).astype(np.float32)
    raw[:, :, :2] = raw[:, :, :2] * 0.5 + 0.25
    raw[:, :, 3] = 1.0
    out = process_window(raw)
    assert out.shape == (15, 99)
    assert out.dtype == np.float32


def test_process_window_removes_z():
    """Processed features don't include z-coordinate."""
    raw = np.random.rand(15, 33, 4).astype(np.float32)
    raw[:, :, 2] = 999.0  # z set to extreme value
    raw[:, :, :2] = raw[:, :, :2] * 0.5 + 0.25
    raw[:, :, 3] = 1.0
    out = process_window(raw)
    # If z were included, 999.0 would survive; 99 dims = 33*3 (x,y,vis)
    assert out.shape == (15, 99)
    assert np.max(np.abs(out)) < 100  # z=999 not present


# ─── Window Creation Tests ───


def _make_poses(n_frames=100):
    """Create dummy pose data with realistic structure."""
    poses = {}
    for i in range(n_frames):
        kp = np.random.rand(33, 4).astype(np.float32)
        kp[:, :2] = kp[:, :2] * 0.3 + 0.35  # keep in center
        # Set realistic shoulder/hip positions for normalization
        kp[11, :2] = [0.4, 0.35]  # L shoulder
        kp[12, :2] = [0.6, 0.35]  # R shoulder
        kp[23, :2] = [0.4, 0.55]  # L hip
        kp[24, :2] = [0.6, 0.55]  # R hip
        kp[:, 3] = 1.0
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
    """Each window has correct keypoint shape (15, 99) with processing."""
    gc = GestureClassifier(model_path="nonexistent", device="cpu")
    poses = _make_poses(50)
    windows = gc.create_windows(poses)
    assert len(windows) > 0
    for w in windows:
        assert w["keypoints"].shape == (WINDOW_FRAMES, 99)


def test_create_windows_skips_sparse():
    """Windows with too many missing frames are skipped."""
    gc = GestureClassifier(model_path="nonexistent", device="cpu")
    poses = {}
    for i in range(0, 100, 5):
        poses[i] = np.random.rand(33, 4).astype(np.float32)
    for i in range(100):
        if i not in poses:
            poses[i] = None
    windows = gc.create_windows(poses)
    assert len(windows) == 0


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
        kp[11, :2] = [0.4, 0.35]
        kp[12, :2] = [0.6, 0.35]
        kp[23, :2] = [0.4, 0.55]
        kp[24, :2] = [0.6, 0.55]
        kp[0, :2] = [0.5, 0.2]
        kp[15, :2] = [0.4, 0.7]
        kp[16, :2] = [0.6, 0.7]
        kp[:, 3] = 1.0
        poses[i] = kp

    windows = gc.create_windows(poses)
    results = gc.classify_windows(windows)
    assert len(results) > 0
    rest_count = sum(1 for r in results if r["gesture_label"] == "Rest")
    assert rest_count / len(results) > 0.5, "Expected mostly Rest"


def test_heuristic_adaptor():
    """Hands near face → Adaptor."""
    gc = GestureClassifier(model_path="nonexistent", device="cpu")
    poses = {}
    for i in range(50):
        kp = np.zeros((33, 4), dtype=np.float32)
        kp[0, :2] = [0.5, 0.2]
        kp[11, :2] = [0.4, 0.35]
        kp[12, :2] = [0.6, 0.35]
        kp[23, :2] = [0.4, 0.55]
        kp[24, :2] = [0.6, 0.55]
        kp[7, :2] = [0.4, 0.2]
        kp[8, :2] = [0.6, 0.2]
        kp[15, :2] = [0.5 + 0.005 * np.sin(i), 0.21]
        kp[16, :2] = [0.6, 0.6]
        kp[:, 3] = 1.0
        poses[i] = kp

    windows = gc.create_windows(poses)
    results = gc.classify_windows(windows)
    assert len(results) > 0
    adaptor_count = sum(1 for r in results if r["gesture_label"] == "Adaptor")
    assert adaptor_count > 0, "Expected at least some Adaptor"


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
    """3 gesture classes defined correctly."""
    assert len(GESTURE_CLASSES) == 3
    assert "Active Gesture" in GESTURE_CLASSES
    assert "Rest" in GESTURE_CLASSES
    assert "Adaptor" in GESTURE_CLASSES
    assert NUM_CLASSES == 3
