"""Tests for pose estimation module."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.body.pose_estimator import (
    PoseEstimator,
    KEYPOINT_NAMES,
    NUM_KEYPOINTS,
    KEYPOINT_DIM,
)


@pytest.fixture
def estimator():
    return PoseEstimator(min_detection_confidence=0.3)


@pytest.fixture
def person_frame():
    """Create a frame with a simple stick-figure-like shape.

    Note: MediaPipe may or may not detect this as a real person.
    We mainly test the interface and shape guarantees.
    """
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a rough humanoid shape
    # Head
    cv2.circle(frame, (320, 100), 30, (200, 180, 160), -1)
    # Torso
    cv2.rectangle(frame, (290, 130), (350, 280), (200, 180, 160), -1)
    # Arms
    cv2.line(frame, (290, 150), (220, 250), (200, 180, 160), 15)
    cv2.line(frame, (350, 150), (420, 250), (200, 180, 160), 15)
    # Legs
    cv2.line(frame, (310, 280), (280, 420), (200, 180, 160), 15)
    cv2.line(frame, (330, 280), (360, 420), (200, 180, 160), 15)
    return frame


class TestPoseEstimator:
    def test_keypoint_names_count(self):
        assert len(KEYPOINT_NAMES) == NUM_KEYPOINTS == 33

    def test_estimate_single_blank_frame(self, estimator):
        """A blank frame should return None (no pose detected)."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        result = estimator.estimate_single(blank)
        assert result is None

    def test_estimate_single_with_bbox(self, estimator, person_frame):
        """Test that bbox cropping doesn't crash."""
        bbox = [200, 50, 450, 430]
        result = estimator.estimate_single(person_frame, bbox=bbox)
        # May or may not detect pose on synthetic image; just verify no crash
        if result is not None:
            assert result.shape == (NUM_KEYPOINTS, KEYPOINT_DIM)

    def test_estimate_single_shape(self, estimator, person_frame):
        """If pose is detected, verify shape is (33, 4)."""
        result = estimator.estimate_single(person_frame)
        if result is not None:
            assert result.shape == (NUM_KEYPOINTS, KEYPOINT_DIM)
            # x, y should be in [0, 1] range (normalized)
            assert np.all(result[:, 0] >= 0) and np.all(result[:, 0] <= 1.5)
            assert np.all(result[:, 1] >= 0) and np.all(result[:, 1] <= 1.5)
            # visibility in [0, 1]
            assert np.all(result[:, 3] >= 0) and np.all(result[:, 3] <= 1.0)

    def test_estimate_single_empty_bbox(self, estimator, person_frame):
        """Zero-area bbox should return None."""
        bbox = [100, 100, 100, 100]  # zero area
        result = estimator.estimate_single(person_frame, bbox=bbox)
        assert result is None


class TestInterpolation:
    def test_interpolate_fills_gap(self, estimator):
        """Test that small gaps are filled via interpolation."""
        kp_start = np.ones((NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32) * 0.0
        kp_end = np.ones((NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32) * 1.0

        poses = {
            0: kp_start,
            1: None,
            2: None,
            3: kp_end,
        }
        result = estimator._interpolate_gaps(poses)

        # Frame 1 should be ~0.333, frame 2 should be ~0.667
        assert result[1] is not None
        assert result[2] is not None
        np.testing.assert_allclose(result[1], kp_start + (1 / 3) * (kp_end - kp_start), atol=1e-5)
        np.testing.assert_allclose(result[2], kp_start + (2 / 3) * (kp_end - kp_start), atol=1e-5)

    def test_interpolate_skips_large_gap(self, estimator):
        """Gaps > 5 frames should NOT be interpolated."""
        kp = np.zeros((NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)
        poses = {0: kp, 1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: kp}
        result = estimator._interpolate_gaps(poses)
        # Gap is 7, > 5 threshold, should remain None
        assert result[3] is None


class TestPoseCache:
    def test_save_and_load_roundtrip(self, tmp_path):
        kp0 = np.random.rand(NUM_KEYPOINTS, KEYPOINT_DIM).astype(np.float32)
        kp2 = np.random.rand(NUM_KEYPOINTS, KEYPOINT_DIM).astype(np.float32)

        poses = {0: kp0, 1: None, 2: kp2}

        cache_dir = str(tmp_path / "poses_cache")
        path = PoseEstimator.save_cache(poses, "test_video", cache_dir=cache_dir)

        loaded = PoseEstimator.load_cache(path)

        assert set(loaded.keys()) == {0, 1, 2}
        np.testing.assert_array_equal(loaded[0], kp0)
        np.testing.assert_array_equal(loaded[2], kp2)
        assert loaded[1] is None
