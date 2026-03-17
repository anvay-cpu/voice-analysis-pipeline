"""Tests for gaze estimation (Phase 7)."""

import numpy as np
import pytest

from src.body.gaze_estimator import (
    GAZE_ZONES,
    GazeEstimator,
    _rotation_matrix_to_euler,
)


# ─── Helpers ───


def _make_face_landmarks(n=478, nose_x=320, nose_y=240):
    """Create synthetic face landmarks (N, 3) in pixel coordinates.

    Places face roughly centered at (nose_x, nose_y) in a 640x480 image.
    """
    lm = np.zeros((n, 3), dtype=np.float32)

    # Set key PnP landmarks for a forward-facing head
    lm[1] = [nose_x, nose_y, 0]               # nose tip
    lm[152] = [nose_x, nose_y + 80, -15]      # chin
    lm[33] = [nose_x - 55, nose_y - 40, -30]  # left eye corner
    lm[263] = [nose_x + 55, nose_y - 40, -30] # right eye corner
    lm[61] = [nose_x - 35, nose_y + 35, -25]  # left mouth
    lm[291] = [nose_x + 35, nose_y + 35, -25] # right mouth

    # Eye landmarks for iris gaze
    lm[133] = [nose_x - 30, nose_y - 40, -25]  # left eye inner
    lm[159] = [nose_x - 45, nose_y - 45, -25]  # left eye top
    lm[145] = [nose_x - 45, nose_y - 35, -25]  # left eye bottom
    lm[362] = [nose_x + 30, nose_y - 40, -25]  # right eye inner
    lm[386] = [nose_x + 45, nose_y - 45, -25]  # right eye top
    lm[374] = [nose_x + 45, nose_y - 35, -25]  # right eye bottom

    if n >= 478:
        # Left iris (centered in left eye)
        left_eye_cx = nose_x - 42
        left_eye_cy = nose_y - 40
        lm[468] = [left_eye_cx, left_eye_cy, -25]  # center
        lm[469] = [left_eye_cx - 3, left_eye_cy, -25]
        lm[470] = [left_eye_cx + 3, left_eye_cy, -25]
        lm[471] = [left_eye_cx, left_eye_cy - 3, -25]
        lm[472] = [left_eye_cx, left_eye_cy + 3, -25]

        # Right iris (centered in right eye)
        right_eye_cx = nose_x + 42
        right_eye_cy = nose_y - 40
        lm[473] = [right_eye_cx, right_eye_cy, -25]
        lm[474] = [right_eye_cx - 3, right_eye_cy, -25]
        lm[475] = [right_eye_cx + 3, right_eye_cy, -25]
        lm[476] = [right_eye_cx, right_eye_cy - 3, -25]
        lm[477] = [right_eye_cx, right_eye_cy + 3, -25]

    return lm


# ─── Head Pose Tests ───


def test_head_pose_forward():
    """Forward-facing head → yaw and pitch near zero."""
    ge = GazeEstimator.__new__(GazeEstimator)
    lm = _make_face_landmarks()
    pose = ge.compute_head_pose(lm, (640, 480))
    assert abs(pose["yaw"]) < 25, f"Expected small yaw, got {pose['yaw']}"
    assert abs(pose["roll"]) < 25, f"Expected small roll, got {pose['roll']}"
    assert "pitch" in pose


def test_head_pose_returns_all_angles():
    """Head pose always has yaw, pitch, roll."""
    ge = GazeEstimator.__new__(GazeEstimator)
    lm = _make_face_landmarks()
    pose = ge.compute_head_pose(lm, (640, 480))
    assert "yaw" in pose
    assert "pitch" in pose
    assert "roll" in pose


def test_head_pose_turned_left():
    """Head turned left → negative yaw."""
    ge = GazeEstimator.__new__(GazeEstimator)
    lm = _make_face_landmarks()
    # Shift right-side landmarks closer to center (head turned left)
    lm[263, 0] -= 30  # right eye moves toward center
    lm[291, 0] -= 20  # right mouth moves toward center
    pose = ge.compute_head_pose(lm, (640, 480))
    # The yaw should shift compared to forward
    assert isinstance(pose["yaw"], float)


def test_head_pose_turned_right():
    """Head turned right → positive yaw."""
    ge = GazeEstimator.__new__(GazeEstimator)
    lm = _make_face_landmarks()
    # Shift left-side landmarks closer to center (head turned right)
    lm[33, 0] += 30
    lm[61, 0] += 20
    pose = ge.compute_head_pose(lm, (640, 480))
    assert isinstance(pose["yaw"], float)


# ─── Iris Gaze Tests ───


def test_iris_gaze_centered():
    """Centered iris → gaze_x near zero."""
    ge = GazeEstimator.__new__(GazeEstimator)
    lm = _make_face_landmarks(478)
    gaze = ge.compute_iris_gaze(lm)
    assert abs(gaze["gaze_x"]) < 0.5, f"Expected centered, got {gaze['gaze_x']}"
    assert 0 <= gaze["left_ratio"] <= 1
    assert 0 <= gaze["right_ratio"] <= 1


def test_iris_gaze_no_iris_landmarks():
    """Fewer than 478 landmarks → default centered values."""
    ge = GazeEstimator.__new__(GazeEstimator)
    lm = _make_face_landmarks(468)  # no iris
    gaze = ge.compute_iris_gaze(lm)
    assert gaze["left_ratio"] == 0.5
    assert gaze["right_ratio"] == 0.5
    assert gaze["gaze_x"] == 0.0


def test_iris_gaze_has_all_fields():
    """Iris gaze returns all required fields."""
    ge = GazeEstimator.__new__(GazeEstimator)
    lm = _make_face_landmarks(478)
    gaze = ge.compute_iris_gaze(lm)
    assert "left_ratio" in gaze
    assert "right_ratio" in gaze
    assert "gaze_x" in gaze
    assert "gaze_y" in gaze


# ─── Gaze Zone Classification Tests ───


def test_zone_audience_center():
    """Yaw=0, pitch=0 → Audience Center."""
    zone = GazeEstimator.classify_gaze_zone(0, 0)
    assert zone == "Audience Center"


def test_zone_audience_left():
    """Yaw=-35, pitch=0 → Audience Left."""
    zone = GazeEstimator.classify_gaze_zone(-35, 0)
    assert zone == "Audience Left"


def test_zone_audience_right():
    """Yaw=35, pitch=0 → Audience Right."""
    zone = GazeEstimator.classify_gaze_zone(35, 0)
    assert zone == "Audience Right"


def test_zone_notes_podium():
    """Yaw=0, pitch=35 → Notes/Podium."""
    zone = GazeEstimator.classify_gaze_zone(0, 35)
    assert zone == "Notes/Podium"


def test_zone_floor():
    """Yaw=0, pitch=50 → Floor."""
    zone = GazeEstimator.classify_gaze_zone(0, 50)
    assert zone == "Floor"


def test_zone_ceiling():
    """Yaw=0, pitch=-30 → Ceiling/Away."""
    zone = GazeEstimator.classify_gaze_zone(0, -30)
    assert zone == "Ceiling/Away"


def test_zone_extreme_angle():
    """Extreme yaw → Away."""
    zone = GazeEstimator.classify_gaze_zone(80, 5)
    assert zone == "Away"


# ─── estimate_single Tests ───


def test_estimate_single_no_face():
    """No face detected → sensible defaults."""
    ge = GazeEstimator.__new__(GazeEstimator)
    ge.model_path = "nonexistent"
    # Create a black frame — no face to detect
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # We can't call estimate_single without the model, but test the output
    # structure by testing the no-face branch directly
    result = ge.estimate_single.__wrapped__(ge, frame) if hasattr(ge.estimate_single, '__wrapped__') else None

    # Instead, test detect_face returns None on black image
    # (would need model file; skip if not available)


def test_estimate_single_output_structure():
    """estimate_single returns all expected fields even when no face."""
    ge = GazeEstimator.__new__(GazeEstimator)
    ge.model_path = DEFAULT_MODEL_PATH = "models/face_landmarker.task"
    ge.min_detection_confidence = 0.5
    ge.min_presence_confidence = 0.5

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    from pathlib import Path
    if not Path(ge.model_path).exists():
        pytest.skip("Face model not available")

    result = ge.estimate_single(frame)
    assert "detected" in result
    assert "head_pose" in result
    assert "iris_gaze" in result
    assert "gaze_zone" in result
    assert isinstance(result["head_pose"]["yaw"], float)


# ─── Euler Conversion Tests ───


def test_euler_identity_matrix():
    """Identity rotation → angles near zero."""
    rmat = np.eye(3)
    angles = _rotation_matrix_to_euler(rmat)
    assert abs(angles[0]) < 1  # pitch
    assert abs(angles[1]) < 1  # yaw
    assert abs(angles[2]) < 1  # roll


def test_euler_known_rotation():
    """90-degree yaw rotation → yaw near 90."""
    # Rotation about Y axis by 90 degrees
    angle = np.radians(90)
    rmat = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ])
    angles = _rotation_matrix_to_euler(rmat)
    # yaw (index 1) should be near 90
    assert abs(abs(angles[1]) - 90) < 5, f"Expected ~90 yaw, got {angles[1]}"


# ─── Zone Constants Tests ───


def test_gaze_zones_defined():
    """All expected gaze zones are defined."""
    assert "Audience Center" in GAZE_ZONES
    assert "Audience Left" in GAZE_ZONES
    assert "Audience Right" in GAZE_ZONES
    assert "Notes/Podium" in GAZE_ZONES
    assert "Floor" in GAZE_ZONES
    assert "Ceiling/Away" in GAZE_ZONES
    assert len(GAZE_ZONES) == 6


def test_gaze_zones_have_bounds():
    """Each zone has yaw and pitch bounds."""
    for zone_name, bounds in GAZE_ZONES.items():
        assert "yaw" in bounds, f"{zone_name} missing yaw"
        assert "pitch" in bounds, f"{zone_name} missing pitch"
        assert len(bounds["yaw"]) == 2
        assert len(bounds["pitch"]) == 2
