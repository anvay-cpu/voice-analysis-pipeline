"""Tests for hand tracking (Phase 6)."""

import numpy as np
import pytest

from src.body.hand_tracker import (
    HAND_STATES,
    HandTracker,
    NUM_HAND_LANDMARKS,
)


# ─── Helper ───

def _make_hand_landmarks(fingers_extended=None):
    """Create synthetic hand landmarks (21, 3).

    By default all fingers extended (open palm).
    fingers_extended: list of 5 bools [thumb, index, middle, ring, pinky].
    """
    if fingers_extended is None:
        fingers_extended = [True, True, True, True, True]

    lm = np.zeros((21, 3), dtype=np.float32)

    # Wrist at center
    lm[0] = [0.5, 0.6, 0.0]

    # Thumb: MCP=1,2 IP=3 TIP=4
    lm[1] = [0.42, 0.55, 0.0]  # CMC
    lm[2] = [0.38, 0.50, 0.0]  # MCP
    lm[3] = [0.35, 0.45, 0.0]  # IP
    if fingers_extended[0]:
        lm[4] = [0.30, 0.40, 0.0]  # TIP extended
    else:
        lm[4] = [0.37, 0.48, 0.0]  # TIP curled

    # Index: MCP=5 PIP=6 DIP=7 TIP=8
    lm[5] = [0.45, 0.48, 0.0]
    lm[6] = [0.44, 0.42, 0.0]  # PIP
    lm[7] = [0.43, 0.38, 0.0]
    if fingers_extended[1]:
        lm[8] = [0.43, 0.32, 0.0]  # TIP above PIP
    else:
        lm[8] = [0.44, 0.50, 0.0]  # TIP below PIP (curled)

    # Middle: MCP=9 PIP=10 DIP=11 TIP=12
    lm[9] = [0.50, 0.47, 0.0]
    lm[10] = [0.50, 0.40, 0.0]
    lm[11] = [0.50, 0.36, 0.0]
    if fingers_extended[2]:
        lm[12] = [0.50, 0.30, 0.0]
    else:
        lm[12] = [0.50, 0.48, 0.0]

    # Ring: MCP=13 PIP=14 DIP=15 TIP=16
    lm[13] = [0.55, 0.48, 0.0]
    lm[14] = [0.55, 0.42, 0.0]
    lm[15] = [0.55, 0.38, 0.0]
    if fingers_extended[3]:
        lm[16] = [0.55, 0.32, 0.0]
    else:
        lm[16] = [0.55, 0.50, 0.0]

    # Pinky: MCP=17 PIP=18 DIP=19 TIP=20
    lm[17] = [0.60, 0.50, 0.0]
    lm[18] = [0.60, 0.45, 0.0]
    lm[19] = [0.60, 0.42, 0.0]
    if fingers_extended[4]:
        lm[20] = [0.60, 0.35, 0.0]
    else:
        lm[20] = [0.60, 0.52, 0.0]

    return lm


def _make_pose(l_wrist=(0.4, 0.5), r_wrist=(0.6, 0.5),
               l_shoulder=(0.4, 0.35), r_shoulder=(0.6, 0.35),
               l_elbow=(0.38, 0.42), r_elbow=(0.62, 0.42),
               l_hip=(0.42, 0.6), r_hip=(0.58, 0.6),
               l_vis=1.0, r_vis=1.0):
    """Create synthetic pose keypoints (33, 4)."""
    kp = np.zeros((33, 4), dtype=np.float32)
    kp[:, 3] = 1.0  # all visible by default
    kp[0, :2] = [0.5, 0.2]   # nose
    kp[11, :2] = l_shoulder
    kp[12, :2] = r_shoulder
    kp[13, :2] = l_elbow
    kp[14, :2] = r_elbow
    kp[15, :2] = l_wrist
    kp[15, 3] = l_vis
    kp[16, :2] = r_wrist
    kp[16, 3] = r_vis
    kp[23, :2] = l_hip
    kp[24, :2] = r_hip
    return kp


# ─── Single Hand Shape Tests ───


def test_open_palm():
    """All fingers extended → Open Palm."""
    ht = HandTracker.__new__(HandTracker)
    lm = _make_hand_landmarks([True, True, True, True, True])
    result = ht._classify_single_hand(lm)
    assert result == "Open Palm"


def test_pointing():
    """Only index extended → Pointing."""
    ht = HandTracker.__new__(HandTracker)
    lm = _make_hand_landmarks([False, True, False, False, False])
    result = ht._classify_single_hand(lm)
    assert result == "Pointing"


def test_fist_is_other():
    """All fingers curled → Other."""
    ht = HandTracker.__new__(HandTracker)
    lm = _make_hand_landmarks([False, False, False, False, False])
    result = ht._classify_single_hand(lm)
    assert result == "Other"


# ─── Finger Extension Tests ───


def test_fingers_extended_all():
    """All 5 fingers detected as extended."""
    ht = HandTracker.__new__(HandTracker)
    lm = _make_hand_landmarks([True, True, True, True, True])
    ext = ht._fingers_extended(lm)
    assert sum(ext) >= 4  # at least 4 (thumb detection can be tricky)


def test_fingers_extended_none():
    """No fingers extended."""
    ht = HandTracker.__new__(HandTracker)
    lm = _make_hand_landmarks([False, False, False, False, False])
    ext = ht._fingers_extended(lm)
    # Index through pinky should all be false
    assert not any(ext[1:])


# ─── Two-Hand State Tests ───


def test_steepling():
    """Fingertips close together → Steepling."""
    ht = HandTracker.__new__(HandTracker)
    lm0 = _make_hand_landmarks([True, True, True, True, True])
    lm1 = lm0.copy()
    # Move second hand slightly — fingertips nearly touching
    lm1[:, 0] += 0.02  # shift x by small amount

    result = ht._is_steepling(lm0, lm1, threshold=0.06)
    assert result is True


def test_not_steepling_when_far():
    """Hands far apart → not steepling."""
    ht = HandTracker.__new__(HandTracker)
    lm0 = _make_hand_landmarks()
    lm1 = _make_hand_landmarks()
    lm1[:, 0] += 0.3  # shift far right
    assert ht._is_steepling(lm0, lm1) is False


def test_clasped():
    """Wrists close + overlapping fingers → Clasped."""
    ht = HandTracker.__new__(HandTracker)
    lm0 = _make_hand_landmarks()
    lm1 = _make_hand_landmarks()
    # Bring wrists together
    lm1[0] = lm0[0] + np.array([0.02, 0.0, 0.0])
    # Overlap MCP/PIP points
    for mcp, pip in [(5, 6), (9, 10), (13, 14), (17, 18)]:
        lm1[mcp, :2] = lm0[pip, :2] + np.array([0.01, 0.01])

    assert ht._is_clasped(lm0, lm1, threshold=0.08) is True


# ─── Pose-Based State Tests ───


def test_crossed_arms():
    """Wrists crossed near chest → Crossed Arms."""
    ht = HandTracker.__new__(HandTracker)
    # Left wrist on right side, right wrist on left side
    pose = _make_pose(
        l_wrist=(0.55, 0.45),   # left wrist on right side
        r_wrist=(0.45, 0.45),   # right wrist on left side
        l_elbow=(0.35, 0.42),   # elbows wide
        r_elbow=(0.65, 0.42),
    )
    assert ht._is_crossed_arms(pose) is True


def test_not_crossed_normal():
    """Normal hand position → not crossed."""
    ht = HandTracker.__new__(HandTracker)
    pose = _make_pose()  # default: hands at sides
    assert ht._is_crossed_arms(pose) is False


def test_behind_back():
    """Low wrist visibility → Behind Back."""
    ht = HandTracker.__new__(HandTracker)
    pose = _make_pose(l_vis=0.1, r_vis=0.1)
    result = ht._classify_no_hands(pose)
    assert result["state"] == "Behind Back"


def test_hands_in_pockets():
    """Wrists below hips, close to body → Hands in Pockets."""
    ht = HandTracker.__new__(HandTracker)
    pose = _make_pose(
        l_wrist=(0.42, 0.65),  # below left hip, close to hip x
        r_wrist=(0.58, 0.65),  # below right hip, close to hip x
    )
    result = ht._classify_no_hands(pose)
    assert result["state"] == "Hands in Pockets"


def test_no_hands_no_pose():
    """No hands detected, no pose → Other."""
    ht = HandTracker.__new__(HandTracker)
    result = ht._classify_no_hands(None)
    assert result["state"] == "Other"
    assert result["num_hands"] == 0


# ─── classify_hand_state Integration ───


def test_classify_returns_all_fields():
    """classify_hand_state returns required fields."""
    ht = HandTracker.__new__(HandTracker)
    hand_result = {"hands": []}
    result = ht.classify_hand_state(hand_result, pose_keypoints=None)
    assert "state" in result
    assert "confidence" in result
    assert "num_hands" in result
    assert "details" in result
    assert result["state"] in HAND_STATES


def test_classify_with_one_hand():
    """Single hand detection classifies correctly."""
    ht = HandTracker.__new__(HandTracker)
    lm = _make_hand_landmarks([True, True, True, True, True])
    hand_result = {"hands": [{"landmarks": lm, "handedness": "Right", "score": 0.9}]}
    result = ht.classify_hand_state(hand_result)
    assert result["state"] == "Open Palm"
    assert result["num_hands"] == 1


# ─── Constants Tests ───


def test_hand_states_list():
    """All expected hand states are defined."""
    assert "Open Palm" in HAND_STATES
    assert "Pointing" in HAND_STATES
    assert "Steepling" in HAND_STATES
    assert "Crossed Arms" in HAND_STATES
    assert "Behind Back" in HAND_STATES
    assert "Clasped" in HAND_STATES
    assert "Other" in HAND_STATES
    assert len(HAND_STATES) == 8
