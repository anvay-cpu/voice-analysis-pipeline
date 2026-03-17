"""Tests for temporal aggregation and output assembly (Phase 10)."""

import json
import os
import tempfile

import numpy as np
import pytest

from src.body.temporal_model import (
    TemporalAggregator,
    aggregate_emotion,
    aggregate_gaze,
    aggregate_gestures,
    aggregate_hand_states,
    aggregate_movement,
    aggregate_posture,
    make_windows,
    majority_vote,
    smooth_scores,
)
from src.body.output_assembler import OutputAssembler, compute_summary


# ─── Smoothing Tests ───


def test_smooth_scores_basic():
    """Smoothing reduces noise."""
    noisy = [50.0, 90.0, 50.0, 90.0, 50.0, 90.0, 50.0]
    smoothed = smooth_scores(noisy, kernel=3)
    assert len(smoothed) == len(noisy)
    # Smoothed values should be less extreme
    assert max(smoothed) < 90.0
    assert min(smoothed) > 50.0


def test_smooth_scores_short():
    """Short sequence returned as-is."""
    short = [1.0, 2.0]
    assert smooth_scores(short, kernel=5) == short


def test_majority_vote():
    """Most common label wins."""
    assert majority_vote(["A", "B", "A", "C", "A"]) == "A"
    assert majority_vote(["X"]) == "X"


def test_majority_vote_empty():
    """Empty list → Unknown."""
    assert majority_vote([]) == "Unknown"


# ─── Window Generation Tests ───


def test_make_windows_basic():
    """100 frames, window=30, 50% overlap → correct windows."""
    windows = make_windows(100, window_size=30, overlap=0.5)
    assert len(windows) > 0
    assert windows[0] == (0, 30)
    # Step size = 15
    assert windows[1] == (15, 45)
    # Last window should reach frame 100
    assert windows[-1][1] == 100


def test_make_windows_empty():
    """Zero frames → empty."""
    assert make_windows(0) == []


def test_make_windows_small():
    """Fewer frames than window → single window."""
    windows = make_windows(10, window_size=30)
    assert len(windows) == 1
    assert windows[0] == (0, 10)


def test_make_windows_no_overlap():
    """No overlap → non-overlapping windows."""
    windows = make_windows(90, window_size=30, overlap=0.0)
    assert windows == [(0, 30), (30, 60), (60, 90)]


# ─── Posture Aggregation Tests ───


def test_aggregate_posture_basic():
    """Aggregates scores correctly."""
    scores = {
        0: {"combined_score": 80.0, "shoulder_alignment": 3.0, "spine_angle": 5.0},
        1: {"combined_score": 90.0, "shoulder_alignment": 2.0, "spine_angle": 4.0},
        2: None,
    }
    result = aggregate_posture(scores, [0, 1, 2])
    assert result["score_mean"] == 85.0
    assert result["shoulder_deg"] == 2.5
    assert result["spine_deg"] == 4.5


def test_aggregate_posture_empty():
    """No valid scores → defaults."""
    result = aggregate_posture({}, [0, 1])
    assert result["score_mean"] == 50.0


# ─── Gesture Aggregation Tests ───


def test_aggregate_gestures_basic():
    """Majority gesture in window."""
    gestures = [
        {"start_frame": 0, "end_frame": 10, "gesture_label": "Active Gesture"},
        {"start_frame": 10, "end_frame": 20, "gesture_label": "Rest"},
        {"start_frame": 20, "end_frame": 30, "gesture_label": "Rest"},
    ]
    result = aggregate_gestures(gestures, 0, 30)
    assert result["dominant"] == "Rest"
    assert "distribution" in result


def test_aggregate_gestures_no_overlap():
    """No gesture overlaps window → Rest fallback."""
    gestures = [{"start_frame": 50, "end_frame": 60, "gesture_label": "Active Gesture"}]
    result = aggregate_gestures(gestures, 0, 30)
    assert result["dominant"] == "Rest"


# ─── Hand State Aggregation Tests ───


def test_aggregate_hand_states_basic():
    """Most frequent hand state."""
    hands = [
        {"frame_idx": 0, "state": "Open Palm"},
        {"frame_idx": 1, "state": "Open Palm"},
        {"frame_idx": 2, "state": "Pointing"},
    ]
    result = aggregate_hand_states(hands, [0, 1, 2])
    assert result["dominant"] == "Open Palm"


def test_aggregate_hand_states_empty():
    """No data → Other fallback."""
    result = aggregate_hand_states([], [0, 1])
    assert result["dominant"] == "Other"


# ─── Gaze Aggregation Tests ───


def test_aggregate_gaze_basic():
    """Zone distribution and engagement."""
    frames = [
        {"frame_idx": 0, "gaze_zone": "Audience Center"},
        {"frame_idx": 1, "gaze_zone": "Audience Center"},
        {"frame_idx": 2, "gaze_zone": "Notes/Podium"},
    ]
    result = aggregate_gaze(frames, [0, 1, 2])
    assert result["engagement_ratio"] == pytest.approx(0.67, abs=0.01)
    assert "Audience Center" in result["distribution"]


def test_aggregate_gaze_empty():
    """No gaze data → Away with 0 engagement."""
    result = aggregate_gaze([], [0, 1])
    assert result["engagement_ratio"] == 0.0


# ─── Emotion Aggregation Tests ───


def test_aggregate_emotion_basic():
    """Majority emotion."""
    emotions = [
        {"frame_idx": 0, "emotion": "Happy", "confidence": 0.8},
        {"frame_idx": 1, "emotion": "Happy", "confidence": 0.7},
        {"frame_idx": 2, "emotion": "Neutral", "confidence": 0.6},
    ]
    result = aggregate_emotion(emotions, [0, 1, 2])
    assert result["dominant"] == "Happy"
    assert result["confidence"] > 0


def test_aggregate_emotion_empty():
    """No emotion data → Neutral."""
    result = aggregate_emotion([], [0, 1])
    assert result["dominant"] == "Neutral"


# ─── Movement Aggregation Tests ───


def test_aggregate_movement_basic():
    """Extracts pattern and velocity for window."""
    movement = {
        "movement_pattern": "Purposeful",
        "centroids": [
            {"frame_idx": 0, "x": 0.3, "y": 0.5},
            {"frame_idx": 5, "x": 0.4, "y": 0.5},
            {"frame_idx": 10, "x": 0.5, "y": 0.5},
        ],
    }
    result = aggregate_movement(movement, 0, 15)
    assert result["pattern"] == "Purposeful"
    assert result["velocity"] > 0


def test_aggregate_movement_no_centroids():
    """No centroids in window → zero velocity."""
    movement = {"movement_pattern": "Anchored", "centroids": []}
    result = aggregate_movement(movement, 0, 30)
    assert result["velocity"] == 0.0


# ─── TemporalAggregator Tests ───


def test_aggregator_full():
    """Full aggregation with all inputs."""
    agg = TemporalAggregator(window_frames=10, overlap=0.0, fps=5.0)
    posture = {i: {"combined_score": 75.0, "shoulder_alignment": 3.0, "spine_angle": 5.0}
               for i in range(20)}
    gestures = [{"start_frame": 0, "end_frame": 20, "gesture_label": "Active Gesture"}]
    hands = [{"frame_idx": i, "state": "Open Palm"} for i in range(20)]
    gaze = {"frames": [{"frame_idx": i, "gaze_zone": "Audience Center"} for i in range(20)]}
    emotions = [{"frame_idx": i, "emotion": "Happy", "confidence": 0.8} for i in range(20)]
    movement = {
        "movement_pattern": "Purposeful",
        "centroids": [{"frame_idx": i, "x": 0.5, "y": 0.5} for i in range(20)],
    }

    segments = agg.aggregate(
        n_frames=20,
        posture_scores=posture,
        gesture_results=gestures,
        hand_results=hands,
        gaze_result=gaze,
        emotion_results=emotions,
        movement_result=movement,
    )

    assert len(segments) == 2  # 20 frames / 10 window / no overlap
    seg = segments[0]
    assert "timestamp_start" in seg
    assert "timestamp_end" in seg
    assert "posture" in seg
    assert "gesture" in seg
    assert "hand_state" in seg
    assert "gaze" in seg
    assert "facial_emotion" in seg
    assert "movement" in seg


def test_aggregator_partial():
    """Works with only some inputs provided."""
    agg = TemporalAggregator(window_frames=10, overlap=0.0, fps=5.0)
    posture = {i: {"combined_score": 80.0, "shoulder_alignment": 2.0, "spine_angle": 4.0}
               for i in range(10)}

    segments = agg.aggregate(n_frames=10, posture_scores=posture)
    assert len(segments) == 1
    assert "posture" in segments[0]
    assert "gesture" not in segments[0]


def test_aggregator_empty():
    """Zero frames → no segments."""
    agg = TemporalAggregator()
    segments = agg.aggregate(n_frames=0)
    assert segments == []


# ─── Summary Tests ───


def test_summary_basic():
    """Summary computed from segments."""
    segments = [
        {
            "posture": {"score_mean": 80.0},
            "gesture": {"dominant": "Active Gesture"},
            "hand_state": {"dominant": "Open Palm"},
            "gaze": {"engagement_ratio": 0.7},
            "facial_emotion": {"dominant": "Happy"},
        },
        {
            "posture": {"score_mean": 90.0},
            "gesture": {"dominant": "Rest"},
            "hand_state": {"dominant": "Open Palm"},
            "gaze": {"engagement_ratio": 0.8},
            "facial_emotion": {"dominant": "Happy"},
        },
    ]
    movement = {"movement_pattern": "Purposeful", "stage_usage_score": 8.0}
    summary = compute_summary(segments, movement)
    assert summary["posture_score"] == 85.0
    assert summary["dominant_gesture"] in ["Active Gesture", "Rest"]
    assert summary["dominant_hand_state"] == "Open Palm"
    assert summary["audience_engagement"] == 0.75
    assert summary["dominant_emotion"] == "Happy"
    assert summary["movement_pattern"] == "Purposeful"
    assert summary["stage_usage_score"] == 8.0


def test_summary_empty():
    """Empty segments → defaults."""
    summary = compute_summary([])
    assert summary["posture_score"] == 50.0
    assert summary["dominant_emotion"] == "Neutral"


# ─── OutputAssembler Tests ───


def test_assembler_output_structure():
    """Assembler produces correct top-level structure."""
    assembler = OutputAssembler(fps=5.0, window_frames=10, overlap=0.0)
    output = assembler.assemble(
        video_path="test.mp4",
        n_frames=20,
        posture_scores={i: {"combined_score": 75.0, "shoulder_alignment": 3.0, "spine_angle": 5.0}
                        for i in range(20)},
    )
    assert output["video"] == "test.mp4"
    assert output["total_frames"] == 20
    assert output["duration_sec"] == 4.0
    assert "summary" in output
    assert "segments" in output
    assert len(output["segments"]) == 2


def test_assembler_save():
    """Assembler saves valid JSON."""
    assembler = OutputAssembler(fps=5.0)
    output = assembler.assemble(video_path="test.mp4", n_frames=10)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        assembler.save(output, path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["video"] == "test.mp4"
    finally:
        os.unlink(path)


def test_assembler_no_inputs():
    """Assembler works with no sub-module outputs."""
    assembler = OutputAssembler(fps=5.0)
    output = assembler.assemble(video_path="empty.mp4", n_frames=0)
    assert output["total_frames"] == 0
    assert output["segments"] == []
    assert "summary" in output
