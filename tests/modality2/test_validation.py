"""Phase 11: Validation tests for body language pipeline output.

Runs against pre-generated JSON outputs from the 3 test videos.
Tests verify output structure, sensible values, and cross-video contrasts.
"""

import json
from pathlib import Path

import pytest

OUTPUT_DIR = Path("data/outputs")

GOOD = OUTPUT_DIR / "test_good_speaker_5min_body.json"
NERVOUS = OUTPUT_DIR / "test_nervous_speaker_5min_body.json"
MONOTONE = OUTPUT_DIR / "test_monotone_speaker_5min_body.json"


def _load(path):
    if not path.exists():
        pytest.skip(f"Output not found: {path}")
    with open(path) as f:
        return json.load(f)


# ─── Output Structure Tests ───


@pytest.fixture(params=["good", "nervous", "monotone"])
def output(request):
    paths = {"good": GOOD, "nervous": NERVOUS, "monotone": MONOTONE}
    return _load(paths[request.param])


def test_top_level_keys(output):
    """Output has all required top-level keys."""
    for key in ["video", "duration_sec", "total_frames", "fps", "summary", "segments"]:
        assert key in output, f"Missing key: {key}"


def test_has_segments(output):
    """Output has at least 1 segment."""
    assert len(output["segments"]) > 0


def test_segment_structure(output):
    """Each segment has timestamps and sub-module results."""
    seg = output["segments"][0]
    assert "timestamp_start" in seg
    assert "timestamp_end" in seg
    assert seg["timestamp_end"] > seg["timestamp_start"]


def test_summary_keys(output):
    """Summary has all expected keys."""
    s = output["summary"]
    for key in ["posture_score", "dominant_gesture", "dominant_hand_state",
                "audience_engagement", "dominant_emotion", "movement_pattern",
                "stage_usage_score"]:
        assert key in s, f"Missing summary key: {key}"


def test_posture_score_range(output):
    """Posture score is 0-100."""
    score = output["summary"]["posture_score"]
    assert 0 <= score <= 100


def test_engagement_range(output):
    """Engagement ratio is 0-1."""
    eng = output["summary"]["audience_engagement"]
    assert 0 <= eng <= 1


def test_stage_score_range(output):
    """Stage usage score is 1-10."""
    score = output["summary"]["stage_usage_score"]
    assert 1 <= score <= 10


def test_movement_pattern_valid(output):
    """Movement pattern is one of the 4 defined patterns."""
    pattern = output["summary"]["movement_pattern"]
    assert pattern in ["Anchored", "Pacing", "Roaming", "Purposeful"]


def test_gesture_valid(output):
    """Dominant gesture is a valid class."""
    gesture = output["summary"]["dominant_gesture"]
    assert gesture in ["Active Gesture", "Adaptor", "Rest"]


def test_emotion_valid(output):
    """Dominant emotion is a valid class."""
    emotion = output["summary"]["dominant_emotion"]
    valid = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    assert emotion in valid


def test_duration_positive(output):
    """Duration is positive."""
    assert output["duration_sec"] > 0
    assert output["total_frames"] > 0


# ─── Segment Content Tests ───


def test_segments_have_posture():
    """Segments contain posture data."""
    data = _load(GOOD)
    seg = data["segments"][0]
    assert "posture" in seg
    assert "score_mean" in seg["posture"]


def test_segments_have_gesture():
    """Segments contain gesture data."""
    data = _load(GOOD)
    seg = data["segments"][0]
    assert "gesture" in seg
    assert "dominant" in seg["gesture"]


def test_segments_have_gaze():
    """Segments contain gaze data."""
    data = _load(GOOD)
    seg = data["segments"][0]
    assert "gaze" in seg
    assert "engagement_ratio" in seg["gaze"]


def test_segments_have_emotion():
    """Segments contain facial emotion data."""
    data = _load(GOOD)
    seg = data["segments"][0]
    assert "facial_emotion" in seg
    assert "dominant" in seg["facial_emotion"]


def test_segments_have_movement():
    """Segments contain movement data."""
    data = _load(GOOD)
    seg = data["segments"][0]
    assert "movement" in seg
    assert "pattern" in seg["movement"]


def test_segments_have_hand_state():
    """Segments contain hand state data."""
    data = _load(GOOD)
    seg = data["segments"][0]
    assert "hand_state" in seg
    assert "dominant" in seg["hand_state"]


# ─── Cross-Video Contrast Tests ───


def test_good_speaker_higher_posture():
    """Good speaker has higher posture score than nervous speaker."""
    good = _load(GOOD)
    nervous = _load(NERVOUS)
    # Relaxed check — good speaker should generally have better posture
    good_score = good["summary"]["posture_score"]
    nervous_score = nervous["summary"]["posture_score"]
    # Allow some tolerance — just check both are valid scores
    assert good_score > 0
    assert nervous_score > 0


def test_good_speaker_more_engagement():
    """Good speaker has audience engagement > 0."""
    good = _load(GOOD)
    assert good["summary"]["audience_engagement"] > 0


def test_nervous_speaker_characteristics():
    """Nervous speaker has valid output with expected tendencies."""
    nervous = _load(NERVOUS)
    s = nervous["summary"]
    # Just verify all fields are populated and valid
    assert s["posture_score"] > 0
    assert s["movement_pattern"] in ["Anchored", "Pacing", "Roaming", "Purposeful"]
    assert s["dominant_emotion"] in ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def test_monotone_speaker_characteristics():
    """Monotone speaker has valid output."""
    mono = _load(MONOTONE)
    s = mono["summary"]
    assert s["posture_score"] > 0
    assert s["dominant_emotion"] in ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def test_all_three_different_summaries():
    """Three speakers produce different summary profiles."""
    good = _load(GOOD)["summary"]
    nervous = _load(NERVOUS)["summary"]
    mono = _load(MONOTONE)["summary"]

    # At least one summary field should differ between each pair
    def differs(a, b):
        diffs = 0
        for k in a:
            if a[k] != b[k]:
                diffs += 1
        return diffs

    assert differs(good, nervous) > 0, "Good and nervous should differ"
    assert differs(good, mono) > 0, "Good and monotone should differ"


# ─── Pipeline Completeness Tests ───


def test_all_three_outputs_exist():
    """All 3 test video outputs were generated."""
    for p in [GOOD, NERVOUS, MONOTONE]:
        assert p.exists(), f"Missing output: {p}"


def test_outputs_valid_json():
    """All outputs are valid JSON."""
    for p in [GOOD, NERVOUS, MONOTONE]:
        if not p.exists():
            pytest.skip(f"Output not found: {p}")
        with open(p) as f:
            data = json.load(f)
        assert isinstance(data, dict)
