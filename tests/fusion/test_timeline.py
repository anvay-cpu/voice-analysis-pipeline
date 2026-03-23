"""Tests for Phase 4.1 — Timeline Alignment."""

import pytest

from src.fusion.timeline_aligner import TimelineAligner


# ── Helpers to build mock modality outputs ──────────────────────

def make_voice_windows(duration_sec: int, hop: float = 1.0, window: float = 2.0):
    """Create mock voice windows (2s window, 1s hop)."""
    windows = []
    pos = 0.0
    while pos + window <= duration_sec + 0.01:
        windows.append({
            "start_sec": round(pos, 3),
            "end_sec": round(pos + window, 3),
            "is_speech": True,
            "transcript": f"word_{int(pos)}",
            "word_count": 2,
            "fillers": [],
            "disfluencies": [],
            "prosody": {
                "available": True,
                "syllable_rate": 4.0 + pos * 0.1,
                "rate_label": "Normal",
                "pitch_cv": 0.2,
                "pitch_mean": 150.0,
                "pitch_assessment": "Normal",
                "pitch_score": 70,
                "volume_assessment": "Normal",
                "volume_score": 75,
            },
            "emotion": {
                "available": True,
                "label": "Confident" if pos < duration_sec / 2 else "Calm",
                "confidence": 0.8,
                "distribution": {"Confident": 0.6, "Calm": 0.4},
            },
            "features": {
                "f0_mean": 150.0,
                "f0_std": 20.0,
                "f0_cv": 0.13,
                "rms_mean_db": -20.0,
                "dynamic_range_db": 15.0,
            },
        })
        pos += hop
    return windows


def make_body_output(duration_sec: int, fps: float = 5.0):
    """Create mock body output with 6s windows, 3s hop."""
    n_frames = int(duration_sec * fps)
    segments = []
    window_frames = 30
    step = 15  # 50% overlap → 3s hop
    pos = 0
    score_val = 60.0
    while pos < n_frames:
        end = min(pos + window_frames, n_frames)
        t_start = round(pos / fps, 2)
        t_end = round(end / fps, 2)
        segments.append({
            "timestamp_start": t_start,
            "timestamp_end": t_end,
            "frame_start": pos,
            "frame_end": end,
            "posture": {
                "score_mean": score_val,
                "score_std": 5.0,
                "shoulder_deg": 2.0,
                "spine_deg": 8.0,
            },
            "gesture": {
                "dominant": "Open Palm" if pos % 30 == 0 else "Rest",
                "distribution": {"Open Palm": 0.6, "Rest": 0.4},
            },
            "hand_state": {
                "dominant": "Visible",
                "distribution": {"Visible": 0.8, "Hidden": 0.2},
            },
            "gaze": {
                "distribution": {"Audience Center": 0.7, "Away": 0.3},
                "engagement_ratio": 0.7,
            },
            "facial_emotion": {
                "dominant": "Happy",
                "confidence": 0.75,
            },
            "movement": {
                "pattern": "Anchored",
                "velocity": 0.02,
            },
        })
        score_val += 2.0  # Gradual increase for interpolation testing
        if end >= n_frames:
            break
        pos += step

    return {
        "video": "test.mp4",
        "duration_sec": duration_sec,
        "total_frames": n_frames,
        "fps": fps,
        "summary": {},
        "segments": segments,
    }


def make_content_output(duration_sec: int):
    """Create mock content output with 3 regime segments."""
    seg_len = duration_sec / 3
    regimes = ["Introduction", "Data Presentation", "Conclusion"]
    tones = ["Informative", "Analytical", "Persuasive"]
    sentiments = ["Positive", "Neutral", "Positive"]

    segments = []
    for i, (regime, tone, sent) in enumerate(zip(regimes, tones, sentiments)):
        segments.append({
            "segment_id": i,
            "timestamp_start": round(i * seg_len, 2),
            "timestamp_end": round((i + 1) * seg_len, 2),
            "transcript_excerpt": f"Segment {i} text...",
            "word_count": 50,
            "regime_type": regime,
            "regime_confidence": 0.85,
            "grammar": {"errors": [], "error_count": 0},
            "readability": {"flesch_kincaid_grade": 8.5},
            "vocabulary": {"ttr": 0.65},
            "sentiment": {
                "polarity": sent,
                "intensity": 0.6,
                "valence": 0.5,
            },
            "tone": {"tone": tone, "confidence": 0.8},
            "argument": {"effectiveness_score": 72.0},
        })

    return {
        "segments": segments,
        "summary": {},
        "metadata": {"duration_sec": duration_sec},
    }


# ── Tests ───────────────────────────────────────────────────────

class TestTimelineAligner:

    @pytest.fixture
    def aligner(self):
        return TimelineAligner()

    @pytest.fixture
    def duration(self):
        return 30  # 30 seconds

    @pytest.fixture
    def voice(self, duration):
        return make_voice_windows(duration)

    @pytest.fixture
    def body(self, duration):
        return make_body_output(duration)

    @pytest.fixture
    def content(self, duration):
        return make_content_output(duration)

    def test_timeline_length(self, aligner, voice, body, content, duration):
        """Timeline should have exactly duration_sec entries."""
        timeline = aligner.align(voice, body, content, duration)
        assert len(timeline) == duration

    def test_time_sec_values(self, aligner, voice, body, content, duration):
        """Each entry should have sequential time_sec values 0..N-1."""
        timeline = aligner.align(voice, body, content, duration)
        for i, entry in enumerate(timeline):
            assert entry["time_sec"] == i

    def test_all_modalities_present(self, aligner, voice, body, content, duration):
        """Every second should have voice, body, and content data."""
        timeline = aligner.align(voice, body, content, duration)
        for entry in timeline:
            assert entry["voice"] is not None, f"Voice missing at t={entry['time_sec']}"
            assert entry["body"] is not None, f"Body missing at t={entry['time_sec']}"
            assert entry["content"] is not None, f"Content missing at t={entry['time_sec']}"

    def test_voice_fields(self, aligner, voice, body, content, duration):
        """Voice entries should have expected fields."""
        timeline = aligner.align(voice, body, content, duration)
        v = timeline[5]["voice"]
        assert "is_speech" in v
        assert "syllable_rate" in v
        assert "pitch_cv" in v
        assert "emotion_label" in v
        assert "filler_count" in v

    def test_body_fields(self, aligner, voice, body, content, duration):
        """Body entries should have expected fields."""
        timeline = aligner.align(voice, body, content, duration)
        b = timeline[5]["body"]
        assert "posture_score" in b
        assert "gesture_dominant" in b
        assert "gaze_engagement" in b
        assert "facial_emotion" in b
        assert "movement_velocity" in b

    def test_content_fields(self, aligner, voice, body, content, duration):
        """Content entries should have expected fields."""
        timeline = aligner.align(voice, body, content, duration)
        c = timeline[5]["content"]
        assert "regime_type" in c
        assert "sentiment_polarity" in c
        assert "tone" in c
        assert "readability_grade" in c

    def test_body_interpolation_smooth(self, aligner, voice, body, content, duration):
        """Body posture scores should not have sudden jumps between adjacent seconds."""
        timeline = aligner.align(voice, body, content, duration)
        scores = [entry["body"]["posture_score"] for entry in timeline]
        for i in range(1, len(scores)):
            diff = abs(scores[i] - scores[i - 1])
            assert diff < 10.0, (
                f"Posture jump of {diff:.1f} between t={i-1} and t={i}"
            )

    def test_regime_boundaries_detected(self, aligner, voice, body, content, duration):
        """Regime boundaries should be at the correct transition points."""
        timeline = aligner.align(voice, body, content, duration)
        boundaries = aligner.find_regime_boundaries(timeline)
        assert len(boundaries) == 2  # 3 regimes → 2 boundaries
        assert boundaries[0]["from_regime"] == "Introduction"
        assert boundaries[0]["to_regime"] == "Data Presentation"
        assert boundaries[1]["from_regime"] == "Data Presentation"
        assert boundaries[1]["to_regime"] == "Conclusion"

    def test_regime_boundary_times(self, aligner, voice, body, content, duration):
        """Boundaries should be at ~10s and ~20s for a 30s speech with 3 equal segments."""
        timeline = aligner.align(voice, body, content, duration)
        boundaries = aligner.find_regime_boundaries(timeline)
        assert abs(boundaries[0]["time_sec"] - 10) <= 1
        assert abs(boundaries[1]["time_sec"] - 20) <= 1

    def test_missing_voice(self, aligner, body, content, duration):
        """Missing voice data should produce None, no crash."""
        timeline = aligner.align([], body, content, duration)
        for entry in timeline:
            assert entry["voice"] is None
            assert entry["body"] is not None
            assert entry["content"] is not None

    def test_missing_body(self, aligner, voice, content, duration):
        """Missing body data should produce None, no crash."""
        empty_body = {"segments": []}
        timeline = aligner.align(voice, empty_body, content, duration)
        for entry in timeline:
            assert entry["voice"] is not None
            assert entry["body"] is None
            assert entry["content"] is not None

    def test_missing_content(self, aligner, voice, body, duration):
        """Missing content data should produce None, no crash."""
        empty_content = {"segments": []}
        timeline = aligner.align(voice, body, empty_content, duration)
        for entry in timeline:
            assert entry["voice"] is not None
            assert entry["body"] is not None
            assert entry["content"] is None

    def test_none_inputs(self, aligner, duration):
        """All None inputs should produce timeline with all None values."""
        timeline = aligner.align([], None, None, duration)
        assert len(timeline) == duration
        for entry in timeline:
            assert entry["voice"] is None
            assert entry["body"] is None
            assert entry["content"] is None

    def test_short_duration(self, aligner):
        """1-second duration should work without errors."""
        voice = make_voice_windows(2)  # need at least 1 window
        body = make_body_output(2)
        content = make_content_output(2)
        timeline = aligner.align(voice, body, content, 1)
        assert len(timeline) == 1
        assert timeline[0]["time_sec"] == 0

    def test_content_constant_within_segment(self, aligner, voice, body, content, duration):
        """Content features should be constant within a regime segment."""
        timeline = aligner.align(voice, body, content, duration)
        # Seconds 0-9 should all be "Introduction"
        for sec in range(0, 10):
            assert timeline[sec]["content"]["regime_type"] == "Introduction"

    def test_no_regime_boundaries_single_regime(self, aligner):
        """Single-regime speech should produce no boundaries."""
        duration = 20
        voice = make_voice_windows(duration)
        body = make_body_output(duration)
        content = {
            "segments": [{
                "segment_id": 0,
                "timestamp_start": 0.0,
                "timestamp_end": 20.0,
                "regime_type": "Introduction",
                "regime_confidence": 0.9,
                "sentiment": {"polarity": "Neutral", "intensity": 0.5, "valence": 0.0},
                "tone": {"tone": "Informative", "confidence": 0.8},
                "readability": {"flesch_kincaid_grade": 8.0},
                "grammar": {"error_count": 0},
                "argument": {},
            }],
            "summary": {},
        }
        timeline = aligner.align(voice, body, content, duration)
        boundaries = aligner.find_regime_boundaries(timeline)
        assert len(boundaries) == 0
