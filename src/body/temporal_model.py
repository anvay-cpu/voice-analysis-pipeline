"""Step 10: Frame-to-segment temporal aggregation.

Aggregates per-frame body language outputs into 6-second windows
(30 frames at 5 fps) with 50% overlap. Applies 1D smoothing before
aggregation to remove single-frame noise.
"""

import logging
from collections import Counter

import numpy as np
from scipy.ndimage import uniform_filter1d

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_FRAMES = 30   # 6 seconds at 5 fps
DEFAULT_OVERLAP = 0.5        # 50% overlap
SMOOTH_KERNEL = 5            # 1D smoothing kernel size


def smooth_scores(scores: list[float], kernel: int = SMOOTH_KERNEL) -> list[float]:
    """Apply 1D uniform filter to smooth a sequence of scores."""
    if len(scores) < kernel:
        return scores
    arr = np.array(scores, dtype=np.float32)
    smoothed = uniform_filter1d(arr, size=kernel, mode="nearest")
    return smoothed.tolist()


def majority_vote(labels: list[str]) -> str:
    """Return the most common label (plurality)."""
    if not labels:
        return "Unknown"
    counts = Counter(labels)
    return counts.most_common(1)[0][0]


def make_windows(
    n_frames: int,
    window_size: int = DEFAULT_WINDOW_FRAMES,
    overlap: float = DEFAULT_OVERLAP,
) -> list[tuple[int, int]]:
    """Generate (start, end) frame index pairs for overlapping windows.

    Args:
        n_frames: Total number of frames.
        window_size: Frames per window.
        overlap: Fraction of overlap between consecutive windows.

    Returns:
        List of (start_idx, end_idx) tuples.
    """
    if n_frames == 0:
        return []
    step = max(1, int(window_size * (1 - overlap)))
    windows = []
    for start in range(0, n_frames, step):
        end = min(start + window_size, n_frames)
        windows.append((start, end))
        if end >= n_frames:
            break
    return windows


def aggregate_posture(
    posture_scores: dict[int, dict | None],
    frame_indices: list[int],
) -> dict:
    """Aggregate posture scores for a window.

    Returns dict with mean/std of combined_score and detail averages.
    """
    scores = []
    shoulder_vals = []
    spine_vals = []
    for idx in frame_indices:
        s = posture_scores.get(idx)
        if s is None:
            continue
        scores.append(s.get("combined_score", s.get("rule_score", 50.0)))
        shoulder_vals.append(s.get("shoulder_alignment", 0.0))
        spine_vals.append(s.get("spine_angle", 0.0))

    if not scores:
        return {"score_mean": 50.0, "score_std": 0.0, "shoulder_deg": 0.0, "spine_deg": 0.0}

    return {
        "score_mean": round(float(np.mean(scores)), 1),
        "score_std": round(float(np.std(scores)), 1),
        "shoulder_deg": round(float(np.mean(shoulder_vals)), 1),
        "spine_deg": round(float(np.mean(spine_vals)), 1),
    }


def aggregate_gestures(
    gesture_results: list[dict],
    start_frame: int,
    end_frame: int,
) -> dict:
    """Aggregate gesture classifications for a window.

    Returns dict with dominant gesture and distribution.
    """
    labels = []
    for g in gesture_results:
        gs = g.get("start_frame", 0)
        ge = g.get("end_frame", 0)
        # Include if gesture overlaps with window
        if ge >= start_frame and gs < end_frame:
            labels.append(g.get("gesture_label", "Rest"))

    if not labels:
        return {"dominant": "Rest", "distribution": {"Rest": 1.0}}

    counts = Counter(labels)
    total = len(labels)
    dist = {k: round(v / total, 2) for k, v in counts.most_common()}
    return {"dominant": counts.most_common(1)[0][0], "distribution": dist}


def aggregate_hand_states(
    hand_results: list[dict],
    frame_indices: list[int],
) -> dict:
    """Aggregate hand states for a window.

    Returns dict with most frequent state and distribution.
    """
    # Build frame_idx -> state lookup
    state_map = {r["frame_idx"]: r["state"] for r in hand_results if "frame_idx" in r}
    states = [state_map[i] for i in frame_indices if i in state_map]

    if not states:
        return {"dominant": "Other", "distribution": {"Other": 1.0}}

    counts = Counter(states)
    total = len(states)
    dist = {k: round(v / total, 2) for k, v in counts.most_common()}
    return {"dominant": counts.most_common(1)[0][0], "distribution": dist}


def aggregate_gaze(
    gaze_frames: list[dict],
    frame_indices: list[int],
) -> dict:
    """Aggregate gaze zones for a window.

    Returns dict with zone distribution and engagement ratio.
    """
    # Build frame_idx -> zone lookup
    zone_map = {}
    for g in gaze_frames:
        if "frame_idx" in g:
            zone_map[g["frame_idx"]] = g.get("gaze_zone", "Away")

    zones = [zone_map[i] for i in frame_indices if i in zone_map]

    if not zones:
        return {"distribution": {"Away": 1.0}, "engagement_ratio": 0.0}

    counts = Counter(zones)
    total = len(zones)
    dist = {k: round(v / total, 2) for k, v in counts.most_common()}

    audience_zones = {"Audience Center", "Audience Left", "Audience Right"}
    engaged = sum(1 for z in zones if z in audience_zones)
    engagement = round(engaged / total, 2)

    return {"distribution": dist, "engagement_ratio": engagement}


def aggregate_emotion(
    emotion_results: list[dict],
    frame_indices: list[int],
) -> dict:
    """Aggregate facial emotions for a window.

    Returns dict with dominant emotion and mean confidence.
    """
    # Build frame_idx -> emotion lookup
    emo_map = {}
    for e in emotion_results:
        if "frame_idx" in e:
            emo_map[e["frame_idx"]] = e

    emotions = []
    confidences = []
    for idx in frame_indices:
        if idx in emo_map:
            emotions.append(emo_map[idx].get("emotion", "Neutral"))
            confidences.append(emo_map[idx].get("confidence", 0.0))

    if not emotions:
        return {"dominant": "Neutral", "confidence": 0.0}

    dominant = majority_vote(emotions)
    mean_conf = round(float(np.mean(confidences)), 3)
    return {"dominant": dominant, "confidence": mean_conf}


def aggregate_movement(
    movement_result: dict,
    start_frame: int,
    end_frame: int,
) -> dict:
    """Extract movement info for a window from the full-video analysis."""
    centroids = movement_result.get("centroids", [])
    window_centroids = [
        c for c in centroids
        if start_frame <= c["frame_idx"] < end_frame
    ]

    if len(window_centroids) < 2:
        return {
            "pattern": movement_result.get("movement_pattern", "Anchored"),
            "velocity": 0.0,
        }

    # Window-level velocity
    total_dist = 0.0
    for i in range(1, len(window_centroids)):
        dx = window_centroids[i]["x"] - window_centroids[i - 1]["x"]
        dy = window_centroids[i]["y"] - window_centroids[i - 1]["y"]
        total_dist += (dx**2 + dy**2) ** 0.5

    span = window_centroids[-1]["frame_idx"] - window_centroids[0]["frame_idx"]
    velocity = total_dist / max(span, 1)

    return {
        "pattern": movement_result.get("movement_pattern", "Anchored"),
        "velocity": round(velocity, 4),
    }


class TemporalAggregator:
    """Aggregate per-frame body language outputs into time windows."""

    def __init__(
        self,
        window_frames: int = DEFAULT_WINDOW_FRAMES,
        overlap: float = DEFAULT_OVERLAP,
        fps: float = 5.0,
    ):
        self.window_frames = window_frames
        self.overlap = overlap
        self.fps = fps

    def aggregate(
        self,
        n_frames: int,
        posture_scores: dict[int, dict | None] | None = None,
        gesture_results: list[dict] | None = None,
        hand_results: list[dict] | None = None,
        gaze_result: dict | None = None,
        emotion_results: list[dict] | None = None,
        movement_result: dict | None = None,
    ) -> list[dict]:
        """Aggregate all per-frame outputs into windowed segments.

        Args:
            n_frames: Total number of frames.
            posture_scores: Frame-indexed posture score dicts.
            gesture_results: List of gesture classification dicts.
            hand_results: List of per-frame hand state dicts.
            gaze_result: Full gaze track_video result dict.
            emotion_results: List of per-frame emotion dicts.
            movement_result: Full stage movement analysis dict.

        Returns:
            List of per-window aggregated dicts.
        """
        windows = make_windows(n_frames, self.window_frames, self.overlap)
        gaze_frames = gaze_result.get("frames", []) if gaze_result else []

        segments = []
        for start, end in windows:
            frame_indices = list(range(start, end))
            t_start = round(start / self.fps, 2)
            t_end = round(end / self.fps, 2)

            segment = {
                "timestamp_start": t_start,
                "timestamp_end": t_end,
                "frame_start": start,
                "frame_end": end,
            }

            if posture_scores is not None:
                segment["posture"] = aggregate_posture(posture_scores, frame_indices)

            if gesture_results is not None:
                segment["gesture"] = aggregate_gestures(gesture_results, start, end)

            if hand_results is not None:
                segment["hand_state"] = aggregate_hand_states(hand_results, frame_indices)

            if gaze_frames:
                segment["gaze"] = aggregate_gaze(gaze_frames, frame_indices)

            if emotion_results is not None:
                segment["facial_emotion"] = aggregate_emotion(emotion_results, frame_indices)

            if movement_result is not None:
                segment["movement"] = aggregate_movement(movement_result, start, end)

            segments.append(segment)

        logger.info(
            "Temporal aggregation: %d frames → %d windows (%.1fs each, %.0f%% overlap)",
            n_frames, len(segments), self.window_frames / self.fps,
            self.overlap * 100,
        )

        return segments
