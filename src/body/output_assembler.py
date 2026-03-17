"""Step 11: JSON output assembly for body language analysis.

Merges all sub-module outputs into a structured JSON report with
per-window segments and video-level summary statistics.
"""

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from src.body.temporal_model import TemporalAggregator

logger = logging.getLogger(__name__)


def compute_summary(segments: list[dict], movement_result: dict | None = None) -> dict:
    """Compute video-level summary from aggregated segments.

    Returns dict with overall scores and dominant patterns.
    """
    if not segments:
        return {
            "posture_score": 50.0,
            "dominant_gesture": "Rest",
            "dominant_hand_state": "Other",
            "audience_engagement": 0.0,
            "dominant_emotion": "Neutral",
            "movement_pattern": "Anchored",
            "stage_usage_score": 5.0,
        }

    # Posture: mean of window means
    posture_scores = [
        s["posture"]["score_mean"]
        for s in segments if "posture" in s
    ]
    avg_posture = round(float(np.mean(posture_scores)), 1) if posture_scores else 50.0

    # Gesture: overall majority vote across all windows
    gesture_labels = [
        s["gesture"]["dominant"]
        for s in segments if "gesture" in s
    ]
    dom_gesture = Counter(gesture_labels).most_common(1)[0][0] if gesture_labels else "Rest"

    # Hand: overall majority
    hand_labels = [
        s["hand_state"]["dominant"]
        for s in segments if "hand_state" in s
    ]
    dom_hand = Counter(hand_labels).most_common(1)[0][0] if hand_labels else "Other"

    # Gaze engagement: mean across windows
    engagements = [
        s["gaze"]["engagement_ratio"]
        for s in segments if "gaze" in s
    ]
    avg_engagement = round(float(np.mean(engagements)), 2) if engagements else 0.0

    # Emotion: overall majority
    emotion_labels = [
        s["facial_emotion"]["dominant"]
        for s in segments if "facial_emotion" in s
    ]
    dom_emotion = Counter(emotion_labels).most_common(1)[0][0] if emotion_labels else "Neutral"

    # Movement: from full-video analysis
    mvmt_pattern = "Anchored"
    stage_score = 5.0
    if movement_result:
        mvmt_pattern = movement_result.get("movement_pattern", "Anchored")
        stage_score = movement_result.get("stage_usage_score", 5.0)

    return {
        "posture_score": avg_posture,
        "dominant_gesture": dom_gesture,
        "dominant_hand_state": dom_hand,
        "audience_engagement": avg_engagement,
        "dominant_emotion": dom_emotion,
        "movement_pattern": mvmt_pattern,
        "stage_usage_score": stage_score,
    }


class OutputAssembler:
    """Assemble body language analysis into structured JSON output."""

    def __init__(self, fps: float = 5.0, window_frames: int = 30, overlap: float = 0.5):
        self.aggregator = TemporalAggregator(
            window_frames=window_frames,
            overlap=overlap,
            fps=fps,
        )
        self.fps = fps

    def assemble(
        self,
        video_path: str,
        n_frames: int,
        posture_scores: dict | None = None,
        gesture_results: list[dict] | None = None,
        hand_results: list[dict] | None = None,
        gaze_result: dict | None = None,
        emotion_results: list[dict] | None = None,
        movement_result: dict | None = None,
    ) -> dict:
        """Assemble complete body language analysis output.

        Returns structured dict ready for JSON serialization.
        """
        # Temporal aggregation
        segments = self.aggregator.aggregate(
            n_frames=n_frames,
            posture_scores=posture_scores,
            gesture_results=gesture_results,
            hand_results=hand_results,
            gaze_result=gaze_result,
            emotion_results=emotion_results,
            movement_result=movement_result,
        )

        # Summary
        summary = compute_summary(segments, movement_result)

        output = {
            "video": video_path,
            "duration_sec": round(n_frames / self.fps, 2),
            "total_frames": n_frames,
            "fps": self.fps,
            "summary": summary,
            "segments": segments,
        }

        logger.info(
            "Output assembled: %d segments, duration=%.1fs",
            len(segments), output["duration_sec"],
        )

        return output

    def save(self, output: dict, output_path: str) -> str:
        """Save assembled output to JSON file.

        Returns the output file path.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info("Saved body language output to %s", output_path)
        return output_path
