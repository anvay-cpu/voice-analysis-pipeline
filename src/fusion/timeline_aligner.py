"""Phase 4.1 — Timeline Alignment.

Aligns voice (2s window, 1s hop), body (6s window, 3s hop), and content
(variable-length regime segments) onto a unified 1-second resolution timeline.
"""

import logging
from copy import deepcopy

logger = logging.getLogger(__name__)

# Body fields that should be linearly interpolated (continuous values)
BODY_CONTINUOUS_FIELDS = {
    "posture": ["score_mean", "score_std", "shoulder_deg", "spine_deg"],
    "gaze": ["engagement_ratio"],
    "facial_emotion": ["confidence"],
    "movement": ["velocity"],
}

# Body fields that hold categorical values (no interpolation)
BODY_CATEGORICAL_FIELDS = {
    "gesture": ["dominant"],
    "hand_state": ["dominant"],
    "facial_emotion": ["dominant"],
    "movement": ["pattern"],
}

# Body fields that are probability distributions (interpolate each key)
BODY_DISTRIBUTION_FIELDS = {
    "gesture": ["distribution"],
    "hand_state": ["distribution"],
    "gaze": ["distribution"],
}


class TimelineAligner:
    """Aligns all three modality outputs onto a 1-second grid.

    Input:
        voice_output: list of dicts, each covering a 2s window with 1s hop
        body_output: dict with 'segments' list, each covering a 6s window
        content_output: dict with 'segments' list, each covering a regime segment

    Output:
        timeline: list of dicts, one per second, containing:
        {
            "time_sec": 42,
            "voice": { prosody, emotion, fillers, ... },
            "body": { posture_score, gesture, gaze, ... },
            "content": { regime_type, sentiment, tone, ... },
        }
    """

    def align(
        self,
        voice_output: list[dict],
        body_output: dict,
        content_output: dict,
        duration_sec: float,
    ) -> list[dict]:
        """Create 1-second aligned timeline from all 3 modalities.

        Args:
            voice_output: Voice pipeline windows (list of dicts with start_sec/end_sec).
            body_output: Body pipeline output dict with 'segments' key.
            content_output: Content pipeline output dict with 'segments' key.
            duration_sec: Total duration in seconds.

        Returns:
            List of per-second dicts with voice/body/content data.
        """
        n_seconds = int(duration_sec)

        voice_1s = self._map_voice_to_1sec(voice_output, n_seconds)
        body_segments = body_output.get("segments", []) if body_output else []
        body_1s = self._interpolate_body_to_1sec(body_segments, n_seconds)
        content_segments = content_output.get("segments", []) if content_output else []
        content_1s = self._map_content_to_1sec(content_segments, n_seconds)

        timeline = []
        for sec in range(n_seconds):
            timeline.append({
                "time_sec": sec,
                "voice": voice_1s[sec] if sec < len(voice_1s) else None,
                "body": body_1s[sec] if sec < len(body_1s) else None,
                "content": content_1s[sec] if sec < len(content_1s) else None,
            })

        logger.info(
            "Timeline aligned: %d seconds, voice=%d body=%d content=%d entries",
            n_seconds, len(voice_1s), len(body_1s), len(content_1s),
        )

        return timeline

    def find_regime_boundaries(self, timeline: list[dict]) -> list[dict]:
        """Find seconds where the regime type changes.

        Args:
            timeline: Aligned timeline from align().

        Returns:
            List of boundary dicts with time_sec, from_regime, to_regime.
        """
        boundaries = []
        for i in range(1, len(timeline)):
            prev_content = timeline[i - 1].get("content")
            curr_content = timeline[i].get("content")

            if prev_content is None or curr_content is None:
                continue

            prev_regime = prev_content.get("regime_type")
            curr_regime = curr_content.get("regime_type")

            if prev_regime != curr_regime:
                boundaries.append({
                    "time_sec": i,
                    "from_regime": prev_regime,
                    "to_regime": curr_regime,
                })

        return boundaries

    # ── Voice mapping ──────────────────────────────────────────────

    def _map_voice_to_1sec(
        self, voice_windows: list[dict], n_seconds: int
    ) -> list[dict | None]:
        """Map voice windows (2s, 1s hop) to 1-second resolution.

        Each second maps to the voice window whose center is closest.
        With 1s hop, most seconds have a direct match.
        """
        if not voice_windows:
            return [None] * n_seconds

        result = [None] * n_seconds

        for sec in range(n_seconds):
            # Find window whose interval contains this second
            best = None
            best_dist = float("inf")

            for w in voice_windows:
                w_start = w.get("start_sec", 0.0)
                w_end = w.get("end_sec", 0.0)
                w_center = (w_start + w_end) / 2.0

                if w_start <= sec + 0.5 < w_end:
                    dist = abs(sec + 0.5 - w_center)
                    if dist < best_dist:
                        best_dist = dist
                        best = w

            if best is None:
                # Fallback: nearest window by center
                best = min(
                    voice_windows,
                    key=lambda w: abs(
                        (w.get("start_sec", 0) + w.get("end_sec", 0)) / 2 - (sec + 0.5)
                    ),
                )

            result[sec] = self._extract_voice_features(best)

        return result

    def _extract_voice_features(self, window: dict) -> dict:
        """Extract relevant voice features from a window dict."""
        prosody = window.get("prosody", {})
        emotion = window.get("emotion", {})
        features = window.get("features", {})

        return {
            "is_speech": window.get("is_speech", False),
            "transcript": window.get("transcript", ""),
            "word_count": window.get("word_count", 0),
            "filler_count": len(window.get("fillers", [])),
            "fillers": window.get("fillers", []),
            "disfluency_count": len(window.get("disfluencies", [])),
            "disfluencies": window.get("disfluencies", []),
            # Prosody
            "syllable_rate": prosody.get("syllable_rate", 0.0),
            "rate_label": prosody.get("rate_label", ""),
            "pitch_cv": prosody.get("pitch_cv", 0.0),
            "pitch_mean": prosody.get("pitch_mean", 0.0),
            "pitch_score": prosody.get("pitch_score", 0),
            "volume_score": prosody.get("volume_score", 0),
            # Emotion
            "emotion_label": emotion.get("label", "Unknown"),
            "emotion_confidence": emotion.get("confidence", 0.0),
            # Acoustic features
            "f0_mean": features.get("f0_mean", 0.0),
            "f0_cv": features.get("f0_cv", 0.0),
            "rms_mean_db": features.get("rms_mean_db", 0.0),
        }

    # ── Body interpolation ─────────────────────────────────────────

    def _interpolate_body_to_1sec(
        self, body_segments: list[dict], n_seconds: int
    ) -> list[dict | None]:
        """Upsample body data from 3s hop to 1s resolution.

        For continuous values: linear interpolation between window centers.
        For categorical values: hold constant within the window.
        For distributions: linear interpolation of each probability.
        """
        if not body_segments:
            return [None] * n_seconds

        result = [None] * n_seconds

        for sec in range(n_seconds):
            sec_mid = sec + 0.5  # center of this 1-second tick

            # Find body segments that cover this second
            covering = [
                s for s in body_segments
                if s.get("timestamp_start", 0) <= sec_mid < s.get("timestamp_end", 0)
            ]

            if len(covering) == 0:
                # Gap: use nearest segment
                nearest = min(
                    body_segments,
                    key=lambda s: min(
                        abs(s.get("timestamp_start", 0) - sec_mid),
                        abs(s.get("timestamp_end", 0) - sec_mid),
                    ),
                )
                result[sec] = self._extract_body_features(nearest)
            elif len(covering) == 1:
                result[sec] = self._extract_body_features(covering[0])
            else:
                # Overlap zone: weighted merge by proximity to center
                result[sec] = self._weighted_merge_body(covering, sec_mid)

        return result

    def _extract_body_features(self, segment: dict) -> dict:
        """Extract body features from a segment dict."""
        posture = segment.get("posture", {})
        gesture = segment.get("gesture", {})
        hand = segment.get("hand_state", {})
        gaze = segment.get("gaze", {})
        emotion = segment.get("facial_emotion", {})
        movement = segment.get("movement", {})

        return {
            "posture_score": posture.get("score_mean", 50.0),
            "posture_std": posture.get("score_std", 0.0),
            "shoulder_deg": posture.get("shoulder_deg", 0.0),
            "spine_deg": posture.get("spine_deg", 0.0),
            "gesture_dominant": gesture.get("dominant", "Rest"),
            "gesture_distribution": gesture.get("distribution", {}),
            "hand_state": hand.get("dominant", "Other"),
            "hand_distribution": hand.get("distribution", {}),
            "gaze_engagement": gaze.get("engagement_ratio", 0.0),
            "gaze_distribution": gaze.get("distribution", {}),
            "facial_emotion": emotion.get("dominant", "Neutral"),
            "facial_emotion_confidence": emotion.get("confidence", 0.0),
            "movement_pattern": movement.get("pattern", "Anchored"),
            "movement_velocity": movement.get("velocity", 0.0),
        }

    def _weighted_merge_body(self, segments: list[dict], sec_mid: float) -> dict:
        """Merge overlapping body segments using distance-weighted averaging.

        Continuous fields are linearly interpolated.
        Categorical fields take the value from the nearest segment.
        Distribution fields are linearly interpolated per key.
        """
        # Compute weights inversely proportional to distance from segment center
        weights = []
        for s in segments:
            center = (s.get("timestamp_start", 0) + s.get("timestamp_end", 0)) / 2
            dist = abs(center - sec_mid) + 1e-6  # avoid division by zero
            weights.append(1.0 / dist)

        total_weight = sum(weights)
        norm_weights = [w / total_weight for w in weights]

        extracted = [self._extract_body_features(s) for s in segments]

        # Start from the nearest segment's features
        nearest_idx = norm_weights.index(max(norm_weights))
        merged = deepcopy(extracted[nearest_idx])

        # Interpolate continuous fields
        continuous_keys = [
            "posture_score", "posture_std", "shoulder_deg", "spine_deg",
            "gaze_engagement", "facial_emotion_confidence", "movement_velocity",
        ]
        for key in continuous_keys:
            merged[key] = round(
                sum(e[key] * w for e, w in zip(extracted, norm_weights)),
                4,
            )

        # Interpolate distributions
        dist_keys = ["gesture_distribution", "hand_distribution", "gaze_distribution"]
        for key in dist_keys:
            merged[key] = self._interpolate_distribution(
                [e[key] for e in extracted], norm_weights
            )

        return merged

    def _interpolate_distribution(
        self, distributions: list[dict], weights: list[float]
    ) -> dict:
        """Weighted average of probability distributions."""
        all_keys = set()
        for d in distributions:
            all_keys.update(d.keys())

        result = {}
        for k in all_keys:
            result[k] = round(
                sum(d.get(k, 0.0) * w for d, w in zip(distributions, weights)),
                3,
            )

        return result

    # ── Content mapping ────────────────────────────────────────────

    def _map_content_to_1sec(
        self, content_segments: list[dict], n_seconds: int
    ) -> list[dict | None]:
        """Map content regime segments to 1-second resolution.

        Content features are constant across the entire segment.
        """
        if not content_segments:
            return [None] * n_seconds

        result = [None] * n_seconds

        for sec in range(n_seconds):
            sec_mid = sec + 0.5

            # Find the segment that contains this second
            match = None
            for seg in content_segments:
                start = seg.get("timestamp_start", 0.0)
                end = seg.get("timestamp_end", 0.0)
                if start <= sec_mid < end:
                    match = seg
                    break

            if match is None:
                # Fallback: nearest segment
                match = min(
                    content_segments,
                    key=lambda s: min(
                        abs(s.get("timestamp_start", 0) - sec_mid),
                        abs(s.get("timestamp_end", 0) - sec_mid),
                    ),
                )

            result[sec] = self._extract_content_features(match)

        return result

    def _extract_content_features(self, segment: dict) -> dict:
        """Extract content features from a segment dict."""
        sentiment = segment.get("sentiment", {})
        tone = segment.get("tone", {})
        readability = segment.get("readability", {})
        argument = segment.get("argument", {})

        return {
            "regime_type": segment.get("regime_type", "Unknown"),
            "regime_confidence": segment.get("regime_confidence", 0.0),
            "sentiment_polarity": sentiment.get("polarity", "Neutral"),
            "sentiment_intensity": sentiment.get("intensity", 0.0),
            "sentiment_valence": sentiment.get("valence", 0.0),
            "tone": tone.get("tone", "Unknown"),
            "tone_confidence": tone.get("confidence", 0.0),
            "readability_grade": readability.get("flesch_kincaid_grade", 0.0),
            "grammar_error_count": segment.get("grammar", {}).get("error_count", 0),
            "argument_effectiveness": argument.get("effectiveness_score"),
        }
