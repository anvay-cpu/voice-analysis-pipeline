"""Phase 4.4 — Emotion Coherence Analysis.

Measures alignment between vocal emotion (M1), facial emotion (M2),
and content sentiment (M3) at each second of the aligned timeline.
"""

import logging
from statistics import mean

logger = logging.getLogger(__name__)

# Valence mappings for vocal emotion labels
VOICE_VALENCE = {
    "Enthusiastic": 1.0,
    "Confident": 0.7,
    "Calm": 0.3,
    "Neutral": 0.0,
    "Nervous": -0.5,
    "Angry": -0.8,
    "Sad": -1.0,
    "Unknown": 0.0,
}

# Valence mappings for facial emotion labels
FACE_VALENCE = {
    "Happy": 1.0,
    "Surprise": 0.3,
    "Neutral": 0.0,
    "Contempt": -0.3,
    "Fear": -0.5,
    "Sad": -0.8,
    "Disgust": -0.9,
    "Anger": -1.0,
    "Unknown": 0.0,
}

# Sentiment polarity to valence
SENTIMENT_VALENCE = {
    "Positive": 1.0,
    "Neutral": 0.0,
    "Negative": -1.0,
}


class EmotionCoherenceAnalyzer:
    """Measures alignment between face, voice, and content emotions."""

    def __init__(self, mismatch_threshold: float = 0.4, cluster_gap_sec: int = 5):
        self.mismatch_threshold = mismatch_threshold
        self.cluster_gap_sec = cluster_gap_sec

    def compute_coherence_at_time(
        self, voice_emo: str, face_emo: str, content_valence: float
    ) -> float:
        """Compute emotion coherence score for one second.

        Args:
            voice_emo: Vocal emotion label.
            face_emo: Facial emotion label.
            content_valence: Content sentiment valence (-1.0 to +1.0).

        Returns:
            Coherence score 0.0 (total mismatch) to 1.0 (perfect alignment).
        """
        v_voice = VOICE_VALENCE.get(voice_emo, 0.0)
        v_face = FACE_VALENCE.get(face_emo, 0.0)
        v_content = content_valence

        # Pairwise agreement: 1.0 when identical, 0.0 when maximally different
        voice_face = 1.0 - abs(v_voice - v_face) / 2.0
        voice_content = 1.0 - abs(v_voice - v_content) / 2.0
        face_content = 1.0 - abs(v_face - v_content) / 2.0

        # Weighted average (voice-face alignment most important)
        coherence = 0.4 * voice_face + 0.3 * voice_content + 0.3 * face_content
        return round(coherence, 4)

    def detect_mismatches(self, timeline: list[dict]) -> list[dict]:
        """Find segments where emotions significantly disagree.

        Args:
            timeline: Aligned 1-second timeline.

        Returns:
            List of clustered mismatch event dicts.
        """
        mismatches = []

        for entry in timeline:
            voice = entry.get("voice") or {}
            body = entry.get("body") or {}
            content = entry.get("content") or {}

            voice_emo = voice.get("emotion_label", "Neutral")
            face_emo = body.get("facial_emotion", "Neutral")
            content_valence = self._get_content_valence(content)

            coherence = self.compute_coherence_at_time(
                voice_emo, face_emo, content_valence
            )

            if coherence < self.mismatch_threshold:
                mismatches.append({
                    "time_sec": entry["time_sec"],
                    "coherence": coherence,
                    "voice_emotion": voice_emo,
                    "face_emotion": face_emo,
                    "content_sentiment": content.get("sentiment_polarity", "Neutral"),
                    "description": self._describe_mismatch(
                        voice_emo, face_emo, content_valence
                    ),
                })

        return self._cluster_mismatches(mismatches)

    def analyze_full(self, timeline: list[dict]) -> dict:
        """Full emotion coherence analysis.

        Args:
            timeline: Aligned 1-second timeline.

        Returns:
            Dict with mean coherence, per-second timeline, mismatch events.
        """
        if not timeline:
            return {
                "mean_coherence": 1.0,
                "coherence_timeline": [],
                "mismatch_events": [],
                "total_mismatch_seconds": 0,
                "coherence_score_0_100": 100,
            }

        per_second = []
        for entry in timeline:
            voice = entry.get("voice") or {}
            body = entry.get("body") or {}
            content = entry.get("content") or {}

            voice_emo = voice.get("emotion_label", "Neutral")
            face_emo = body.get("facial_emotion", "Neutral")
            content_valence = self._get_content_valence(content)

            per_second.append(
                self.compute_coherence_at_time(voice_emo, face_emo, content_valence)
            )

        mismatches = self.detect_mismatches(timeline)
        mean_coherence = mean(per_second) if per_second else 1.0

        return {
            "mean_coherence": round(mean_coherence, 4),
            "coherence_timeline": per_second,
            "mismatch_events": mismatches,
            "total_mismatch_seconds": sum(
                m.get("duration_sec", 1) for m in mismatches
            ),
            "coherence_score_0_100": round(mean_coherence * 100),
        }

    # ── Internal ───────────────────────────────────────────────────

    def _get_content_valence(self, content: dict) -> float:
        """Extract valence from content sentiment data."""
        # Try direct valence first
        valence = content.get("sentiment_valence")
        if valence is not None:
            return float(valence)

        # Fall back to polarity mapping
        polarity = content.get("sentiment_polarity", "Neutral")
        return SENTIMENT_VALENCE.get(polarity, 0.0)

    def _describe_mismatch(
        self, voice_emo: str, face_emo: str, content_valence: float
    ) -> str:
        """Generate human-readable description of the mismatch."""
        descriptions = []

        if voice_emo == "Enthusiastic" and face_emo in ("Neutral", "Sad"):
            descriptions.append("Voice sounds excited but face appears flat")
        if face_emo == "Happy" and content_valence < -0.3:
            descriptions.append("Smiling while delivering negative content")
        if voice_emo == "Nervous" and face_emo == "Happy":
            descriptions.append("Voice betrays nervousness despite smiling")
        if voice_emo == "Neutral" and content_valence > 0.5:
            descriptions.append("Monotone delivery of enthusiastic content")
        if voice_emo == "Angry" and face_emo == "Happy":
            descriptions.append("Angry voice but happy face")
        if voice_emo == "Confident" and face_emo in ("Fear", "Sad"):
            descriptions.append("Confident voice but anxious/sad expression")
        if face_emo == "Anger" and content_valence > 0.3:
            descriptions.append("Angry expression while delivering positive content")

        return "; ".join(descriptions) if descriptions else "Emotion channels disagree"

    def _cluster_mismatches(self, mismatches: list[dict]) -> list[dict]:
        """Merge consecutive mismatches within gap_sec into events."""
        if not mismatches:
            return []

        clusters = []
        current = [mismatches[0]]

        for m in mismatches[1:]:
            if m["time_sec"] - current[-1]["time_sec"] <= self.cluster_gap_sec:
                current.append(m)
            else:
                clusters.append(self._summarize_cluster(current))
                current = [m]

        clusters.append(self._summarize_cluster(current))
        return clusters

    def _summarize_cluster(self, cluster: list[dict]) -> dict:
        """Summarize a cluster of consecutive mismatches into one event."""
        coherences = [m["coherence"] for m in cluster]
        return {
            "time_sec": cluster[0]["time_sec"],
            "end_sec": cluster[-1]["time_sec"],
            "duration_sec": cluster[-1]["time_sec"] - cluster[0]["time_sec"] + 1,
            "mean_coherence": round(mean(coherences), 4),
            "min_coherence": round(min(coherences), 4),
            "primary_voice_emotion": cluster[0]["voice_emotion"],
            "primary_face_emotion": cluster[0]["face_emotion"],
            "description": cluster[0]["description"],
        }
