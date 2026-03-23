PHASE 4.4— EMOTION COHERENCE
================================

AGENT ROLE: Emotion Alignment Specialist
DEPENDS ON: Phase 4.1 (aligned timeline)
DELIVERS TO: Phase 5.1 (scoring)
RUNS ON: Local M1
ESTIMATED TIME: 20 minutes (agent), 0 minutes (user)

OBJECTIVE:
Build src/fusion/emotion_coherence.py that checks whether the
speaker's facial emotion, vocal emotion, and content sentiment
are aligned or contradictory.

═══════════════════════════════════════════════════════════════
THE THREE EMOTION CHANNELS
═══════════════════════════════════════════════════════════════

At each second of the timeline, we have:

1. VOCAL EMOTION (from Modality 1):
   Classes: Neutral, Enthusiastic, Nervous, Angry, Sad
   Map to valence: Enthusiastic=+1, Neutral=0, Nervous=-0.5,
                   Angry=-0.8, Sad=-1.0

2. FACIAL EMOTION (from Modality 2):
   Classes: Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt
   Map to valence: Happy=+1, Surprise=+0.3, Neutral=0,
                   Contempt=-0.3, Fear=-0.5, Sad=-0.8,
                   Disgust=-0.9, Anger=-1.0

3. CONTENT SENTIMENT (from Modality 3):
   Polarity: Positive (+1), Neutral (0), Negative (-1)
   Already mapped to valence.

═══════════════════════════════════════════════════════════════
COHERENCE COMPUTATION
═══════════════════════════════════════════════════════════════

```python
class EmotionCoherenceAnalyzer:
    """Measures alignment between face, voice, and content emotions."""

    VOICE_VALENCE = {
        "Enthusiastic": 1.0, "Neutral": 0.0, "Nervous": -0.5,
        "Angry": -0.8, "Sad": -1.0,
    }

    FACE_VALENCE = {
        "Happy": 1.0, "Surprise": 0.3, "Neutral": 0.0,
        "Contempt": -0.3, "Fear": -0.5, "Sad": -0.8,
        "Disgust": -0.9, "Anger": -1.0,
    }

    def compute_coherence_at_time(self, voice_emo, face_emo, content_sentiment):
        """Compute emotion coherence score for one second.

        Returns 0.0 (total mismatch) to 1.0 (perfect alignment).
        """
        v_voice = self.VOICE_VALENCE.get(voice_emo, 0.0)
        v_face = self.FACE_VALENCE.get(face_emo, 0.0)
        v_content = content_sentiment  # Already -1 to +1

        # Pairwise agreement scores
        voice_face = 1.0 - abs(v_voice - v_face) / 2.0
        voice_content = 1.0 - abs(v_voice - v_content) / 2.0
        face_content = 1.0 - abs(v_face - v_content) / 2.0

        # Weighted average (voice-face most important)
        coherence = 0.4 * voice_face + 0.3 * voice_content + 0.3 * face_content
        return coherence

    def detect_mismatches(self, timeline, threshold=0.4) -> list[dict]:
        """Find segments where emotions significantly disagree."""
        mismatches = []
        for entry in timeline:
            v = entry.get("voice", {})
            b = entry.get("body", {})
            c = entry.get("content", {})

            voice_emo = v.get("vocal_emotion", {}).get("label", "Neutral")
            face_emo = b.get("facial_emotion", {}).get("label", "Neutral")
            content_val = c.get("sentiment", {}).get("valence", 0.0)

            coherence = self.compute_coherence_at_time(
                voice_emo, face_emo, content_val
            )

            if coherence < threshold:
                mismatches.append({
                    "time_sec": entry["time_sec"],
                    "coherence": coherence,
                    "voice_emotion": voice_emo,
                    "face_emotion": face_emo,
                    "content_sentiment": "Positive" if content_val > 0.3 else
                                        "Negative" if content_val < -0.3 else "Neutral",
                    "description": self._describe_mismatch(
                        voice_emo, face_emo, content_val
                    ),
                })

        return self._cluster_mismatches(mismatches)

    def _describe_mismatch(self, voice, face, content_val):
        """Human-readable description of the mismatch."""
        descriptions = []
        if voice == "Enthusiastic" and face in ["Neutral", "Sad"]:
            descriptions.append("Voice sounds excited but face appears flat")
        if face == "Happy" and content_val < -0.3:
            descriptions.append("Smiling while delivering negative content")
        if voice == "Nervous" and face == "Happy":
            descriptions.append("Voice betrays nervousness despite smiling")
        if voice == "Neutral" and content_val > 0.5:
            descriptions.append("Monotone delivery of enthusiastic content")
        return "; ".join(descriptions) if descriptions else "Emotion channels disagree"

    def _cluster_mismatches(self, mismatches, gap_sec=5):
        """Merge consecutive mismatches within gap_sec into single events."""
        if not mismatches:
            return []
        clusters = []
        current = [mismatches[0]]
        for m in mismatches[1:]:
            if m["time_sec"] - current[-1]["time_sec"] <= gap_sec:
                current.append(m)
            else:
                clusters.append(self._summarize_cluster(current))
                current = [m]
        clusters.append(self._summarize_cluster(current))
        return clusters

    def analyze_full(self, timeline) -> dict:
        """Full emotion coherence analysis."""
        per_second = []
        for entry in timeline:
            v = entry.get("voice", {}).get("vocal_emotion", {}).get("label", "Neutral")
            f = entry.get("body", {}).get("facial_emotion", {}).get("label", "Neutral")
            c = entry.get("content", {}).get("sentiment", {}).get("valence", 0.0)
            per_second.append(self.compute_coherence_at_time(v, f, c))

        mismatches = self.detect_mismatches(timeline)

        return {
            "mean_coherence": np.mean(per_second),
            "coherence_timeline": per_second,
            "mismatch_events": mismatches,
            "total_mismatch_seconds": sum(
                m.get("duration_sec", 1) for m in mismatches
            ),
            "coherence_score_0_100": round(np.mean(per_second) * 100),
        }
```

NO USER INPUT REQUIRED. Pure computation.

COMPLETION: ✓ when coherence scores and mismatch events are generated correctly.