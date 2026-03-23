"""Phase 5.1 — 6-Dimension Scoring.

Computes 6 coaching scores (0-100) from fused multimodal data:
1. Vocal Clarity
2. Body Language
3. Content Structure
4. Audience Engagement
5. Emotional Expressiveness
6. Regime Adaptability
"""

import logging
import math
from collections import Counter
from statistics import mean

logger = logging.getLogger(__name__)


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _entropy(distribution: dict) -> float:
    """Shannon entropy of a probability distribution."""
    total = sum(distribution.values())
    if total == 0:
        return 0.0
    probs = [v / total for v in distribution.values() if v > 0]
    return -sum(p * math.log2(p) for p in probs)


class DimensionScorer:
    """Computes 6 coaching dimension scores from fused data."""

    def score_all(self, fusion_output: dict) -> dict:
        """Compute all 6 dimension scores.

        Args:
            fusion_output: Output from FusionEngine.fuse().

        Returns:
            Dict with 6 dimension scores (0-100) and overall.
        """
        timeline = fusion_output.get("timeline", [])
        transitions = fusion_output.get("transitions", {})
        recovery = fusion_output.get("recovery", {})
        coherence = fusion_output.get("emotion_coherence", {})

        scores = {
            "vocal_clarity": self._score_vocal_clarity(timeline),
            "body_language": self._score_body_language(timeline),
            "content_structure": self._score_content_structure(timeline, transitions),
            "audience_engagement": self._score_engagement(timeline),
            "emotional_expressiveness": self._score_expressiveness(timeline, coherence),
            "regime_adaptability": self._score_adaptability(transitions, recovery),
        }

        # Round all scores
        scores = {k: round(v, 1) for k, v in scores.items()}

        # Overall = weighted mean (equal weights by default)
        overall = mean(scores.values())
        scores["overall"] = round(overall, 1)

        logger.info("Dimension scores: %s", scores)
        return scores

    # ── Dimension 1: Vocal Clarity ─────────────────────────────────

    def _score_vocal_clarity(self, timeline: list[dict]) -> float:
        """Score vocal clarity based on rate, variety, fillers, disfluencies."""
        voice_entries = [e["voice"] for e in timeline if e.get("voice")]
        if not voice_entries:
            return 50.0

        # Speaking rate (optimal ~4.25 syl/sec)
        rates = [v.get("syllable_rate", 0.0) for v in voice_entries if v.get("syllable_rate", 0) > 0]
        avg_rate = mean(rates) if rates else 4.0
        rate_score = _clamp(100 - abs(avg_rate - 4.25) * 30)

        # Pitch variety (optimal f0_cv 0.15-0.25)
        cvs = [v.get("pitch_cv", 0.0) for v in voice_entries if v.get("pitch_cv", 0) > 0]
        avg_cv = mean(cvs) if cvs else 0.2
        if 0.15 <= avg_cv <= 0.25:
            variety_score = 100.0
        elif avg_cv < 0.15:
            variety_score = _clamp(100 - (0.15 - avg_cv) * 500)
        else:
            variety_score = _clamp(100 - (avg_cv - 0.25) * 300)

        # Filler penalty
        duration_min = len(timeline) / 60.0 if timeline else 1.0
        total_fillers = sum(v.get("filler_count", 0) for v in voice_entries)
        fillers_per_min = total_fillers / max(duration_min, 0.1)
        filler_penalty = min(fillers_per_min * 5, 40)

        # Disfluency penalty
        total_disfluencies = sum(v.get("disfluency_count", 0) for v in voice_entries)
        disfluency_penalty = min(total_disfluencies * 3, 20)

        vocal_clarity = (
            rate_score * 0.25
            + variety_score * 0.25
            + (100 - filler_penalty) * 0.25
            + (100 - disfluency_penalty) * 0.25
        )

        return _clamp(vocal_clarity)

    # ── Dimension 2: Body Language ─────────────────────────────────

    def _score_body_language(self, timeline: list[dict]) -> float:
        """Score body language from posture, gestures, and hand states."""
        body_entries = [e["body"] for e in timeline if e.get("body")]
        if not body_entries:
            return 50.0

        # Posture (already 0-100)
        posture_scores = [b.get("posture_score", 50.0) for b in body_entries]
        avg_posture = mean(posture_scores)

        # Gesture variety (entropy of gesture distribution)
        gesture_counts = Counter(b.get("gesture_dominant", "Rest") for b in body_entries)
        total = sum(gesture_counts.values())
        gesture_dist = {k: v / total for k, v in gesture_counts.items()}
        max_entropy = math.log2(max(len(gesture_counts), 1)) if len(gesture_counts) > 1 else 1.0
        actual_entropy = _entropy(gesture_dist)
        gesture_variety = _clamp((actual_entropy / max(max_entropy, 0.01)) * 100)

        # Rest penalty (too much "Rest" gesture)
        rest_pct = gesture_counts.get("Rest", 0) / max(total, 1)
        rest_penalty = _clamp(max(0, rest_pct - 0.4) * 100, 0, 20)

        body_score = (
            avg_posture * 0.4
            + gesture_variety * 0.3
            + (100 - rest_penalty) * 0.3
        )

        return _clamp(body_score)

    # ── Dimension 3: Content Structure ─────────────────────────────

    def _score_content_structure(
        self, timeline: list[dict], transitions: dict
    ) -> float:
        """Score content structure from grammar, readability, regime sequence."""
        content_entries = [e["content"] for e in timeline if e.get("content")]
        if not content_entries:
            return 50.0

        # Grammar (inverse of error count)
        total_errors = sum(c.get("grammar_error_count", 0) for c in content_entries)
        duration_min = len(timeline) / 60.0 if timeline else 1.0
        errors_per_min = total_errors / max(duration_min, 0.1)
        grammar_score = _clamp(100 - errors_per_min * 10)

        # Readability (optimal FKGL ~10)
        grades = [c.get("readability_grade", 10.0) for c in content_entries if c.get("readability_grade")]
        avg_grade = mean(grades) if grades else 10.0
        readability_score = _clamp(100 - abs(avg_grade - 10) * 8)

        # Structure: check for intro, conclusion, variety
        regimes = [c.get("regime_type", "Unknown") for c in content_entries]
        unique_regimes = set(regimes)
        structure_score = 0.0
        if "Introduction" in unique_regimes:
            structure_score += 15
        if "Conclusion" in unique_regimes or "Call to Action" in unique_regimes:
            structure_score += 15
        if "Call to Action" in unique_regimes:
            structure_score += 10
        structure_score += min(len(unique_regimes) * 5, 30)
        # Bonus for data or anecdotes
        if "Data Presentation" in unique_regimes or "Anecdote" in unique_regimes:
            structure_score += 10
        structure_score = _clamp(structure_score)

        # Transition quality
        transition_scores = transitions.get("transitions", [])
        if transition_scores:
            avg_transition = mean(t["score"] for t in transition_scores) * 100
        else:
            avg_transition = 50.0

        content_score = (
            grammar_score * 0.2
            + readability_score * 0.2
            + structure_score * 0.3
            + avg_transition * 0.3
        )

        return _clamp(content_score)

    # ── Dimension 4: Audience Engagement ───────────────────────────

    def _score_engagement(self, timeline: list[dict]) -> float:
        """Score audience engagement from gaze, pace, and content interaction."""
        body_entries = [e["body"] for e in timeline if e.get("body")]
        voice_entries = [e["voice"] for e in timeline if e.get("voice")]

        if not body_entries and not voice_entries:
            return 50.0

        # Gaze engagement (audience gaze ratio → 0-100)
        gaze_ratios = [b.get("gaze_engagement", 0.0) for b in body_entries]
        gaze_score = _clamp(mean(gaze_ratios) * 100 if gaze_ratios else 50.0)

        # Pace score (optimal WPM ~145, from syllable rate ~4.25 → ~145 WPM)
        if voice_entries:
            rates = [v.get("syllable_rate", 0.0) for v in voice_entries if v.get("syllable_rate", 0) > 0]
            avg_rate = mean(rates) if rates else 4.0
            wpm_approx = avg_rate * 60 / 1.75  # ~1.75 syllables per word
            pace_score = _clamp(100 - abs(wpm_approx - 145) * 0.5)
        else:
            pace_score = 50.0

        # Content interaction: count questions and direct address in transcripts
        question_count = 0
        direct_count = 0
        for v in voice_entries:
            text = v.get("transcript", "").lower()
            if "?" in text:
                question_count += 1
            for word in ["you", "we", "your", "our", "us"]:
                if word in text.split():
                    direct_count += 1
                    break

        interaction_score = _clamp(min(question_count * 10, 20))
        direct_score = _clamp(min(direct_count * 2, 15))

        engagement = (
            gaze_score * 0.4
            + interaction_score * 0.2
            + direct_score * 0.2
            + pace_score * 0.2
        )

        return _clamp(engagement)

    # ── Dimension 5: Emotional Expressiveness ──────────────────────

    def _score_expressiveness(
        self, timeline: list[dict], coherence: dict
    ) -> float:
        """Score emotional expressiveness from variety, coherence, and range."""
        voice_entries = [e["voice"] for e in timeline if e.get("voice")]
        body_entries = [e["body"] for e in timeline if e.get("body")]

        # Voice emotion variety
        voice_emotions = [v.get("emotion_label", "Neutral") for v in voice_entries]
        unique_voice = len(set(voice_emotions))
        voice_variety = _clamp((unique_voice / 5) * 100)  # 5 possible voice emotions

        # Face emotion variety
        face_emotions = [b.get("facial_emotion", "Neutral") for b in body_entries]
        unique_face = len(set(face_emotions))
        face_variety = _clamp((unique_face / 8) * 100)  # 8 possible face emotions

        # Coherence score (already 0-100)
        coherence_score = coherence.get("coherence_score_0_100", 50)

        # Not monotone penalty (% of time neutral)
        neutral_voice = sum(1 for e in voice_emotions if e in ("Neutral", "Unknown"))
        neutral_pct = neutral_voice / max(len(voice_emotions), 1)
        not_monotone = _clamp(100 - neutral_pct * 100)

        expressiveness = (
            voice_variety * 0.2
            + face_variety * 0.2
            + coherence_score * 0.35
            + not_monotone * 0.25
        )

        return _clamp(expressiveness)

    # ── Dimension 6: Regime Adaptability ───────────────────────────

    def _score_adaptability(self, transitions: dict, recovery: dict) -> float:
        """Score regime adaptability from transitions, composure, and resilience."""
        # Transition quality
        transition_list = transitions.get("transitions", [])
        if transition_list:
            transition_score = mean(t["score"] for t in transition_list) * 100
        else:
            transition_score = 50.0

        # Composure from recovery
        composure_score = recovery.get("mean_composure", 1.0) * 100

        # Resilience: fewer disruptions = better
        disruptions_per_min = recovery.get("disruptions_per_minute", 0)
        resilience = _clamp(100 - disruptions_per_min * 20)

        adaptability = (
            transition_score * 0.4
            + composure_score * 0.35
            + resilience * 0.25
        )

        return _clamp(adaptability)
