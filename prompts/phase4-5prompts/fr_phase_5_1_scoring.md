PHASE 5.1 — 6-DIMENSION SCORING
==================================

AGENT ROLE: Assessment Designer
DEPENDS ON: Phase 4.5 (fusion output)
DELIVERS TO: Phase 5.2 (coaching writer), Phase 5.3 (charts)
RUNS ON: Local M1
ESTIMATED TIME: 20 minutes (agent), 0 minutes (user)

OBJECTIVE:
Build src/scoring/dimension_scorer.py that computes 6 scores (0-100)
from the fused multimodal data. These 6 scores form the radar chart
in the final report.

═══════════════════════════════════════════════════════════════
THE 6 DIMENSIONS
═══════════════════════════════════════════════════════════════

```python
class DimensionScorer:
    """Computes 6 coaching dimension scores from fused data."""

    def score_all(self, fusion_output: dict) -> dict:
        timeline = fusion_output["timeline"]
        transitions = fusion_output["transitions"]
        recovery = fusion_output["recovery"]
        coherence = fusion_output["emotion_coherence"]

        return {
            "vocal_clarity": self._score_vocal_clarity(timeline),
            "body_language": self._score_body_language(timeline),
            "content_structure": self._score_content_structure(timeline, transitions),
            "audience_engagement": self._score_engagement(timeline),
            "emotional_expressiveness": self._score_expressiveness(timeline, coherence),
            "regime_adaptability": self._score_adaptability(transitions, recovery),
        }
```

DIMENSION 1: VOCAL CLARITY (0-100)
  Inputs: speaking_rate, f0_cv, filler_rate, disfluency_count, jitter
  Formula:
    rate_score = 100 - abs(rate - 4.25) * 30  # Optimal = 4.25 syl/sec
    variety_score = 100 if 0.15 < f0_cv < 0.25 else decay
    filler_penalty = min(fillers_per_min * 5, 40)
    disfluency_penalty = min(disfluencies * 3, 20)
    tension_penalty = max(0, (jitter - 1.0) * 10)
    VOCAL_CLARITY = max(0, rate_score * 0.25 + variety_score * 0.25
                        + (100 - filler_penalty) * 0.25
                        + (100 - disfluency_penalty - tension_penalty) * 0.25)

DIMENSION 2: BODY LANGUAGE (0-100)
  Inputs: posture_score, gesture_distribution, hand_states
  Formula:
    posture = mean(posture_scores)  # Already 0-100
    gesture_variety = entropy(gesture_distribution) / max_entropy * 100
    adaptor_penalty = min(adaptor_pct * 200, 30)
    rest_penalty = min(max(0, rest_pct - 0.4) * 100, 20)
    BODY_LANGUAGE = posture * 0.4 + gesture_variety * 0.3
                    + (100 - adaptor_penalty - rest_penalty) * 0.3

DIMENSION 3: CONTENT STRUCTURE (0-100)
  Inputs: grammar_score, readability, regime_sequence, transition_scores
  Formula:
    grammar = mean(grammar_scores)  # Already 0-100
    readability = 100 - abs(fkgl - 10) * 8  # Optimal FKGL = 10
    structure = has_intro * 15 + has_conclusion * 15
                + has_call_to_action * 10 + regime_count * 5
    transition_quality = mean(transition_scores) * 100
    CONTENT_STRUCTURE = grammar * 0.2 + readability * 0.2
                        + structure * 0.3 + transition_quality * 0.3

DIMENSION 4: AUDIENCE ENGAGEMENT (0-100)
  Inputs: gaze_at_audience, question_count, direct_address, wpm
  Formula:
    gaze_score = audience_gaze_ratio * 100  # 0-100
    interaction = min(question_count * 10, 20)  # Up to 20 for questions
    direct = min(direct_address_count * 5, 15)  # "you", "we"
    pace_score = 100 - abs(wpm - 145) * 0.5
    AUDIENCE_ENGAGEMENT = gaze_score * 0.4 + interaction * 0.2
                          + direct * 0.2 + pace_score * 0.2

DIMENSION 5: EMOTIONAL EXPRESSIVENESS (0-100)
  Inputs: emotion_variety, coherence_score, face_emotion_range
  Formula:
    voice_variety = count_unique_voice_emotions / 5 * 100
    face_variety = count_unique_face_emotions / 8 * 100
    coherence = coherence_score  # Already 0-100
    not_monotone = 100 - neutral_pct * 100  # Penalty for always neutral
    EMOTIONAL_EXPRESSIVENESS = voice_variety * 0.2 + face_variety * 0.2
                               + coherence * 0.35 + not_monotone * 0.25

DIMENSION 6: REGIME ADAPTABILITY (0-100)
  Inputs: transition_scores, recovery_composure, disruption_rate
  Formula:
    transition = mean(transition_scores) * 100
    composure = mean_composure * 100  # From recovery analysis
    resilience = max(0, 100 - disruptions_per_min * 20)
    REGIME_ADAPTABILITY = transition * 0.4 + composure * 0.35
                          + resilience * 0.25

OVERALL SCORE:
  overall = weighted_mean(all 6 dimensions)
  Default weights: equal (1/6 each)
  Can be customized per coaching focus

NO USER INPUT REQUIRED. Pure mathematical scoring.

COMPLETION: ✓ when all 6 scores compute correctly on fusion output.


═══════════════════════════════════════════════════════════════════
═══════════════════════════════════════════════════════════════════