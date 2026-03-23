PHASE 4.2 — REGIME TRANSITION SCORING
========================================

AGENT ROLE: Transition Quality Analyst
DEPENDS ON: Phase 4.1 (aligned timeline + regime boundaries)
DELIVERS TO: Phase 5.1 (scoring)
RUNS ON: Local M1 (pure computation + optional Claude API)
ESTIMATED TIME: 25 minutes (agent), 0 minutes (user)

OBJECTIVE:
Build src/fusion/regime_transition_scorer.py that evaluates HOW WELL
the speaker adapted their delivery at each regime transition.

═══════════════════════════════════════════════════════════════
THE CONCEPT
═══════════════════════════════════════════════════════════════

When a speech moves from "Introduction" to "Main Argument", a skilled
speaker changes their delivery: voice becomes more assertive, gestures
shift from open-welcome to purposeful-pointing, pace may increase.
A poor speaker makes NO changes — same flat delivery across transitions.

For each regime boundary, we measure change in 6 channels and score
whether the direction of change was APPROPRIATE for that transition type.

═══════════════════════════════════════════════════════════════
THE 6 CHANNELS TO MEASURE
═══════════════════════════════════════════════════════════════

At each regime boundary (e.g., second 62), compare the 10 seconds
BEFORE the boundary with the 10 seconds AFTER:

1. VOCAL ENERGY: mean(f0_mean + rms_db) before vs after
   Δ_energy = energy_after - energy_before

2. SPEAKING RATE: mean(syl_per_sec) before vs after
   Δ_rate = rate_after - rate_before

3. PITCH VARIETY: mean(f0_cv) before vs after
   Δ_variety = variety_after - variety_before

4. GESTURE INTENSITY: fraction of "Active Gesture" frames before vs after
   Δ_gesture = gesture_pct_after - gesture_pct_before

5. AUDIENCE ENGAGEMENT: mean(audience_gaze_ratio) before vs after
   Δ_gaze = gaze_after - gaze_before

6. POSTURE SHIFT: mean(posture_score) before vs after
   Δ_posture = posture_after - posture_before

═══════════════════════════════════════════════════════════════
EXPECTED DIRECTION MATRIX
═══════════════════════════════════════════════════════════════

Different transitions SHOULD produce different changes.
This matrix encodes expert knowledge:

```python
EXPECTED_DIRECTION = {
    # (from_regime, to_regime): {channel: expected_sign}
    # +1 = should increase, -1 = should decrease, 0 = neutral

    ("Introduction", "Main Argument"): {
        "energy": +1, "rate": +1, "variety": 0,
        "gesture": +1, "gaze": 0, "posture": 0,
    },
    ("Main Argument", "Anecdote"): {
        "energy": 0, "rate": -1, "variety": +1,
        "gesture": -1, "gaze": +1, "posture": 0,
    },
    ("Anecdote", "Data Presentation"): {
        "energy": -1, "rate": 0, "variety": -1,
        "gesture": -1, "gaze": -1, "posture": 0,
    },
    ("Data Presentation", "Main Argument"): {
        "energy": +1, "rate": 0, "variety": +1,
        "gesture": +1, "gaze": +1, "posture": 0,
    },
    ("Main Argument", "Call to Action"): {
        "energy": +1, "rate": -1, "variety": +1,
        "gesture": +1, "gaze": +1, "posture": +1,
    },
    ("*", "Conclusion"): {
        "energy": +1, "rate": -1, "variety": +1,
        "gesture": +1, "gaze": +1, "posture": +1,
    },
}

# Default for unknown transitions: expect no specific direction
DEFAULT_EXPECTED = {
    "energy": 0, "rate": 0, "variety": 0,
    "gesture": 0, "gaze": 0, "posture": 0,
}
```

═══════════════════════════════════════════════════════════════
SCORING FORMULA
═══════════════════════════════════════════════════════════════

For each transition:
1. Compute the 6 deltas (before vs after)
2. Look up expected directions for this transition type
3. For each channel where expected != 0:
   - If actual delta matches expected sign → +1 point
   - If actual delta is opposite → -1 point
   - If actual delta is near zero (< 5% change) → 0 points
4. Transition score = (total_points + max_possible) / (2 * max_possible)
   Normalized to 0.0 - 1.0 range.

Overall Regime Adaptability Score (for the 6-dimension scoring):
  R_adapt = mean(all transition scores) * 100

═══════════════════════════════════════════════════════════════
IMPLEMENTATION
═══════════════════════════════════════════════════════════════

```python
class RegimeTransitionScorer:
    def __init__(self, window_sec=10):
        self.window = window_sec  # Seconds before/after boundary

    def score_transition(self, timeline, boundary) -> dict:
        t = boundary["time_sec"]
        before = timeline[max(0, t - self.window):t]
        after = timeline[t:t + self.window]

        deltas = self._compute_deltas(before, after)
        expected = self._get_expected(boundary["from_regime"],
                                       boundary["to_regime"])
        score = self._score_alignment(deltas, expected)

        return {
            "time_sec": t,
            "from_regime": boundary["from_regime"],
            "to_regime": boundary["to_regime"],
            "deltas": deltas,
            "expected": expected,
            "score": score,  # 0.0-1.0
            "channels_aligned": self._which_aligned(deltas, expected),
            "channels_misaligned": self._which_misaligned(deltas, expected),
        }

    def score_all_transitions(self, timeline, boundaries) -> dict:
        scores = [self.score_transition(timeline, b) for b in boundaries]
        return {
            "transitions": scores,
            "mean_score": np.mean([s["score"] for s in scores]),
            "best_transition": max(scores, key=lambda s: s["score"]),
            "worst_transition": min(scores, key=lambda s: s["score"]),
        }
```

NO USER INPUT REQUIRED. NO TRAINING. NO GPU.

COMPLETION: ✓ when transitions are scored correctly on mock timeline data.


═══════════════════════════════════════════════════════════════════
═══════════════════════════════════════════════════════════════════