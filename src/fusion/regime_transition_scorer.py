"""Phase 4.2 — Regime Transition Scoring.

Evaluates how well the speaker adapted their delivery at each regime
transition by measuring 6 channels and comparing against expected
direction-of-change for each transition type.
"""

import logging
from statistics import mean

logger = logging.getLogger(__name__)

# Expected direction of change for known transitions.
# +1 = should increase, -1 = should decrease, 0 = neutral (don't score)
EXPECTED_DIRECTION = {
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
    ("Introduction", "Data Presentation"): {
        "energy": 0, "rate": 0, "variety": -1,
        "gesture": -1, "gaze": 0, "posture": 0,
    },
    ("Introduction", "Anecdote"): {
        "energy": 0, "rate": -1, "variety": +1,
        "gesture": 0, "gaze": +1, "posture": 0,
    },
    ("Data Presentation", "Conclusion"): {
        "energy": +1, "rate": -1, "variety": +1,
        "gesture": +1, "gaze": +1, "posture": +1,
    },
    ("Anecdote", "Main Argument"): {
        "energy": +1, "rate": +1, "variety": -1,
        "gesture": +1, "gaze": 0, "posture": 0,
    },
    ("Anecdote", "Conclusion"): {
        "energy": +1, "rate": -1, "variety": +1,
        "gesture": +1, "gaze": +1, "posture": +1,
    },
    ("Main Argument", "Conclusion"): {
        "energy": +1, "rate": -1, "variety": +1,
        "gesture": +1, "gaze": +1, "posture": +1,
    },
}

# Default: no specific expectations for unknown transitions
DEFAULT_EXPECTED = {
    "energy": 0, "rate": 0, "variety": 0,
    "gesture": 0, "gaze": 0, "posture": 0,
}

# Threshold below which a delta is considered "near zero"
NEAR_ZERO_THRESHOLD = 0.05


class RegimeTransitionScorer:
    """Score how well the speaker adapts delivery across regime transitions."""

    def __init__(self, window_sec: int = 10):
        """
        Args:
            window_sec: Number of seconds before/after boundary to compare.
        """
        self.window = window_sec

    def score_transition(self, timeline: list[dict], boundary: dict) -> dict:
        """Score a single regime transition.

        Args:
            timeline: Aligned 1-second timeline from TimelineAligner.
            boundary: Dict with time_sec, from_regime, to_regime.

        Returns:
            Dict with deltas, expected directions, score (0-1), and alignment info.
        """
        t = boundary["time_sec"]
        before = timeline[max(0, t - self.window):t]
        after = timeline[t:min(len(timeline), t + self.window)]

        if not before or not after:
            return {
                "time_sec": t,
                "from_regime": boundary["from_regime"],
                "to_regime": boundary["to_regime"],
                "deltas": {},
                "expected": DEFAULT_EXPECTED,
                "score": 0.5,
                "channels_aligned": [],
                "channels_misaligned": [],
            }

        deltas = self._compute_deltas(before, after)
        expected = self._get_expected(boundary["from_regime"], boundary["to_regime"])
        score = self._score_alignment(deltas, expected)

        return {
            "time_sec": t,
            "from_regime": boundary["from_regime"],
            "to_regime": boundary["to_regime"],
            "deltas": deltas,
            "expected": expected,
            "score": score,
            "channels_aligned": self._which_aligned(deltas, expected),
            "channels_misaligned": self._which_misaligned(deltas, expected),
        }

    def score_all_transitions(
        self, timeline: list[dict], boundaries: list[dict]
    ) -> dict:
        """Score all transitions in the speech.

        Args:
            timeline: Aligned 1-second timeline.
            boundaries: List of boundary dicts from TimelineAligner.find_regime_boundaries.

        Returns:
            Dict with per-transition scores, mean score, best/worst transitions.
        """
        if not boundaries:
            return {
                "transitions": [],
                "mean_score": 0.5,
                "adaptability_score": 50.0,
                "best_transition": None,
                "worst_transition": None,
            }

        scores = [self.score_transition(timeline, b) for b in boundaries]
        mean_score = mean(s["score"] for s in scores)

        return {
            "transitions": scores,
            "mean_score": round(mean_score, 4),
            "adaptability_score": round(mean_score * 100, 1),
            "best_transition": max(scores, key=lambda s: s["score"]),
            "worst_transition": min(scores, key=lambda s: s["score"]),
        }

    # ── Internal methods ───────────────────────────────────────────

    def _compute_deltas(
        self, before: list[dict], after: list[dict]
    ) -> dict[str, float]:
        """Compute change in each channel between before and after windows."""
        before_vals = self._aggregate_channels(before)
        after_vals = self._aggregate_channels(after)

        deltas = {}
        for channel in before_vals:
            b = before_vals[channel]
            a = after_vals.get(channel, b)
            # Normalize delta relative to the before value to get fractional change
            if abs(b) > 1e-6:
                deltas[channel] = (a - b) / abs(b)
            else:
                deltas[channel] = a - b

        return deltas

    def _aggregate_channels(self, window: list[dict]) -> dict[str, float]:
        """Extract mean values for 6 channels from a window of timeline entries."""
        energies = []
        rates = []
        varieties = []
        gesture_active = []
        gaze_engagement = []
        posture_scores = []

        for entry in window:
            voice = entry.get("voice")
            body = entry.get("body")

            if voice:
                # Energy: combine f0_mean and rms_mean_db (normalized)
                f0 = voice.get("f0_mean", 0.0)
                rms = voice.get("rms_mean_db", -30.0)
                # Normalize: f0 in ~80-300 Hz, rms in ~-40 to -10 dB
                energy = (f0 / 200.0) + ((rms + 40) / 30.0)
                energies.append(energy)

                rates.append(voice.get("syllable_rate", 0.0))
                varieties.append(voice.get("pitch_cv", 0.0))

            if body:
                # Gesture intensity: 1 if not "Rest", 0 if "Rest"
                gesture = body.get("gesture_dominant", "Rest")
                gesture_active.append(0.0 if gesture == "Rest" else 1.0)

                gaze_engagement.append(body.get("gaze_engagement", 0.0))
                posture_scores.append(body.get("posture_score", 50.0))

        return {
            "energy": mean(energies) if energies else 0.0,
            "rate": mean(rates) if rates else 0.0,
            "variety": mean(varieties) if varieties else 0.0,
            "gesture": mean(gesture_active) if gesture_active else 0.0,
            "gaze": mean(gaze_engagement) if gaze_engagement else 0.0,
            "posture": mean(posture_scores) if posture_scores else 50.0,
        }

    def _get_expected(self, from_regime: str, to_regime: str) -> dict[str, int]:
        """Look up expected direction for this transition type."""
        # Exact match
        key = (from_regime, to_regime)
        if key in EXPECTED_DIRECTION:
            return EXPECTED_DIRECTION[key]

        # Wildcard match: ("*", to_regime)
        wildcard_key = ("*", to_regime)
        if wildcard_key in EXPECTED_DIRECTION:
            return EXPECTED_DIRECTION[wildcard_key]

        return DEFAULT_EXPECTED.copy()

    def _score_alignment(
        self, deltas: dict[str, float], expected: dict[str, int]
    ) -> float:
        """Score how well actual deltas match expected directions.

        Returns 0.0 - 1.0 score.
        """
        points = 0
        max_possible = 0

        for channel, exp_sign in expected.items():
            if exp_sign == 0:
                continue  # No expectation → skip

            max_possible += 1
            actual_delta = deltas.get(channel, 0.0)

            if abs(actual_delta) < NEAR_ZERO_THRESHOLD:
                points += 0  # Near zero: neutral
            elif (actual_delta > 0 and exp_sign > 0) or (actual_delta < 0 and exp_sign < 0):
                points += 1  # Correct direction
            else:
                points -= 1  # Wrong direction

        if max_possible == 0:
            return 0.5  # No expectations → neutral score

        # Normalize to 0.0-1.0: (points + max) / (2 * max)
        return (points + max_possible) / (2 * max_possible)

    def _which_aligned(
        self, deltas: dict[str, float], expected: dict[str, int]
    ) -> list[str]:
        """Return list of channels that changed in the expected direction."""
        aligned = []
        for channel, exp_sign in expected.items():
            if exp_sign == 0:
                continue
            actual = deltas.get(channel, 0.0)
            if abs(actual) >= NEAR_ZERO_THRESHOLD:
                if (actual > 0 and exp_sign > 0) or (actual < 0 and exp_sign < 0):
                    aligned.append(channel)
        return aligned

    def _which_misaligned(
        self, deltas: dict[str, float], expected: dict[str, int]
    ) -> list[str]:
        """Return list of channels that changed opposite to expected direction."""
        misaligned = []
        for channel, exp_sign in expected.items():
            if exp_sign == 0:
                continue
            actual = deltas.get(channel, 0.0)
            if abs(actual) >= NEAR_ZERO_THRESHOLD:
                if (actual > 0 and exp_sign < 0) or (actual < 0 and exp_sign > 0):
                    misaligned.append(channel)
        return misaligned
