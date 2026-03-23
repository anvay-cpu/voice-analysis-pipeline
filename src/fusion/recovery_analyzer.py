"""Phase 4.3 — Recovery Analysis.

Detects disruptions (filler bursts, disfluencies, long pauses, pace spikes,
posture collapses) and measures how quickly the speaker recovers across
all modalities.
"""

import logging
from statistics import mean

logger = logging.getLogger(__name__)


class RecoveryAnalyzer:
    """Detect disruptions and measure recovery speed."""

    def __init__(
        self,
        filler_burst_threshold: int = 3,
        filler_burst_window: int = 10,
        pause_threshold_sec: int = 3,
        pace_spike_pct: float = 0.40,
        posture_drop_threshold: float = 15.0,
    ):
        self.filler_burst_threshold = filler_burst_threshold
        self.filler_burst_window = filler_burst_window
        self.pause_threshold = pause_threshold_sec
        self.pace_spike_pct = pace_spike_pct
        self.posture_drop = posture_drop_threshold

    def detect_disruptions(self, timeline: list[dict]) -> list[dict]:
        """Find all disruption events in the timeline.

        Returns sorted, de-overlapped list of disruption dicts.
        """
        disruptions = []
        disruptions += self._detect_filler_bursts(timeline)
        disruptions += self._detect_disfluencies(timeline)
        disruptions += self._detect_long_pauses(timeline)
        disruptions += self._detect_pace_spikes(timeline)
        disruptions += self._detect_posture_collapses(timeline)

        disruptions.sort(key=lambda d: d["time_sec"])
        return self._merge_overlapping(disruptions)

    def measure_recovery(self, timeline: list[dict], disruption: dict) -> dict:
        """Measure how quickly the speaker recovered from a disruption.

        Args:
            timeline: Aligned 1-second timeline.
            disruption: Disruption dict with time_sec, type, duration_sec.

        Returns:
            Recovery metrics dict.
        """
        t = disruption["time_sec"]
        pre = timeline[max(0, t - 10):t]
        post = timeline[t:min(len(timeline), t + 15)]

        vocal_recovery = self._measure_vocal_recovery(pre, post)
        physical_recovery = self._measure_physical_recovery(pre, post)
        content_maintained = self._measure_content_recovery(pre, post)
        composure = self._compute_composure_score(
            vocal_recovery, physical_recovery, content_maintained
        )

        return {
            "disruption": disruption,
            "vocal_recovery_sec": vocal_recovery,
            "physical_recovery_sec": physical_recovery,
            "content_maintained": content_maintained,
            "composure_score": composure,
        }

    def analyze_all(self, timeline: list[dict]) -> dict:
        """Run full disruption detection and recovery analysis.

        Returns:
            Summary dict with disruption counts, recovery metrics, and composure.
        """
        disruptions = self.detect_disruptions(timeline)
        recoveries = [self.measure_recovery(timeline, d) for d in disruptions]

        duration_min = len(timeline) / 60.0 if timeline else 1.0

        return {
            "total_disruptions": len(disruptions),
            "disruptions_per_minute": round(len(disruptions) / duration_min, 2),
            "recoveries": recoveries,
            "mean_composure": round(
                mean(r["composure_score"] for r in recoveries), 3
            ) if recoveries else 1.0,
            "fastest_recovery": min(
                recoveries, key=lambda r: r["vocal_recovery_sec"]
            ) if recoveries else None,
            "slowest_recovery": max(
                recoveries, key=lambda r: r["vocal_recovery_sec"]
            ) if recoveries else None,
        }

    # ── Disruption detectors ───────────────────────────────────────

    def _detect_filler_bursts(self, timeline: list[dict]) -> list[dict]:
        """Detect 3+ filler words within a sliding window."""
        disruptions = []
        n = len(timeline)

        for i in range(n):
            window_end = min(i + self.filler_burst_window, n)
            total_fillers = sum(
                self._get_voice(timeline[j]).get("filler_count", 0)
                for j in range(i, window_end)
            )
            if total_fillers >= self.filler_burst_threshold:
                disruptions.append({
                    "time_sec": i,
                    "type": "filler_burst",
                    "severity": min(total_fillers / self.filler_burst_threshold, 3.0),
                    "duration_sec": self.filler_burst_window,
                    "detail": f"{total_fillers} fillers in {window_end - i}s",
                })
                # Skip ahead to avoid duplicate detections
                break_at = window_end
                while i < break_at and i < n - 1:
                    i += 1

        return disruptions

    def _detect_disfluencies(self, timeline: list[dict]) -> list[dict]:
        """Detect disfluency events (stuttering, repetition, etc.)."""
        disruptions = []
        for i, entry in enumerate(timeline):
            voice = self._get_voice(entry)
            count = voice.get("disfluency_count", 0)
            if count > 0:
                disruptions.append({
                    "time_sec": i,
                    "type": "disfluency",
                    "severity": min(count, 3.0),
                    "duration_sec": 1,
                    "detail": f"{count} disfluencies",
                })
        return disruptions

    def _detect_long_pauses(self, timeline: list[dict]) -> list[dict]:
        """Detect silence > threshold seconds (not at regime boundaries)."""
        disruptions = []
        consecutive_silence = 0
        silence_start = 0

        for i, entry in enumerate(timeline):
            voice = self._get_voice(entry)
            is_speech = voice.get("is_speech", True)

            if not is_speech:
                if consecutive_silence == 0:
                    silence_start = i
                consecutive_silence += 1
            else:
                if consecutive_silence >= self.pause_threshold:
                    # Check if this is at a regime boundary (which is expected)
                    if not self._is_regime_boundary_pause(timeline, silence_start, i):
                        disruptions.append({
                            "time_sec": silence_start,
                            "type": "long_pause",
                            "severity": min(consecutive_silence / self.pause_threshold, 3.0),
                            "duration_sec": consecutive_silence,
                            "detail": f"{consecutive_silence}s silence",
                        })
                consecutive_silence = 0

        # Handle trailing silence
        if consecutive_silence >= self.pause_threshold:
            if not self._is_regime_boundary_pause(timeline, silence_start, len(timeline)):
                disruptions.append({
                    "time_sec": silence_start,
                    "type": "long_pause",
                    "severity": min(consecutive_silence / self.pause_threshold, 3.0),
                    "duration_sec": consecutive_silence,
                    "detail": f"{consecutive_silence}s silence",
                })

        return disruptions

    def _detect_pace_spikes(self, timeline: list[dict]) -> list[dict]:
        """Detect speaking rate jumps > threshold above/below running average."""
        disruptions = []
        rates = []

        for i, entry in enumerate(timeline):
            voice = self._get_voice(entry)
            rate = voice.get("syllable_rate", 0.0)

            if not voice.get("is_speech", True):
                continue

            if rate <= 0:
                continue

            # Build running average from previous entries
            if len(rates) >= 5:
                avg_rate = mean(rates[-10:])  # last 10 entries
                if avg_rate > 0:
                    deviation = abs(rate - avg_rate) / avg_rate
                    if deviation > self.pace_spike_pct:
                        direction = "speed_up" if rate > avg_rate else "slow_down"
                        disruptions.append({
                            "time_sec": i,
                            "type": "pace_spike",
                            "severity": min(deviation / self.pace_spike_pct, 3.0),
                            "duration_sec": 1,
                            "detail": f"{direction}: {rate:.1f} vs avg {avg_rate:.1f}",
                        })

            rates.append(rate)

        return disruptions

    def _detect_posture_collapses(self, timeline: list[dict]) -> list[dict]:
        """Detect posture score drops > threshold in 3 seconds."""
        disruptions = []
        n = len(timeline)

        for i in range(3, n):
            body_now = self._get_body(timeline[i])
            body_prev = self._get_body(timeline[i - 3])

            score_now = body_now.get("posture_score", 50.0)
            score_prev = body_prev.get("posture_score", 50.0)

            drop = score_prev - score_now
            if drop >= self.posture_drop:
                disruptions.append({
                    "time_sec": i,
                    "type": "posture_collapse",
                    "severity": min(drop / self.posture_drop, 3.0),
                    "duration_sec": 3,
                    "detail": f"posture dropped {drop:.1f} points",
                })

        return disruptions

    # ── Recovery measurement ───────────────────────────────────────

    def _measure_vocal_recovery(
        self, pre: list[dict], post: list[dict]
    ) -> float:
        """Seconds until voice metrics return to pre-disruption levels."""
        if not pre or not post:
            return 15.0  # max recovery time

        # Pre-disruption baselines
        pre_rates = [self._get_voice(e).get("syllable_rate", 0.0) for e in pre]
        pre_f0 = [self._get_voice(e).get("f0_mean", 0.0) for e in pre]
        pre_cv = [self._get_voice(e).get("pitch_cv", 0.0) for e in pre]

        avg_rate = mean(pre_rates) if pre_rates else 0.0
        avg_f0 = mean(pre_f0) if pre_f0 else 0.0
        avg_cv = mean(pre_cv) if pre_cv else 0.0

        # Check each post-disruption second
        for i, entry in enumerate(post):
            voice = self._get_voice(entry)
            rate = voice.get("syllable_rate", 0.0)
            f0 = voice.get("f0_mean", 0.0)
            cv = voice.get("pitch_cv", 0.0)

            rate_ok = self._within_pct(rate, avg_rate, 0.10)
            f0_ok = self._within_pct(f0, avg_f0, 0.10)
            cv_ok = self._within_pct(cv, avg_cv, 0.10)

            if rate_ok and f0_ok and cv_ok:
                return float(i)

        return float(len(post))

    def _measure_physical_recovery(
        self, pre: list[dict], post: list[dict]
    ) -> float:
        """Seconds until posture returns to pre-disruption level."""
        if not pre or not post:
            return 15.0

        pre_posture = [self._get_body(e).get("posture_score", 50.0) for e in pre]
        avg_posture = mean(pre_posture) if pre_posture else 50.0

        for i, entry in enumerate(post):
            body = self._get_body(entry)
            score = body.get("posture_score", 50.0)
            if abs(score - avg_posture) <= 5.0:
                return float(i)

        return float(len(post))

    def _measure_content_recovery(
        self, pre: list[dict], post: list[dict]
    ) -> bool:
        """Check if the speaker maintained their topic after disruption."""
        if not pre or not post:
            return True

        pre_regimes = [
            self._get_content(e).get("regime_type", "Unknown")
            for e in pre if self._get_content(e)
        ]
        post_regimes = [
            self._get_content(e).get("regime_type", "Unknown")
            for e in post[:10] if self._get_content(e)
        ]

        if not pre_regimes or not post_regimes:
            return True

        # Check if regime changed within 10 seconds after disruption
        pre_regime = pre_regimes[-1]
        for regime in post_regimes:
            if regime != pre_regime:
                return False  # Topic abandoned

        return True

    def _compute_composure_score(
        self,
        vocal_recovery: float,
        physical_recovery: float,
        content_maintained: bool,
    ) -> float:
        """Compute composure score 0.0-1.0 based on recovery metrics."""
        max_recovery = max(vocal_recovery, physical_recovery)

        if max_recovery <= 3.0 and content_maintained:
            return 1.0
        elif max_recovery <= 5.0 and content_maintained:
            return 0.8
        elif max_recovery <= 10.0 and content_maintained:
            return 0.6
        elif max_recovery <= 10.0 and not content_maintained:
            return 0.4
        elif max_recovery > 10.0 and content_maintained:
            return 0.3
        else:
            return 0.1

    # ── Helpers ────────────────────────────────────────────────────

    def _merge_overlapping(self, disruptions: list[dict]) -> list[dict]:
        """Merge disruptions that overlap in time (within 3 seconds)."""
        if not disruptions:
            return []

        merged = [disruptions[0]]
        for d in disruptions[1:]:
            last = merged[-1]
            last_end = last["time_sec"] + last.get("duration_sec", 1)
            if d["time_sec"] <= last_end + 3:
                # Merge: extend duration, take higher severity
                new_end = max(last_end, d["time_sec"] + d.get("duration_sec", 1))
                last["duration_sec"] = new_end - last["time_sec"]
                last["severity"] = max(last["severity"], d["severity"])
                if d["type"] != last["type"]:
                    last["type"] = f"{last['type']}+{d['type']}"
                    last["detail"] = f"{last['detail']}; {d['detail']}"
            else:
                merged.append(d)

        return merged

    def _is_regime_boundary_pause(
        self, timeline: list[dict], start: int, end: int
    ) -> bool:
        """Check if a pause coincides with a regime boundary (expected pause)."""
        for i in range(max(1, start), min(len(timeline), end + 1)):
            prev = self._get_content(timeline[i - 1])
            curr = self._get_content(timeline[i])
            if prev and curr:
                if prev.get("regime_type") != curr.get("regime_type"):
                    return True
        return False

    @staticmethod
    def _within_pct(value: float, baseline: float, pct: float) -> bool:
        """Check if value is within pct% of baseline."""
        if abs(baseline) < 1e-6:
            return abs(value) < 1e-6
        return abs(value - baseline) / abs(baseline) <= pct

    @staticmethod
    def _get_voice(entry: dict) -> dict:
        return entry.get("voice") or {}

    @staticmethod
    def _get_body(entry: dict) -> dict:
        return entry.get("body") or {}

    @staticmethod
    def _get_content(entry: dict) -> dict:
        return entry.get("content") or {}
