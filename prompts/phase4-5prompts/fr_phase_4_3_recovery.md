PHASE 4.3— RECOVERY ANALYSIS
================================

AGENT ROLE: Disruption Recovery Specialist
DEPENDS ON: Phase 4.1 (aligned timeline)
DELIVERS TO: Phase 5.1 (scoring)
RUNS ON: Local M1
ESTIMATED TIME: 20 minutes (agent), 0 minutes (user)

OBJECTIVE:
Build src/fusion/recovery_analyzer.py that detects moments of disruption
(filler bursts, disfluencies, lost train of thought) and measures how
quickly the speaker recovered across all modalities.

═══════════════════════════════════════════════════════════════
DISRUPTION TYPES
═══════════════════════════════════════════════════════════════

1. FILLER BURST: 3+ filler words within 10 seconds
   Source: voice.filler_words from timeline

2. DISFLUENCY EVENT: any stuttering event (repetition, prolongation, block)
   Source: voice.disfluencies from timeline

3. LONG PAUSE: silence > 3 seconds (not at a regime boundary)
   Source: voice.is_speech == False for 3+ consecutive seconds

4. PACE SPIKE: speaking rate jumps > 40% above or below the
   speaker's running average (panic speed-up or freeze slow-down)
   Source: voice.speaking_rate_syl_per_sec from timeline

5. POSTURE COLLAPSE: posture score drops > 15 points in 3 seconds
   Source: body.posture_score from timeline

═══════════════════════════════════════════════════════════════
RECOVERY METRICS (measured for each disruption)
═══════════════════════════════════════════════════════════════

After detecting a disruption at time T:

1. VOCAL RECOVERY TIME: seconds until speaking_rate, f0_mean, and
   f0_cv return to within 10% of their pre-disruption running average.

2. PHYSICAL RECOVERY TIME: seconds until posture_score returns to
   within 5 points of pre-disruption level AND adaptor gestures
   (face-touching) stop.

3. CONTENT RECOVERY: did the speaker successfully continue their
   argument, or did they abandon the point and move to something new?
   Check if regime changed within 10 seconds after disruption.

4. COMPOSURE SCORE (0-1):
   - Voice steady within 3s AND posture recovered AND no regime jump → 1.0
   - Recovery took 5-10s → 0.6
   - Recovery took >10s or topic abandoned → 0.3
   - Multiple cascading disruptions → 0.1

═══════════════════════════════════════════════════════════════
IMPLEMENTATION
═══════════════════════════════════════════════════════════════

```python
class RecoveryAnalyzer:
    def __init__(self, filler_burst_threshold=3, filler_burst_window=10,
                 pause_threshold_sec=3, pace_spike_pct=0.40,
                 posture_drop_threshold=15):
        self.filler_burst_threshold = filler_burst_threshold
        self.filler_burst_window = filler_burst_window
        self.pause_threshold = pause_threshold_sec
        self.pace_spike_pct = pace_spike_pct
        self.posture_drop = posture_drop_threshold

    def detect_disruptions(self, timeline) -> list[dict]:
        """Find all disruption events in the timeline."""
        disruptions = []
        disruptions += self._detect_filler_bursts(timeline)
        disruptions += self._detect_disfluencies(timeline)
        disruptions += self._detect_long_pauses(timeline)
        disruptions += self._detect_pace_spikes(timeline)
        disruptions += self._detect_posture_collapses(timeline)
        # Sort by time, merge overlapping disruptions
        return self._merge_overlapping(sorted(disruptions, key=lambda d: d["time_sec"]))

    def measure_recovery(self, timeline, disruption) -> dict:
        """Measure how quickly the speaker recovered from a disruption."""
        t = disruption["time_sec"]
        pre = timeline[max(0, t-10):t]  # 10 seconds before
        post = timeline[t:min(len(timeline), t+15)]  # 15 seconds after

        vocal_recovery = self._measure_vocal_recovery(pre, post)
        physical_recovery = self._measure_physical_recovery(pre, post)
        content_recovery = self._measure_content_recovery(pre, post)
        composure = self._compute_composure_score(
            vocal_recovery, physical_recovery, content_recovery
        )

        return {
            "disruption": disruption,
            "vocal_recovery_sec": vocal_recovery,
            "physical_recovery_sec": physical_recovery,
            "content_maintained": content_recovery,
            "composure_score": composure,
        }

    def analyze_all(self, timeline) -> dict:
        disruptions = self.detect_disruptions(timeline)
        recoveries = [self.measure_recovery(timeline, d) for d in disruptions]
        return {
            "total_disruptions": len(disruptions),
            "disruptions_per_minute": len(disruptions) / (len(timeline) / 60),
            "recoveries": recoveries,
            "mean_composure": np.mean([r["composure_score"] for r in recoveries]) if recoveries else 1.0,
            "fastest_recovery": min(recoveries, key=lambda r: r["vocal_recovery_sec"]) if recoveries else None,
            "slowest_recovery": max(recoveries, key=lambda r: r["vocal_recovery_sec"]) if recoveries else None,
        }
```

NO USER INPUT REQUIRED. Pure computation on timeline data.

COMPLETION: ✓ when disruptions are detected and recovery metrics computed.


═══════════════════════════════════════════════════════════════════
═══════════════════════════════════════════════════════════════════