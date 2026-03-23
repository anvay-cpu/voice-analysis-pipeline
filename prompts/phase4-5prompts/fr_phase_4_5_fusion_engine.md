PHASE 4.5 — FUSION ENGINE
===========================

AGENT ROLE: Systems Integrator
DEPENDS ON: Phases 4.1-4.4
DELIVERS TO: Phase 5.1
RUNS ON: Local M1
ESTIMATED TIME: 15 minutes (agent), 0 minutes (user)

OBJECTIVE:
Build src/fusion/fusion_engine.py and src/fusion/pipeline.py that
wire timeline alignment, regime transition scoring, recovery analysis,
and emotion coherence into a single fusion pipeline.

TASKS:

1. Build FusionEngine class:
```python
class FusionEngine:
    def __init__(self, config):
        self.aligner = TimelineAligner()
        self.transition_scorer = RegimeTransitionScorer(window_sec=config["transition_window"])
        self.recovery_analyzer = RecoveryAnalyzer()
        self.emotion_analyzer = EmotionCoherenceAnalyzer()

    def fuse(self, voice_output, body_output, content_output, duration_sec) -> dict:
        # Step 1: Align on shared timeline
        timeline = self.aligner.align(voice_output, body_output,
                                       content_output, duration_sec)

        # Step 2: Find regime boundaries
        boundaries = self.aligner.find_regime_boundaries(timeline)

        # Step 3: Score regime transitions
        transitions = self.transition_scorer.score_all_transitions(
            timeline, boundaries
        )

        # Step 4: Analyze disruptions and recovery
        recovery = self.recovery_analyzer.analyze_all(timeline)

        # Step 5: Compute emotion coherence
        coherence = self.emotion_analyzer.analyze_full(timeline)

        return {
            "timeline": timeline,
            "duration_sec": duration_sec,
            "regime_boundaries": boundaries,
            "transitions": transitions,
            "recovery": recovery,
            "emotion_coherence": coherence,
        }
```

2. Build fusion config in configs/fusion_config.yaml:
```yaml
fusion:
  transition_window: 10
  recovery:
    filler_burst_threshold: 3
    filler_burst_window: 10
    pause_threshold_sec: 3
    pace_spike_pct: 0.40
    posture_drop_threshold: 15
  coherence:
    mismatch_threshold: 0.4
    cluster_gap_sec: 5
```

3. Write tests/fusion/test_fusion_engine.py with mock data from all 3 modalities.

NO USER INPUT REQUIRED. Pure integration code.

COMPLETION: ✓ when fusion engine produces combined output from mock modality data.


═══════════════════════════════════════════════════════════════════
═══════════════════════════════════════════════════════════════════