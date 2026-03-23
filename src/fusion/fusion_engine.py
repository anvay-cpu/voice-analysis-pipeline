"""Phase 4.5 — Fusion Engine.

Wires together timeline alignment, regime transition scoring,
recovery analysis, and emotion coherence into a single fusion pipeline.
"""

import logging

from src.fusion.timeline_aligner import TimelineAligner
from src.fusion.regime_transition_scorer import RegimeTransitionScorer
from src.fusion.recovery_analyzer import RecoveryAnalyzer
from src.fusion.emotion_coherence import EmotionCoherenceAnalyzer

logger = logging.getLogger(__name__)


class FusionEngine:
    """Multimodal fusion pipeline combining all Phase 4 components."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        fusion_cfg = config.get("fusion", {})
        recovery_cfg = fusion_cfg.get("recovery", {})
        coherence_cfg = fusion_cfg.get("coherence", {})

        self.aligner = TimelineAligner()
        self.transition_scorer = RegimeTransitionScorer(
            window_sec=fusion_cfg.get("transition_context_sec", 10),
        )
        self.recovery_analyzer = RecoveryAnalyzer(
            filler_burst_threshold=recovery_cfg.get("filler_burst_threshold", 3),
            filler_burst_window=recovery_cfg.get("filler_burst_window", 10),
            pause_threshold_sec=recovery_cfg.get("pause_threshold_sec", 3),
            pace_spike_pct=recovery_cfg.get("pace_spike_pct", 0.40),
            posture_drop_threshold=recovery_cfg.get("posture_drop_threshold", 15),
        )
        self.emotion_analyzer = EmotionCoherenceAnalyzer(
            mismatch_threshold=coherence_cfg.get("mismatch_threshold", 0.4),
            cluster_gap_sec=coherence_cfg.get("cluster_gap_sec", 5),
        )

    def fuse(
        self,
        voice_output: list[dict],
        body_output: dict,
        content_output: dict,
        duration_sec: float,
    ) -> dict:
        """Run the full fusion pipeline.

        Args:
            voice_output: Voice pipeline per-window output list.
            body_output: Body pipeline output dict with 'segments'.
            content_output: Content pipeline output dict with 'segments'.
            duration_sec: Total speech duration in seconds.

        Returns:
            Dict with aligned timeline, transitions, recovery, coherence.
        """
        logger.info("Fusion engine: starting alignment for %.1fs speech", duration_sec)

        # Step 1: Align on shared 1-second timeline
        timeline = self.aligner.align(
            voice_output, body_output, content_output, duration_sec
        )

        # Step 2: Find regime boundaries
        boundaries = self.aligner.find_regime_boundaries(timeline)
        logger.info("Found %d regime boundaries", len(boundaries))

        # Step 3: Score regime transitions
        transitions = self.transition_scorer.score_all_transitions(
            timeline, boundaries
        )

        # Step 4: Analyze disruptions and recovery
        recovery = self.recovery_analyzer.analyze_all(timeline)
        logger.info(
            "Detected %d disruptions, mean composure=%.2f",
            recovery["total_disruptions"], recovery["mean_composure"],
        )

        # Step 5: Compute emotion coherence
        coherence = self.emotion_analyzer.analyze_full(timeline)
        logger.info("Emotion coherence score: %d/100", coherence["coherence_score_0_100"])

        return {
            "timeline": timeline,
            "duration_sec": duration_sec,
            "regime_boundaries": boundaries,
            "transitions": transitions,
            "recovery": recovery,
            "emotion_coherence": coherence,
        }
