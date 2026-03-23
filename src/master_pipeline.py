"""Phase 5.5 — Master Pipeline Orchestrator.

Complete AI Public Speaking Assistant Pipeline:
Input: video file (MP4)
Output: coaching report (HTML + PDF) with 6-dimension scores,
        charts, per-segment feedback, and practice plan.

Runs: 3 modality pipelines → fusion → scoring → coaching → charts → report
"""

import logging
import os
import time

import yaml

from src.fusion.fusion_engine import FusionEngine
from src.scoring.dimension_scorer import DimensionScorer
from src.report.coaching_writer import CoachingWriter
from src.report.chart_generator import ChartGenerator
from src.report.report_builder import ReportBuilder

logger = logging.getLogger(__name__)


class SpeechCoachPipeline:
    """Complete AI Public Speaking Assistant Pipeline.

    Input: video file (MP4)
    Output: coaching report (HTML + PDF) with:
        - 6-dimension scores (0-100)
        - Timeline heatmap
        - Per-segment coaching feedback
        - Emotion arc chart
        - Practice plan

    Runs 3 modality pipelines → fusion → scoring → report.
    """

    def __init__(self, config_path: str = "configs/fusion_config.yaml"):
        self.config = self._load_config(config_path)

        # Pipeline components (lazy-loaded for modality pipelines)
        self.fusion_engine = FusionEngine(self.config)
        self.scorer = DimensionScorer()
        self.coaching_writer = CoachingWriter(use_api=True)
        self.chart_generator = ChartGenerator()
        self.report_builder = ReportBuilder()

        # Modality pipelines loaded on demand
        self._voice_pipeline = None
        self._body_pipeline = None
        self._content_pipeline = None

    def process(
        self,
        video_path: str,
        output_dir: str = "reports/",
    ) -> dict:
        """Full end-to-end pipeline.

        Args:
            video_path: Path to speech video (MP4).
            output_dir: Directory for report output.

        Returns:
            Dict with report paths and scores.
        """
        start = time.time()
        os.makedirs(output_dir, exist_ok=True)

        print("=" * 60)
        print(f"Processing: {video_path}")
        print("=" * 60)

        # Step 1: Voice analysis
        print("\n[1/8] Voice analysis...")
        voice_output = self._run_voice_pipeline(video_path)

        # Step 2: Body language analysis
        print("[2/8] Body language analysis...")
        body_output = self._run_body_pipeline(video_path)

        # Step 3: Content analysis
        print("[3/8] Content analysis...")
        content_output = self._run_content_pipeline(voice_output)

        # Determine duration
        duration = self._get_duration(voice_output, body_output)

        # Step 4: Multimodal fusion
        print("[4/8] Multimodal fusion...")
        voice_windows = self._extract_voice_windows(voice_output)
        fusion_output = self.fusion_engine.fuse(
            voice_windows, body_output, content_output, duration
        )

        # Step 5: Scoring
        print("[5/8] Computing scores...")
        scores = self.scorer.score_all(fusion_output)
        print(f"  Overall score: {scores['overall']:.0f}/100")
        for dim, val in scores.items():
            if dim != "overall":
                print(f"  {dim}: {val:.0f}/100")

        # Step 6: Coaching feedback
        print("[6/8] Generating coaching feedback...")
        coaching = self.coaching_writer.generate_full_report(
            all_segments=[],
            overall_scores=scores,
            fusion_output=fusion_output,
            speech_metadata={
                "duration_sec": duration,
                "video_path": video_path,
            },
        )

        # Step 7: Charts
        print("[7/8] Generating charts...")
        chart_dir = os.path.join(output_dir, "charts")
        self.chart_generator.output_dir = chart_dir
        os.makedirs(chart_dir, exist_ok=True)
        charts = self.chart_generator.generate_all(
            scores, fusion_output["timeline"], fusion_output
        )

        # Step 8: Report
        print("[8/8] Building report...")
        elapsed = time.time() - start
        report = self.report_builder.build_and_save(
            coaching_data=coaching,
            scores=scores,
            charts=charts,
            metadata={
                "duration_sec": duration,
                "video_path": video_path,
                "processing_time_sec": elapsed,
            },
            output_dir=output_dir,
        )

        final_elapsed = time.time() - start
        print(f"\n{'=' * 60}")
        print(f"COMPLETE in {final_elapsed:.0f} seconds")
        print(f"HTML report: {report['html']}")
        if report.get("pdf"):
            print(f"PDF report:  {report['pdf']}")
        print(f"{'=' * 60}")

        return {
            "report": report,
            "scores": scores,
            "charts": charts,
            "duration_sec": duration,
            "processing_time_sec": final_elapsed,
        }

    def process_from_outputs(
        self,
        voice_output: dict | list,
        body_output: dict,
        content_output: dict,
        duration_sec: float,
        output_dir: str = "reports/",
        video_path: str = "unknown",
    ) -> dict:
        """Run fusion → scoring → report from pre-computed modality outputs.

        Use this when modality pipelines were already run (e.g., on Colab)
        and you just need the fusion + report generation locally.
        """
        start = time.time()
        os.makedirs(output_dir, exist_ok=True)

        # Extract windows if voice_output is the full pipeline dict
        voice_windows = self._extract_voice_windows(voice_output)

        print("[1/5] Multimodal fusion...")
        fusion_output = self.fusion_engine.fuse(
            voice_windows, body_output, content_output, duration_sec
        )

        print("[2/5] Computing scores...")
        scores = self.scorer.score_all(fusion_output)
        print(f"  Overall: {scores['overall']:.0f}/100")

        print("[3/5] Generating coaching feedback...")
        coaching = self.coaching_writer.generate_full_report(
            all_segments=[],
            overall_scores=scores,
            fusion_output=fusion_output,
            speech_metadata={"duration_sec": duration_sec, "video_path": video_path},
        )

        print("[4/5] Generating charts...")
        chart_dir = os.path.join(output_dir, "charts")
        self.chart_generator.output_dir = chart_dir
        os.makedirs(chart_dir, exist_ok=True)
        charts = self.chart_generator.generate_all(
            scores, fusion_output["timeline"], fusion_output
        )

        print("[5/5] Building report...")
        elapsed = time.time() - start
        report = self.report_builder.build_and_save(
            coaching_data=coaching,
            scores=scores,
            charts=charts,
            metadata={
                "duration_sec": duration_sec,
                "video_path": video_path,
                "processing_time_sec": elapsed,
            },
            output_dir=output_dir,
        )

        print(f"COMPLETE in {time.time() - start:.0f}s → {report['html']}")
        return {"report": report, "scores": scores, "charts": charts}

    # ── Modality pipeline wrappers ─────────────────────────────────

    def _run_voice_pipeline(self, video_path: str) -> dict:
        """Run the voice (Modality 1) pipeline."""
        try:
            if self._voice_pipeline is None:
                from src.pipeline import VoiceAnalysisPipeline
                self._voice_pipeline = VoiceAnalysisPipeline()
            return self._voice_pipeline.process(video_path)
        except Exception as e:
            logger.warning("Voice pipeline not available (%s: %s), using empty output",
                          type(e).__name__, e)
            return {"windows": [], "summary": {}, "metadata": {"duration_sec": 0}}

    def _run_body_pipeline(self, video_path: str) -> dict:
        """Run the body (Modality 2) pipeline."""
        try:
            if self._body_pipeline is None:
                from src.body.pipeline import BodyAnalysisPipeline
                self._body_pipeline = BodyAnalysisPipeline()
            return self._body_pipeline.process(video_path)
        except Exception as e:
            logger.warning("Body pipeline not available (%s: %s), using empty output",
                          type(e).__name__, e)
            return {"segments": [], "summary": {}}

    def _run_content_pipeline(self, voice_output: dict) -> dict:
        """Run the content (Modality 3) pipeline."""
        try:
            if self._content_pipeline is None:
                from src.content.pipeline import ContentAnalysisPipeline
                self._content_pipeline = ContentAnalysisPipeline()
            transcript = voice_output.get("transcript", voice_output)
            return self._content_pipeline.process(transcript)
        except Exception as e:
            logger.warning("Content pipeline not available (%s: %s), using empty output",
                          type(e).__name__, e)
            return {"segments": [], "summary": {}}

    # ── Helpers ────────────────────────────────────────────────────

    def _extract_voice_windows(self, voice_output) -> list[dict]:
        """Extract voice windows from various output formats."""
        if isinstance(voice_output, list):
            return voice_output
        if isinstance(voice_output, dict):
            return voice_output.get("windows", [])
        return []

    def _get_duration(self, voice_output: dict, body_output: dict) -> float:
        """Get speech duration from available outputs."""
        if isinstance(voice_output, dict):
            d = voice_output.get("metadata", {}).get("duration_sec")
            if d:
                return float(d)
            d = voice_output.get("summary", {}).get("duration_sec")
            if d:
                return float(d)
        if isinstance(body_output, dict):
            d = body_output.get("duration_sec")
            if d:
                return float(d)
        return 300.0  # Default 5 minutes

    @staticmethod
    def _load_config(config_path: str) -> dict:
        """Load YAML config file."""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.info("Config not found at %s, using defaults", config_path)
            return {}


# CLI interface
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="AI Public Speaking Coach")
    parser.add_argument("--video", required=True, help="Path to speech video (MP4)")
    parser.add_argument("--output", default="reports/", help="Output directory")
    parser.add_argument("--config", default="configs/fusion_config.yaml", help="Config file")
    args = parser.parse_args()

    pipeline = SpeechCoachPipeline(config_path=args.config)
    pipeline.process(args.video, args.output)
