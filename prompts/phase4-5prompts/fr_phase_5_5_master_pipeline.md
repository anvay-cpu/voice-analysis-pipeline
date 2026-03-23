PHASE 5.5 — FULL PIPELINE ORCHESTRATOR
=========================================

AGENT ROLE: Systems Architect
DEPENDS ON: ALL previous phases
DELIVERS TO: Phase 5.6 (testing)
RUNS ON: Local M1 (orchestration) + Colab T4 (heavy processing)
ESTIMATED TIME: 20 minutes (agent), 0 minutes (user)

OBJECTIVE:
Build the MASTER PIPELINE that takes a video file and produces
the complete coaching report end-to-end.

```python
# src/master_pipeline.py

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

    def __init__(self, config_path="configs/master_config.yaml"):
        self.voice_pipeline = VoiceAnalysisPipeline(config_path)
        self.body_pipeline = BodyAnalysisPipeline(config_path)
        self.content_pipeline = ContentAnalysisPipeline(config_path)
        self.fusion_engine = FusionEngine(config_path)
        self.scorer = DimensionScorer()
        self.coaching_writer = CoachingWriter()  # Uses proxy at 8080
        self.chart_generator = ChartGenerator()
        self.report_builder = ReportBuilder()

    def process(self, video_path: str, output_dir: str = "reports/") -> dict:
        """Full end-to-end pipeline.

        Processing order:
        1. Voice pipeline (audio extraction + Whisper + prosody + emotion)
        2. Body pipeline (frame extraction + pose + gesture + gaze)
           ↑ These two run the SAME video, can be parallelized
        3. Content pipeline (takes transcript from voice pipeline)
        4. Fusion (aligns all 3 on timeline + novel analyses)
        5. Scoring (6 dimensions)
        6. Coaching (Claude API writes feedback)
        7. Charts (matplotlib generates visuals)
        8. Report (HTML/PDF assembly)
        """
        import time
        start = time.time()
        os.makedirs(output_dir, exist_ok=True)

        # Step 1+2: Run voice and body pipelines
        print("="*60)
        print(f"Processing: {video_path}")
        print("="*60)

        print("\n[1/8] Voice analysis...")
        voice_output = self.voice_pipeline.process(video_path)

        print("[2/8] Body language analysis...")
        body_output = self.body_pipeline.process(video_path)

        print("[3/8] Content analysis...")
        transcript = voice_output.get("transcript", {})
        content_output = self.content_pipeline.process(transcript)

        duration = voice_output.get("duration_sec",
                    body_output.get("duration_sec", 600))

        # Step 4: Fusion
        print("[4/8] Multimodal fusion...")
        fusion_output = self.fusion_engine.fuse(
            voice_output, body_output, content_output, duration
        )

        # Step 5: Scoring
        print("[5/8] Computing scores...")
        scores = self.scorer.score_all(fusion_output)
        overall = sum(scores.values()) / len(scores)
        scores["overall"] = round(overall, 1)

        print(f"  Overall score: {overall:.0f}/100")
        for dim, val in scores.items():
            if dim != "overall":
                print(f"  {dim}: {val:.0f}/100")

        # Step 6: Coaching feedback
        print("[6/8] Generating coaching feedback...")
        coaching = self.coaching_writer.generate_full_report(
            all_segments=fusion_output.get("timeline_segments", []),
            overall_scores=scores,
            speech_metadata={"duration_sec": duration,
                             "video_path": video_path},
        )

        # Step 7: Charts
        print("[7/8] Generating charts...")
        chart_dir = os.path.join(output_dir, "charts")
        self.chart_generator.output_dir = chart_dir
        charts = self.chart_generator.generate_all(
            scores, fusion_output["timeline"], fusion_output
        )

        # Step 8: Report
        print("[8/8] Building report...")
        report = self.report_builder.build_and_save(
            coaching_data=coaching,
            scores=scores,
            charts=charts,
            metadata={
                "duration_sec": duration,
                "video_path": video_path,
                "processing_time_sec": time.time() - start,
            },
            output_dir=output_dir,
        )

        elapsed = time.time() - start
        print(f"\n{'='*60}")
        print(f"COMPLETE in {elapsed:.0f} seconds")
        print(f"HTML report: {report['html']}")
        if report.get('pdf'):
            print(f"PDF report:  {report['pdf']}")
        print(f"{'='*60}")

        return report


# CLI interface
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Public Speaking Coach")
    parser.add_argument("--video", required=True, help="Path to speech video")
    parser.add_argument("--output", default="reports/", help="Output directory")
    args = parser.parse_args()

    pipeline = SpeechCoachPipeline()
    pipeline.process(args.video, args.output)
```

USAGE:
```bash
# Process a speech and generate report
python -m src.master_pipeline --video data/raw/videos/ted_talk.mp4 --output reports/ted_talk/
# Opens: reports/ted_talk/speech_report.html
```

NO USER INPUT REQUIRED.

COMPLETION: ✓ when master pipeline chains all steps without errors.


═══════════════════════════════════════════════════════════════════
═══════════════════════════════════════════════════════════════════