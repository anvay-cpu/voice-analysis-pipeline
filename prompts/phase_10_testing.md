PHASE 10 — TESTING & VALIDATION AGENT
========================================

AGENT ROLE: Quality Assurance Engineer
DEPENDS ON: Phase 9 (integrated pipeline)
DELIVERS TO: Final output — ready for Modality 2
ESTIMATED TIME: 20 min (agent) + 30 min (user provides test videos)

OBJECTIVE:
Run comprehensive validation on the complete voice pipeline.
Verify all models meet targets. Run end-to-end on real speeches.

═══════════════════════════════════════════════════════════════
TASK 1: Build tests/test_integration.py
═══════════════════════════════════════════════════════════════

End-to-end integration tests:
- test_full_pipeline_short: 1-minute clip → valid JSON, all fields present
- test_full_pipeline_long: 10-minute clip → no OOM, memory bounded
- test_graceful_degradation: run with missing models → no crash
- test_output_schema: validate JSON matches expected schema
- test_timestamps_monotonic: all windows are sequential, no gaps
- test_speech_ratio: speech_ratio between 0 and 1

═══════════════════════════════════════════════════════════════
TASK 2: Model metric verification
═══════════════════════════════════════════════════════════════

Run final evaluation on each model and report:
- Filler MLP: load, evaluate on held-out test set → report F1
- Disfluency: load, evaluate on SEP-28k test split → report Macro F1
- Vocal Emotion: load, evaluate on combined test set → report UAR
- Print pass/fail against targets

═══════════════════════════════════════════════════════════════
TASK 3: Real-world speech tests
═══════════════════════════════════════════════════════════════

Process 3 different speech types and inspect outputs:

Test A: "Good Speaker" — a TED talk from a polished speaker
  Expected: high pitch variation, few fillers, good prosody scores,
  mostly Enthusiastic/Neutral emotion, few disfluencies

Test B: "Nervous Speaker" — a student presentation or first-time speaker
  Expected: more fillers, higher jitter/shimmer, Nervous emotion detected,
  possibly faster speaking rate, some disfluencies

Test C: "Monotone Speaker" — a dry lecture or reading
  Expected: low pitch CV, low emotion variety, mostly Neutral emotion,
  adequate content but flat delivery scores

═══════════════════════════════════════════════════════════════
BRIDGE AGENT INTERVENTION POINT
═══════════════════════════════════════════════════════════════

🔔 USER INPUT REQUIRED — Phase 10: Testing

WHAT IS HAPPENING:
The Testing Agent needs 3 real speech videos to validate the pipeline
on diverse speaking styles.

WHY YOUR INPUT IS NEEDED:
Synthetic tests verify code works. Real videos verify the pipeline
produces meaningful, useful coaching data.

WHAT YOU NEED TO DO:

Step 1: Find 3 videos (YouTube or your own recordings):
  Video A: A polished TED talk or keynote (5-10 min)
  Video B: A nervous/beginner presentation (3-10 min)
  Video C: A monotone lecture or reading (5-10 min)

Step 2: Download them:
  yt-dlp -o "data/raw/test_good_speaker.mp4" "URL_A"
  yt-dlp -o "data/raw/test_nervous_speaker.mp4" "URL_B"
  yt-dlp -o "data/raw/test_monotone_speaker.mp4" "URL_C"

Step 3: Place the 3 files in:
  ~/Desktop/Claude-assistant/data/raw/
  Files: test_good_speaker.mp4, test_nervous_speaker.mp4, test_monotone_speaker.mp4

WHAT HAPPENS NEXT:
The agent will run the full pipeline on all 3 videos,
inspect outputs, and generate a validation report at
docs/validation_report.md

═══════════════════════════════════════════════════════════════
TASK 4: Generate validation report
═══════════════════════════════════════════════════════════════

Write docs/validation_report.md containing:
- Model metrics table (all 3 models with pass/fail)
- Per-video summary (scores, detected issues, emotion timeline)
- Processing time per video
- Memory usage peak
- Known issues or edge cases found
- Recommendation: ready for Modality 2? YES/NO

═══════════════════════════════════════════════════════════════
TASK 5: Final PROGRESS.md update
═══════════════════════════════════════════════════════════════

Update PROGRESS.md with all phases complete.
Mark Modality 1 as DONE.

COMPLETION: ✓ when all 3 test videos produce valid, sensible JSON outputs
            and all model metrics meet targets.

FINAL OUTPUT:
The voice pipeline is complete and ready to feed into the
Multimodal Fusion Layer alongside Modality 2 (Body Language)
and Modality 3 (Content Analysis).
