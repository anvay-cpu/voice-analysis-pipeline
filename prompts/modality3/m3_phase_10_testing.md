PHASE 10 — TESTING & VALIDATION
==================================

AGENT ROLE: QA Engineer
DEPENDS ON: Phase 9 (integrated pipeline)
DELIVERS TO: Multimodal Fusion Layer
ESTIMATED TIME: 25 min (agent) + 10 min (user provides test data)

OBJECTIVE:
Validate the complete content analysis pipeline on real speech transcripts.

═══════════════════════════════════════════════════════════════
TEST SCENARIOS
═══════════════════════════════════════════════════════════════

Test A: "Well-structured TED talk"
  Expected: clear regime sequence (Intro → Main → Anecdote → Data → Conclusion),
  smooth transitions, high grammar score, moderate readability,
  mostly Persuasive/Inspirational tone, positive sentiment in conclusion.

Test B: "Rambling student presentation"
  Expected: unclear regime boundaries, abrupt transitions, more grammar errors,
  lower vocabulary diversity, mostly Informative tone, neutral sentiment.

Test C: "Data-heavy technical talk"
  Expected: Data Presentation regime dominant, high readability grade (12+),
  high AWL coverage, Informative/Assertive tone, strong evidence quality.

═══════════════════════════════════════════════════════════════
BRIDGE AGENT INTERVENTION
═══════════════════════════════════════════════════════════════

🔔 USER INPUT REQUIRED — Phase 10: Testing

WHAT IS HAPPENING:
The pipeline needs real speech transcripts to validate.

WHAT YOU NEED TO DO:
Use the SAME 3 test videos from Modality 1 and 2 testing.
The voice pipeline (Modality 1) should have already produced
transcripts for these. If not:

Step 1: Run Modality 1 pipeline on your 3 test videos:
  python -m src.pipeline --video data/raw/videos/test_good_speaker.mp4
  This produces transcript JSON in data/outputs/

Step 2: Or manually transcribe a short speech (~3 minutes)
  and save as data/test_transcript.json in Whisper output format.

WHAT HAPPENS NEXT:
The agent runs the content pipeline on all 3 transcripts and
generates a validation report.

═══════════════════════════════════════════════════════════════
MODEL METRICS VERIFICATION
═══════════════════════════════════════════════════════════════

- [ ] Grammar checker: catches real errors, doesn't flag speech artifacts
- [ ] Readability: FKGL in 8-14 range for typical speeches
- [ ] Vocabulary: TTR in 0.4-0.7 range, reasonable AWL coverage
- [ ] Regime detection: boundaries at obvious topic changes
- [ ] Sentiment: matches actual content polarity
- [ ] Tone: matches perceived speaking style
- [ ] Argument (if API key): reasonable claims and evidence extraction
- [ ] Full pipeline: processes 10-min transcript in < 30 seconds on M1
- [ ] JSON output: all fields populated, no NaN/null crashes

Generate docs/validation_report_m3.md with results.

COMPLETION: ✓ when all test transcripts produce valid, sensible output.
Modality 3 ready for fusion layer integration.