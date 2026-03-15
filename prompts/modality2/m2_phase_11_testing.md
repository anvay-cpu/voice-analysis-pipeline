PHASE 11 — TESTING & VALIDATION
=================================

AGENT ROLE: QA Engineer
DEPENDS ON: Phase 10 (integrated pipeline)
DELIVERS TO: Multimodal fusion layer
ESTIMATED TIME: 25 min (agent) + 15 min (user provides test videos)

OBJECTIVE:
Validate the complete body language pipeline on real speech videos.

═══════════════════════════════════════════════════════════════
BRIDGE AGENT INTERVENTION
═══════════════════════════════════════════════════════════════

🔔 USER INPUT REQUIRED — Phase 11: Testing

WHAT IS HAPPENING:
The pipeline needs real videos to validate all components work together.

WHAT YOU NEED TO DO:
Use the SAME 3 test videos from Modality 1 testing.
If you don't have them yet:

  yt-dlp -f "bestvideo[height<=720]+bestaudio" --merge-output-format mp4 \
      -o "data/raw/videos/test_%(title)s.%(ext)s" \
      "URL_GOOD_SPEAKER" "URL_NERVOUS_SPEAKER" "URL_MONOTONE_SPEAKER"

Place at: ~/Desktop/Claude-assistant/data/raw/videos/

═══════════════════════════════════════════════════════════════
TEST SCENARIOS
═══════════════════════════════════════════════════════════════

Test A: Good Speaker (TED talk)
  Expected: high posture score (>80), frequent illustrators,
  audience gaze >75%, happy/neutral face, purposeful movement

Test B: Nervous Speaker
  Expected: more adaptors, lower posture score, more floor gaze,
  nervous facial emotion, pacing movement pattern

Test C: Monotone/Static Speaker
  Expected: anchored movement, rest hand position, neutral face,
  good posture but minimal gestures

═══════════════════════════════════════════════════════════════
MODEL METRICS VERIFICATION
═══════════════════════════════════════════════════════════════

- [ ] Posture MLP: MAE ≤ 0.6 vs rule-based scores
- [ ] Gesture Transformer: F1 ≥ 0.65 on held-out data
- [ ] Facial Emotion: Accuracy ≥ 0.60 on FER+ test split
- [ ] Pipeline processes 10-min video without OOM on M1
- [ ] JSON output has all fields populated

Generate docs/validation_report_m2.md with results.

COMPLETION: ✓ when all test videos produce valid, sensible output.
Modality 2 ready for fusion layer integration.