PHASE 9 — INTEGRATION & ASSEMBLY AGENT
=========================================

AGENT ROLE: Systems Integrator
DEPENDS ON: ALL previous phases (2-8)
DELIVERS TO: Phase 10 (testing)
ESTIMATED TIME: 30 minutes (agent), 0 minutes (user)

OBJECTIVE:
Build src/output_assembler.py and src/pipeline.py that wire all
10 steps together into a single end-to-end pipeline.

═══════════════════════════════════════════════════════════════
TASK 1: Build src/output_assembler.py (Step 10)
═══════════════════════════════════════════════════════════════

Build the output merger:
- assemble_voice_output() → list of per-window JSON dicts
  Iterates 2-second windows with 1-second overlap
  For each window, collects:
    transcript, word_count, is_speech,
    speaking_rate, f0_mean/std/cv, volume_rms/range,
    jitter, shimmer, filler_words, disfluencies, vocal_emotion

- generate_summary(windows) → summary dict
  Computes: avg speaking rate, filler rate per minute,
  total disfluencies, pitch variation CV, dominant emotion,
  emotion variety score

- save_output(windows, summary, metadata, output_path)
  Saves complete JSON with metadata, summary, and window array

═══════════════════════════════════════════════════════════════
TASK 2: Build src/pipeline.py (Master Orchestrator)
═══════════════════════════════════════════════════════════════

Build VoiceAnalysisPipeline class:
- __init__(config_path) → loads config, loads all models
- process(video_path) → complete output dict
  Chains all 10 steps:
    1. extract_audio()
    2. detect_speech_segments()
    3. reduce_noise()
    4. transcribe()
    5. extract_features()
    6. detect_fillers() + verify_fillers()
    7. detect_disfluencies()
    8. analyze_prosody()
    9. predict_emotions()
    10. assemble_output()
- CLI: python -m src.pipeline --video path/to/video.mp4 --output data/outputs/

Include timing for each step (print elapsed time per module).
Handle missing models gracefully: if a model isn't trained yet,
skip that step and mark it as "unavailable" in the output.

═══════════════════════════════════════════════════════════════
TASK 3: Build configs for graceful degradation
═══════════════════════════════════════════════════════════════

The pipeline should work even with missing models:
- No filler verifier? → Use regex-only (Stage 1) with lower confidence
- No disfluency model? → Skip disfluency, output empty list
- No emotion model? → Skip emotion, output "unavailable"
This lets you test early before all training is complete.

NO USER INPUT REQUIRED.

COMPLETION: ✓ when pipeline.py processes a test video end-to-end
            and produces valid JSON output.
