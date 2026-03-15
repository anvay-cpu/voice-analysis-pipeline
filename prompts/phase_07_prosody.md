PHASE 7 — PROSODY ANALYSIS AGENT
===================================

AGENT ROLE: Prosody Analyst
DEPENDS ON: Phase 4 (acoustic features)
DELIVERS TO: Phase 9 (output assembly)
ESTIMATED TIME: 15 minutes (agent), 0 minutes (user)

OBJECTIVE:
Build src/prosody.py — statistical analysis of speaking rate,
pitch dynamics, and volume control. Pure computation, no training.

TASKS:
1. Build functions:
   - estimate_syllable_rate(audio, sr, window_sec) → list of per-window rates
     Uses energy-peak counting (Mermelstein method)
   - classify_rate(syl_per_sec) → "Too slow"/"Slow"/"Optimal"/"Fast"/"Too fast"
   - assess_pitch_dynamics(f0_data) → dict with cv, range, assessment, score(0-100)
   - assess_volume_dynamics(energy_data) → dict with dynamic_range, issues, score(0-100)
   - analyze_prosody(audio, features, config) → complete prosody dict per window

2. Scoring thresholds (from pipeline_config.yaml):
   - Speaking rate optimal: 3.5-5.0 syl/sec
   - Pitch CV optimal: 0.15-0.25 (below = monotone, above = erratic)
   - Volume dynamic range optimal: 6-30 dB

3. Write tests/test_prosody.py:
   - Metronome test: fixed-rate clicks → verify rate detection accuracy
   - Monotone test: constant pitch audio → verify low CV score
   - Dynamic test: varied pitch → verify good CV score

NO USER INPUT REQUIRED. NO TRAINING REQUIRED.

COMPLETION: ✓ when analyze_prosody() returns valid scores and assessments.
