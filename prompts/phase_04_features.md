PHASE 4 — FEATURE EXTRACTION AGENT
=====================================

AGENT ROLE: Audio DSP Engineer
DEPENDS ON: Phase 2 (clean audio)
DELIVERS TO: Phase 5 (filler verification), Phase 7 (prosody), Phase 9 (fusion)
ESTIMATED TIME: 20 minutes (agent), 0 minutes (user)

OBJECTIVE:
Build src/features.py — an AcousticFeatureExtractor class that extracts
handcrafted audio features using librosa and parselmouth.

TASKS:
1. Build AcousticFeatureExtractor class with methods:
   - extract_mfcc(audio) → 39-dim MFCCs (13 + delta + delta-delta)
   - extract_f0(audio) → dict with f0_contour, f0_mean, f0_std, f0_range, f0_cv
   - extract_energy(audio) → dict with rms_contour_db, rms_mean_db, dynamic_range_db
   - extract_spectral(audio) → dict with centroid, bandwidth, rolloff
   - extract_voice_quality(audio) → dict with jitter_percent, shimmer_percent, hnr_db
   - extract_all(audio) → combined dict of all features
   - extract_windowed(audio, window_sec, hop_sec) → list of per-window feature dicts

2. Configuration from pipeline_config.yaml:
   frame_ms=25, hop_ms=10, n_mfcc=13, f0_min=75, f0_max=600

3. Edge case handling:
   - Short segments (< 50ms): return None/defaults, don't crash
   - Unvoiced segments: F0 returns 0, voice quality returns defaults
   - Silent segments: energy returns -inf dB, skip jitter/shimmer

4. Write tests/test_features.py:
   - Generate a 440Hz sine wave → verify F0 = 440 ± 1Hz
   - Generate silence → verify energy is very low
   - Process a real speech clip → verify MFCCs are 39-dim

NO USER INPUT REQUIRED.
NO TRAINING REQUIRED. Pure signal processing.

COMPLETION: ✓ when extract_all() produces valid features on test audio
            and all edge cases are handled without crashes.
