PHASE 2 — AUDIO PREPROCESSING AGENT
======================================

AGENT ROLE: Audio Engineer
DEPENDS ON: Phase 1 (environment ready)
DELIVERS TO: Phase 3 (Whisper), Phase 4 (Features), Phase 6 (Disfluency), Phase 8 (Emotion)
ESTIMATED TIME: 20 minutes (agent), 0 minutes (user)

OBJECTIVE:
Build three source files that handle the first three pipeline steps:
  Step 1: Audio extraction from video (FFmpeg)
  Step 2: Voice activity detection (Silero VAD)
  Step 3: Noise reduction (spectral gating)

No user input needed. No training needed. Pure code.

═══════════════════════════════════════════════════════════════
TASK 1: Build src/audio_extractor.py (Step 1)
═══════════════════════════════════════════════════════════════

REQUIREMENTS:
- Accept MP4, WebM, MKV, AVI, MOV video files
- Validate file exists, is under 2GB, under 1 hour
- Extract audio to 16kHz mono WAV using FFmpeg subprocess
- Return path to the extracted WAV file
- Include get_duration() helper using ffprobe
- Raise clear errors on failure

FUNCTION SIGNATURES:
  validate_input(video_path: str) -> dict
    Returns: {"duration_sec": float, "size_mb": float, "format": str}
    Raises: ValueError if invalid

  extract_audio(video_path: str, output_dir: str) -> str
    Returns: path to 16kHz mono WAV
    Raises: RuntimeError if FFmpeg fails

INPUT:  Video file path (MP4/WebM/etc.)
OUTPUT: data/audio/{video_stem}.wav (16kHz, mono, PCM16)

═══════════════════════════════════════════════════════════════
TASK 2: Build src/vad.py (Step 2)
═══════════════════════════════════════════════════════════════

REQUIREMENTS:
- Load Silero VAD from torch.hub (cache after first download)
- Process audio file and return speech segment timestamps
- Configurable threshold, min_speech_duration, min_silence_duration
- Also compute pause durations between speech segments
  (pauses > 3 seconds are used by the fusion layer for disruption detection)
- Load config from pipeline_config.yaml

FUNCTION SIGNATURES:
  load_vad_model() -> tuple
    Returns: (model, get_speech_timestamps_fn, read_audio_fn)

  detect_speech_segments(audio_path: str, config: dict) -> dict
    Returns: {
      "speech_segments": [{"start_sec", "end_sec", "duration_sec"}, ...],
      "pauses": [{"start_sec", "end_sec", "duration_sec"}, ...],
      "total_speech_sec": float,
      "total_silence_sec": float,
      "speech_ratio": float
    }

INPUT:  16kHz WAV audio file
OUTPUT: Speech/silence segment timestamps + pause list

═══════════════════════════════════════════════════════════════
TASK 3: Build src/noise_reduction.py (Step 3)
═══════════════════════════════════════════════════════════════

REQUIREMENTS:
- Use noisereduce library with spectral gating
- Build noise profile from non-speech segments (from VAD output)
- If no non-speech segments, use first 0.5 seconds as noise estimate
- Include SNR estimation function to decide if noise reduction is needed
- Skip noise reduction if SNR > 30dB (clean recording)
- Apply if SNR < 20dB, optional between 20-30dB
- Save cleaned audio to disk

FUNCTION SIGNATURES:
  estimate_snr(audio: np.ndarray, speech_segments: list, sr: int) -> float
    Returns: SNR in dB

  reduce_noise(audio_path: str, speech_segments: list,
               output_path: str, prop_decrease: float = 0.8) -> str
    Returns: path to cleaned audio file

  preprocess_audio(video_path: str, output_dir: str, config: dict) -> dict
    This is the MASTER function that chains Steps 1→2→3.
    Returns: {
      "raw_audio_path": str,
      "clean_audio_path": str,
      "speech_segments": list,
      "pauses": list,
      "duration_sec": float,
      "snr_db": float,
      "noise_reduced": bool
    }

INPUT:  16kHz WAV + VAD segments
OUTPUT: Cleaned WAV + preprocessing metadata dict

═══════════════════════════════════════════════════════════════
TASK 4: Build test for preprocessing
═══════════════════════════════════════════════════════════════

Write tests/test_preprocessing.py with:
- test_extract_audio: Extract from a small test video
- test_vad: Run on a known audio clip with pauses
- test_noise_reduction: Add synthetic noise, verify SNR improves
- test_full_preprocess: Chain all three steps

For testing without a real video, create a synthetic test:
  Generate a 10-second sine wave with 2 seconds of silence in the middle.
  Verify VAD detects 2 speech segments and 1 pause.

═══════════════════════════════════════════════════════════════
COMPLETION CRITERIA
═══════════════════════════════════════════════════════════════

Phase 2 is COMPLETE when:
✓ src/audio_extractor.py exists and handles all video formats
✓ src/vad.py loads Silero VAD and returns speech segments
✓ src/noise_reduction.py applies spectral gating correctly
✓ preprocess_audio() chains all three steps end-to-end
✓ tests/test_preprocessing.py passes
✓ PROGRESS.md updated

NO USER INPUT REQUIRED for this phase.

OUTPUT TO NEXT PHASES:
- clean_audio_path → Phase 3 (Whisper), Phase 6 (Disfluency), Phase 8 (Emotion)
- speech_segments → Phase 3 (filtering), Phase 5 (fillers), Phase 7 (prosody)
- pauses → Phase 9 (fusion layer disruption detection)
