# Voice Analysis Pipeline

A real-time voice analysis pipeline that takes a speech video and produces timestamped coaching data — transcription, filler words, disfluencies, prosody metrics, and vocal emotion classification.

**Modality 1** of a larger AI Public Speaking Assistant.

## What It Does

Given a video of someone speaking, the pipeline outputs per-window (2s) JSON with:

- **Transcription** — word-level timestamps via Whisper
- **Filler detection** — "um", "uh", "like", "you know", etc.
- **Disfluency detection** — repetitions, prolongations, blocks, interjections
- **Prosody analysis** — speaking rate, pitch variation, volume assessment
- **Vocal emotion** — Neutral, Enthusiastic, Nervous, Angry, Sad
- **Acoustic features** — F0, energy, jitter, shimmer, HNR

## Pipeline Steps

1. Audio extraction + resampling to 16kHz mono
2. Voice Activity Detection (Silero VAD)
3. Noise reduction
4. Speech-to-text (Whisper small)
5. Acoustic feature extraction (librosa + parselmouth)
6. Filler word detection (regex + optional MLP verifier)
7. Disfluency detection (Wav2Vec2 + classifier head)
8. Prosody analysis (syllable rate, pitch CV, volume scoring)
9. Vocal emotion classification (ECAPA-TDNN + MLP head)
10. Output assembly into timestamped JSON

## Quick Start

```bash
# Create conda environment
conda create -n voice-pipeline python=3.10 -y
conda activate voice-pipeline

# Install dependencies
pip install torch torchaudio transformers librosa praat-parselmouth speechbrain pyyaml psutil noisereduce soundfile

# Run on a video
python -m src.pipeline --video path/to/speech.mp4
```

Output is saved to `data/outputs/<filename>.json`.

## Project Structure

```
src/
  audio_extractor.py   — FFmpeg audio extraction
  vad.py               — Silero Voice Activity Detection
  noise_reduction.py   — Spectral noise reduction
  transcriber.py       — Whisper speech-to-text
  features.py          — F0, energy, MFCC, voice quality
  filler_detector.py   — Regex-based filler detection
  filler_verifier.py   — MLP filler verification (optional)
  disfluency_model.py  — Wav2Vec2 disfluency classifier
  prosody.py           — Speaking rate, pitch, volume analysis
  vocal_emotion.py     — ECAPA-TDNN emotion classifier
  output_assembler.py  — Per-window JSON assembly
  pipeline.py          — Main pipeline orchestrator

models/
  disfluency/          — Wav2Vec2 checkpoint (361MB, not in repo)
  vocal_emotion/       — ECAPA-TDNN checkpoint

configs/
  pipeline_config.yaml — All pipeline settings

tests/                 — Unit and integration tests
training/              — Model training scripts
notebooks/             — Colab notebooks for GPU training/testing
docs/                  — Validation report
```

## Models

| Model | Architecture | Metric | Notes |
|-------|-------------|--------|-------|
| Transcription | Whisper small (244M) | Pretrained | OpenAI |
| VAD | Silero VAD | Pretrained | Silero |
| Filler Verifier | MLP (200K) | F1 = 0.82 | Optional |
| Disfluency | Wav2Vec2-base + head (95M) | Macro F1 = 0.50 | Trained on SEP-28k |
| Vocal Emotion | ECAPA-TDNN + MLP (15M) | UAR = 0.54 | Trained on RAVDESS + CREMA-D |

Large model files (>100MB) are stored on Google Drive, not in the repo.

## Output Format

```json
{
  "metadata": { "duration_sec": 300, "timings": {...} },
  "summary": {
    "words_per_minute": 128.4,
    "fillers_per_minute": 0.6,
    "disfluency_count": 19,
    "avg_pitch_cv": 0.42,
    "dominant_emotion": "Neutral"
  },
  "windows": [
    {
      "start_sec": 0.0,
      "end_sec": 2.0,
      "transcript": "welcome to today's talk",
      "fillers": [],
      "disfluencies": [],
      "prosody": { "syllable_rate": 3.5, "pitch_cv": 0.21 },
      "emotion": { "label": "Enthusiastic", "confidence": 0.72 },
      "features": { "f0_mean": 180.2, "hnr_db": 15.3 }
    }
  ]
}
```

## Hardware

- **Developed on:** Apple M1 Mac 16GB (MPS backend)
- **Tested on:** Google Colab T4 GPU (~2.5 min per 5-min video)
- **CPU fallback:** Supported but slower

## Tests

```bash
python -m pytest tests/ -v
```

## License

MIT
