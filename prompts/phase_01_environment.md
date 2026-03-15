PHASE 1 — ENVIRONMENT SETUP AGENT
====================================

AGENT ROLE: Environment Architect
DEPENDS ON: Nothing (first phase)
DELIVERS TO: All subsequent phases
ESTIMATED TIME: 15 minutes (agent) + 5 minutes (user)

OBJECTIVE:
Create the complete project directory structure, Python environment,
install all dependencies, and verify hardware (M1 GPU / MPS) works.

═══════════════════════════════════════════════════════════════
TASK 1: Create Directory Structure
═══════════════════════════════════════════════════════════════

Create the following directory tree at ~/Desktop/Claude-assistant/:

```
Claude-assistant/
├── src/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── audio_extractor.py
│   ├── vad.py
│   ├── noise_reduction.py
│   ├── transcriber.py
│   ├── features.py
│   ├── filler_detector.py
│   ├── filler_verifier.py
│   ├── disfluency_model.py
│   ├── prosody.py
│   ├── vocal_emotion.py
│   └── output_assembler.py
├── training/
│   ├── train_filler_verifier.py
│   ├── train_disfluency.py
│   ├── train_vocal_emotion.py
│   └── configs/
│       ├── filler.yaml
│       ├── disfluency.yaml
│       └── emotion.yaml
├── models/
│   ├── filler_verifier/
│   ├── disfluency/
│   └── vocal_emotion/
├── data/
│   ├── raw/
│   ├── audio/
│   ├── datasets/
│   │   ├── sep28k/
│   │   ├── ravdess/
│   │   └── cremad/
│   ├── filler_embeddings/
│   └── outputs/
├── tests/
│   └── fixtures/
├── configs/
│   └── pipeline_config.yaml
├── docs/
├── prompts/
├── requirements.txt
├── setup.py
├── PROGRESS.md
└── README.md
```

═══════════════════════════════════════════════════════════════
TASK 2: Create requirements.txt
═══════════════════════════════════════════════════════════════

Write this file at ~/Desktop/Claude-assistant/requirements.txt:

```
# Core ML
torch>=2.1.0
torchvision>=0.16.0
torchaudio>=2.1.0

# HuggingFace
transformers>=4.36.0
datasets>=2.16.0
accelerate>=0.25.0
evaluate>=0.4.0

# Audio processing
librosa>=0.10.1
praat-parselmouth>=0.4.3
noisereduce>=3.0.0
soundfile>=0.12.1
ffmpeg-python>=0.2.0

# Speech models
speechbrain>=1.0.0

# Scientific
numpy>=1.24.0
scipy>=1.11.0

# Utilities
pyyaml>=6.0
tqdm>=4.65.0

# Experiment tracking (optional)
wandb>=0.16.0
```

═══════════════════════════════════════════════════════════════
TASK 3: Create setup script
═══════════════════════════════════════════════════════════════

Write ~/Desktop/Claude-assistant/setup.sh:

```bash
#!/bin/bash
set -e

echo "=== AI Public Speaking Assistant — Environment Setup ==="

# Check Python
python3 --version || { echo "ERROR: Python 3 not found"; exit 1; }

# Create conda environment
echo "Creating conda environment..."
conda create -n voice-pipeline python=3.10 -y
conda activate voice-pipeline

# Install PyTorch (M1 optimized)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio

# Install all dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install FFmpeg
echo "Checking FFmpeg..."
if ! command -v ffmpeg &> /dev/null; then
    echo "FFmpeg not found. Installing via Homebrew..."
    brew install ffmpeg
fi

# Verify MPS
echo "Verifying M1 GPU (MPS)..."
python3 -c "
import torch
mps = torch.backends.mps.is_available()
print(f'MPS available: {mps}')
if not mps:
    print('WARNING: MPS not available. Training will use CPU (slower).')
    print('Ensure you have macOS 12.3+ and PyTorch 2.0+')
"

# Set MPS fallback
echo "export PYTORCH_ENABLE_MPS_FALLBACK=1" >> ~/.zshrc

echo ""
echo "=== Setup Complete ==="
echo "Activate with: conda activate voice-pipeline"
```

═══════════════════════════════════════════════════════════════
TASK 4: Create pipeline config
═══════════════════════════════════════════════════════════════

Write ~/Desktop/Claude-assistant/configs/pipeline_config.yaml:

```yaml
# Voice Pipeline Configuration
pipeline:
  name: "voice_analysis_v1"
  version: "1.0.0"
  device: "mps"  # "mps" for M1, "cuda" for Colab, "cpu" for fallback

audio:
  sample_rate: 16000
  channels: 1
  max_duration_sec: 3600  # 1 hour max

vad:
  threshold: 0.5
  min_speech_duration_ms: 250
  min_silence_duration_ms: 100

whisper:
  model_id: "openai/whisper-small"
  language: "en"
  chunk_length_s: 30
  batch_size: 1

features:
  frame_ms: 25
  hop_ms: 10
  n_mfcc: 13
  f0_min: 75
  f0_max: 600

filler:
  threshold: 0.5
  patterns:
    hesitation: ["um", "uh", "erm", "ah", "mm"]
    discourse: ["like", "you know", "basically", "i mean",
                "sort of", "kind of", "right", "okay so", "actually"]

disfluency:
  model_path: "models/disfluency/best_model.pt"
  window_sec: 3.0
  hop_sec: 1.5

prosody:
  rate_window_sec: 10.0
  optimal_rate_range: [3.5, 5.0]  # syllables/sec
  pitch_cv_optimal: [0.15, 0.25]

emotion:
  model_path: "models/vocal_emotion/best_model.pt"
  window_sec: 2.0
  classes: ["Neutral", "Enthusiastic", "Nervous", "Angry", "Sad"]

output:
  window_sec: 2.0
  hop_sec: 1.0
  output_dir: "data/outputs"
```

═══════════════════════════════════════════════════════════════
TASK 5: Initialize PROGRESS.md
═══════════════════════════════════════════════════════════════

```markdown
# Voice Pipeline Build Progress

## Status
- [ ] Phase 1: Environment Setup
- [ ] Phase 2: Audio Preprocessing
- [ ] Phase 3: Speech-to-Text (Whisper)
- [ ] Phase 4: Feature Extraction
- [ ] Phase 5: Filler Detection
- [ ] Phase 6: Disfluency Detection
- [ ] Phase 7: Prosody Analysis
- [ ] Phase 8: Vocal Emotion
- [ ] Phase 9: Integration & Assembly
- [ ] Phase 10: Testing & Validation

## Models
| Model | Status | Metric | Target | Pass? |
|-------|--------|--------|--------|-------|
| Filler MLP | — | — | F1≥0.90 | — |
| Disfluency | — | — | MacroF1≥0.72 | — |
| Vocal Emotion | — | — | UAR≥0.58 | — |

## User Actions Pending
None yet.
```

═══════════════════════════════════════════════════════════════
BRIDGE AGENT INTERVENTION POINT
═══════════════════════════════════════════════════════════════

🔔 USER INPUT REQUIRED — Phase 1: Environment Setup

WHAT IS HAPPENING:
The Environment Agent has created all project files and configs.
Now you need to run the setup script to install dependencies.

WHY YOUR INPUT IS NEEDED:
Installing system software (conda, FFmpeg) requires your terminal access.

WHAT YOU NEED TO DO:
Step 1: Open Terminal
Step 2: cd ~/Desktop/Claude-assistant
Step 3: chmod +x setup.sh
Step 4: ./setup.sh
Step 5: Verify output shows "MPS available: True"

IF CONDA IS NOT INSTALLED:
Go to https://docs.conda.io/en/latest/miniconda.html
Download the macOS ARM64 installer
Run: bash Miniconda3-latest-MacOSX-arm64.sh

WHAT HAPPENS NEXT:
Once setup.sh completes successfully, Phase 1 is done.
Phase 2 (Audio Preprocessing) will begin immediately.
No more user input needed until Phase 5.

═══════════════════════════════════════════════════════════════
COMPLETION CRITERIA
═══════════════════════════════════════════════════════════════

Phase 1 is COMPLETE when:
✓ All directories exist
✓ requirements.txt is written
✓ pipeline_config.yaml is written
✓ conda environment "voice-pipeline" is active
✓ torch.backends.mps.is_available() returns True
✓ ffmpeg is accessible from command line
✓ PROGRESS.md updated to show Phase 1 complete

OUTPUT TO NEXT PHASE:
- Working Python environment with all packages
- Project directory structure ready
- Config file at configs/pipeline_config.yaml
