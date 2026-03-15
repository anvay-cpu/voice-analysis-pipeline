# Voice Pipeline (Modality 1) — Model Guide

---

## Overview

The voice pipeline has 10 processing steps. Only 3 require training.
The rest are either pretrained downloads or pure code with no model at all.

---

## Pretrained — Download and Use

These models work out of the box. No training, no fine-tuning, no datasets needed.

### Whisper Small (Speech-to-Text)

| | |
|---|---|
| **What it does** | Converts speech audio to text with word-level timestamps |
| **Parameters** | 244M |
| **Source** | `openai/whisper-small` on HuggingFace |
| **Why no training** | Pretrained on 680,000 hours of multilingual audio. 7.7% WER on English is sufficient for our pipeline. Filler detection and content analysis correct remaining errors downstream. |
| **Memory** | ~2GB |
| **M1 inference speed** | ~1.5x real-time (15-20s per minute of audio) |

```python
from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    device="mps",
)

result = pipe(
    "speech.wav",
    return_timestamps="word",
    chunk_length_s=30,
    batch_size=1,
)
# result["chunks"] = [{"text": "Good", "timestamp": [0.0, 0.38]}, ...]
```

### Silero VAD (Voice Activity Detection)

| | |
|---|---|
| **What it does** | Segments audio into speech vs silence/noise regions |
| **Parameters** | ~2M |
| **Source** | `snakers4/silero-vad` via torch.hub |
| **Why no training** | Tiny, robust, pretrained on massive data. Works perfectly on speech recordings. |
| **Memory** | ~8MB |
| **Speed** | Faster than real-time on CPU |

```python
model, utils = torch.hub.load(
    "snakers4/silero-vad", "silero_vad"
)
get_speech_timestamps, _, read_audio, _, _ = utils

wav = read_audio("speech.wav", sampling_rate=16000)
timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)
# timestamps = [{"start": 0, "end": 48000}, {"start": 52000, "end": 96000}, ...]
```

---

## No Model Needed — Pure Code

These steps use libraries and statistical computation. Nothing to download or train.

### Audio Extraction (FFmpeg)

| | |
|---|---|
| **What it does** | Extracts 16kHz mono WAV from video files |
| **Install** | `brew install ffmpeg` |
| **Code** | `ffmpeg -i input.mp4 -ar 16000 -ac 1 -f wav output.wav` |

### Noise Reduction (noisereduce)

| | |
|---|---|
| **What it does** | Removes stationary background noise using spectral gating |
| **Install** | `pip install noisereduce` |
| **How it works** | Uses non-speech segments (from VAD) as noise profile, then subtracts from speech segments. No neural network involved. |

### Acoustic Feature Extraction (librosa + parselmouth)

| | |
|---|---|
| **What it does** | Extracts MFCCs, F0 pitch contour, RMS energy, jitter, shimmer, HNR |
| **Install** | `pip install librosa praat-parselmouth` |
| **How it works** | Signal processing algorithms. MFCCs via FFT, F0 via autocorrelation, jitter/shimmer via Praat. |

### Prosody Analysis

| | |
|---|---|
| **What it does** | Computes speaking rate (syllables/sec), pitch variation (CV), volume dynamics |
| **Install** | Nothing extra beyond librosa |
| **How it works** | Statistical formulas on top of the features from the previous step. Speaking rate uses energy peak counting. Pitch variation is coefficient of variation of F0. |

### Filler Word Detection — Stage 1 (Regex)

| | |
|---|---|
| **What it does** | Finds candidate filler words in the Whisper transcript |
| **Install** | Nothing (Python standard library) |
| **How it works** | Pattern matching on known filler words: um, uh, like, basically, you know, right, actually, I mean, sort of, kind of. Outputs candidates for the trained verifier (Stage 2). |

### Output Assembly

| | |
|---|---|
| **What it does** | Merges all sub-module outputs into timestamped JSON windows |
| **Install** | Nothing |
| **How it works** | Iterates through 2-second windows with 1-second overlap, collects transcript words, fillers, disfluencies, prosody stats, and emotion labels for each window. |

---

## Train on M1 — Small Models That Fit Comfortably

These models are small enough that M1 training is practical and you avoid Colab session risks.

### Filler Verification MLP (Stage 2)

| | |
|---|---|
| **What it does** | Verifies whether a candidate filler word (from regex) is actually a filler based on acoustic context. "I was like totally confused" → filler. "I like pizza" → not a filler. |
| **Architecture** | 3-layer MLP (768 → 256 → 64 → 1) with BatchNorm and Dropout |
| **Parameters** | ~200K |
| **Input** | 768-dim Whisper encoder embedding for a 0.5s window around the candidate word |
| **Output** | Probability of being a filler (sigmoid) |
| **Training data** | Custom dataset you build: 500+ labeled clips (250 fillers + 250 non-fillers). See `train_filler_verifier.py` for step-by-step dataset creation instructions. |
| **Loss** | BCEWithLogitsLoss with positive class weighting |
| **Optimizer** | AdamW, lr=1e-3, weight_decay=0.01 |
| **M1 training time** | ~1 hour (after embedding pre-extraction) |
| **Target metric** | F1 ≥ 0.90 |
| **Script** | `train_filler_verifier.py` |

**Why M1:** The model is tiny. Uploading data to Colab and managing sessions would take longer than just training locally. The embedding pre-extraction step (running Whisper encoder on all 500 clips) takes ~20 minutes on M1, then training the MLP itself takes minutes.

**Training commands:**
```bash
# Step 1: Pre-extract Whisper embeddings (run once, ~20 min)
python training/train_filler_verifier.py --preextract \
    --data data/filler_labels.json \
    --embeddings_dir data/filler_embeddings \
    --device mps

# Step 2: Train the MLP (~1 hour)
python training/train_filler_verifier.py \
    --embeddings_dir data/filler_embeddings \
    --device mps \
    --epochs 50 \
    --batch_size 64
```

**Output:** `models/filler_verifier/best_model.pt`

---

### Vocal Emotion Classifier (ECAPA-TDNN)

| | |
|---|---|
| **What it does** | Classifies the speaker's vocal emotion every 2 seconds into 5 categories: Neutral, Enthusiastic, Nervous, Angry, Sad |
| **Architecture** | ECAPA-TDNN encoder (pretrained, frozen/partially unfrozen) + 2-layer MLP head |
| **Parameters** | 6.2M total. Phase 1: ~33K trainable (head only). Phase 2: ~2M trainable (top TDNN layers + head). |
| **Input** | Raw 16kHz audio waveform (2-4 second clips) |
| **Output** | 5-class probability distribution |
| **Training data** | RAVDESS (1,440 clips, free download) + CREMA-D (7,442 clips, free download) + optionally IEMOCAP (12 hrs, requires registration) |
| **Loss** | CrossEntropyLoss with inverse-frequency class weights |
| **Optimizer** | Phase 1: AdamW lr=1e-3. Phase 2: differential LR — 1e-5 encoder, 5e-4 head. |
| **M1 training time** | Phase 1: ~30 minutes. Phase 2: ~6 hours. Total: ~7 hours. |
| **Target metric** | UAR (Unweighted Average Recall) ≥ 0.58 |
| **Script** | `train_vocal_emotion.py` |

**Why M1:** Phase 1 trains in 30 minutes because it only trains a tiny MLP on pre-extracted 192-dim embeddings. Phase 2 takes 6 hours but the model is small (6.2M params, ~800MB memory) so it runs comfortably on M1 with no OOM risk. Colab would be faster (~1.2 hours total) but the session management overhead isn't worth it for a model this size.

**Two-phase training explained:**
- Phase 1 freezes the entire ECAPA-TDNN encoder and trains only the classification head. This gives you a working model fast so you can test the pipeline early.
- Phase 2 unfreezes the top TDNN block and fine-tunes it alongside the head. This improves UAR by 3-8% because the encoder learns emotion-specific features instead of just speaker features.

**Training commands:**
```bash
# Download datasets first
# RAVDESS: https://zenodo.org/record/1188976
# CREMA-D: https://github.com/CheyneyComputerScience/CREMA-D

# Run all phases sequentially (~7 hours total)
python training/train_vocal_emotion.py \
    --phase all \
    --ravdess_dir data/ravdess \
    --cremad_dir data/cremad \
    --device mps

# Or run phases individually:
# Phase 0: Extract embeddings (~15 min)
python training/train_vocal_emotion.py --phase extract --device mps

# Phase 1: Train head (~30 min)
python training/train_vocal_emotion.py --phase phase1 --device mps

# Phase 2: Fine-tune encoder (~6 hours)
python training/train_vocal_emotion.py --phase phase2 --device mps
```

**Output:** `models/vocal_emotion/best_model.pt`

---

## Train on Colab — Too Slow on M1

This model benefits significantly from Colab's T4 GPU with FP16 tensor cores.

### Disfluency Detector (Wav2Vec2-base)

| | |
|---|---|
| **What it does** | Detects stuttering events at the acoustic level: repetitions ("b-b-because"), prolongations ("sssso"), blocks (silent tension mid-word), and interjections ("um" inserted mid-sentence). The transcript alone cannot capture these patterns — you need acoustic-level analysis. |
| **Architecture** | Wav2Vec2-base encoder (12 transformer layers, first 3 frozen) + 2-layer MLP classifier |
| **Parameters** | 95M total, ~72M trainable (layers 4-12 + classifier head) |
| **Input** | Raw 16kHz audio waveform (3-second clips) |
| **Output** | 5-class prediction: Fluent, Repetition, Prolongation, Block, Interjection |
| **Training data** | SEP-28k (28,177 clips from Apple's stuttering dataset) + optionally FluencyBank |
| **Loss** | Weighted CrossEntropyLoss (class weights: Fluent=1.0, Repetition=6.5, Prolongation=8.0, Block=12.0, Interjection=5.5) |
| **Optimizer** | AdamW with differential LR — 2e-5 for encoder, 1e-3 for classifier head |
| **Scheduler** | Cosine with linear warmup (10% of steps) |
| **Gradient accumulation** | 2 steps on Colab (effective batch=32), 4 steps on M1 (effective batch=32) |
| **Colab training time** | ~4 hours with FP16 on T4 |
| **M1 training time** | ~36 hours |
| **Target metric** | Macro F1 ≥ 0.72 across all 5 classes |
| **Script** | `train_disfluency.py` |

**Why Colab:** This is 95M parameters with 72M trainable. On M1 with FP32 and batch_size=8, each epoch takes ~2.4 hours — 15 epochs = 36 hours. On Colab T4 with FP16 tensor cores, the same training takes ~4 hours. The T4's FP16 tensor cores give a genuine 9x speedup for this model. The entire training run fits within one 12-hour Colab session with room to spare, so disconnect risk is low.

**Colab notebook setup:**

```python
# Cell 1: Install dependencies
!pip install transformers datasets soundfile accelerate

# Cell 2: Mount Drive for checkpoints
from google.colab import drive
drive.mount('/content/drive')

# Cell 3: Upload or clone your code
# Option A: Upload train_disfluency.py manually
# Option B: Clone your repo
# !git clone https://github.com/YOUR_REPO.git

# Cell 4: Download SEP-28k dataset
!git clone https://github.com/apple/ml-stuttering-events-dataset.git
!mv ml-stuttering-events-dataset/SEP-28k /content/data/sep28k

# Cell 5: Train
!python train_disfluency.py \
    --platform colab \
    --audio_dir /content/data/sep28k/audio \
    --labels_csv /content/data/sep28k/SEP-28k_episodes.csv \
    --epochs 15

# Cell 6: Save to Drive
!mkdir -p "/content/drive/MyDrive/voice_pipeline_models"
!cp models/disfluency/best_model.pt \
    "/content/drive/MyDrive/voice_pipeline_models/disfluency_best.pt"
!cp models/disfluency/history.json \
    "/content/drive/MyDrive/voice_pipeline_models/disfluency_history.json"
```

**Colab survival checklist:**
- Keep the browser tab open and active during training
- Checkpoints save every epoch automatically — if Colab disconnects, resume from the last epoch
- Use FP16 (the script enables it automatically with `--platform colab`)
- Monitor GPU memory with `!nvidia-smi` — should show ~10-12GB used out of 16GB

**If you must train on M1:**
```bash
python training/train_disfluency.py \
    --platform m1 \
    --audio_dir data/sep28k/audio \
    --labels_csv data/sep28k/SEP-28k_episodes.csv \
    --epochs 15
```
Start it Friday evening. It will checkpoint every epoch. Check Monday morning.

**Output:** `models/disfluency/best_model.pt`

---

## Summary Table

| Step | Component | Type | Where | Time |
|------|-----------|------|-------|------|
| 1 | Audio extraction | Pure code (FFmpeg) | — | — |
| 2 | Silero VAD | **Pretrained** | Download | 10 sec |
| 3 | Noise reduction | Pure code (noisereduce) | — | — |
| 4 | Whisper Small | **Pretrained** | Download | 2 min |
| 5 | Feature extraction | Pure code (librosa) | — | — |
| 6a | Filler regex | Pure code | — | — |
| 6b | Filler verifier MLP | **Train on M1** | M1 | 1 hour |
| 7 | Disfluency detector | **Train on Colab** | Colab T4 | 4 hours |
| 8 | Prosody analysis | Pure code | — | — |
| 9 | Vocal emotion | **Train on M1** | M1 | 7 hours |
| 10 | Output assembly | Pure code | — | — |

**Total training time:** ~12 hours (8 hours on M1 + 4 hours on Colab, run in parallel)

---

## Training Order

```
Day 1 Morning:   Train filler MLP on M1          [1 hour]
Day 1 Afternoon:  Start vocal emotion on M1        [7 hours, runs into evening]
Day 2 Morning:   Train disfluency on Colab        [4 hours]
Day 2 Afternoon:  Integration testing              [2 hours]
```

The filler MLP finishes first so you can test the filler detection pipeline immediately. Vocal emotion runs overnight on M1. Disfluency runs on Colab the next morning while you test the other components locally. By Day 2 afternoon everything is trained and you can wire the full pipeline together.

---

## Model Files After Training

```
models/
├── filler_verifier/
│   ├── best_model.pt          # 768→256→64→1 MLP (~1MB)
│   └── history.json
├── disfluency/
│   ├── best_model.pt          # Wav2Vec2 + classifier (~380MB)
│   └── history.json
└── vocal_emotion/
    ├── phase1_best_head.pt    # MLP head only (~0.1MB)
    ├── best_model.pt          # ECAPA-TDNN + head (~25MB)
    └── labels.pt              # Pre-extracted embeddings
```

---

## Target Metrics Checklist

Before moving to Modality 2, verify all three trained models pass:

- [ ] Filler MLP: F1 ≥ 0.90 on test set
- [ ] Disfluency: Macro F1 ≥ 0.72 on SEP-28k test split
- [ ] Vocal emotion: UAR ≥ 0.58 on combined test set

And verify pretrained models work:

- [ ] Whisper produces word-level timestamps on a test video
- [ ] Silero VAD correctly segments speech vs silence
- [ ] Feature extraction outputs valid F0, MFCCs, energy values
- [ ] Prosody analysis produces reasonable speaking rate (3.5-5.0 syl/sec for normal speech)
- [ ] Full pipeline produces valid JSON output for a 1-minute test clip
