<p align="center">
  <h1 align="center">AI Public Speaking Coach</h1>
  <p align="center">
    <strong>Multimodal speech analysis pipeline that watches you speak and coaches you like a pro.</strong>
  </p>
  <p align="center">
    Voice &bull; Body Language &bull; Content &bull; Fusion &bull; Real-Time Coaching
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10-blue?logo=python&logoColor=white" alt="Python 3.10" />
  <img src="https://img.shields.io/badge/PyTorch-2.1+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Next.js-16-black?logo=next.js&logoColor=white" alt="Next.js" />
  <img src="https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=white" alt="React" />
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License" />
</p>

---

## What It Does

Upload a video of someone giving a speech or presentation. The system analyzes **three modalities** in parallel — voice, body language, and content — then **fuses** them into a unified timeline, computes **six coaching dimension scores**, and delivers **actionable feedback** through a Bloomberg Terminal-inspired web UI.

```
                          ┌──────────────┐
                          │  Input Video │
                          └──────┬───────┘
                                 │
                 ┌───────────────┼───────────────┐
                 ▼               ▼               ▼
        ┌────────────┐  ┌────────────┐  ┌────────────┐
        │  MODALITY 1 │  │  MODALITY 2 │  │  MODALITY 3 │
        │    Voice    │  │    Body     │  │   Content   │
        │  Analysis   │  │  Language   │  │  Analysis   │
        └──────┬─────┘  └──────┬─────┘  └──────┬─────┘
               │               │               │
               └───────────────┼───────────────┘
                               ▼
                    ┌────────────────────┐
                    │  MULTIMODAL FUSION │
                    │  1-second timeline │
                    └─────────┬──────────┘
                              ▼
                    ┌────────────────────┐
                    │   6-DIMENSION      │
                    │   SCORING (0-100)  │
                    └─────────┬──────────┘
                              ▼
                 ┌────────────┴────────────┐
                 ▼                         ▼
        ┌────────────────┐       ┌────────────────┐
        │  Report Engine │       │  Bloomberg UI  │
        │  HTML / PDF    │       │  (Next.js App) │
        └────────────────┘       └────────────────┘
```

### Example Output (5-minute speech)

| Dimension | Score | Description |
|-----------|-------|-------------|
| Vocal Clarity | 78/100 | Low filler rate, clear articulation |
| Body Language | 65/100 | Active gestures, posture needs work |
| Content Structure | 82/100 | Strong argument flow, good readability |
| Audience Engagement | 71/100 | Solid eye contact, varied energy |
| Emotional Expressiveness | 69/100 | Pitch variety present, emotion coherence fair |
| Regime Adaptability | 74/100 | Smooth transitions, fast recovery from disruptions |
| **Overall** | **73/100** | |

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Modality 1 — Voice Analysis](#modality-1--voice-analysis)
- [Modality 2 — Body Language Analysis](#modality-2--body-language-analysis)
- [Modality 3 — Content Analysis](#modality-3--content-analysis)
- [Multimodal Fusion](#multimodal-fusion)
- [6-Dimension Scoring](#6-dimension-scoring)
- [Report Generation](#report-generation)
- [Web Application](#web-application)
- [API Server](#api-server)
- [Models](#models)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Testing](#testing)
- [Training](#training)
- [Hardware Requirements](#hardware-requirements)
- [License](#license)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MASTER PIPELINE                              │
│                                                                     │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐            │
│  │ Voice Pipeline│   │ Body Pipeline │   │Content Pipeline│           │
│  │  (2s window)  │   │  (6s window)  │   │ (regime-level) │           │
│  │              │   │              │   │              │            │
│  │ Audio Extract│   │Frame Extract │   │Transcript    │            │
│  │ Silero VAD   │   │YOLOv8 Detect │   │Processing    │            │
│  │ Noise Reduce │   │MediaPipe Pose│   │Grammar Check │            │
│  │ Whisper STT  │   │Posture Score │   │Readability   │            │
│  │ Features     │   │Gesture Class │   │Vocabulary    │            │
│  │ Filler Det.  │   │Hand Tracking │   │Regime Detect │            │
│  │ Disfluency   │   │Gaze Estimate │   │Sentiment     │            │
│  │ Prosody      │   │Face Emotion  │   │Tone + Argument│           │
│  │ Vocal Emotion│   │Stage Movement│   │              │            │
│  │ Assembly     │   │Temporal Agg. │   │Assembly      │            │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘            │
│         │                  │                  │                     │
│         └──────────────────┼──────────────────┘                     │
│                            ▼                                        │
│                   ┌────────────────┐                                 │
│                   │ Fusion Engine  │                                 │
│                   │ Timeline Align │                                 │
│                   │ Transitions    │                                 │
│                   │ Recovery       │                                 │
│                   │ Emotion Coher. │                                 │
│                   └───────┬────────┘                                 │
│                           ▼                                         │
│                   ┌────────────────┐                                 │
│                   │Dimension Scorer│──▶ 6 scores (0-100)            │
│                   └───────┬────────┘                                 │
│                           ▼                                         │
│              ┌────────────┴────────────┐                            │
│              │      Report Builder     │                            │
│              │ Charts + Coaching + PDF │                            │
│              └─────────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────┘
         ▲                                          │
         │  Upload                                  │ Report JSON
    ┌────┴────┐                                ┌────▼────┐
    │ FastAPI │◄──────── REST API ─────────────▶│ Next.js │
    │ Backend │                                │ Web App │
    └─────────┘                                └─────────┘
```

---

## Modality 1 — Voice Analysis

Extracts audio from the input video and runs a 10-step pipeline producing per-window (2-second, 1-second hop) analysis.

### Pipeline Steps

| # | Step | Method | Output |
|---|------|--------|--------|
| 1 | Audio Extraction | FFmpeg → 16kHz mono WAV | Raw audio |
| 2 | Voice Activity Detection | Silero VAD | Speech/silence segments |
| 3 | Noise Reduction | Spectral gating (noisereduce) | Cleaned audio |
| 4 | Transcription | OpenAI Whisper (small, 244M) | Word-level timestamps |
| 5 | Feature Extraction | librosa + Praat (parselmouth) | F0, energy, MFCC, jitter, shimmer, HNR |
| 6 | Filler Detection | Regex patterns + optional MLP verifier | "um", "uh", "like", "you know" with timestamps |
| 7 | Disfluency Detection | Wav2Vec2-base + classifier head | Repetitions, prolongations, blocks |
| 8 | Prosody Analysis | Rule-based syllable rate, pitch CV | Speaking rate, pitch variation, volume scores |
| 9 | Vocal Emotion | ECAPA-TDNN + MLP head | 5 classes: Neutral, Enthusiastic, Nervous, Angry, Sad |
| 10 | Output Assembly | JSON builder | Per-window structured output |

### Voice Output Schema

```json
{
  "metadata": {
    "duration_sec": 300.0,
    "models": { "whisper": "available", "disfluency": "available", "emotion": "available" },
    "timings": { "preprocessing": 34.7, "transcription": 110.8 }
  },
  "summary": {
    "word_count": 585,
    "words_per_minute": 117.0,
    "filler_count": 4,
    "disfluency_count": 16,
    "disfluency_types": { "SoundRep": 4, "Prolongation": 10, "Block": 2 },
    "dominant_emotion": "Neutral",
    "emotion_variety": 0.42
  },
  "windows": [
    {
      "start_sec": 0.0,
      "end_sec": 2.0,
      "is_speech": true,
      "transcript": "welcome to today's talk",
      "fillers": [],
      "disfluencies": [{ "type": "SoundRep", "confidence": 0.77 }],
      "prosody": {
        "syllable_rate": 5.7,
        "pitch_cv": 0.17,
        "pitch_score": 85,
        "volume_score": 70
      },
      "emotion": { "label": "Enthusiastic", "confidence": 0.71 },
      "features": { "f0_mean": 128.9, "hnr_db": 15.3 }
    }
  ]
}
```

### Filler Word Categories

| Category | Patterns |
|----------|----------|
| Hesitation | um, uh, erm, ah |
| Discourse markers | like, you know, basically, actually, literally |

### Disfluency Types

| Type | Description | Example |
|------|-------------|---------|
| SoundRep | Sound/syllable repetition | "I-I-I think" |
| WordRep | Whole word repetition | "the the results" |
| Prolongation | Extended sound duration | "soooo what happened" |
| Block | Speech block/stoppage | Abnormal pause mid-word |
| Interjection | Non-lexical insertion | "well, anyway" |

---

## Modality 2 — Body Language Analysis

Extracts video frames and runs computer vision models to analyze posture, gestures, gaze, facial emotion, and stage movement in 6-second windows.

### Pipeline Steps

| # | Step | Method | Output |
|---|------|--------|--------|
| 1 | Frame Extraction | OpenCV at 5 FPS | Video frames |
| 2 | Person Detection | YOLOv8n | Bounding boxes |
| 3 | Pose Estimation | MediaPipe Pose (33 keypoints) | Skeleton landmarks |
| 4 | Posture Scoring | Rule-based + MLP hybrid | Score 0-100 (shoulder alignment, spine angle) |
| 5 | Gesture Classification | Transformer model | 3 classes: Active Gesture, Adaptor, Rest |
| 6 | Hand State Tracking | MediaPipe Hands + rules | 8 states (see below) |
| 7 | Gaze Estimation | MediaPipe Face Mesh | 6 gaze zones + engagement ratio |
| 8 | Facial Emotion | EfficientNet-B0 (FER2013) | 7 classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral |
| 9 | Stage Movement | Centroid tracking + metrics | 4 patterns (see below) |
| 10 | Temporal Aggregation | 6s windows, 50% overlap | Smoothed per-window summaries |
| 11 | Output Assembly | JSON builder | Structured body output |

### Hand State Taxonomy (8 states)

```
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│   Open Palm     │    Pointing     │   Steepling     │  Crossed Arms   │
│   (engaging)    │  (directing)    │  (authority)     │  (defensive)    │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ Hands in Pockets│  Behind Back    │    Clasped      │     Other       │
│   (casual)      │   (formal)      │  (restrained)   │                 │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

### Gesture Classification (3-class)

| Class | Description | Speaking Impact |
|-------|-------------|-----------------|
| **Active Gesture** | Purposeful hand/arm movements that emphasize speech | Positive — reinforces message |
| **Adaptor** | Self-touching, fidgeting, grooming | Negative — signals nervousness |
| **Rest** | Hands still at sides or on podium | Neutral — missed opportunity |

### Stage Movement Patterns

```
  ┌───────────────────────────────────────────┐
  │                                           │
  │    Anchored (5)      Purposeful (8)       │    Movement Pattern Scores
  │    ┌───┐              ───▶ ───▶           │    (stage usage 0-10)
  │    │ ● │             ●         ●          │
  │    └───┘              ◀─── ◀───           │    Purposeful = 8 (best)
  │                                           │    Roaming    = 7
  │    Pacing (3)         Roaming (7)         │    Anchored   = 5
  │    ● ◀──▶ ●          ●  ●                │    Pacing     = 3 (worst)
  │                        ●    ●             │
  │                          ●                │
  └───────────────────────────────────────────┘
```

### Gaze Zones (6 zones)

| Zone | Description |
|------|-------------|
| Center | Direct audience engagement |
| Left / Right | Scanning audience sections |
| Up / Down | Notes, ceiling, floor |
| Away | Off-camera or turned away |

### Body Output Schema

```json
{
  "video": "path/to/video.mp4",
  "duration_sec": 300.0,
  "summary": {
    "posture_score": 64.7,
    "dominant_gesture": "Active Gesture",
    "dominant_hand_state": "Open Palm",
    "audience_engagement": 0.65,
    "dominant_emotion": "Happy",
    "movement_pattern": "Purposeful",
    "stage_usage_score": 8.0
  },
  "segments": [
    {
      "timestamp_start": 0.0,
      "timestamp_end": 6.0,
      "posture": { "score_mean": 75.0, "shoulder_deg": 2.1, "spine_deg": 5.3 },
      "gesture": { "dominant": "Active Gesture", "distribution": { "Active Gesture": 0.6, "Rest": 0.4 } },
      "hand_state": { "dominant": "Open Palm", "distribution": { "Open Palm": 0.7, "Other": 0.3 } },
      "gaze": { "engagement_ratio": 0.8, "distribution": { "Center": 0.6, "Left": 0.2, "Right": 0.2 } },
      "facial_emotion": { "dominant": "Happy", "confidence": 0.72 },
      "movement": { "pattern": "Purposeful", "velocity": 0.15 }
    }
  ]
}
```

---

## Modality 3 — Content Analysis

Processes the transcript to evaluate linguistic quality, argument structure, and narrative flow.

### Pipeline Steps

| # | Step | Method | Output |
|---|------|--------|--------|
| 1 | Transcript Processing | spaCy sentence splitting | Cleaned, segmented text |
| 2 | Grammar Checking | LanguageTool integration | Error types and counts |
| 3 | Readability Scoring | Flesch-Kincaid + Coleman-Liau | Grade level (FKGL), readability index |
| 4 | Vocabulary Analysis | Type-Token Ratio (TTR) + AWL | Lexical diversity, academic word usage |
| 5 | Regime Detection | Sentence-transformers embeddings | Semantic content boundaries (intro, data, conclusion) |
| 6 | Sentiment Analysis | RoBERTa (Twitter sentiment) | Positive / negative / neutral per segment |
| 7 | Tone Classification | LLM-based (Claude API) | Formal, casual, persuasive, conversational |
| 8 | Argument Analysis | LLM evaluation | Argument strength, structure, evidence quality |
| 9 | Output Assembly | JSON builder | Segment-level content metrics |

### Content Metrics

| Metric | Range | What It Measures |
|--------|-------|------------------|
| **FKGL** | 0-18+ | Flesch-Kincaid Grade Level (lower = simpler) |
| **Coleman-Liau** | 0-18+ | Character-based readability index |
| **TTR** | 0.0-1.0 | Type-Token Ratio (vocabulary diversity) |
| **Grammar Score** | 0-100 | Correctness based on LanguageTool errors |
| **Argument Effectiveness** | 0-100 | Structure, evidence, persuasiveness |

### Regime Detection

The system automatically detects semantic boundaries in the speech to identify content "regimes":

```
TIME ──────────────────────────────────────────────────▶

│ Introduction  │   Main Point 1   │ Main Point 2 │ Conclusion │
│  (greeting,   │  (core argument, │ (supporting  │  (summary, │
│   context)    │   evidence)      │  evidence)   │   CTA)     │
│               │                  │              │            │
0:00          1:30              3:00           4:15         5:00
```

---

## Multimodal Fusion

The fusion engine aligns all three modalities onto a **unified 1-second timeline** and performs cross-modal analysis.

### Fusion Steps

```
Voice (2s windows, 1s hop)   ──▶ ┌──────────────────────┐
                                  │  Timeline Aligner    │
Body  (6s windows, 3s hop)  ──▶ │  (1-second grid)     │ ──▶ Unified Timeline
                                  │  Interpolation +     │
Content (variable segments) ──▶ │  carry-forward       │
                                  └──────────┬───────────┘
                                             │
                          ┌──────────────────┼──────────────────┐
                          ▼                  ▼                  ▼
                 ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
                 │  Transition    │ │  Recovery      │ │   Emotion      │
                 │  Scoring       │ │  Analysis      │ │   Coherence    │
                 └────────────────┘ └────────────────┘ └────────────────┘
```

### Timeline Alignment

Each second of the speech gets a unified data point:

```json
{
  "time_sec": 42,
  "voice": {
    "syllable_rate": 4.2,
    "pitch_cv": 0.25,
    "emotion_label": "Enthusiastic",
    "filler_count": 0,
    "disfluency_count": 1
  },
  "body": {
    "posture_score": 75.0,
    "gesture_dominant": "Active Gesture",
    "gaze_engagement": 0.8,
    "facial_emotion": "Happy"
  },
  "content": {
    "regime_type": "Main Point",
    "grammar_error_count": 0
  }
}
```

### Disruption Detection & Recovery

The system identifies four types of performance disruptions and measures recovery:

| Disruption Type | Detection Criteria | Recovery Measurement |
|-----------------|-------------------|---------------------|
| **Filler Burst** | 3+ fillers in 10s window | Time until filler rate normalizes |
| **Pace Spike** | 40% syllable rate increase | Time until rate stabilizes |
| **Posture Drop** | 15+ point posture decrease | Time until score recovers |
| **Pause Anomaly** | Silence > 3s during speech | Time until speech resumes |

### Emotion Coherence

Cross-modal validation across three emotion channels:

```
                    ┌─── Voice Emotion (vocal energy, pitch)
                    │
Emotion Coherence ──┼─── Facial Emotion (expression, micro-expressions)
                    │
                    └─── Text Sentiment (word-level sentiment polarity)

         Coherent: Enthusiastic voice + Happy face + Positive text  ✓
         Mismatch: Angry tone + Happy face + Positive text          ✗
```

---

## 6-Dimension Scoring

All modalities feed into six coaching dimensions, each scored 0-100:

```
    ┌──────────────────────────────────────────────────────────┐
    │                   DIMENSION SCORES                       │
    │                                                          │
    │  VOCAL CLARITY ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░  78/100       │
    │   └─ fillers, disfluency, articulation (Voice)           │
    │                                                          │
    │  BODY LANGUAGE ▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░  65/100       │
    │   └─ posture, gestures, gaze (Body)                      │
    │                                                          │
    │  CONTENT       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░  82/100       │
    │   └─ readability, grammar, argument flow (Content)       │
    │                                                          │
    │  ENGAGEMENT    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░  71/100       │
    │   └─ eye contact, energy, gesture activity (Body+Voice)  │
    │                                                          │
    │  EXPRESSIVENESS▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░  69/100       │
    │   └─ pitch variety, emotion coherence (Voice+Body+Text)  │
    │                                                          │
    │  ADAPTABILITY  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░  74/100       │
    │   └─ transitions, recovery, composure (Fusion)           │
    │                                                          │
    │  ─────────────────────────────────────────               │
    │  OVERALL       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░  73/100       │
    └──────────────────────────────────────────────────────────┘
```

### Scoring Weights

| Dimension | Weight | Source Modalities |
|-----------|--------|-------------------|
| Delivery (Vocal Clarity) | 20% | Voice prosody, voice emotion |
| Body Language | 15% | Body posture, gestures, gaze |
| Content Quality | 20% | Grammar, readability, argument |
| Engagement | 15% | Gaze, voice emotion, content tone |
| Structure | 15% | Content regime flow, transitions |
| Confidence | 15% | Fillers, disfluency, posture, gestures |

---

## Report Generation

The report engine produces self-contained HTML (and optional PDF) reports with embedded charts.

### Report Components

| Component | Description |
|-----------|-------------|
| **Executive Summary** | One-paragraph overview of strengths and areas for improvement |
| **Dimension Radar Chart** | 6-axis radar showing score profile |
| **Timeline Heatmap** | Per-second quality heatmap across all dimensions |
| **Emotion Arc** | 3-line chart (voice, face, text valence over time) |
| **Regime Flow** | Visual timeline of content regime transitions |
| **Coaching Feedback** | Per-segment actionable advice (LLM-generated or templated) |
| **Disruption Log** | Timestamped list of detected disruptions with recovery times |
| **Practice Plan** | Personalized exercises targeting lowest-scoring dimensions |

### Chart Visualizations

```
   Radar Chart                     Emotion Arc
   ───────────                     ───────────
       Vocal                       1.0 ┤    ╱╲
      ╱     ╲                          │   ╱  ╲   ╱╲
   Flow      Body                 0.5 ┤──╱────╲─╱──╲───
     │    ●    │                       │ ╱      ╳    ╲
   Expr.    Content               0.0 ┤╱──────╱╲─────╲
      ╲     ╱                          │      ╱   ╲
       Engage                    -0.5 ┤─────╱─────╲──
                                       └──────────────
                                       0    60   120  180  240  300s
                                       — Voice  ─── Face  ··· Text

   Timeline Heatmap (ASCII)
   ─────────────────────────
   VOCAL  ████▓▓██▓▓░░██████▓▓▓▓████████░░██▓▓████████████████
   BODY   ▓▓▓▓██████▓▓████▓▓██████████████████▓▓██████████████
   PACE   ████████▓▓████████████████▓▓▓▓██████████████▓▓██████
   ENERGY ██████████▓▓▓▓██████████████████████████████████████
          0:00      1:00      2:00      3:00      4:00    5:00
```

---

## Web Application

A **Bloomberg Terminal-inspired** Next.js web application for viewing analysis results.

### Tech Stack

- **Next.js 16** with App Router
- **React 19** with TypeScript 5.9
- **Tailwind CSS 4**
- Dark theme with amber/orange accents (`#FFB000`)

### Views (F-key navigation)

| Key | View | Description |
|-----|------|-------------|
| **F1** | Dashboard | Performance matrix with LED bars, sparkline trends, AI growth suggestions |
| **F2** | Summary | Executive summary, regime timeline, key event log, pitch oscillator |
| **F3** | Timeline | ASCII heatmaps, 3-channel emotion arc SVG, disruption event markers |
| **F4** | Coaching | Per-event coaching cards with priority badges, navigable by timestamp |
| **F5** | Data | Full transcript with highlighted filler words, raw metrics, export buttons |
| **F6** | Video | Video player with waveform, synchronized coaching diagnostics overlay |
| **F7** | Practice | Personalized exercises, milestone tracking, progress indicators |
| **F8** | Upload | Drag-and-drop video upload with processing progress tracker |

### UI Components (14 total)

```
┌──────────────────────────────────────────────────────────────────┐
│  BLOOMBERG CHROME                                [HH:MM:SS]     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ [F1] DASH  [F2] SUMM  [F3] TIME  [F4] COACH  ...      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────┐  ┌───────────────────────────────────────────────┐    │
│  │ NAV  │  │                                               │    │
│  │ PANEL│  │              ACTIVE VIEW                      │    │
│  │      │  │                                               │    │
│  │ F1 ● │  │  ┌────────────────┐  ┌────────────────┐      │    │
│  │ F2   │  │  │ LED Bar:       │  │ LED Bar:       │      │    │
│  │ F3   │  │  │ VOCAL ████░░   │  │ BODY  ███░░░   │      │    │
│  │ F4   │  │  │        7.8     │  │       6.5      │      │    │
│  │ F5   │  │  └────────────────┘  └────────────────┘      │    │
│  │ F6   │  │                                               │    │
│  │ F7   │  │  Sparkline ╱╲╱╲╱╲   Trend: ▲ +0.3           │    │
│  │ F8   │  │                                               │    │
│  └──────┘  └───────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ STATUS BAR    Session: TED Talk Analysis    Score: 73   │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

### Dashboard View Detail

```
┌────────────────────────────────────────────────────────┐
│  PERFORMANCE MATRIX              ┌──────────────────┐  │
│                                  │ AI GROWTH SUGGEST│  │
│  VOCAL  [████████░░] 7.8  ╱╲╱   │                  │  │
│  PACE   [█████████░] 8.2  ╱╲╱   │ ● Reduce filler  │  │
│  TONE   [███████░░░] 6.9  ╱╲╱   │   words by 40%   │  │
│  BODY   [██████░░░░] 6.5  ╱╲╱   │ ● Maintain eye   │  │
│  ENERGY [███████░░░] 7.1  ╱╲╱   │   contact during  │  │
│  FLOW   [███████░░░] 7.4  ╱╲╱   │   transitions    │  │
│                                  │ ● Vary pitch in  │  │
│  ──────────────────────────────  │   conclusion     │  │
│  TEMPORAL INTENSITY HEATMAP      └──────────────────┘  │
│  ▓█▓▓█████▓▓░▓▓████████▓▓███████░░▓▓██████████████    │
│  0:00    1:00    2:00    3:00    4:00    5:00           │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │ SYSTEM ALERTS                                    │  │
│  │ ▸ Filler burst detected at 2:15 (4 in 8s)       │  │
│  │ ▸ Posture drop at 3:42 → recovered in 5s        │  │
│  │ ▸ Emotion mismatch at 4:01 (angry/happy)        │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

---

## API Server

FastAPI backend that handles video uploads, pipeline execution, and report serving.

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload/` | Upload video file + metadata, returns `job_id` |
| `GET` | `/jobs/{job_id}` | Poll job status (queued → running → complete) |
| `GET` | `/sessions/` | List all analysis sessions |
| `GET` | `/sessions/{id}/report` | Fetch completed report JSON |
| `GET` | `/reports/{id}` | Serve HTML/PDF report |
| `POST` | `/enrich/` | LLM enrichment of coaching feedback |
| `GET` | `/uploads/{filename}` | Serve uploaded video files |

### Job Lifecycle

```
Upload ──▶ Queued ──▶ Running ──▶ Complete
                        │
                        ├── Step 1/7: Extracting audio...
                        ├── Step 2/7: Voice analysis...
                        ├── Step 3/7: Body analysis...
                        ├── Step 4/7: Content analysis...
                        ├── Step 5/7: Fusing modalities...
                        ├── Step 6/7: Scoring dimensions...
                        └── Step 7/7: Generating report...
```

### Start the Server

```bash
conda activate voice-pipeline
uvicorn src.api_server:app --host 0.0.0.0 --port 8080
```

---

## Models

### Pretrained Models (no training required)

| Model | Architecture | Size | Source |
|-------|-------------|------|--------|
| Whisper | Encoder-Decoder (small) | 244M | OpenAI |
| Silero VAD | CNN | ~2MB | Silero |
| YOLOv8n | YOLO v8 Nano | 6.5MB | Ultralytics |
| MediaPipe Pose | BlazePose | 5.8MB | Google |
| MediaPipe Hands | HandLandmarker | 7.8MB | Google |
| MediaPipe Face | FaceMesh | 3.7MB | Google |

### Trained Models

| Model | Architecture | Dataset | Size | Metric |
|-------|-------------|---------|------|--------|
| Filler Verifier | MLP (2-layer) | Auto-labeled (regex) | ~200KB | F1 = 0.82 |
| Disfluency | Wav2Vec2-base + head | SEP-28k (Kaggle) | ~95MB | Macro F1 = 0.50 |
| Vocal Emotion | ECAPA-TDNN + MLP | RAVDESS + CREMA-D | ~15MB | UAR = 0.54 |
| Gesture Classifier | Transformer | MediaPipe landmarks | ~5MB | Val F1 = 0.55 |
| Posture MLP | MLP (3-layer) | Rule-based labels | ~2MB | MAE = 1.32 |
| Facial Emotion | EfficientNet-B0 + head | FER2013 | ~17MB | Acc = 0.545 |

> Large model files (>100MB) are stored on Google Drive, not in the repository.

---

## Quick Start

### Prerequisites

- Python 3.10+
- conda (recommended) or virtualenv
- FFmpeg installed (`brew install ffmpeg` on macOS)
- Node.js 18+ (for the web app)

### 1. Python Environment

```bash
# Create and activate conda environment
conda create -n voice-pipeline python=3.10 -y
conda activate voice-pipeline

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Full Pipeline (CLI)

```bash
# Analyze a video
python -m src.master_pipeline --video path/to/speech.mp4

# Run just the voice pipeline
python -m src.pipeline --video path/to/speech.mp4

# Run just the body pipeline
python -m src.body.pipeline --video path/to/speech.mp4
```

Output is saved to `data/outputs/` and `data/reports/`.

### 3. Start the API Server

```bash
uvicorn src.api_server:app --host 0.0.0.0 --port 8080
```

### 4. Start the Web App

```bash
cd webapp
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## Project Structure

```
.
├── src/                           # All source code
│   ├── pipeline.py                # Voice analysis pipeline (Modality 1)
│   ├── audio_extractor.py         # FFmpeg audio extraction
│   ├── vad.py                     # Silero Voice Activity Detection
│   ├── noise_reduction.py         # Spectral noise reduction
│   ├── transcriber.py             # Whisper speech-to-text
│   ├── features.py                # F0, energy, MFCC, voice quality
│   ├── filler_detector.py         # Regex-based filler detection
│   ├── filler_verifier.py         # MLP filler verification
│   ├── disfluency_model.py        # Wav2Vec2 disfluency classifier
│   ├── prosody.py                 # Speaking rate, pitch, volume
│   ├── vocal_emotion.py           # ECAPA-TDNN emotion classifier
│   ├── output_assembler.py        # Voice output JSON assembly
│   │
│   ├── body/                      # Body language pipeline (Modality 2)
│   │   ├── pipeline.py            # Body pipeline orchestrator
│   │   ├── frame_extractor.py     # Video → frames
│   │   ├── person_detector.py     # YOLOv8 person detection
│   │   ├── pose_estimator.py      # MediaPipe 33-point skeleton
│   │   ├── posture_scorer.py      # Rule-based + MLP posture scoring
│   │   ├── gesture_classifier.py  # Transformer gesture classification
│   │   ├── hand_tracker.py        # 8-state hand classification
│   │   ├── gaze_estimator.py      # 6-zone gaze estimation
│   │   ├── facial_emotion.py      # EfficientNet-B0 face emotion
│   │   ├── stage_movement.py      # Movement pattern detection
│   │   ├── temporal_model.py      # 6s window aggregation
│   │   └── output_assembler.py    # Body output JSON assembly
│   │
│   ├── content/                   # Content analysis pipeline (Modality 3)
│   │   ├── pipeline.py            # Content pipeline orchestrator
│   │   ├── transcript_processor.py# Sentence splitting + segmentation
│   │   ├── grammar_checker.py     # LanguageTool integration
│   │   ├── readability_scorer.py  # FKGL, Coleman-Liau
│   │   ├── vocabulary_analyzer.py # TTR, academic word detection
│   │   ├── regime_detector.py     # Semantic boundary detection
│   │   ├── sentiment_analyzer.py  # RoBERTa sentiment
│   │   ├── tone_classifier.py     # LLM tone classification
│   │   ├── argument_analyzer.py   # Argument structure analysis
│   │   └── output_assembler.py    # Content output JSON assembly
│   │
│   ├── fusion/                    # Multimodal fusion engine
│   │   ├── fusion_engine.py       # Main fusion orchestrator
│   │   ├── timeline_aligner.py    # 1-second grid alignment
│   │   ├── regime_transition_scorer.py
│   │   ├── recovery_analyzer.py   # Disruption detection + recovery
│   │   └── emotion_coherence.py   # Cross-modal emotion validation
│   │
│   ├── scoring/                   # Dimension scoring
│   │   └── dimension_scorer.py    # 6-dimension score computation
│   │
│   ├── report/                    # Report generation
│   │   ├── report_builder.py      # HTML/PDF report assembly
│   │   ├── coaching_writer.py     # LLM-based coaching feedback
│   │   ├── chart_generator.py     # Matplotlib chart generation
│   │   └── templates/             # HTML/CSS report templates
│   │
│   ├── master_pipeline.py         # End-to-end orchestrator
│   ├── api_server.py              # FastAPI backend
│   └── report_transformer.py      # Output format conversion
│
├── webapp/                        # Next.js web application
│   └── src/
│       ├── app/                   # Next.js app router
│       │   ├── page.tsx           # Home page
│       │   ├── api/               # API routes (enrich, report)
│       │   └── report/[id]/       # Report page
│       ├── components/            # 14 React components
│       │   ├── BloombergChrome.tsx # Main layout shell
│       │   ├── DashboardView.tsx   # Performance matrix + LED bars
│       │   ├── SummaryView.tsx     # Executive summary
│       │   ├── TimelineView.tsx    # Heatmaps + emotion arc
│       │   ├── CoachingView.tsx    # Per-event coaching
│       │   ├── DataView.tsx        # Transcript + raw data
│       │   ├── VideoView.tsx       # Video player + waveform
│       │   ├── PracticeView.tsx    # Practice exercises
│       │   ├── UploadView.tsx      # Video upload
│       │   ├── ProcessingView.tsx  # Progress tracking
│       │   ├── HistoryView.tsx     # Session history
│       │   ├── SettingsView.tsx    # Configuration
│       │   ├── NavPanel.tsx        # Side navigation
│       │   ├── LEDBar.tsx          # Reusable LED bar
│       │   └── FunctionKeyBar.tsx  # F-key bar
│       └── lib/
│           ├── types.ts           # TypeScript interfaces
│           ├── api.ts             # API client
│           ├── utils.ts           # Helpers
│           └── sample-data.ts     # Development mock data
│
├── configs/                       # Configuration files
│   ├── pipeline_config.yaml       # Voice pipeline config
│   ├── body_pipeline_config.yaml  # Body pipeline config
│   ├── content_pipeline_config.yaml
│   └── fusion_config.yaml         # Fusion + scoring + report config
│
├── models/                        # Trained model checkpoints
│   ├── disfluency/                # Wav2Vec2 checkpoint
│   ├── vocal_emotion/             # ECAPA-TDNN checkpoint
│   ├── gesture_transformer/       # Gesture model
│   ├── posture_mlp/               # Posture scoring model
│   ├── facial_emotion/            # EfficientNet-B0 model
│   ├── filler_verifier/           # Optional filler MLP
│   ├── pose_landmarker_lite.task  # MediaPipe Pose
│   ├── hand_landmarker.task       # MediaPipe Hands
│   └── face_landmarker.task       # MediaPipe Face
│
├── training/                      # Model training scripts
│   ├── train_disfluency.py
│   ├── train_vocal_emotion.py
│   ├── train_filler_verifier.py
│   ├── colab_disfluency.ipynb
│   ├── colab_vocal_emotion.ipynb
│   └── modality2/                 # Body model training
│       ├── train_gesture_transformer.py
│       ├── train_posture_mlp.py
│       ├── train_facial_emotion.py
│       ├── colab_gesture_transformer.ipynb
│       └── colab_facial_emotion.ipynb
│
├── tests/                         # Test suite (150+ tests)
│   ├── test_*.py                  # Voice pipeline tests
│   ├── modality2/                 # Body language tests
│   ├── modality3/                 # Content analysis tests
│   └── fusion/                    # Fusion integration tests
│
├── data/
│   ├── outputs/                   # Pipeline output JSONs
│   ├── raw/                       # Input videos
│   ├── uploads/                   # User uploads
│   ├── sessions/                  # Session tracking
│   └── resources/                 # Academic Word List, etc.
│
├── prompts/                       # Phase-based build prompts
├── docs/                          # Validation reports
├── notebooks/                     # Jupyter/Colab notebooks
├── requirements.txt               # Python dependencies
└── setup.py                       # Package setup
```

---

## Configuration

All pipeline behavior is controlled via YAML config files in `configs/`.

### Voice Pipeline (`pipeline_config.yaml`)

```yaml
whisper:
  model_id: "openai/whisper-small"
  language: "en"

filler:
  patterns:
    hesitation: ["um", "uh", "erm", "ah"]
    discourse: ["like", "you know", "basically"]

prosody:
  optimal_rate_range: [3.5, 5.0]  # syllables/sec

emotion:
  classes: ["Neutral", "Enthusiastic", "Nervous", "Angry", "Sad"]
```

### Fusion (`fusion_config.yaml`)

```yaml
timeline:
  resolution_sec: 1          # 1-second grid
  voice_window_sec: 2.0
  body_window_sec: 6.0

fusion:
  transition_context_sec: 5  # Context around regime transitions
  recovery_window_sec: 10    # Recovery detection window
  coherence_window_sec: 5    # Emotion coherence window

scoring:
  dimensions:
    delivery:    { weight: 0.20, sources: [voice_prosody, voice_emotion] }
    body_language: { weight: 0.15, sources: [body_posture, body_gesture, body_gaze] }
    content_quality: { weight: 0.20, sources: [content_grammar, content_readability] }
    engagement:  { weight: 0.15, sources: [body_gaze, voice_emotion, content_tone] }
    structure:   { weight: 0.15, sources: [content_regime, content_transitions] }
    confidence:  { weight: 0.15, sources: [voice_fillers, voice_disfluency, body_posture] }
```

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run by modality
python -m pytest tests/test_*.py -v              # Voice (58 tests)
python -m pytest tests/modality2/ -v              # Body (95 tests)
python -m pytest tests/modality3/ -v              # Content
python -m pytest tests/fusion/ -v                 # Fusion

# Run with coverage
python -m pytest tests/ -v --cov=src
```

### Validation Results

Tested against three 5-minute speaker profiles on Google Colab (T4 GPU):

| Speaker Profile | Fillers | Disfluencies | Dominant Emotion | Posture | Movement |
|----------------|---------|-------------|-----------------|---------|----------|
| Good Speaker | 4 | 15 | Neutral | 64.7 | Roaming (7.0) |
| Nervous Speaker | 12 | 71 | Nervous | 48.2 | Pacing (3.0) |
| Monotone Speaker | 2 | 8 | Neutral | 72.1 | Purposeful (8.0) |

---

## Training

All models were trained using Google Colab T4 GPUs. Training scripts are in `training/` with corresponding `.ipynb` notebooks for Colab.

### Train Disfluency Model

```bash
# Local (MPS/CUDA)
python training/train_disfluency.py

# Or use the Colab notebook
# training/colab_disfluency.ipynb
```

### Train Vocal Emotion Model

```bash
python training/train_vocal_emotion.py
```

### Train Body Language Models

```bash
python training/modality2/train_gesture_transformer.py
python training/modality2/train_posture_mlp.py
python training/modality2/train_facial_emotion.py
```

### Datasets

| Model | Dataset | Size |
|-------|---------|------|
| Disfluency | SEP-28k (Kaggle) | ~28,000 clips |
| Vocal Emotion | RAVDESS + CREMA-D | ~8,000 clips |
| Gesture | Auto-labeled from MediaPipe | ~5,000 sequences |
| Facial Emotion | FER2013 | ~35,000 images |
| Posture | Rule-based auto-labels | ~3,000 frames |

---

## Hardware Requirements

| Environment | Specs | Processing Time (5-min video) |
|-------------|-------|------------------------------|
| **Apple M1 Mac** | 16GB RAM, MPS GPU | ~4 min |
| **Google Colab** | T4 GPU, 12GB VRAM | ~2.5 min |
| **CPU only** | Any modern CPU | ~12 min |

The pipeline automatically selects the best available device (`cuda` → `mps` → `cpu`).

---

## Known Limitations

- **Gaze engagement** scores low when the speaker's face is small in wide-angle stage shots
- **Vocal emotion** model can misclassify high vocal energy as "Angry" instead of "Enthusiastic"
- **Grammar checking** requires LanguageTool server running (gracefully skips if unavailable)
- **Tone/argument analysis** requires Claude API key for LLM-based evaluation (falls back to templates)

---

## License

MIT
