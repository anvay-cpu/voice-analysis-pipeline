PHASE 8 — VOCAL EMOTION TRAINING AGENT
=========================================

AGENT ROLE: Emotion Recognition Specialist
DEPENDS ON: Phase 1 (environment)
DELIVERS TO: Phase 9 (output assembly)
ESTIMATED TIME: 30 min (agent code) + 30 min (user downloads) + 7 hours (training)

OBJECTIVE:
Build the vocal emotion classifier (ECAPA-TDNN + MLP head) with 2-phase
training on RAVDESS + CREMA-D datasets. Trains on M1.

═══════════════════════════════════════════════════════════════
TASK 1: Build src/vocal_emotion.py
═══════════════════════════════════════════════════════════════

Build VocalEmotionClassifier class:
- ECAPA-TDNN encoder from SpeechBrain (pretrained, 192-dim output)
- Emotion head: 192 → 128 → 5 classes
- predict(audio) → {"label", "confidence", "distribution"}
- 5 classes: Neutral, Enthusiastic, Nervous, Angry, Sad
- load_emotion_model(model_path) → model

═══════════════════════════════════════════════════════════════
TASK 2: Build training/train_vocal_emotion.py
═══════════════════════════════════════════════════════════════

Complete training script with:
- RAVDESSDataset: parse filenames for emotion codes, map to 5 classes
- CREMADDataset: parse filenames for emotion codes, map to 5 classes
- IEMOCAPDataset: (optional) parse evaluation files
- Embedding pre-extraction for Phase 1 (train head on frozen embeddings)
- Phase 1: head-only training (~30 min), AdamW lr=1e-3
- Phase 2: fine-tune top TDNN layers + head (~6 hrs), differential LR
- Focal loss or weighted CrossEntropy for class imbalance
- UAR (Unweighted Average Recall) as primary metric
- CLI: --phase all|extract|phase1|phase2

═══════════════════════════════════════════════════════════════
BRIDGE AGENT INTERVENTION POINT
═══════════════════════════════════════════════════════════════

🔔 USER INPUT REQUIRED — Phase 8: Vocal Emotion

WHAT IS HAPPENING:
The Emotion Agent has built the model and training scripts.
It needs you to download two emotion speech datasets.

WHY YOUR INPUT IS NEEDED:
RAVDESS and CREMA-D are hosted externally. You need to download
and extract them to the correct folders.

WHAT YOU NEED TO DO:

Step 1: Download RAVDESS
  Go to: https://zenodo.org/record/1188976
  Download: "Audio_Speech_Actors_01-24.zip" (~200MB)
  Extract to: ~/Desktop/Claude-assistant/data/datasets/ravdess/
  You should see folders: Actor_01/, Actor_02/, ... Actor_24/

Step 2: Download CREMA-D
  Go to: https://github.com/CheyneyComputerScience/CREMA-D
  Download the AudioWAV folder (~1.5GB)
  Extract to: ~/Desktop/Claude-assistant/data/datasets/cremad/
  You should see files: 1001_DFA_ANG_XX.wav, 1001_DFA_DIS_XX.wav, ...

Step 3: Verify:
  ls data/datasets/ravdess/Actor_01/ | head -5
  ls data/datasets/cremad/*.wav | wc -l  (should show ~7442)

WHAT HAPPENS NEXT:
The agent will run the full training pipeline:
  Phase 0: Pre-extract embeddings (~15 min)
  Phase 1: Train head only (~30 min)
  Phase 2: Fine-tune encoder (~6 hours)
  Total: ~7 hours (can run overnight on M1)

═══════════════════════════════════════════════════════════════
TASK 3: Execute training (after user downloads data)
═══════════════════════════════════════════════════════════════

Run: python training/train_vocal_emotion.py \
         --phase all \
         --ravdess_dir data/datasets/ravdess \
         --cremad_dir data/datasets/cremad \
         --device mps

Verify: UAR ≥ 0.58 on test set.

COMPLETION: ✓ when models/vocal_emotion/best_model.pt exists
            and test UAR ≥ 0.58.
