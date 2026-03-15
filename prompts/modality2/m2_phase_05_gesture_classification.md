PHASE 5 — GESTURE CLASSIFICATION (TRANSFORMER)
================================================

AGENT ROLE: Gesture Recognition Specialist
DEPENDS ON: Phase 3 (pose keypoints)
DELIVERS TO: Phase 10 (output assembly)
ESTIMATED TIME: 30 min (agent code) + 1 hr (user data) + 12-18 hrs (M1 training)

OBJECTIVE:
Build a Transformer encoder that classifies gesture types from
sequences of pose keypoints.

═══════════════════════════════════════════════════════════════
MODEL ARCHITECTURE
═══════════════════════════════════════════════════════════════

Input: Sliding window of 15 frames (3 seconds at 5fps)
       Each frame = 33 keypoints × 4 = 132 dimensions
       Total input: (15, 132)

Transformer Encoder:
  - Positional encoding (learnable)
  - 4 layers, 4 attention heads
  - Hidden dim 256, FFN dim 512
  - Dropout 0.1

Classification head:
  - Mean pool across time → 256-dim
  - Linear 256 → 5 classes

5 GESTURE CLASSES:
  0 = Illustrator   → Accompanying speech (pointing, showing size)
  1 = Emblem        → Culturally defined (thumbs up, open palm)
  2 = Beat          → Rhythmic hand movements for emphasis
  3 = Adaptor       → Self-touching (face, hair, clothing) — nervousness
  4 = Rest          → Hands at sides, on podium, clasped

Parameters: ~3.8M
Loss: CrossEntropyLoss with class weights
Training time: 12-18 hours on M1, ~3 hours on Colab

═══════════════════════════════════════════════════════════════
TRAINING DATA STRATEGY
═══════════════════════════════════════════════════════════════

Option A: SEMI-AUTOMATIC LABELING (recommended)
1. Use the 10 WAV files from Modality 1 — you need the VIDEOS, not audio.
   User must re-download as MP4 (not audio-only).
2. Extract pose sequences from all videos
3. Apply heuristic pre-labeling:
   - Upper body movement > threshold → Illustrator
   - Hands near face/hair → Adaptor
   - Repetitive symmetric hand motion → Beat
   - Low hand movement, hands below waist → Rest
   - Specific hand shapes (open palm, thumbs up) → Emblem
4. User reviews/corrects ~20% of pre-labels
5. Expected: ~2000 labeled windows from 70 minutes of video

Option B: NTU RGB+D SUBSET (no manual labeling)
1. Download gesture-relevant classes from NTU RGB+D 120
   (actions 6, 7, 8, 27, 28, 29 = waving, clapping, pointing, etc.)
2. Map NTU actions to our 5 gesture classes
3. Convert NTU skeleton format to MediaPipe format
Downside: NTU is filmed in a lab, not real speeches.
Upside: Zero manual labeling, thousands of samples.

RECOMMENDATION: Start with Option A (real speech data)
               for higher relevance. Add Option B later if needed.

═══════════════════════════════════════════════════════════════
BRIDGE AGENT INTERVENTION
═══════════════════════════════════════════════════════════════

🔔 USER INPUT REQUIRED — Phase 5: Gesture Classification

WHAT IS HAPPENING:
The Gesture Agent needs video files (MP4) of speeches to extract
pose sequences and create a gesture training dataset.

WHY YOUR INPUT IS NEEDED:
In Modality 1 you downloaded audio-only WAV files. For gesture
analysis we need the VIDEO with visual content.

WHAT YOU NEED TO DO:
Step 1: Re-download your 10 speech clips as VIDEO (MP4):
  yt-dlp -f "bestvideo[height<=720]+bestaudio" --merge-output-format mp4 \
      -o "data/raw/videos/%(title)s.%(ext)s" \
      "URL1" "URL2" ... (same 10 URLs from Modality 1)

Step 2: Place MP4 files in: ~/Desktop/Claude-assistant/data/raw/videos/

Step 3: The agent will:
  - Extract frames + poses from all 10 videos
  - Auto-label gesture windows using heuristics
  - Ask you to review ~400 uncertain labels (optional, ~30 min)
  - Train the Transformer

ESTIMATED USER TIME: 15 min (download) + optional 30 min (label review)
TRAINING TIME ON M1: 12-18 hours (run overnight)

═══════════════════════════════════════════════════════════════
TRAINING SCRIPT
═══════════════════════════════════════════════════════════════

Build training/modality2/train_gesture_transformer.py:
- GestureDataset class: sliding windows of pose keypoints
- Auto-labeling heuristics for gesture classes
- Transformer encoder model
- CrossEntropyLoss with inverse-frequency class weights
- AdamW lr=5e-4, cosine schedule, 30 epochs
- Augmentation: random joint noise, temporal jitter, horizontal flip
- Save best model by validation F1

Target: F1 ≥ 0.65 (gesture classification is inherently noisy)

COMPLETION: ✓ when models/gesture_transformer/best_model.pt exists.