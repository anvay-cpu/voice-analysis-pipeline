PHASE 8 — FACIAL EMOTION RECOGNITION (EFFICIENTNET-B0)
========================================================

AGENT ROLE: Facial Emotion Specialist
DEPENDS ON: Phase 2 (frames, face crops)
DELIVERS TO: Phase 10 (output assembly)
ESTIMATED TIME: 25 min (agent code) + 20 min (user download) + 3-4 hrs (Colab)

OBJECTIVE:
Build facial emotion classifier using EfficientNet-B0 (5.3M params)
fine-tuned on AffectNet or FER+. Trains on Colab T4.

═══════════════════════════════════════════════════════════════
MODEL ARCHITECTURE
═══════════════════════════════════════════════════════════════

Base: EfficientNet-B0 (pretrained on ImageNet)
Replace final FC layer: 1280 → 256 → 8 classes + 2 (valence, arousal)

8 discrete classes: Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt
2 continuous: valence [-1, 1], arousal [-1, 1]

Multi-task loss:
  L = L_CE(discrete) + 0.5 * L_CCC(valence) + 0.5 * L_CCC(arousal)

where CCC = concordance correlation coefficient loss

Input: 224×224 face crop (RGB)
Parameters: 5.3M
Training: 3-4 hours on Colab T4 with FP16

═══════════════════════════════════════════════════════════════
DATASET OPTIONS
═══════════════════════════════════════════════════════════════

OPTION A: FER+ (easiest — available on Kaggle)
  - 35,887 grayscale 48×48 face images
  - 8 emotion classes with crowd-sourced labels (10 annotators per image)
  - Download: kaggle datasets download -d msambare/fer2013
  - Upside: Easy to download, well-studied
  - Downside: Low resolution (48×48), grayscale, some noisy labels

OPTION B: AffectNet (best quality — requires registration)
  - 450K labeled face images (in the wild)
  - 8 discrete emotions + valence/arousal continuous labels
  - Download: http://mohammadmahoor.com/affectnet/ (academic registration)
  - Upside: High quality, large, includes valence/arousal
  - Downside: Registration takes 1-3 days

RECOMMENDATION: Start with FER+ (instant download).
Upgrade to AffectNet later for better production quality.

═══════════════════════════════════════════════════════════════
BRIDGE AGENT INTERVENTION
═══════════════════════════════════════════════════════════════

🔔 USER INPUT REQUIRED — Phase 8: Facial Emotion

WHAT IS HAPPENING:
The Facial Emotion Agent needs a labeled face dataset to fine-tune
EfficientNet-B0.

WHAT YOU NEED TO DO:

Option A — FER+ (fastest, ~5 minutes):
  1. Go to: https://www.kaggle.com/datasets/msambare/fer2013
  2. Download fer2013.csv (~90MB)
  3. Place at: ~/Desktop/Claude-assistant/data/datasets/fer_plus/

Option B — AffectNet (better quality, takes days for approval):
  1. Register at: http://mohammadmahoor.com/affectnet/
  2. Download when approved
  3. Place at: ~/Desktop/Claude-assistant/data/datasets/affectnet/

TRAINING PLATFORM: Google Colab (T4 GPU with FP16)
  The model is 5.3M params — trainable on M1 but Colab is 4x faster.
  Colab time: 3-4 hours. M1 time: 18-24 hours.

COLAB CELLS PROVIDED in training/modality2/train_facial_emotion.py

═══════════════════════════════════════════════════════════════
TRAINING DETAILS
═══════════════════════════════════════════════════════════════

Build training/modality2/train_facial_emotion.py:

- FERPlusDataset: parse CSV, convert 48×48 grayscale → 224×224 RGB
  (resize + convert to 3-channel by repeating)
- Use torchvision.models.efficientnet_b0(pretrained=True)
- Replace classifier head: 1280 → 256 → 8
- Freeze first 5 of 7 blocks, fine-tune last 2 + head
- AdamW: 1e-5 encoder, 5e-4 head
- Data augmentation: random horizontal flip, rotation ±15°,
  color jitter, random erasing
- FP16 on Colab
- 15 epochs, patience 4

Target: Accuracy ≥ 0.60 on FER+ test set
        (state-of-art is ~0.65, so 0.60 is realistic)

COMPLETION: ✓ when models/facial_emotion/best_model.pt exists
            and test accuracy ≥ 0.60.