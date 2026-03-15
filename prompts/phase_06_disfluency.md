PHASE 6 — DISFLUENCY DETECTION AGENT
=======================================

AGENT ROLE: Disfluency Model Trainer
DEPENDS ON: Phase 1 (environment), Phase 2 (audio preprocessing)
DELIVERS TO: Phase 9 (output assembly)
ESTIMATED TIME: 30 min (agent code) + 30 min (user Colab setup) + 4 hours (training)

OBJECTIVE:
Build the disfluency detection model (Wav2Vec2-base fine-tuned for stuttering)
and its training script. This model TRAINS ON COLAB due to size (95M params).

═══════════════════════════════════════════════════════════════
TASK 1: Build src/disfluency_model.py
═══════════════════════════════════════════════════════════════

Build the DisfluencyDetector class:
- Wav2Vec2Model encoder with first 3 layers frozen
- 2-layer MLP classifier (768 → 256 → 5)
- forward(input_values) → logits
- 5 classes: Fluent, Repetition, Prolongation, Block, Interjection

Build inference functions:
- load_disfluency_model(model_path, device) → (model, processor)
- predict_disfluency(model, processor, audio, device) → list of events
  Uses sliding window (3s window, 1.5s hop)
  Returns: [{"start_sec", "end_sec", "type", "confidence"}, ...]

═══════════════════════════════════════════════════════════════
TASK 2: Build training/train_disfluency.py
═══════════════════════════════════════════════════════════════

Complete training script with:
- DisfluencyDataset class for SEP-28k CSV format
- Audio augmentation (speed perturbation, noise injection, pitch shift)
- Weighted CrossEntropyLoss with class weights [1.0, 6.5, 8.0, 12.0, 5.5]
- Differential learning rates: 2e-5 encoder, 1e-3 head
- Cosine warmup scheduler
- FP16 mixed precision support (for Colab T4)
- Gradient accumulation (2 steps Colab, 4 steps M1)
- Per-class F1 and Macro F1 evaluation
- Checkpoint saving every epoch
- CLI: --platform m1|colab

═══════════════════════════════════════════════════════════════
TASK 3: Build Colab notebook template
═══════════════════════════════════════════════════════════════

Create training/colab_disfluency.py (a single-file version for Colab):
All code in one file so the user can copy-paste into a Colab notebook.
Include: install commands, Drive mount, dataset download, training, save.

═══════════════════════════════════════════════════════════════
BRIDGE AGENT INTERVENTION POINT
═══════════════════════════════════════════════════════════════

🔔 USER INPUT REQUIRED — Phase 6: Disfluency Detection

WHAT IS HAPPENING:
The Disfluency Agent has built the model code and training scripts.
Training must happen on Google Colab because the model is too large
for efficient M1 training (36 hours M1 vs 4 hours Colab).

WHY YOUR INPUT IS NEEDED:
You need to open Google Colab, upload the script, download the
SEP-28k dataset, and run the training cells.

WHAT YOU NEED TO DO:

Step 1: Go to https://colab.research.google.com
        Create a new notebook
        Set runtime: Runtime → Change runtime type → T4 GPU

Step 2: In Cell 1, install dependencies:
        !pip install transformers datasets soundfile accelerate

Step 3: Mount Google Drive:
        from google.colab import drive
        drive.mount('/content/drive')

Step 4: Download SEP-28k dataset:
        !git clone https://github.com/apple/ml-stuttering-events-dataset.git
        !ls ml-stuttering-events-dataset/SEP-28k/

Step 5: Upload training/colab_disfluency.py to Colab
        OR copy-paste its contents into a cell

Step 6: Run training cell (takes ~4 hours):
        !python colab_disfluency.py \
            --audio_dir /content/ml-stuttering-events-dataset/SEP-28k/clips \
            --labels_csv /content/ml-stuttering-events-dataset/SEP-28k/SEP-28k_episodes.csv \
            --epochs 15

Step 7: After training completes, save to Drive:
        !cp models/disfluency/best_model.pt \
            "/content/drive/MyDrive/voice_pipeline_models/"

Step 8: Download best_model.pt from Google Drive to your Mac:
        Place it at: ~/Desktop/Claude-assistant/models/disfluency/best_model.pt

COLAB SURVIVAL TIPS:
- Keep browser tab OPEN and ACTIVE during 4-hour training
- Model checkpoints save every epoch automatically
- If disconnected, resume from last checkpoint
- Verify GPU: !nvidia-smi (should show Tesla T4)

WHAT HAPPENS NEXT:
Once best_model.pt is at models/disfluency/best_model.pt,
the agent will verify it loads correctly and run inference test.

ALTERNATIVE (if Colab unavailable):
Train on M1 with: python training/train_disfluency.py --platform m1
This takes ~36 hours. Start Friday evening, check Monday.

═══════════════════════════════════════════════════════════════
TASK 4: Verify trained model (after user provides .pt file)
═══════════════════════════════════════════════════════════════

Write tests/test_disfluency.py:
- Load model from models/disfluency/best_model.pt
- Run inference on a short audio clip
- Verify output format: list of disfluency events with timestamps
- Verify model loads without errors on M1 MPS device

COMPLETION: ✓ when models/disfluency/best_model.pt exists,
            loads on M1, and inference produces valid output.
