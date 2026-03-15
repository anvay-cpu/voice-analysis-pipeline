PHASE 5 — FILLER DETECTION AGENT
===================================

AGENT ROLE: Filler Detection Specialist
DEPENDS ON: Phase 3 (transcript), Phase 4 (features)
DELIVERS TO: Phase 9 (output assembly)
ESTIMATED TIME: 30 min (agent code) + 2 hours (user labeling) + 1 hour (training)

OBJECTIVE:
Build the two-stage filler detection system:
  Stage 1: Regex-based transcript matching (no training)
  Stage 2: Acoustic verification MLP (REQUIRES TRAINING on custom data)

═══════════════════════════════════════════════════════════════
TASK 1: Build src/filler_detector.py (Stage 1 — no training)
═══════════════════════════════════════════════════════════════

Build the regex-based filler finder:
- FillerDetection dataclass: word, start_sec, end_sec, confidence, category
- FILLER_PATTERNS dict with "hesitation" and "discourse" categories
- detect_fillers_from_transcript(whisper_output) → list[FillerDetection]
- compute_filler_rate(fillers, duration, word_count) → stats dict
  with fillers_per_minute, filler_percentage, counts by category

═══════════════════════════════════════════════════════════════
TASK 2: Build src/filler_verifier.py (Stage 2 — needs training)
═══════════════════════════════════════════════════════════════

Build the MLP verifier model:
- FillerVerifierMLP class: 768 → 256 → 64 → 1 with BatchNorm + Dropout
- load_filler_verifier(model_path) → model
- verify_fillers(model, candidates, whisper_model) → verified list
  Only candidates with P(filler) > 0.5 pass verification

═══════════════════════════════════════════════════════════════
TASK 3: Build training/train_filler_verifier.py
═══════════════════════════════════════════════════════════════

Build the complete training script with:
- Dataset class for pre-extracted Whisper embeddings
- Pre-extraction function (runs Whisper encoder on audio clips)
- Training loop with BCEWithLogitsLoss, early stopping, checkpointing
- Evaluation with F1, precision, recall
- CLI interface with --preextract and training modes

═══════════════════════════════════════════════════════════════
TASK 4: Build dataset creation helper
═══════════════════════════════════════════════════════════════

Build training/create_filler_dataset.py that:
1. Takes a list of audio files
2. Runs Whisper to get word-level timestamps
3. Finds all candidate filler words
4. Outputs a JSON file with candidates labeled is_filler=-1 (unlabeled)
5. Prints instructions for the user on how to label them

═══════════════════════════════════════════════════════════════
BRIDGE AGENT INTERVENTION POINT
═══════════════════════════════════════════════════════════════

🔔 USER INPUT REQUIRED — Phase 5: Filler Detection

WHAT IS HAPPENING:
The Filler Detection Agent has built all the code and a dataset creation
helper. Now it needs labeled training data to train the verification MLP.

WHY YOUR INPUT IS NEEDED:
The filler verifier needs to learn the difference between "like" as a
filler vs "like" as a verb. This requires YOU to listen to audio clips
and label each one. No public dataset exists for this specific task.

WHAT YOU NEED TO DO:

Step 1: Find 10+ speech videos (TED talks, presentations, lectures)
        Download audio using:
        yt-dlp -x --audio-format wav "VIDEO_URL"
        Place WAV files in: ~/Desktop/Claude-assistant/data/raw/

Step 2: Run the dataset creation helper:
        cd ~/Desktop/Claude-assistant
        conda activate voice-pipeline
        python training/create_filler_dataset.py \
            --audio_dir data/raw/ \
            --output data/filler_candidates.json

Step 3: Open data/filler_candidates.json in a text editor
        For each sample, listen to the audio context and set:
          "is_filler": 1    ← if the word IS a filler
          "is_filler": 0    ← if the word is NOT a filler
        Remove any samples where "is_filler" is still -1

        Target: 250 fillers (is_filler=1) + 250 non-fillers (is_filler=0)
        This takes approximately 2 hours of manual work.

Step 4: Save the labeled file at:
        ~/Desktop/Claude-assistant/data/filler_labels.json

FORMAT REQUIRED:
```json
{
  "samples": [
    {
      "audio_path": "data/raw/ted_talk_1.wav",
      "word": "like",
      "start_sec": 42.3,
      "end_sec": 42.8,
      "is_filler": 1,
      "context": "I was like totally confused"
    }
  ]
}
```

WHAT HAPPENS NEXT:
Once the labeled JSON exists, the agent will:
  1. Pre-extract Whisper embeddings (~20 min on M1)
  2. Train the MLP (~1 hour on M1)
  3. Evaluate on test split
  4. Save model to models/filler_verifier/best_model.pt
  5. Update PROGRESS.md

WHILE WAITING FOR USER:
The Bridge Agent should SKIP to Phase 6 and Phase 7 (which don't depend
on this phase) and return here when the user provides the labeled data.

═══════════════════════════════════════════════════════════════
TASK 5: Train the model (after user provides data)
═══════════════════════════════════════════════════════════════

Execute training:
  python training/train_filler_verifier.py --preextract \
      --data data/filler_labels.json \
      --embeddings_dir data/filler_embeddings \
      --device mps

  python training/train_filler_verifier.py \
      --embeddings_dir data/filler_embeddings \
      --device mps --epochs 50 --batch_size 64

Verify: F1 ≥ 0.90 on test set.

COMPLETION: ✓ when models/filler_verifier/best_model.pt exists
            and test F1 ≥ 0.90.
