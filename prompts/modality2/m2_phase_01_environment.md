PHASE 1 — ENVIRONMENT & DEPENDENCIES (MODALITY 2)
====================================================

AGENT ROLE: Environment Architect
DEPENDS ON: Modality 1 environment (conda env voice-pipeline already exists)
DELIVERS TO: All subsequent Modality 2 phases
ESTIMATED TIME: 10 minutes (agent) + 3 minutes (user)

OBJECTIVE:
Extend the existing project environment with video processing dependencies.
Create Modality 2 directory structure and config files.

═══════════════════════════════════════════════════════════════
TASK 1: Install Additional Dependencies
═══════════════════════════════════════════════════════════════

Add to requirements.txt and install:

```
# Video processing (Modality 2)
opencv-python>=4.8.0
mediapipe>=0.10.9
ultralytics>=8.1.0        # YOLOv8

# Image processing
Pillow>=10.0.0

# Already installed from Modality 1 (verify):
# torch, torchvision, numpy, scipy, tqdm, pyyaml
```

Install command:
```bash
conda activate voice-pipeline
pip install opencv-python mediapipe ultralytics Pillow
```

═══════════════════════════════════════════════════════════════
TASK 2: Create Directory Structure
═══════════════════════════════════════════════════════════════

```
src/body/
├── __init__.py
├── frame_extractor.py      # Step 1: Video → frames
├── person_detector.py      # Step 2: Detect + track speaker
├── pose_estimator.py       # Step 3: 33-point skeleton
├── posture_scorer.py       # Step 4: Posture quality
├── gesture_classifier.py   # Step 5: Gesture types
├── hand_tracker.py         # Step 6: Hand states
├── gaze_estimator.py       # Step 7: Gaze direction
├── facial_emotion.py       # Step 8: Face emotion
├── stage_movement.py       # Step 9: Proxemics
├── temporal_model.py       # Step 10: Frame→segment aggregation
├── output_assembler.py     # Step 11: JSON output
└── pipeline.py             # Master orchestrator

training/modality2/
├── train_posture_mlp.py
├── train_gesture_transformer.py
├── train_facial_emotion.py
├── create_gesture_dataset.py
└── configs/
    ├── posture.yaml
    ├── gesture.yaml
    └── facial_emotion.yaml

models/
├── posture_mlp/
├── gesture_transformer/
└── facial_emotion/

tests/modality2/
├── test_frame_extraction.py
├── test_pose.py
├── test_posture.py
├── test_gesture.py
├── test_facial_emotion.py
└── test_integration.py
```

═══════════════════════════════════════════════════════════════
TASK 3: Create Modality 2 Config
═══════════════════════════════════════════════════════════════

Write configs/body_pipeline_config.yaml:

```yaml
body_pipeline:
  name: "body_analysis_v1"
  version: "1.0.0"
  device: "mps"

video:
  frame_rate: 5              # Extract at 5 fps
  resolution: [1280, 720]    # Resize to 720p
  max_duration_sec: 3600

person_detection:
  model: "yolov8n"           # YOLOv8-nano
  confidence_threshold: 0.5
  select_strategy: "largest_bbox"  # Assume largest person is speaker

pose:
  model: "mediapipe"
  num_keypoints: 33
  min_detection_confidence: 0.5
  min_tracking_confidence: 0.5

posture:
  model_path: "models/posture_mlp/best_model.pt"
  shoulder_angle_threshold_deg: 5.0
  spine_deviation_threshold_deg: 10.0
  weight_ratio_range: [0.85, 1.15]
  rule_weight: 0.3           # lambda in hybrid scoring

gesture:
  model_path: "models/gesture_transformer/best_model.pt"
  window_frames: 15          # 3 seconds at 5fps
  hop_frames: 8              # ~1.6 second hop
  classes: ["Illustrator", "Emblem", "Beat", "Adaptor", "Rest"]

hand:
  model: "mediapipe"
  min_detection_confidence: 0.5
  states: ["Open palm", "Pointing", "Crossed arms",
           "Hands in pockets", "Steepling", "Behind back", "Other"]

gaze:
  model: "mediapipe_face_mesh"
  zones:
    audience_center: {yaw_range: [-20, 20], pitch_range: [-15, 15]}
    audience_left: {yaw_range: [-60, -20], pitch_range: [-15, 15]}
    audience_right: {yaw_range: [20, 60], pitch_range: [-15, 15]}
    notes_podium: {pitch_range: [25, 60]}
    floor: {pitch_range: [40, 90]}

facial_emotion:
  model_path: "models/facial_emotion/best_model.pt"
  classes: ["Neutral", "Happy", "Sad", "Surprise", "Fear",
            "Disgust", "Anger", "Contempt"]
  face_crop_margin: 0.2

stage_movement:
  frozen_threshold_px_per_sec: 2.0
  pacing_autocorrelation_threshold: 0.6
  classifications: ["Roaming", "Anchored", "Pacing", "Purposeful"]

temporal:
  window_frames: 30          # 6 seconds at 5fps
  hop_frames: 15             # 3-second hop (50% overlap)
  smoothing_kernel: 5

output:
  window_sec: 6.0
  hop_sec: 3.0
  output_dir: "data/outputs"
```

═══════════════════════════════════════════════════════════════
TASK 4: Initialize Progress Tracker
═══════════════════════════════════════════════════════════════

Write PROGRESS_M2.md:

```markdown
# Body Language Pipeline (Modality 2) Build Progress

## Status
- [ ] Phase 1: Environment Setup
- [ ] Phase 2: Video Preprocessing
- [ ] Phase 3: Pose Estimation
- [ ] Phase 4: Posture Scoring
- [ ] Phase 5: Gesture Classification
- [ ] Phase 6: Hand Tracking
- [ ] Phase 7: Gaze Estimation
- [ ] Phase 8: Facial Emotion
- [ ] Phase 9: Stage Movement
- [ ] Phase 10: Temporal Modeling & Assembly
- [ ] Phase 11: Testing & Validation

## Models
| Model | Status | Metric | Target | Pass? |
|-------|--------|--------|--------|-------|
| Posture MLP | — | — | MAE≤0.6 | — |
| Gesture Transformer | — | — | F1≥0.65 | — |
| Facial Emotion | — | — | Acc≥0.60 | — |
```

═══════════════════════════════════════════════════════════════
BRIDGE AGENT INTERVENTION
═══════════════════════════════════════════════════════════════

🔔 USER INPUT REQUIRED — Phase 1: Environment

WHAT IS HAPPENING:
The agent has created all directories and config files.
You need to install the new video processing packages.

WHAT YOU NEED TO DO:
```bash
conda activate voice-pipeline
pip install opencv-python mediapipe ultralytics Pillow
python -c "import cv2; import mediapipe; print('OK')"
```

WHAT HAPPENS NEXT:
Phase 2 starts immediately (no more user input until Phase 5).

COMPLETION: ✓ when cv2, mediapipe, ultralytics all import without error.
