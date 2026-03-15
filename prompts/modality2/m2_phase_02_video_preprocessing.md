PHASE 2 — VIDEO PREPROCESSING (FRAME EXTRACTION + PERSON DETECTION)
=====================================================================

AGENT ROLE: Video Engineer
DEPENDS ON: Phase 1 (environment ready)
DELIVERS TO: Phase 3-9 (all downstream body analysis)
ESTIMATED TIME: 25 minutes (agent), 0 minutes (user)

OBJECTIVE:
Build frame extraction from video and person detection/tracking.
No training needed — uses OpenCV and pretrained YOLOv8-nano.

═══════════════════════════════════════════════════════════════
TASK 1: Build src/body/frame_extractor.py
═══════════════════════════════════════════════════════════════

REQUIREMENTS:
- Accept MP4, WebM, AVI, MOV video files
- Extract frames at configurable fps (default 5 fps)
- Resize to target resolution (default 1280x720)
- Save frames as JPEG to data/frames/{video_stem}/
- Return frame metadata: total frames, fps, duration, resolution

FUNCTION SIGNATURES:
  extract_frames(video_path, output_dir, target_fps=5,
                 target_resolution=(1280,720)) -> dict
    Returns: {
      "frame_dir": str,
      "frame_paths": list[str],
      "total_frames": int,
      "source_fps": float,
      "target_fps": int,
      "duration_sec": float,
      "resolution": tuple
    }

IMPLEMENTATION NOTES:
- Use cv2.VideoCapture for reading
- Calculate frame skip interval: skip = source_fps / target_fps
- Use cv2.resize with INTER_AREA for downsampling (best quality)
- Name frames as frame_00000.jpg, frame_00001.jpg, etc.
- For a 10-min video at 5fps = 3,000 frames = ~150MB JPEG

═══════════════════════════════════════════════════════════════
TASK 2: Build src/body/person_detector.py
═══════════════════════════════════════════════════════════════

REQUIREMENTS:
- Load YOLOv8-nano (6MB, pretrained, downloads automatically)
- Detect all persons in each frame
- Select the PRIMARY SPEAKER using strategy:
  a. Default: largest bounding box (assumes speaker is closest/most prominent)
  b. Optional: user-specified region of interest
- Track the speaker across frames using simple IoU-based tracking
  (if YOLO loses detection for a few frames, extrapolate from last known position)
- Return per-frame bounding boxes for the speaker

FUNCTION SIGNATURES:
  class PersonDetector:
    __init__(model_name="yolov8n", confidence=0.5)
    detect_persons(frame) -> list[dict]  # All persons in frame
    select_speaker(detections, strategy="largest") -> dict  # Primary speaker bbox
    track_speaker(frames_dir, frame_paths) -> list[dict]
      Returns per-frame: {"frame_idx", "bbox": [x1,y1,x2,y2],
                          "confidence", "detected": bool}

IMPLEMENTATION NOTES:
- from ultralytics import YOLO; model = YOLO("yolov8n.pt")
- YOLOv8 class 0 = person
- IoU tracking: if IoU between current and previous bbox > 0.3, same person
- If detection drops for < 5 frames, interpolate bbox linearly
- If detection drops for > 5 frames, log a warning (speaker may have left frame)

═══════════════════════════════════════════════════════════════
TASK 3: Build tests
═══════════════════════════════════════════════════════════════

Write tests/modality2/test_frame_extraction.py:
- Test frame extraction from a short video clip
- Verify correct number of frames at 5fps
- Verify resolution matches target
- Test person detection on a frame with a known person

NO USER INPUT REQUIRED. NO TRAINING REQUIRED.

COMPLETION: ✓ when frames extract correctly and YOLO detects persons.
