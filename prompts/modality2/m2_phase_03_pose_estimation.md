PHASE 3 — POSE ESTIMATION (MEDIAPIPE)
=======================================

AGENT ROLE: Pose Specialist
DEPENDS ON: Phase 2 (frames + speaker bbox)
DELIVERS TO: Phase 4 (posture), Phase 5 (gesture), Phase 9 (movement)
ESTIMATED TIME: 20 minutes (agent), 0 minutes (user)

OBJECTIVE:
Build src/body/pose_estimator.py using MediaPipe Pose (pretrained, no training).
Extracts 33 keypoints per frame for the detected speaker.

TASKS:
1. Build PoseEstimator class:
   - Uses mediapipe.solutions.pose
   - Crops frame to speaker bbox before pose estimation (better accuracy)
   - Extracts 33 keypoints, each with (x, y, z, visibility)
   - Output per frame: np.array of shape (33, 4)
   - Batch process all frames: returns dict of {frame_idx: keypoints}
   - Cache results to data/poses/{video_stem}.npz (avoid recomputing)

2. Keypoint reference (MediaPipe 33-point):
   0=nose, 1=left_eye_inner, 2=left_eye, 3=left_eye_outer,
   4=right_eye_inner, 5=right_eye, 6=right_eye_outer,
   7=left_ear, 8=right_ear, 9=mouth_left, 10=mouth_right,
   11=left_shoulder, 12=right_shoulder, 13=left_elbow, 14=right_elbow,
   15=left_wrist, 16=right_wrist, 17=left_pinky, 18=right_pinky,
   19=left_index, 20=right_index, 21=left_thumb, 22=right_thumb,
   23=left_hip, 24=right_hip, 25=left_knee, 26=right_knee,
   27=left_ankle, 28=right_ankle, 29=left_heel, 30=right_heel,
   31=left_foot_index, 32=right_foot_index

3. Handle edge cases:
   - Frame with no pose detected → return None, interpolate from neighbors
   - Low visibility keypoints → flag but don't discard
   - Speaker partially out of frame → use available keypoints only

4. Performance: MediaPipe runs ~15ms/frame on CPU = 3,000 frames in 45 seconds.
   NO GPU needed for this step.

NO USER INPUT REQUIRED. PRETRAINED MODEL (downloads automatically).

COMPLETION: ✓ when pose estimation returns 33 keypoints for test frames.