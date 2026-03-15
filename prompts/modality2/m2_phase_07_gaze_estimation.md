PHASE 7 — GAZE & HEAD POSE ESTIMATION
========================================

AGENT ROLE: Gaze Analyst
DEPENDS ON: Phase 2 (frames)
DELIVERS TO: Phase 10 (output assembly)
ESTIMATED TIME: 20 minutes (agent), 0 minutes (user)

OBJECTIVE:
Build src/body/gaze_estimator.py using MediaPipe Face Mesh (pretrained).
Determine where the speaker is looking: audience, notes, floor, etc.

TASKS:
1. Use mediapipe.solutions.face_mesh to extract 468 face landmarks
2. Compute HEAD POSE (yaw, pitch, roll) using PnP solver:
   - 6 reference landmarks: nose tip (1), chin (152), left eye corner (33),
     right eye corner (263), left mouth (61), right mouth (291)
   - Generic 3D face model coordinates
   - cv2.solvePnP → rotation vector → Euler angles

3. Estimate GAZE DIRECTION from iris landmarks (landmarks 468-477):
   - Iris center relative to eye corner positions
   - Combined with head pose for final gaze vector

4. Classify into GAZE ZONES (from config):
   - Audience center: yaw ±20°, pitch ±15°
   - Audience left: yaw -60° to -20°
   - Audience right: yaw 20° to 60°
   - Notes/podium: pitch 25°-60° downward
   - Floor: pitch > 40° downward
   - Ceiling/away: pitch > 20° upward or yaw > 45°

5. Compute AUDIENCE ENGAGEMENT RATIO:
   R_gaze = frames_at_audience / total_frames
   Target for effective speakers: R_gaze ≥ 0.75

NO USER INPUT REQUIRED. PRETRAINED MODEL.

COMPLETION: ✓ when gaze zones are classified and engagement ratio computed.