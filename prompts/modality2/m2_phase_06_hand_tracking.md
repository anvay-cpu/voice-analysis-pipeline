PHASE 6 — HAND TRACKING & PLACEMENT
======================================

AGENT ROLE: Hand Analysis Specialist
DEPENDS ON: Phase 2 (frames), Phase 3 (pose)
DELIVERS TO: Phase 10 (output assembly)
ESTIMATED TIME: 20 minutes (agent), 0 minutes (user)

OBJECTIVE:
Build src/body/hand_tracker.py using MediaPipe Hands (pretrained).
Classify hand states relevant to public speaking.

TASKS:
1. Use mediapipe.solutions.hands to extract 21 landmarks per hand
2. Classify hand state using RULE-BASED geometry:

   OPEN PALM: All fingers extended (finger tip y < finger MCP y)
   POINTING: Index extended, others curled
   STEEPLING: Fingertips of both hands touching (small distance)
   CROSSED ARMS: Both wrists near chest midline, elbows wide
     (from pose keypoints: wrist x near chest center, elbow x far from center)
   HANDS IN POCKETS: Wrist keypoints below hip keypoints and close to body
   BEHIND BACK: Wrist visibility < 0.3 and wrist y > hip y
   CLASPED: Both hands visible, small distance between wrists, fingers interleaved
   OTHER: Default fallback

3. No training needed — pure geometry on hand landmarks + pose keypoints.

4. Handle: one hand visible, no hands visible, partial occlusion.

NO USER INPUT REQUIRED. PRETRAINED MODEL.

COMPLETION: ✓ when hand states are classified correctly on test frames.