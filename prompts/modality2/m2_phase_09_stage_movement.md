PHASE 9 — STAGE MOVEMENT & PROXEMICS
=======================================

AGENT ROLE: Spatial Analyst
DEPENDS ON: Phase 3 (pose keypoints)
DELIVERS TO: Phase 10 (output assembly)
ESTIMATED TIME: 15 minutes (agent), 0 minutes (user)

OBJECTIVE:
Build src/body/stage_movement.py — track the speaker's position
on stage and classify movement patterns. Pure computation, no training.

TASKS:
1. Compute speaker CENTROID per frame:
   centroid = midpoint of (left_hip, right_hip) keypoints
   Normalize to 0-1 range relative to frame width/height.

2. Compute MOVEMENT METRICS:
   - Total displacement (sum of frame-to-frame distances)
   - Convex hull area of all centroid positions
   - Mean velocity (pixels/sec, normalized)
   - Directional entropy (how random is the movement direction)
   - Autocorrelation (detects repetitive pacing)

3. CLASSIFY movement pattern:
   - ANCHORED: convex hull area < 5% of frame area, low velocity
   - PACING: autocorrelation > 0.6, repetitive back-and-forth
   - ROAMING: high convex hull area, high directional entropy
   - PURPOSEFUL: moderate movement, low autocorrelation
     (moving to specific spots, then pausing)

4. SCORE stage usage:
   - Roaming + purposeful = high score (using the space)
   - Anchored = medium score (acceptable but static)
   - Pacing = low score (nervous repetitive movement)

NO USER INPUT REQUIRED. NO TRAINING REQUIRED.

COMPLETION: ✓ when movement classification and scoring work on test video.