PHASE 4 — POSTURE SCORING (RULE-BASED + MLP)
==============================================

AGENT ROLE: Posture Analyst
DEPENDS ON: Phase 3 (pose keypoints)
DELIVERS TO: Phase 10 (output assembly)
ESTIMATED TIME: 25 min (agent code) + 30 min (training on M1)

OBJECTIVE:
Build src/body/posture_scorer.py — a hybrid rule-based + learned model
that scores posture quality on a 0-100 scale.

═══════════════════════════════════════════════════════════════
TASK 1: Rule-Based Features (no training)
═══════════════════════════════════════════════════════════════

Implement these geometric posture metrics:

1. SHOULDER ALIGNMENT:
   angle = |atan2(y_right_shoulder - y_left_shoulder,
                   x_right_shoulder - x_left_shoulder)|
   Score: 100 if < 3°, linear decay to 0 at 15°

2. SPINE ANGLE:
   Midpoint of shoulders vs midpoint of hips, relative to vertical.
   Score: 100 if < 5°, linear decay to 0 at 25°

3. WEIGHT DISTRIBUTION:
   ratio = |x_left_hip - x_center| / |x_right_hip - x_center|
   Score: 100 if 0.9-1.1, linear decay outside

4. HEAD POSITION:
   Distance from nose to projected shoulder midpoint (pixels, normalized)
   Score: 100 if < 3% of torso height, decay beyond

5. SHOULDER TENSION:
   y_shoulder relative to y_ear — hunched shoulders score lower
   Score: 100 if shoulders > 20% below ears, 0 if at ear level

Combined rule score: weighted average of all 5 metrics.

═══════════════════════════════════════════════════════════════
TASK 2: MLP Refinement Model (needs training on M1)
═══════════════════════════════════════════════════════════════

Architecture: (33*4=132) → 64 → 32 → 1
Input: flattened 33 keypoints × 4 values = 132 features
Output: posture score 0-100 (regression)
Parameters: ~10K — trains in 30 minutes on M1

Training data: YOU GENERATE IT SYNTHETICALLY.
1. Take pose keypoints from your test videos
2. Apply the rule-based scorer to get automatic labels
3. Add noise to create training pairs
4. The MLP learns to smooth and generalize the rule-based scores

This is self-supervised — no manual labeling needed.

Loss: MSE between MLP output and rule-based score
Target: MAE ≤ 0.6 on held-out frames

Final combined score:
  S_posture = 0.3 * S_rules + 0.7 * S_mlp

═══════════════════════════════════════════════════════════════
TASK 3: Build training/modality2/train_posture_mlp.py
═══════════════════════════════════════════════════════════════

- Extract poses from all available videos
- Compute rule-based scores as training labels
- Add Gaussian noise to keypoints for augmentation
- Train MLP with AdamW, lr=1e-3, 50 epochs
- Save to models/posture_mlp/best_model.pt

NO USER INPUT REQUIRED (self-supervised from rule-based labels).
TRAIN ON M1: ~30 minutes.

COMPLETION: ✓ when posture_scorer returns scores 0-100 for test frames.