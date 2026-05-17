"""Step 4: Posture quality scoring (rule-based + MLP hybrid)."""

import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# MediaPipe keypoint indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
NOSE = 0
LEFT_EAR = 7
RIGHT_EAR = 8

NUM_KEYPOINTS = 33
KEYPOINT_DIM = 4
MLP_INPUT_DIM = NUM_KEYPOINTS * KEYPOINT_DIM  # 132 flattened keypoint features


# ── Rule-Based Scoring ────────────────────────────────────────


def _linear_decay(value: float, good_thresh: float, bad_thresh: float) -> float:
    """Score 100 if value < good_thresh, linearly decay to 0 at bad_thresh."""
    if value <= good_thresh:
        return 100.0
    if value >= bad_thresh:
        return 0.0
    return 100.0 * (bad_thresh - value) / (bad_thresh - good_thresh)


def shoulder_alignment_score(keypoints: np.ndarray) -> float:
    """Score shoulder alignment (0-100). Level shoulders = 100."""
    ls = keypoints[LEFT_SHOULDER]
    rs = keypoints[RIGHT_SHOULDER]
    if ls[3] < 0.3 or rs[3] < 0.3:  # low visibility
        return 50.0  # neutral
    angle_deg = abs(math.degrees(math.atan2(rs[1] - ls[1], rs[0] - ls[0])))
    return _linear_decay(angle_deg, 3.0, 15.0)


def spine_angle_score(keypoints: np.ndarray) -> float:
    """Score spine verticality (0-100). Upright = 100."""
    ls, rs = keypoints[LEFT_SHOULDER], keypoints[RIGHT_SHOULDER]
    lh, rh = keypoints[LEFT_HIP], keypoints[RIGHT_HIP]
    if min(ls[3], rs[3], lh[3], rh[3]) < 0.3:
        return 50.0

    shoulder_mid = np.array([(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2])
    hip_mid = np.array([(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2])

    # Angle from vertical (y-axis goes down in image coords)
    dx = shoulder_mid[0] - hip_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]  # positive = shoulder above hip
    if dy <= 0:
        return 0.0
    angle_deg = abs(math.degrees(math.atan2(dx, dy)))
    return _linear_decay(angle_deg, 5.0, 25.0)


def weight_distribution_score(keypoints: np.ndarray) -> float:
    """Score weight distribution symmetry (0-100). Even = 100."""
    lh, rh = keypoints[LEFT_HIP], keypoints[RIGHT_HIP]
    if lh[3] < 0.3 or rh[3] < 0.3:
        return 50.0

    center_x = (lh[0] + rh[0]) / 2
    dist_left = abs(lh[0] - center_x)
    dist_right = abs(rh[0] - center_x)

    if dist_right < 1e-6:
        return 50.0
    ratio = dist_left / dist_right

    # Score: 100 if ratio in [0.9, 1.1], decay outside
    deviation = abs(ratio - 1.0)
    return _linear_decay(deviation, 0.1, 0.5)


def head_position_score(keypoints: np.ndarray) -> float:
    """Score head alignment over torso (0-100). Centered = 100."""
    nose = keypoints[NOSE]
    ls, rs = keypoints[LEFT_SHOULDER], keypoints[RIGHT_SHOULDER]
    lh, rh = keypoints[LEFT_HIP], keypoints[RIGHT_HIP]
    if min(nose[3], ls[3], rs[3], lh[3], rh[3]) < 0.3:
        return 50.0

    shoulder_mid_x = (ls[0] + rs[0]) / 2
    hip_mid_y = (lh[1] + rh[1]) / 2
    shoulder_mid_y = (ls[1] + rs[1]) / 2
    torso_height = abs(hip_mid_y - shoulder_mid_y)
    if torso_height < 1e-6:
        return 50.0

    # Horizontal offset of nose from shoulder midpoint, as % of torso height
    offset_pct = abs(nose[0] - shoulder_mid_x) / torso_height * 100
    return _linear_decay(offset_pct, 3.0, 15.0)


def shoulder_tension_score(keypoints: np.ndarray) -> float:
    """Score shoulder relaxation (0-100). Relaxed/dropped = 100."""
    ls, rs = keypoints[LEFT_SHOULDER], keypoints[RIGHT_SHOULDER]
    le, re = keypoints[LEFT_EAR], keypoints[RIGHT_EAR]
    if min(ls[3], rs[3], le[3], re[3]) < 0.3:
        return 50.0

    shoulder_y = (ls[1] + rs[1]) / 2
    ear_y = (le[1] + re[1]) / 2

    # In image coords, y increases downward, so shoulder_y > ear_y means below
    ear_shoulder_dist = shoulder_y - ear_y
    torso_ref = abs((keypoints[LEFT_HIP][1] + keypoints[RIGHT_HIP][1]) / 2 - shoulder_y)
    if torso_ref < 1e-6:
        return 50.0

    # Fraction of torso height that shoulders are below ears
    frac = ear_shoulder_dist / torso_ref
    # 100 if frac >= 0.2 (shoulders well below ears), 0 if frac <= 0
    return _linear_decay(-frac, -0.2, 0.0)


# Weights for combining rule scores
RULE_WEIGHTS = {
    "shoulder_alignment": 0.25,
    "spine_angle": 0.25,
    "weight_distribution": 0.15,
    "head_position": 0.20,
    "shoulder_tension": 0.15,
}


def compute_rule_score(keypoints: np.ndarray) -> dict:
    """Compute all rule-based posture scores.

    Returns:
        Dict with individual scores and weighted 'combined' score (0-100).
    """
    scores = {
        "shoulder_alignment": shoulder_alignment_score(keypoints),
        "spine_angle": spine_angle_score(keypoints),
        "weight_distribution": weight_distribution_score(keypoints),
        "head_position": head_position_score(keypoints),
        "shoulder_tension": shoulder_tension_score(keypoints),
    }
    scores["combined"] = sum(
        RULE_WEIGHTS[k] * scores[k] for k in RULE_WEIGHTS
    )
    return scores


# ── MLP Model ─────────────────────────────────────────────────


class PostureMLP(nn.Module):
    """Small MLP to refine posture scoring from raw keypoints."""

    def __init__(self, input_dim: int = MLP_INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # output in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── Hybrid Scorer ─────────────────────────────────────────────


class PostureScorer:
    """Hybrid posture scorer combining rule-based + MLP."""

    def __init__(
        self,
        model_path: str | None = None,
        rule_weight: float = 0.3,
        device: str = "cpu",
    ):
        self.rule_weight = rule_weight
        self.mlp_weight = 1.0 - rule_weight
        self.device = device
        self.mlp = None

        if model_path and Path(model_path).exists():
            self.mlp = PostureMLP()
            state = torch.load(model_path, map_location=device, weights_only=False)
            self.mlp.load_state_dict(state)
            self.mlp.to(device)
            self.mlp.eval()
            logger.info("Loaded PostureMLP from %s", model_path)
        else:
            if model_path:
                logger.warning(
                    "MLP model not found at %s; using rule-based only", model_path
                )

    def score_frame(self, keypoints: np.ndarray) -> dict:
        """Score posture for a single frame.

        Args:
            keypoints: np.ndarray of shape (33, 4).

        Returns:
            Dict with 'rule_score', 'mlp_score' (if available),
            'combined_score', and individual rule metrics.
        """
        rule_result = compute_rule_score(keypoints)
        rule_score = rule_result["combined"]

        result = {
            "rule_score": round(rule_score, 1),
            "shoulder_alignment": round(rule_result["shoulder_alignment"], 1),
            "spine_angle": round(rule_result["spine_angle"], 1),
            "weight_distribution": round(rule_result["weight_distribution"], 1),
            "head_position": round(rule_result["head_position"], 1),
            "shoulder_tension": round(rule_result["shoulder_tension"], 1),
        }

        if self.mlp is not None:
            flat = torch.tensor(
                keypoints.flatten(), dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            with torch.no_grad():
                mlp_raw = self.mlp(flat).item()
            mlp_score = mlp_raw * 100.0  # Sigmoid output [0,1] → [0,100]
            result["mlp_score"] = round(mlp_score, 1)
            result["combined_score"] = round(
                self.rule_weight * rule_score + self.mlp_weight * mlp_score, 1
            )
        else:
            result["mlp_score"] = None
            result["combined_score"] = round(rule_score, 1)

        return result

    def score_batch(
        self, poses: dict[int, np.ndarray | None]
    ) -> dict[int, dict | None]:
        """Score posture for all frames.

        Args:
            poses: Dict mapping frame_idx to keypoints (33, 4) or None.

        Returns:
            Dict mapping frame_idx to posture score dict or None.
        """
        results = {}
        for idx in sorted(poses.keys()):
            kp = poses[idx]
            if kp is None:
                results[idx] = None
                continue
            results[idx] = self.score_frame(kp)
        return results
