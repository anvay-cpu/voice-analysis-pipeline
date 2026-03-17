"""Step 6: Hand state tracking via MediaPipe Hands.

Extracts 21 landmarks per hand and classifies hand states using
rule-based geometry on hand landmarks + pose keypoints.
"""

import logging
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

# MediaPipe hand landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

NUM_HAND_LANDMARKS = 21

# Finger tip and MCP pairs for extension detection
FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
FINGER_PIPS = [THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]

# Hand state labels
HAND_STATES = [
    "Open Palm",
    "Pointing",
    "Steepling",
    "Crossed Arms",
    "Hands in Pockets",
    "Behind Back",
    "Clasped",
    "Other",
]

DEFAULT_MODEL_PATH = "models/hand_landmarker.task"

# Pose keypoint indices (from MediaPipe Pose 33-point)
POSE_L_WRIST, POSE_R_WRIST = 15, 16
POSE_L_SHOULDER, POSE_R_SHOULDER = 11, 12
POSE_L_ELBOW, POSE_R_ELBOW = 13, 14
POSE_L_HIP, POSE_R_HIP = 23, 24


class HandTracker:
    """Track and classify hand states from video frames.

    Uses MediaPipe HandLandmarker for 21-point hand landmarks and
    rule-based geometry to classify hand states relevant to public speaking.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
    ):
        self.model_path = model_path
        self.min_detection_confidence = min_detection_confidence
        self.min_presence_confidence = min_presence_confidence

        if not Path(model_path).exists():
            logger.warning("Hand model not found at %s", model_path)

    def _create_landmarker(self):
        """Create a HandLandmarker instance."""
        opts = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_presence_confidence,
        )
        return mp.tasks.vision.HandLandmarker.create_from_options(opts)

    def detect_hands(self, frame: np.ndarray) -> dict:
        """Detect hand landmarks in a single frame.

        Args:
            frame: BGR image.

        Returns:
            Dict with 'hands' list (each has 'landmarks' (21, 3),
            'handedness' str, 'score' float) or empty list.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        landmarker = self._create_landmarker()
        try:
            result = landmarker.detect(mp_image)
        finally:
            landmarker.close()

        hands = []
        if result.hand_landmarks:
            for i, landmarks in enumerate(result.hand_landmarks):
                lm_array = np.zeros((NUM_HAND_LANDMARKS, 3), dtype=np.float32)
                for j, lm in enumerate(landmarks):
                    if j >= NUM_HAND_LANDMARKS:
                        break
                    lm_array[j] = [lm.x, lm.y, lm.z]

                handedness = "Left"
                score = 0.0
                if result.handedness and i < len(result.handedness):
                    h = result.handedness[i][0]
                    handedness = h.category_name
                    score = h.score

                hands.append({
                    "landmarks": lm_array,
                    "handedness": handedness,
                    "score": score,
                })

        return {"hands": hands}

    def classify_hand_state(
        self,
        hand_result: dict,
        pose_keypoints: np.ndarray | None = None,
    ) -> dict:
        """Classify the hand state for a frame.

        Args:
            hand_result: Output of detect_hands().
            pose_keypoints: Optional (33, 4) pose keypoints for body context.

        Returns:
            Dict with 'state', 'confidence', 'num_hands', 'details'.
        """
        hands = hand_result["hands"]
        num_hands = len(hands)

        # No hands visible — use pose keypoints for context
        if num_hands == 0:
            return self._classify_no_hands(pose_keypoints)

        # Classify individual hand shapes
        hand_shapes = []
        for h in hands:
            shape = self._classify_single_hand(h["landmarks"])
            hand_shapes.append({
                "handedness": h["handedness"],
                "shape": shape,
                "landmarks": h["landmarks"],
            })

        # Two-hand states
        if num_hands >= 2:
            two_hand = self._classify_two_hands(hand_shapes, pose_keypoints)
            if two_hand is not None:
                return two_hand

        # Single hand state (use the first/primary hand)
        primary = hand_shapes[0]
        return {
            "state": primary["shape"],
            "confidence": 0.70,
            "num_hands": num_hands,
            "details": {h["handedness"]: h["shape"] for h in hand_shapes},
        }

    def _classify_single_hand(self, landmarks: np.ndarray) -> str:
        """Classify a single hand's shape from its 21 landmarks.

        Args:
            landmarks: (21, 3) normalized hand landmarks.

        Returns:
            Shape string: "Open Palm", "Pointing", or "Other".
        """
        extended = self._fingers_extended(landmarks)

        # Open Palm: all 5 fingers extended
        if sum(extended) >= 4:
            return "Open Palm"

        # Pointing: only index extended
        if extended[1] and sum(extended) <= 2:
            return "Pointing"

        return "Other"

    def _classify_two_hands(
        self,
        hand_shapes: list[dict],
        pose_keypoints: np.ndarray | None,
    ) -> dict | None:
        """Classify two-hand states.

        Returns result dict or None if no special two-hand state detected.
        """
        lm0 = hand_shapes[0]["landmarks"]
        lm1 = hand_shapes[1]["landmarks"]

        # Steepling: fingertips of both hands close together
        if self._is_steepling(lm0, lm1):
            return {
                "state": "Steepling",
                "confidence": 0.75,
                "num_hands": 2,
                "details": {"left": hand_shapes[0]["shape"],
                             "right": hand_shapes[1]["shape"]},
            }

        # Clasped: wrists close, hands overlapping
        if self._is_clasped(lm0, lm1):
            return {
                "state": "Clasped",
                "confidence": 0.70,
                "num_hands": 2,
                "details": {"left": hand_shapes[0]["shape"],
                             "right": hand_shapes[1]["shape"]},
            }

        # Crossed arms: use pose keypoints
        if pose_keypoints is not None and self._is_crossed_arms(pose_keypoints):
            return {
                "state": "Crossed Arms",
                "confidence": 0.65,
                "num_hands": 2,
                "details": {"left": hand_shapes[0]["shape"],
                             "right": hand_shapes[1]["shape"]},
            }

        return None

    def _classify_no_hands(self, pose_keypoints: np.ndarray | None) -> dict:
        """Classify state when no hands are detected, using pose keypoints."""
        if pose_keypoints is None:
            return {
                "state": "Other",
                "confidence": 0.30,
                "num_hands": 0,
                "details": {},
            }

        # Check behind back: low wrist visibility + wrists below shoulders
        l_vis = pose_keypoints[POSE_L_WRIST, 3]
        r_vis = pose_keypoints[POSE_R_WRIST, 3]
        l_wrist_y = pose_keypoints[POSE_L_WRIST, 1]
        r_wrist_y = pose_keypoints[POSE_R_WRIST, 1]
        l_hip_y = pose_keypoints[POSE_L_HIP, 1]
        r_hip_y = pose_keypoints[POSE_R_HIP, 1]

        if l_vis < 0.3 and r_vis < 0.3:
            return {
                "state": "Behind Back",
                "confidence": 0.60,
                "num_hands": 0,
                "details": {"l_wrist_vis": float(l_vis),
                             "r_wrist_vis": float(r_vis)},
            }

        # Check hands in pockets: wrists below hips + close to body midline
        hip_y = (l_hip_y + r_hip_y) / 2
        l_hip_x = pose_keypoints[POSE_L_HIP, 0]
        r_hip_x = pose_keypoints[POSE_R_HIP, 0]
        l_wrist_x = pose_keypoints[POSE_L_WRIST, 0]
        r_wrist_x = pose_keypoints[POSE_R_WRIST, 0]

        l_near_hip = (l_wrist_y > l_hip_y and
                      abs(l_wrist_x - l_hip_x) < 0.08)
        r_near_hip = (r_wrist_y > r_hip_y and
                      abs(r_wrist_x - r_hip_x) < 0.08)

        if l_near_hip and r_near_hip:
            return {
                "state": "Hands in Pockets",
                "confidence": 0.55,
                "num_hands": 0,
                "details": {},
            }

        # Crossed arms from pose only
        if self._is_crossed_arms(pose_keypoints):
            return {
                "state": "Crossed Arms",
                "confidence": 0.60,
                "num_hands": 0,
                "details": {},
            }

        return {
            "state": "Other",
            "confidence": 0.40,
            "num_hands": 0,
            "details": {},
        }

    def _fingers_extended(self, landmarks: np.ndarray) -> list[bool]:
        """Check which fingers are extended.

        Args:
            landmarks: (21, 3) hand landmarks.

        Returns:
            List of 5 bools: [thumb, index, middle, ring, pinky].
        """
        extended = []

        # Thumb: compare tip x to IP x (depends on hand orientation)
        # Use distance from MCP as a simpler heuristic
        thumb_tip = landmarks[THUMB_TIP]
        thumb_ip = landmarks[THUMB_IP]
        thumb_mcp = landmarks[THUMB_MCP]
        thumb_dist_tip = np.linalg.norm(thumb_tip[:2] - thumb_mcp[:2])
        thumb_dist_ip = np.linalg.norm(thumb_ip[:2] - thumb_mcp[:2])
        extended.append(thumb_dist_tip > thumb_dist_ip * 1.2)

        # Index through pinky: tip y < PIP y means extended
        # (in normalized coords, y increases downward)
        for tip_idx, pip_idx in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
            extended.append(landmarks[tip_idx, 1] < landmarks[pip_idx, 1])

        return extended

    def _is_steepling(
        self, lm0: np.ndarray, lm1: np.ndarray, threshold: float = 0.06
    ) -> bool:
        """Check if fingertips of both hands are touching (steepling)."""
        tip_dists = []
        for tip_idx in FINGER_TIPS[1:]:  # skip thumb
            d = np.linalg.norm(lm0[tip_idx, :2] - lm1[tip_idx, :2])
            tip_dists.append(d)

        # At least 3 fingertip pairs must be close
        close_count = sum(1 for d in tip_dists if d < threshold)
        return close_count >= 3

    def _is_clasped(
        self, lm0: np.ndarray, lm1: np.ndarray, threshold: float = 0.08
    ) -> bool:
        """Check if hands are clasped (wrists close, fingers interleaved)."""
        wrist_dist = np.linalg.norm(lm0[WRIST, :2] - lm1[WRIST, :2])
        if wrist_dist > threshold:
            return False

        # Check finger overlap: MCP points of one hand near PIPs of other
        overlap_count = 0
        for mcp_idx, pip_idx in zip(
            [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP],
            [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP],
        ):
            d1 = np.linalg.norm(lm0[mcp_idx, :2] - lm1[pip_idx, :2])
            d2 = np.linalg.norm(lm1[mcp_idx, :2] - lm0[pip_idx, :2])
            if min(d1, d2) < 0.05:
                overlap_count += 1

        return overlap_count >= 2

    @staticmethod
    def _is_crossed_arms(pose_keypoints: np.ndarray) -> bool:
        """Detect crossed arms from pose keypoints.

        Crossed arms: wrists near opposite shoulders / chest midline,
        elbows flared out.
        """
        l_wrist = pose_keypoints[POSE_L_WRIST, :2]
        r_wrist = pose_keypoints[POSE_R_WRIST, :2]
        l_shoulder = pose_keypoints[POSE_L_SHOULDER, :2]
        r_shoulder = pose_keypoints[POSE_R_SHOULDER, :2]
        l_elbow = pose_keypoints[POSE_L_ELBOW, :2]
        r_elbow = pose_keypoints[POSE_R_ELBOW, :2]

        chest_x = (l_shoulder[0] + r_shoulder[0]) / 2
        shoulder_width = abs(r_shoulder[0] - l_shoulder[0])

        if shoulder_width < 0.01:
            return False

        # Both wrists near chest midline
        l_wrist_near_center = abs(l_wrist[0] - chest_x) < shoulder_width * 0.5
        r_wrist_near_center = abs(r_wrist[0] - chest_x) < shoulder_width * 0.5

        # Wrists crossed: left wrist on right side, right wrist on left side
        l_crossed = l_wrist[0] > chest_x
        r_crossed = r_wrist[0] < chest_x

        # Elbows wider than wrists
        elbow_spread = abs(r_elbow[0] - l_elbow[0])
        wrist_spread = abs(r_wrist[0] - l_wrist[0])
        elbows_wide = elbow_spread > wrist_spread * 1.2

        return bool(l_wrist_near_center and r_wrist_near_center and
                    l_crossed and r_crossed and elbows_wide)

    def track_video(
        self,
        frame_paths: list[str],
        poses: dict[int, np.ndarray | None] | None = None,
    ) -> list[dict]:
        """Track hand states across all frames.

        Args:
            frame_paths: Ordered list of frame image paths.
            poses: Optional pose keypoints per frame from PoseEstimator.

        Returns:
            List of per-frame hand state dicts.
        """
        results = []

        for idx, fpath in enumerate(frame_paths):
            frame = cv2.imread(fpath)
            if frame is None:
                logger.warning("Could not read frame: %s", fpath)
                results.append({
                    "frame_idx": idx,
                    "state": "Other",
                    "confidence": 0.0,
                    "num_hands": 0,
                    "details": {},
                })
                continue

            hand_result = self.detect_hands(frame)
            pose_kps = None
            if poses is not None and idx in poses:
                pose_kps = poses[idx]

            state = self.classify_hand_state(hand_result, pose_kps)
            state["frame_idx"] = idx
            results.append(state)

            if (idx + 1) % 100 == 0:
                logger.info("Hand tracking: %d / %d frames", idx + 1, len(frame_paths))

        # Log summary
        from collections import Counter
        state_counts = Counter(r["state"] for r in results)
        logger.info(
            "Hand tracking complete: %d frames, distribution: %s",
            len(results), dict(state_counts),
        )

        return results
