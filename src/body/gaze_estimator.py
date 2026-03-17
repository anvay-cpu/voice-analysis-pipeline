"""Step 7: Gaze direction & head pose via MediaPipe Face Mesh.

Extracts 478 face landmarks (468 mesh + 10 iris), computes head pose
(yaw/pitch/roll) via PnP solver, estimates gaze direction from iris
position, and classifies into gaze zones for speaker coaching.
"""

import logging
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "models/face_landmarker.task"

# MediaPipe face landmark indices for head pose PnP
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263
LEFT_MOUTH = 61
RIGHT_MOUTH = 291

# Iris landmark indices (478-landmark model: 468..477)
LEFT_IRIS = [468, 469, 470, 471, 472]    # center + 4 boundary
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Eye corner indices for iris-relative gaze
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263

# Generic 3D face model points (approximate, in arbitrary units)
# Matches the 6 PnP reference landmarks above
FACE_3D_MODEL = np.array([
    [0.0, 0.0, 0.0],           # nose tip
    [0.0, -63.6, -12.5],       # chin
    [-43.3, 32.7, -26.0],      # left eye corner
    [43.3, 32.7, -26.0],       # right eye corner
    [-28.9, -28.9, -24.1],     # left mouth
    [28.9, -28.9, -24.1],      # right mouth
], dtype=np.float64)

# Gaze zone definitions (degrees)
GAZE_ZONES = {
    "Floor": {"yaw": (-90, 90), "pitch": (40, 90)},
    "Ceiling/Away": {"yaw": (-90, 90), "pitch": (-90, -20)},
    "Notes/Podium": {"yaw": (-30, 30), "pitch": (25, 40)},
    "Audience Center": {"yaw": (-20, 20), "pitch": (-15, 15)},
    "Audience Left": {"yaw": (-60, -20), "pitch": (-20, 20)},
    "Audience Right": {"yaw": (20, 60), "pitch": (-20, 20)},
}


class GazeEstimator:
    """Estimate gaze direction and head pose from video frames.

    Uses MediaPipe FaceLandmarker for 478 face+iris landmarks,
    PnP solver for head pose (yaw/pitch/roll), and iris position
    for gaze direction refinement.
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
            logger.warning("Face model not found at %s", model_path)

    def _create_landmarker(self):
        """Create a FaceLandmarker instance."""
        opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=self.min_detection_confidence,
            min_face_presence_confidence=self.min_presence_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        return mp.tasks.vision.FaceLandmarker.create_from_options(opts)

    def detect_face(self, frame: np.ndarray) -> dict | None:
        """Detect face landmarks in a single frame.

        Args:
            frame: BGR image.

        Returns:
            Dict with 'landmarks' (N, 3) or None if no face.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        landmarker = self._create_landmarker()
        try:
            result = landmarker.detect(mp_image)
        finally:
            landmarker.close()

        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        n_lm = len(landmarks)
        lm_array = np.zeros((n_lm, 3), dtype=np.float32)
        for i, lm in enumerate(landmarks):
            lm_array[i] = [lm.x * w, lm.y * h, lm.z * w]

        return {"landmarks": lm_array, "image_size": (w, h)}

    def compute_head_pose(
        self, landmarks: np.ndarray, image_size: tuple[int, int]
    ) -> dict:
        """Compute head pose (yaw, pitch, roll) using PnP solver.

        Args:
            landmarks: (N, 3) face landmarks in pixel coordinates.
            image_size: (width, height).

        Returns:
            Dict with 'yaw', 'pitch', 'roll' in degrees.
        """
        w, h = image_size

        # 2D points from landmarks
        face_2d = np.array([
            landmarks[NOSE_TIP, :2],
            landmarks[CHIN, :2],
            landmarks[LEFT_EYE_CORNER, :2],
            landmarks[RIGHT_EYE_CORNER, :2],
            landmarks[LEFT_MOUTH, :2],
            landmarks[RIGHT_MOUTH, :2],
        ], dtype=np.float64)

        # Camera matrix (approximate from image dimensions)
        focal_length = w
        camera_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            FACE_3D_MODEL, face_2d, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

        # Convert rotation vector to rotation matrix, then to Euler angles
        rmat, _ = cv2.Rodrigues(rvec)
        angles = _rotation_matrix_to_euler(rmat)

        return {
            "yaw": float(angles[1]),
            "pitch": float(angles[0]),
            "roll": float(angles[2]),
        }

    def compute_iris_gaze(self, landmarks: np.ndarray) -> dict:
        """Compute gaze offset from iris position relative to eye corners.

        Args:
            landmarks: (N, 3) face landmarks (must have 478+ for iris).

        Returns:
            Dict with 'left_ratio', 'right_ratio', 'gaze_x', 'gaze_y'.
            Ratios: 0.0 = looking left, 0.5 = center, 1.0 = looking right.
        """
        n_lm = landmarks.shape[0]
        if n_lm < 478:
            return {
                "left_ratio": 0.5, "right_ratio": 0.5,
                "gaze_x": 0.0, "gaze_y": 0.0,
            }

        # Left eye
        left_iris_center = landmarks[LEFT_IRIS[0], :2]
        left_inner = landmarks[LEFT_EYE_INNER, :2]
        left_outer = landmarks[LEFT_EYE_OUTER, :2]
        left_width = np.linalg.norm(left_inner - left_outer)
        if left_width > 0:
            left_ratio = np.linalg.norm(left_iris_center - left_outer) / left_width
        else:
            left_ratio = 0.5

        # Right eye
        right_iris_center = landmarks[RIGHT_IRIS[0], :2]
        right_inner = landmarks[RIGHT_EYE_INNER, :2]
        right_outer = landmarks[RIGHT_EYE_OUTER, :2]
        right_width = np.linalg.norm(right_inner - right_outer)
        if right_width > 0:
            right_ratio = np.linalg.norm(right_iris_center - right_outer) / right_width
        else:
            right_ratio = 0.5

        # Average gaze offset: -1 = full left, 0 = center, +1 = full right
        avg_ratio = (left_ratio + right_ratio) / 2
        gaze_x = (avg_ratio - 0.5) * 2

        # Vertical: iris center y relative to eye top/bottom
        left_eye_top = landmarks[159, 1]   # upper eyelid
        left_eye_bot = landmarks[145, 1]   # lower eyelid
        eye_height = left_eye_bot - left_eye_top
        if eye_height > 0:
            gaze_y = (left_iris_center[1] - left_eye_top) / eye_height - 0.5
        else:
            gaze_y = 0.0

        return {
            "left_ratio": float(np.clip(left_ratio, 0, 1)),
            "right_ratio": float(np.clip(right_ratio, 0, 1)),
            "gaze_x": float(np.clip(gaze_x, -1, 1)),
            "gaze_y": float(np.clip(gaze_y, -1, 1)),
        }

    @staticmethod
    def classify_gaze_zone(yaw: float, pitch: float) -> str:
        """Classify head pose into a gaze zone.

        Args:
            yaw: Head yaw in degrees (negative=left, positive=right).
            pitch: Head pitch in degrees (positive=down, negative=up).

        Returns:
            Gaze zone string.
        """
        # Check zones in priority order
        for zone_name, bounds in GAZE_ZONES.items():
            yaw_lo, yaw_hi = bounds["yaw"]
            pitch_lo, pitch_hi = bounds["pitch"]
            if yaw_lo <= yaw <= yaw_hi and pitch_lo <= pitch <= pitch_hi:
                return zone_name

        return "Away"

    def estimate_single(self, frame: np.ndarray) -> dict:
        """Full gaze estimation for a single frame.

        Args:
            frame: BGR image.

        Returns:
            Dict with 'head_pose', 'iris_gaze', 'gaze_zone', 'detected'.
        """
        face = self.detect_face(frame)

        if face is None:
            return {
                "detected": False,
                "head_pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
                "iris_gaze": {
                    "left_ratio": 0.5, "right_ratio": 0.5,
                    "gaze_x": 0.0, "gaze_y": 0.0,
                },
                "gaze_zone": "Away",
            }

        lm = face["landmarks"]
        img_size = face["image_size"]

        head_pose = self.compute_head_pose(lm, img_size)
        iris_gaze = self.compute_iris_gaze(lm)
        gaze_zone = self.classify_gaze_zone(head_pose["yaw"], head_pose["pitch"])

        return {
            "detected": True,
            "head_pose": head_pose,
            "iris_gaze": iris_gaze,
            "gaze_zone": gaze_zone,
        }

    def track_video(self, frame_paths: list[str]) -> dict:
        """Track gaze across all frames and compute engagement metrics.

        Args:
            frame_paths: Ordered list of frame image paths.

        Returns:
            Dict with 'frames' (per-frame results), 'engagement_ratio',
            'zone_distribution'.
        """
        frame_results = []

        for idx, fpath in enumerate(frame_paths):
            frame = cv2.imread(fpath)
            if frame is None:
                logger.warning("Could not read frame: %s", fpath)
                frame_results.append({
                    "frame_idx": idx,
                    "detected": False,
                    "head_pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
                    "iris_gaze": {
                        "left_ratio": 0.5, "right_ratio": 0.5,
                        "gaze_x": 0.0, "gaze_y": 0.0,
                    },
                    "gaze_zone": "Away",
                })
                continue

            result = self.estimate_single(frame)
            result["frame_idx"] = idx
            frame_results.append(result)

            if (idx + 1) % 100 == 0:
                logger.info("Gaze estimation: %d / %d frames", idx + 1, len(frame_paths))

        # Compute engagement metrics
        detected = [r for r in frame_results if r["detected"]]
        total_detected = len(detected)

        audience_zones = {"Audience Center", "Audience Left", "Audience Right"}
        audience_frames = sum(
            1 for r in detected if r["gaze_zone"] in audience_zones
        )
        engagement_ratio = (
            audience_frames / total_detected if total_detected > 0 else 0.0
        )

        # Zone distribution
        from collections import Counter
        zone_counts = Counter(r["gaze_zone"] for r in detected)
        zone_distribution = {
            zone: count / total_detected if total_detected > 0 else 0.0
            for zone, count in zone_counts.items()
        }

        logger.info(
            "Gaze tracking complete: %d frames, %d detected, "
            "engagement=%.2f",
            len(frame_results), total_detected, engagement_ratio,
        )

        return {
            "frames": frame_results,
            "engagement_ratio": round(engagement_ratio, 4),
            "zone_distribution": zone_distribution,
            "total_frames": len(frame_results),
            "detected_frames": total_detected,
        }


def _rotation_matrix_to_euler(rmat: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to Euler angles (pitch, yaw, roll) in degrees."""
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)

    if sy > 1e-6:
        pitch = np.arctan2(rmat[2, 1], rmat[2, 2])
        yaw = np.arctan2(-rmat[2, 0], sy)
        roll = np.arctan2(rmat[1, 0], rmat[0, 0])
    else:
        pitch = np.arctan2(-rmat[1, 2], rmat[1, 1])
        yaw = np.arctan2(-rmat[2, 0], sy)
        roll = 0.0

    return np.degrees(np.array([pitch, yaw, roll]))
