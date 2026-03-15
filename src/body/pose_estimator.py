"""Step 3: 33-point skeleton extraction via MediaPipe Pose Landmarker."""

import logging
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

# MediaPipe 33 keypoint names for reference
KEYPOINT_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

NUM_KEYPOINTS = 33
KEYPOINT_DIM = 4  # x, y, z, visibility

# Default model path (relative to project root)
DEFAULT_MODEL_PATH = "models/pose_landmarker_lite.task"


class PoseEstimator:
    """Extract 33-point pose keypoints using MediaPipe PoseLandmarker."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.model_path = model_path
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

    def _create_landmarker(self):
        """Create a PoseLandmarker instance for IMAGE mode."""
        base_options = mp.tasks.BaseOptions(model_asset_path=self.model_path)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_pose_presence_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        return mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def estimate_single(
        self,
        frame: np.ndarray,
        bbox: list | None = None,
    ) -> np.ndarray | None:
        """Estimate pose keypoints for a single frame.

        Args:
            frame: BGR image as numpy array.
            bbox: Optional [x1, y1, x2, y2] to crop to speaker region.

        Returns:
            np.ndarray of shape (33, 4) with (x, y, z, visibility) per keypoint,
            with x/y in original frame coordinates (normalized). None if no pose.
        """
        h, w = frame.shape[:2]

        # Crop to speaker bbox if provided
        if bbox is not None:
            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(w, int(bbox[2]))
            y2 = min(h, int(bbox[3]))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            crop_w, crop_h = x2 - x1, y2 - y1
        else:
            crop = frame
            x1, y1 = 0, 0
            crop_w, crop_h = w, h

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        landmarker = self._create_landmarker()
        try:
            result = landmarker.detect(mp_image)
        finally:
            landmarker.close()

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        landmarks = result.pose_landmarks[0]  # first (only) pose

        keypoints = np.zeros((NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)
        for i, lm in enumerate(landmarks):
            if i >= NUM_KEYPOINTS:
                break
            # Convert normalized crop coords back to original frame coords
            keypoints[i, 0] = (lm.x * crop_w + x1) / w
            keypoints[i, 1] = (lm.y * crop_h + y1) / h
            keypoints[i, 2] = lm.z
            keypoints[i, 3] = lm.visibility if lm.visibility is not None else 0.0

        return keypoints

    def estimate_batch(
        self,
        frame_paths: list[str],
        tracking_results: list[dict] | None = None,
    ) -> dict[int, np.ndarray | None]:
        """Estimate pose for all frames, optionally cropping to speaker bbox.

        Args:
            frame_paths: Ordered list of frame image paths.
            tracking_results: Optional per-frame tracking from PersonDetector.

        Returns:
            Dict mapping frame_idx to keypoints (33, 4) or None.
        """
        poses = {}

        for idx, fpath in enumerate(frame_paths):
            frame = cv2.imread(fpath)
            if frame is None:
                logger.warning("Could not read frame: %s", fpath)
                poses[idx] = None
                continue

            bbox = None
            if tracking_results is not None and idx < len(tracking_results):
                bbox = tracking_results[idx].get("bbox")

            keypoints = self.estimate_single(frame, bbox=bbox)
            poses[idx] = keypoints

            if (idx + 1) % 100 == 0:
                logger.info("Pose estimated: %d / %d frames", idx + 1, len(frame_paths))

        detected = sum(1 for v in poses.values() if v is not None)
        logger.info(
            "Pose estimation complete: %d/%d frames with pose (%.1f%%)",
            detected,
            len(poses),
            100 * detected / max(len(poses), 1),
        )

        # Interpolate gaps
        poses = self._interpolate_gaps(poses)

        return poses

    def _interpolate_gaps(
        self, poses: dict[int, np.ndarray | None]
    ) -> dict[int, np.ndarray | None]:
        """Fill small gaps by linearly interpolating between neighbor poses."""
        indices = sorted(poses.keys())
        if not indices:
            return poses

        detected_indices = [i for i in indices if poses[i] is not None]
        if len(detected_indices) < 2:
            return poses

        for i in range(len(detected_indices) - 1):
            start_idx = detected_indices[i]
            end_idx = detected_indices[i + 1]
            gap = end_idx - start_idx

            if gap <= 1:
                continue
            # Only interpolate small gaps (up to 5 frames)
            if gap > 5:
                continue

            start_kp = poses[start_idx]
            end_kp = poses[end_idx]

            for g in range(1, gap):
                fill_idx = start_idx + g
                if fill_idx in poses and poses[fill_idx] is None:
                    t = g / gap
                    poses[fill_idx] = start_kp + t * (end_kp - start_kp)

        return poses

    @staticmethod
    def save_cache(
        poses: dict[int, np.ndarray | None],
        video_stem: str,
        cache_dir: str = "data/poses",
    ) -> str:
        """Save pose results to an npz cache file."""
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        out_file = cache_path / f"{video_stem}.npz"

        arrays = {}
        none_indices = []
        for idx, kp in poses.items():
            if kp is not None:
                arrays[f"frame_{idx:05d}"] = kp
            else:
                none_indices.append(idx)

        np.savez_compressed(str(out_file), none_indices=np.array(none_indices), **arrays)
        logger.info("Cached %d poses to %s", len(arrays), out_file)
        return str(out_file)

    @staticmethod
    def load_cache(cache_path: str) -> dict[int, np.ndarray | None]:
        """Load pose results from an npz cache file."""
        data = np.load(cache_path, allow_pickle=False)

        none_indices = set(data["none_indices"].tolist()) if "none_indices" in data else set()

        poses = {}
        for key in data.files:
            if key == "none_indices":
                continue
            idx = int(key.replace("frame_", ""))
            poses[idx] = data[key]

        for idx in none_indices:
            poses[idx] = None

        return poses
