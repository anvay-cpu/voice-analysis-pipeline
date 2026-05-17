"""Master orchestrator for the body language analysis pipeline.

Chains: frames → detection → pose → posture → gesture →
        hands → gaze → face emotion → movement → assemble

Usage:
    python -m src.body.pipeline --video input.mp4
"""

import argparse
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class BodyAnalysisPipeline:
    """End-to-end body language analysis from video.

    Gracefully degrades: skips any module whose model is missing.
    """

    def __init__(self, device: str = "cpu", target_fps: int = 5):
        self.device = device
        self.target_fps = target_fps
        self._modules = {}
        self._load_modules()

    def _load_modules(self):
        """Load all sub-modules, skipping those that fail."""
        # Frame extraction (no model needed)
        try:
            from src.body.frame_extractor import extract_frames
            self._modules["frame_extractor"] = extract_frames
            logger.info("Loaded: frame_extractor")
        except ImportError as e:
            logger.warning("Skipping frame_extractor: %s", e)

        # Person detection
        try:
            from src.body.person_detector import PersonDetector
            self._modules["person_detector"] = PersonDetector()
            logger.info("Loaded: person_detector")
        except Exception as e:
            logger.warning("Skipping person_detector: %s", e)

        # Pose estimation
        try:
            from src.body.pose_estimator import PoseEstimator
            pe = PoseEstimator()
            if Path(pe.model_path).exists():
                self._modules["pose_estimator"] = pe
                logger.info("Loaded: pose_estimator")
            else:
                logger.warning("Skipping pose_estimator: model not found")
        except Exception as e:
            logger.warning("Skipping pose_estimator: %s", e)

        # Posture scoring
        try:
            from src.body.posture_scorer import PostureScorer
            self._modules["posture_scorer"] = PostureScorer(device=self.device)
            logger.info("Loaded: posture_scorer")
        except Exception as e:
            logger.warning("Skipping posture_scorer: %s", e)

        # Gesture classification
        try:
            from src.body.gesture_classifier import GestureClassifier
            self._modules["gesture_classifier"] = GestureClassifier(device=self.device)
            logger.info("Loaded: gesture_classifier")
        except Exception as e:
            logger.warning("Skipping gesture_classifier: %s", e)

        # Hand tracking
        try:
            from src.body.hand_tracker import HandTracker
            ht = HandTracker()
            if Path(ht.model_path).exists():
                self._modules["hand_tracker"] = ht
                logger.info("Loaded: hand_tracker")
            else:
                logger.warning("Skipping hand_tracker: model not found")
        except Exception as e:
            logger.warning("Skipping hand_tracker: %s", e)

        # Gaze estimation
        try:
            from src.body.gaze_estimator import GazeEstimator
            ge = GazeEstimator()
            if Path(ge.model_path).exists():
                self._modules["gaze_estimator"] = ge
                logger.info("Loaded: gaze_estimator")
            else:
                logger.warning("Skipping gaze_estimator: model not found")
        except Exception as e:
            logger.warning("Skipping gaze_estimator: %s", e)

        # Facial emotion
        try:
            from src.body.facial_emotion import FacialEmotionClassifier
            self._modules["facial_emotion"] = FacialEmotionClassifier(device=self.device)
            logger.info("Loaded: facial_emotion")
        except Exception as e:
            logger.warning("Skipping facial_emotion: %s", e)

        # Stage movement (no model needed)
        try:
            from src.body.stage_movement import StageMovementAnalyzer
            self._modules["stage_movement"] = StageMovementAnalyzer()
            logger.info("Loaded: stage_movement")
        except ImportError as e:
            logger.warning("Skipping stage_movement: %s", e)

        # Output assembler
        try:
            from src.body.output_assembler import OutputAssembler
            self._modules["output_assembler"] = OutputAssembler(fps=self.target_fps)
            logger.info("Loaded: output_assembler")
        except ImportError as e:
            logger.warning("Skipping output_assembler: %s", e)

        logger.info("Pipeline loaded: %d/%d modules", len(self._modules), 10)

    def process(self, video_path: str, output_dir: str = "data/outputs") -> dict:
        """Run complete body language analysis on a video.

        Args:
            video_path: Path to input video file.
            output_dir: Directory for output JSON.

        Returns:
            Assembled analysis dict.
        """
        start_time = time.time()
        video_stem = Path(video_path).stem
        logger.info("Starting body language analysis: %s", video_path)

        # Step 1: Extract frames
        frame_info = None
        frame_paths = []
        if "frame_extractor" in self._modules:
            try:
                logger.info("[1/9] Extracting frames...")
                frame_info = self._modules["frame_extractor"](
                    video_path,
                    output_dir=f"data/frames/{video_stem}",
                    target_fps=self.target_fps,
                )
                frame_paths = frame_info.get("frame_paths", [])
                logger.info("  → %d frames at %d fps", len(frame_paths), self.target_fps)
            except Exception as e:
                logger.error("[1/9] Frame extraction failed: %s", e)

        if not frame_paths:
            logger.error("No frames extracted — cannot proceed with body analysis")
            return {"segments": [], "summary": {}}

        n_frames = len(frame_paths)

        # Step 2: Person detection
        tracking = None
        if "person_detector" in self._modules:
            try:
                logger.info("[2/9] Detecting speaker...")
                tracking = self._modules["person_detector"].track_speaker(frame_paths)
                detected = sum(1 for t in tracking if t["detected"])
                logger.info("  → Speaker detected in %d/%d frames", detected, n_frames)
            except Exception as e:
                logger.warning("[2/9] Person detection failed: %s", e)

        # Step 3: Pose estimation
        poses = None
        if "pose_estimator" in self._modules:
            try:
                logger.info("[3/9] Estimating poses...")
                poses = self._modules["pose_estimator"].estimate_batch(frame_paths, tracking)
                detected = sum(1 for v in poses.values() if v is not None)
                logger.info("  → Pose in %d/%d frames", detected, n_frames)
            except Exception as e:
                logger.warning("[3/9] Pose estimation failed: %s", e)

        # Step 4: Posture scoring (needs poses)
        posture_scores = None
        if "posture_scorer" in self._modules and poses is not None:
            try:
                logger.info("[4/9] Scoring posture...")
                posture_scores = self._modules["posture_scorer"].score_batch(poses)
                scored = sum(1 for v in posture_scores.values() if v is not None)
                logger.info("  → Posture scored for %d frames", scored)
            except Exception as e:
                logger.warning("[4/9] Posture scoring failed: %s", e)

        # Step 5: Gesture classification (needs poses)
        gesture_results = None
        if "gesture_classifier" in self._modules and poses is not None:
            try:
                logger.info("[5/9] Classifying gestures...")
                gesture_results = self._modules["gesture_classifier"].classify_video(
                    poses, fps=self.target_fps
                )
                logger.info("  → %d gesture windows", len(gesture_results))
            except Exception as e:
                logger.warning("[5/9] Gesture classification failed: %s", e)

        # Step 6: Hand tracking (needs frames, optionally poses)
        hand_results = None
        if "hand_tracker" in self._modules:
            try:
                logger.info("[6/9] Tracking hands...")
                hand_results = self._modules["hand_tracker"].track_video(
                    frame_paths, poses=poses
                )
                logger.info("  → %d hand states", len(hand_results))
            except Exception as e:
                logger.warning("[6/9] Hand tracking failed: %s", e)

        # Step 7: Gaze estimation (needs frames)
        gaze_result = None
        if "gaze_estimator" in self._modules:
            try:
                logger.info("[7/9] Estimating gaze...")
                gaze_result = self._modules["gaze_estimator"].track_video(frame_paths)
                logger.info(
                    "  → Engagement: %.2f, detected %d/%d",
                    gaze_result["engagement_ratio"],
                    gaze_result["detected_frames"],
                    gaze_result["total_frames"],
                )
            except Exception as e:
                logger.warning("[7/9] Gaze estimation failed: %s", e)

        # Step 8: Facial emotion (needs frames)
        emotion_results = None
        if "facial_emotion" in self._modules:
            try:
                logger.info("[8/9] Classifying facial emotions...")
                emotion_results = self._modules["facial_emotion"].track_video(frame_paths)
                logger.info("  → %d emotion classifications", len(emotion_results))
            except Exception as e:
                logger.warning("[8/9] Facial emotion failed: %s", e)

        # Step 9: Stage movement (needs poses)
        movement_result = None
        if "stage_movement" in self._modules and poses is not None:
            try:
                logger.info("[9/9] Analyzing stage movement...")
                movement_result = self._modules["stage_movement"].analyze(
                    poses, fps=self.target_fps
                )
                logger.info(
                    "  → Pattern: %s, score: %.1f",
                    movement_result["movement_pattern"],
                    movement_result["stage_usage_score"],
                )
            except Exception as e:
                logger.warning("[9/9] Stage movement failed: %s", e)

        # Assemble output
        if "output_assembler" in self._modules:
            output = self._modules["output_assembler"].assemble(
                video_path=video_path,
                n_frames=n_frames,
                posture_scores=posture_scores,
                gesture_results=gesture_results,
                hand_results=hand_results,
                gaze_result=gaze_result,
                emotion_results=emotion_results,
                movement_result=movement_result,
            )

            # Save
            out_path = str(Path(output_dir) / f"{video_stem}_body.json")
            self._modules["output_assembler"].save(output, out_path)
        else:
            output = {
                "video": video_path,
                "total_frames": n_frames,
                "error": "output_assembler not available",
            }

        elapsed = time.time() - start_time
        logger.info(
            "Body language analysis complete in %.1fs (%.1f frames/sec)",
            elapsed, n_frames / max(elapsed, 0.1),
        )

        return output


def main():
    parser = argparse.ArgumentParser(description="Body Language Analysis Pipeline")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--device", default="cpu", help="Device: cpu, mps, cuda")
    parser.add_argument("--fps", type=int, default=5, help="Target FPS for frame extraction")
    parser.add_argument("--output-dir", default="data/outputs", help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    pipeline = BodyAnalysisPipeline(device=args.device, target_fps=args.fps)
    result = pipeline.process(args.video, output_dir=args.output_dir)

    print(json.dumps(result.get("summary", {}), indent=2))


if __name__ == "__main__":
    main()
