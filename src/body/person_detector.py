"""Step 2: Detect and track speaker using YOLOv8-nano."""

import logging
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

PERSON_CLASS_ID = 0
MAX_INTERPOLATION_GAP = 5


def _iou(box_a, box_b):
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def _bbox_area(bbox):
    """Compute area of [x1, y1, x2, y2] bbox."""
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _interpolate_bbox(bbox_start, bbox_end, t):
    """Linearly interpolate between two bboxes. t in [0, 1]."""
    return [
        bbox_start[i] + (bbox_end[i] - bbox_start[i]) * t
        for i in range(4)
    ]


class PersonDetector:
    """Detect persons in frames and track the primary speaker."""

    def __init__(self, model_name: str = "yolov8n", confidence: float = 0.5):
        self.confidence = confidence
        model_file = f"{model_name}.pt"
        logger.info("Loading YOLO model: %s", model_file)
        self.model = YOLO(model_file)

    def detect_persons(self, frame: np.ndarray) -> list[dict]:
        """Detect all persons in a single frame.

        Args:
            frame: BGR image as numpy array.

        Returns:
            List of dicts with keys: bbox [x1,y1,x2,y2], confidence.
        """
        results = self.model(frame, verbose=False, conf=self.confidence)

        persons = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                if cls_id != PERSON_CLASS_ID:
                    continue
                conf = float(boxes.conf[i].item())
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                persons.append({"bbox": bbox, "confidence": conf})

        return persons

    def select_speaker(
        self, detections: list[dict], strategy: str = "largest"
    ) -> dict | None:
        """Select the primary speaker from detected persons.

        Args:
            detections: List of person detections from detect_persons().
            strategy: Selection strategy — "largest" picks the largest bbox.

        Returns:
            The selected person dict, or None if no detections.
        """
        if not detections:
            return None

        if strategy == "largest":
            return max(detections, key=lambda d: _bbox_area(d["bbox"]))

        raise ValueError(f"Unknown strategy: {strategy}")

    def track_speaker(
        self,
        frame_paths: list[str],
        strategy: str = "largest",
        iou_threshold: float = 0.3,
    ) -> list[dict]:
        """Track the primary speaker across all extracted frames.

        Uses IoU-based tracking: if the best candidate in the current frame
        has IoU > threshold with the previous speaker bbox, consider it the
        same person. If detection drops for <= MAX_INTERPOLATION_GAP frames,
        interpolate the bbox linearly.

        Args:
            frame_paths: Ordered list of frame image paths.
            strategy: Speaker selection strategy.
            iou_threshold: Minimum IoU to consider same person.

        Returns:
            Per-frame list of dicts:
                frame_idx, bbox [x1,y1,x2,y2], confidence, detected (bool)
        """
        tracking_results = []
        last_detected_bbox = None
        last_detected_idx = None

        for idx, fpath in enumerate(frame_paths):
            frame = cv2.imread(fpath)
            if frame is None:
                logger.warning("Could not read frame: %s", fpath)
                tracking_results.append({
                    "frame_idx": idx,
                    "bbox": None,
                    "confidence": 0.0,
                    "detected": False,
                })
                continue

            persons = self.detect_persons(frame)
            speaker = self.select_speaker(persons, strategy=strategy)

            if speaker is not None:
                # If we have a previous bbox, verify IoU continuity
                if last_detected_bbox is not None:
                    iou = _iou(speaker["bbox"], last_detected_bbox)
                    if iou < iou_threshold and len(persons) > 1:
                        # Try to find a better match among all detections
                        best_match = max(
                            persons,
                            key=lambda d: _iou(d["bbox"], last_detected_bbox),
                        )
                        if _iou(best_match["bbox"], last_detected_bbox) >= iou_threshold:
                            speaker = best_match

                # Backfill interpolation for any gap
                if (
                    last_detected_idx is not None
                    and idx - last_detected_idx > 1
                    and idx - last_detected_idx <= MAX_INTERPOLATION_GAP
                ):
                    gap = idx - last_detected_idx
                    for g in range(1, gap):
                        t = g / gap
                        interp_bbox = _interpolate_bbox(
                            last_detected_bbox, speaker["bbox"], t
                        )
                        fill_idx = last_detected_idx + g
                        tracking_results[fill_idx] = {
                            "frame_idx": fill_idx,
                            "bbox": interp_bbox,
                            "confidence": 0.0,
                            "detected": False,  # interpolated, not detected
                        }

                last_detected_bbox = speaker["bbox"]
                last_detected_idx = idx

                tracking_results.append({
                    "frame_idx": idx,
                    "bbox": speaker["bbox"],
                    "confidence": speaker["confidence"],
                    "detected": True,
                })
            else:
                # No detection this frame
                if (
                    last_detected_idx is not None
                    and idx - last_detected_idx > MAX_INTERPOLATION_GAP
                ):
                    logger.warning(
                        "Speaker lost for >%d frames at frame %d",
                        MAX_INTERPOLATION_GAP,
                        idx,
                    )

                tracking_results.append({
                    "frame_idx": idx,
                    "bbox": None,
                    "confidence": 0.0,
                    "detected": False,
                })

            if (idx + 1) % 100 == 0:
                logger.info("Tracked %d / %d frames", idx + 1, len(frame_paths))

        detected_count = sum(1 for r in tracking_results if r["detected"])
        logger.info(
            "Tracking complete: %d/%d frames with detection (%.1f%%)",
            detected_count,
            len(tracking_results),
            100 * detected_count / max(len(tracking_results), 1),
        )

        return tracking_results
