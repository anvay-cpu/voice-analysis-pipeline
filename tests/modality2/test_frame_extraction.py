"""Tests for frame extraction and person detection modules."""

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.body.frame_extractor import extract_frames
from src.body.person_detector import PersonDetector, _iou


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def synthetic_video(tmp_path):
    """Create a short synthetic video (3 seconds, 30 fps, 640x480)."""
    video_path = tmp_path / "test_clip.mp4"
    fps = 30
    duration_sec = 3
    width, height = 640, 480
    total_frames = fps * duration_sec

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for i in range(total_frames):
        # Gradient frame so each frame is slightly different
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        shade = int(255 * i / total_frames)
        frame[:, :] = (shade, 128, 255 - shade)

        # Draw a rectangle to simulate a "person" region
        x1, y1 = 200, 100
        x2, y2 = 440, 400
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), -1)

        writer.write(frame)

    writer.release()
    return str(video_path)


# ── Frame Extraction Tests ────────────────────────────────────


class TestFrameExtraction:
    def test_extract_basic(self, synthetic_video, tmp_path):
        out_dir = str(tmp_path / "frames_out")
        result = extract_frames(
            synthetic_video, output_dir=out_dir, target_fps=5
        )

        assert result["total_frames"] > 0
        assert result["source_fps"] == 30.0
        assert result["target_fps"] == 5
        assert result["duration_sec"] == 3.0

        # At 5 fps for 3 seconds we expect 15 frames
        assert result["total_frames"] == 15
        assert len(result["frame_paths"]) == 15

    def test_frames_exist_on_disk(self, synthetic_video, tmp_path):
        out_dir = str(tmp_path / "frames_out2")
        result = extract_frames(
            synthetic_video, output_dir=out_dir, target_fps=5
        )

        for fpath in result["frame_paths"]:
            assert os.path.isfile(fpath), f"Frame file missing: {fpath}"

    def test_resolution_matches_target(self, synthetic_video, tmp_path):
        out_dir = str(tmp_path / "frames_out3")
        target_res = (320, 240)
        result = extract_frames(
            synthetic_video,
            output_dir=out_dir,
            target_fps=5,
            target_resolution=target_res,
        )

        frame = cv2.imread(result["frame_paths"][0])
        h, w = frame.shape[:2]
        assert (w, h) == target_res

    def test_frame_naming(self, synthetic_video, tmp_path):
        out_dir = str(tmp_path / "frames_out4")
        result = extract_frames(
            synthetic_video, output_dir=out_dir, target_fps=5
        )

        for i, fpath in enumerate(result["frame_paths"]):
            assert Path(fpath).name == f"frame_{i:05d}.jpg"

    def test_invalid_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            extract_frames("/nonexistent/video.mp4", output_dir=str(tmp_path))

    def test_unsupported_format_raises(self, tmp_path):
        fake = tmp_path / "video.gif"
        fake.touch()
        with pytest.raises(ValueError, match="Unsupported format"):
            extract_frames(str(fake), output_dir=str(tmp_path))

    def test_target_fps_clamped(self, synthetic_video, tmp_path):
        """target_fps higher than source should be clamped."""
        out_dir = str(tmp_path / "frames_out5")
        result = extract_frames(
            synthetic_video, output_dir=out_dir, target_fps=60
        )
        # Should clamp to source fps (30)
        assert result["target_fps"] == 30


# ── IoU Utility Test ──────────────────────────────────────────


class TestIoU:
    def test_identical_boxes(self):
        assert _iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0

    def test_no_overlap(self):
        assert _iou([0, 0, 5, 5], [10, 10, 20, 20]) == 0.0

    def test_partial_overlap(self):
        iou = _iou([0, 0, 10, 10], [5, 5, 15, 15])
        assert 0.0 < iou < 1.0


# ── Person Detection Tests (requires model download) ─────────


class TestPersonDetector:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.detector = PersonDetector(model_name="yolov8n", confidence=0.3)

    def test_detect_on_blank_frame(self):
        """A blank frame should return no persons."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = self.detector.detect_persons(blank)
        assert isinstance(detections, list)
        # Blank frame — likely no persons
        assert len(detections) == 0

    def test_select_speaker_empty(self):
        result = self.detector.select_speaker([])
        assert result is None

    def test_select_speaker_largest(self):
        detections = [
            {"bbox": [0, 0, 50, 50], "confidence": 0.9},
            {"bbox": [0, 0, 200, 200], "confidence": 0.8},
        ]
        speaker = self.detector.select_speaker(detections, strategy="largest")
        assert speaker["bbox"] == [0, 0, 200, 200]
