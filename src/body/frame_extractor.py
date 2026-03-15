"""Step 1: Video → frames extraction using OpenCV."""

import os
import logging
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".mp4", ".webm", ".avi", ".mov"}


def extract_frames(
    video_path: str,
    output_dir: str = "data/frames",
    target_fps: int = 5,
    target_resolution: tuple = (1280, 720),
) -> dict:
    """Extract frames from a video at a configurable fps and resolution.

    Args:
        video_path: Path to the input video file.
        output_dir: Base directory for extracted frames.
        target_fps: Frames per second to extract (default 5).
        target_resolution: (width, height) to resize frames to.

    Returns:
        Dictionary with frame metadata:
            frame_dir, frame_paths, total_frames, source_fps,
            target_fps, duration_sec, resolution
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    suffix = video_path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{suffix}'. Supported: {SUPPORTED_FORMATS}"
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_video_frames / source_fps if source_fps > 0 else 0.0

    if target_fps > source_fps:
        logger.warning(
            "target_fps (%d) > source_fps (%.1f); clamping to source",
            target_fps,
            source_fps,
        )
        target_fps = int(source_fps)

    frame_interval = source_fps / target_fps

    # Create output directory: output_dir/video_stem/
    frame_dir = Path(output_dir) / video_path.stem
    frame_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = []
    extracted_count = 0
    frame_idx = 0

    logger.info(
        "Extracting frames: %s (%.1fs, %.1f fps → %d fps)",
        video_path.name,
        duration_sec,
        source_fps,
        target_fps,
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only keep frames at the target interval
        if frame_idx >= extracted_count * frame_interval:
            # Resize if needed
            h, w = frame.shape[:2]
            tw, th = target_resolution
            if (w, h) != (tw, th):
                frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)

            fname = f"frame_{extracted_count:05d}.jpg"
            fpath = str(frame_dir / fname)
            cv2.imwrite(fpath, frame)
            frame_paths.append(fpath)
            extracted_count += 1

        frame_idx += 1

    cap.release()

    logger.info("Extracted %d frames to %s", extracted_count, frame_dir)

    return {
        "frame_dir": str(frame_dir),
        "frame_paths": frame_paths,
        "total_frames": extracted_count,
        "source_fps": source_fps,
        "target_fps": target_fps,
        "duration_sec": round(duration_sec, 2),
        "resolution": target_resolution,
    }
