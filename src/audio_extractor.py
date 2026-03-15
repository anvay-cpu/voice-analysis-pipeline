"""Extract audio from video files and convert to 16kHz mono WAV."""

import json
import subprocess
from pathlib import Path

SUPPORTED_FORMATS = {".mp4", ".webm", ".mkv", ".avi", ".mov"}
MAX_SIZE_MB = 2048  # 2 GB
MAX_DURATION_SEC = 3600  # 1 hour


def _run_ffprobe(file_path: str) -> dict:
    """Run ffprobe and return parsed JSON output."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(file_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return json.loads(result.stdout)


def get_duration(file_path: str) -> float:
    """Get the duration of a media file in seconds."""
    info = _run_ffprobe(file_path)
    return float(info["format"]["duration"])


def validate_input(video_path: str) -> dict:
    """Validate video file exists, is a supported format, and within limits.

    Returns:
        dict with duration_sec, size_mb, format keys.

    Raises:
        ValueError: If the file is invalid, too large, or too long.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(video_path)

    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{suffix}'. Supported: {SUPPORTED_FORMATS}"
        )

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_SIZE_MB:
        raise ValueError(f"File too large: {size_mb:.1f}MB (max {MAX_SIZE_MB}MB)")

    duration_sec = get_duration(video_path)
    if duration_sec > MAX_DURATION_SEC:
        raise ValueError(
            f"File too long: {duration_sec:.0f}s (max {MAX_DURATION_SEC}s)"
        )

    return {
        "duration_sec": duration_sec,
        "size_mb": size_mb,
        "format": suffix,
    }


def extract_audio(video_path: str, output_dir: str = "data/audio") -> str:
    """Extract audio from video as 16kHz mono WAV using FFmpeg.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory to save the extracted WAV.

    Returns:
        Path to the output WAV file.

    Raises:
        RuntimeError: If FFmpeg extraction fails.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{video_path.stem}.wav"

    cmd = [
        "ffmpeg",
        "-y",              # overwrite output
        "-i", str(video_path),
        "-vn",             # no video
        "-acodec", "pcm_s16le",
        "-ar", "16000",    # 16kHz
        "-ac", "1",        # mono
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr}")

    if not output_path.exists():
        raise RuntimeError(f"Output file was not created: {output_path}")

    return str(output_path)
