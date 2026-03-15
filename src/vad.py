"""Voice Activity Detection using Silero VAD."""

import torch
import soundfile as sf
import numpy as np

_vad_cache = None


def load_vad_model() -> tuple:
    """Load Silero VAD model from torch.hub (cached after first download).

    Returns:
        Tuple of (model, get_speech_timestamps function, read_audio function).
    """
    global _vad_cache

    if _vad_cache is not None:
        return _vad_cache

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    get_speech_timestamps = utils[0]
    read_audio = utils[2]

    _vad_cache = (model, get_speech_timestamps, read_audio)
    return _vad_cache


def detect_speech_segments(audio_path: str, config: dict) -> dict:
    """Detect speech and silence segments in an audio file.

    Args:
        audio_path: Path to 16kHz mono WAV file.
        config: Pipeline config dict (uses 'vad' section).

    Returns:
        Dictionary containing:
            - speech_segments: list of {start_sec, end_sec, duration_sec}
            - pauses: list of {start_sec, end_sec, duration_sec}
            - total_speech_sec: total speech duration
            - total_silence_sec: total silence duration
            - speech_ratio: ratio of speech to total duration
    """
    vad_config = config.get("vad", {})
    threshold = vad_config.get("threshold", 0.5)
    min_speech_ms = vad_config.get("min_speech_duration_ms", 250)
    min_silence_ms = vad_config.get("min_silence_duration_ms", 100)

    model, get_speech_timestamps, _ = load_vad_model()

    # Load audio using soundfile (avoids torchcodec dependency)
    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)  # to mono
    if sr != 16000:
        # Resample with scipy
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(16000, sr)
        data = resample_poly(data, 16000 // g, sr // g).astype(np.float32)
        sr = 16000
    wav = torch.from_numpy(data)

    total_duration = len(wav) / sr

    # Get speech timestamps (in samples)
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        threshold=threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
        sampling_rate=sr,
    )

    # Convert to seconds
    speech_segments = []
    for ts in speech_timestamps:
        start_sec = ts["start"] / sr
        end_sec = ts["end"] / sr
        speech_segments.append({
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
            "duration_sec": round(end_sec - start_sec, 3),
        })

    # Compute pauses between speech segments
    pauses = []
    for i in range(1, len(speech_segments)):
        pause_start = speech_segments[i - 1]["end_sec"]
        pause_end = speech_segments[i]["start_sec"]
        if pause_end > pause_start:
            pauses.append({
                "start_sec": round(pause_start, 3),
                "end_sec": round(pause_end, 3),
                "duration_sec": round(pause_end - pause_start, 3),
            })

    total_speech = sum(s["duration_sec"] for s in speech_segments)
    total_silence = total_duration - total_speech

    return {
        "speech_segments": speech_segments,
        "pauses": pauses,
        "total_speech_sec": round(total_speech, 3),
        "total_silence_sec": round(total_silence, 3),
        "speech_ratio": round(total_speech / total_duration, 3) if total_duration > 0 else 0.0,
    }
