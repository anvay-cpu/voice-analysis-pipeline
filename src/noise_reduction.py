"""Noise reduction for audio preprocessing using spectral gating."""

import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from pathlib import Path

from src.audio_extractor import extract_audio, validate_input
from src.vad import detect_speech_segments


def estimate_snr(audio: np.ndarray, speech_segments: list, sr: int) -> float:
    """Estimate Signal-to-Noise Ratio from speech vs non-speech regions.

    Args:
        audio: Audio signal as numpy array.
        speech_segments: List of dicts with start_sec/end_sec keys.
        sr: Sample rate.

    Returns:
        Estimated SNR in dB.
    """
    # Collect speech and noise samples
    speech_samples = []
    noise_samples = []

    for seg in speech_segments:
        start = int(seg["start_sec"] * sr)
        end = int(seg["end_sec"] * sr)
        speech_samples.append(audio[start:end])

    # Build noise mask (everything not speech)
    speech_mask = np.zeros(len(audio), dtype=bool)
    for seg in speech_segments:
        start = int(seg["start_sec"] * sr)
        end = int(seg["end_sec"] * sr)
        speech_mask[start:end] = True
    noise_samples_arr = audio[~speech_mask]

    if len(noise_samples_arr) == 0 or len(speech_samples) == 0:
        return 30.0  # assume clean if we can't estimate

    speech_concat = np.concatenate(speech_samples)
    speech_power = np.mean(speech_concat ** 2)
    noise_power = np.mean(noise_samples_arr ** 2)

    if noise_power < 1e-10:
        return 60.0  # essentially no noise

    snr_db = 10 * np.log10(speech_power / noise_power)
    return round(float(snr_db), 1)


def reduce_noise(
    audio_path: str,
    speech_segments: list,
    output_path: str,
    prop_decrease: float = 0.8,
) -> str:
    """Apply spectral gating noise reduction.

    Uses non-speech segments as noise profile. If none available,
    uses the first 0.5 seconds as noise estimate.

    Args:
        audio_path: Path to input WAV file.
        speech_segments: VAD speech segments.
        output_path: Path for cleaned output WAV.
        prop_decrease: Noise reduction strength (0-1).

    Returns:
        Path to the cleaned audio file.
    """
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)

    snr = estimate_snr(audio, speech_segments, sr)

    if snr > 30:
        # Clean enough — just copy
        sf.write(output_path, audio, sr)
        return output_path

    # Build noise clip from non-speech regions
    noise_clip = _get_noise_clip(audio, speech_segments, sr)

    # Apply spectral gating
    cleaned = nr.reduce_noise(
        y=audio,
        sr=sr,
        y_noise=noise_clip,
        prop_decrease=prop_decrease,
        n_fft=2048,
        hop_length=512,
    )

    sf.write(output_path, cleaned, sr)
    return output_path


def _get_noise_clip(
    audio: np.ndarray, speech_segments: list, sr: int
) -> np.ndarray:
    """Extract noise clip from non-speech regions or fallback to first 0.5s."""
    speech_mask = np.zeros(len(audio), dtype=bool)
    for seg in speech_segments:
        start = int(seg["start_sec"] * sr)
        end = int(seg["end_sec"] * sr)
        speech_mask[start:end] = True

    noise = audio[~speech_mask]

    if len(noise) < sr // 4:  # less than 0.25 seconds of noise
        # Fallback: use first 0.5 seconds
        noise = audio[: int(0.5 * sr)]

    return noise


def preprocess_audio(video_path: str, output_dir: str, config: dict) -> dict:
    """Master preprocessing function: extract -> VAD -> denoise.

    Chains Steps 1 (extraction), 2 (VAD), and 3 (noise reduction).

    Args:
        video_path: Path to input video file.
        output_dir: Base output directory for audio files.
        config: Full pipeline config dict.

    Returns:
        Dictionary with preprocessing results:
            - raw_audio_path: path to extracted WAV
            - clean_audio_path: path to noise-reduced WAV
            - speech_segments: list of speech segment dicts
            - pauses: list of pause dicts
            - duration_sec: total audio duration
            - snr_db: estimated SNR
            - noise_reduced: whether noise reduction was applied
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Validate and extract audio
    info = validate_input(video_path)
    raw_audio_path = extract_audio(video_path, str(output_dir))

    # Step 2: Voice activity detection
    vad_result = detect_speech_segments(raw_audio_path, config)

    # Step 3: Noise reduction
    audio, sr = librosa.load(raw_audio_path, sr=16000, mono=True)
    snr = estimate_snr(audio, vad_result["speech_segments"], sr)

    clean_audio_path = str(output_dir / f"{Path(video_path).stem}_clean.wav")

    if snr > 30:
        # Clean recording — skip noise reduction
        sf.write(clean_audio_path, audio, sr)
        noise_reduced = False
    else:
        reduce_noise(
            raw_audio_path,
            vad_result["speech_segments"],
            clean_audio_path,
        )
        noise_reduced = True

    return {
        "raw_audio_path": raw_audio_path,
        "clean_audio_path": clean_audio_path,
        "speech_segments": vad_result["speech_segments"],
        "pauses": vad_result["pauses"],
        "duration_sec": info["duration_sec"],
        "snr_db": snr,
        "noise_reduced": noise_reduced,
    }
