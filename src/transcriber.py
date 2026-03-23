"""Speech-to-text transcription using Whisper with word-level timestamps."""

import torch
from transformers import pipeline as hf_pipeline


_whisper_pipe = None


def load_whisper(model_id: str = "openai/whisper-small", device: str = "auto"):
    """Load Whisper model as a HuggingFace pipeline (cached after first call).

    Args:
        model_id: HuggingFace model identifier.
        device: Device string — "mps", "cuda", or "cpu".

    Returns:
        HuggingFace automatic-speech-recognition pipeline.
    """
    global _whisper_pipe

    if _whisper_pipe is not None:
        return _whisper_pipe

    # Auto-detect device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # CUDA can use float16 for speed; MPS/CPU need float32
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=model_id,
        torch_dtype=torch_dtype,
        device=device,
    )

    _whisper_pipe = pipe
    return pipe


def transcribe(pipe, audio_path: str, language: str = "en",
               chunk_length_s: int = 30, stride_length_s: int = 5,
               batch_size: int = 1) -> dict:
    """Transcribe audio file with word-level timestamps.

    Args:
        pipe: Whisper HuggingFace pipeline.
        audio_path: Path to 16kHz mono WAV file.
        language: Language code for Whisper.
        chunk_length_s: Chunk length in seconds for long audio.
        stride_length_s: Stride overlap in seconds.
        batch_size: Batch size (1 recommended for M1).

    Returns:
        Dictionary with:
            - text: full transcript string
            - words: list of {"word": str, "start": float, "end": float}
    """
    result = pipe(
        audio_path,
        return_timestamps="word",
        chunk_length_s=chunk_length_s,
        stride_length_s=(stride_length_s, stride_length_s),
        batch_size=batch_size,
        generate_kwargs={"language": language},
    )

    # Parse chunks into word list
    words = []
    if "chunks" in result:
        for chunk in result["chunks"]:
            ts = chunk.get("timestamp", (None, None))
            if ts and ts[0] is not None:
                words.append({
                    "word": chunk["text"].strip(),
                    "start": round(float(ts[0]), 3),
                    "end": round(float(ts[1]), 3) if ts[1] is not None else round(float(ts[0]) + 0.5, 3),
                })

    return {
        "text": result.get("text", "").strip(),
        "words": words,
    }


def filter_by_vad(transcript: dict, speech_segments: list,
                  tolerance_sec: float = 0.15) -> dict:
    """Filter transcript words to only those within VAD speech regions.

    Removes Whisper hallucinations that fall outside detected speech regions.

    Args:
        transcript: Output from transcribe() with "text" and "words" keys.
        speech_segments: VAD speech segments, each with "start_sec" and "end_sec".
        tolerance_sec: Tolerance for matching word edges to segment boundaries.

    Returns:
        Filtered transcript dict with same structure.
    """
    if not speech_segments or not transcript.get("words"):
        return transcript

    filtered_words = []
    for word in transcript["words"]:
        word_mid = (word["start"] + word["end"]) / 2
        in_speech = any(
            (seg["start_sec"] - tolerance_sec) <= word_mid <= (seg["end_sec"] + tolerance_sec)
            for seg in speech_segments
        )
        if in_speech:
            filtered_words.append(word)

    filtered_text = " ".join(w["word"] for w in filtered_words)

    return {
        "text": filtered_text,
        "words": filtered_words,
    }
