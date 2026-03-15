"""Assemble all analysis results into timestamped JSON output."""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path


def assemble_voice_output(
    duration_sec: float,
    transcript: dict,
    speech_segments: list[dict],
    fillers: list,
    disfluencies: list,
    prosody_results: list,
    emotion_results: list,
    features_windowed: list[dict],
    config: dict,
) -> list[dict]:
    """Assemble per-window JSON dicts from all analysis modules.

    Iterates 2-second windows with 1-second hop. For each window, collects
    transcript words, filler words, disfluencies, prosody, emotion, and
    acoustic features that overlap with that window.

    Args:
        duration_sec: Total audio duration in seconds.
        transcript: Dict with "text" and "words" keys.
        speech_segments: VAD speech segments.
        fillers: List of FillerDetection or VerifiedFiller objects.
        disfluencies: List of DisfluencyEvent objects.
        prosody_results: List of ProsodyResult objects.
        emotion_results: List of EmotionResult or dicts.
        features_windowed: List of per-window feature dicts.
        config: Pipeline config dict.

    Returns:
        List of per-window dicts with all analysis data.
    """
    output_cfg = config.get("output", {})
    window_sec = output_cfg.get("window_sec", 2.0)
    hop_sec = output_cfg.get("hop_sec", 1.0)

    words = transcript.get("words", [])
    windows = []
    pos = 0.0

    while pos + window_sec <= duration_sec + 0.01:
        win_start = round(pos, 3)
        win_end = round(pos + window_sec, 3)

        # Words in this window
        win_words = [w for w in words
                     if w["start"] >= win_start - 0.05 and w["end"] <= win_end + 0.05]
        win_text = " ".join(w["word"] for w in win_words)

        # Is speech present?
        is_speech = any(
            seg["start_sec"] < win_end and seg["end_sec"] > win_start
            for seg in speech_segments
        )

        # Fillers in this window
        win_fillers = [
            {"word": f.word, "start": f.start_sec, "end": f.end_sec,
             "confidence": f.confidence, "category": f.category}
            for f in fillers
            if f.start_sec >= win_start - 0.05 and f.start_sec < win_end
        ]

        # Disfluencies in this window
        win_disfluencies = [
            {"type": d.type, "start": d.start_sec, "end": d.end_sec,
             "confidence": d.confidence}
            for d in disfluencies
            if d.start_sec >= win_start - 0.05 and d.start_sec < win_end
        ]

        # Prosody overlapping this window (prosody uses larger windows)
        win_prosody = _find_overlapping_prosody(prosody_results, win_start, win_end)

        # Emotion overlapping this window
        win_emotion = _find_overlapping_emotion(emotion_results, win_start, win_end)

        # Acoustic features overlapping this window
        win_features = _find_overlapping_features(features_windowed, win_start, win_end)

        window_dict = {
            "start_sec": win_start,
            "end_sec": win_end,
            "is_speech": is_speech,
            "transcript": win_text,
            "word_count": len(win_words),
            "fillers": win_fillers,
            "disfluencies": win_disfluencies,
            "prosody": win_prosody,
            "emotion": win_emotion,
            "features": win_features,
        }
        windows.append(window_dict)
        pos += hop_sec

    return windows


def _find_overlapping_prosody(prosody_results, win_start, win_end):
    """Find the prosody result that best overlaps with this window."""
    if not prosody_results:
        return {"available": False}

    best = None
    best_overlap = 0.0
    for p in prosody_results:
        overlap_start = max(p.start_sec, win_start)
        overlap_end = min(p.end_sec, win_end)
        overlap = max(0, overlap_end - overlap_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best = p

    if best is None:
        return {"available": False}

    return {
        "available": True,
        "syllable_rate": best.syllable_rate,
        "rate_label": best.rate_label,
        "pitch_cv": best.pitch_cv,
        "pitch_mean": best.pitch_mean,
        "pitch_assessment": best.pitch_assessment,
        "pitch_score": best.pitch_score,
        "volume_assessment": best.volume_assessment,
        "volume_score": best.volume_score,
    }


def _find_overlapping_emotion(emotion_results, win_start, win_end):
    """Find the emotion result that best overlaps with this window."""
    if not emotion_results:
        return {"available": False}

    best = None
    best_overlap = 0.0
    for e in emotion_results:
        # Support both EmotionResult objects and dicts
        e_start = getattr(e, "start_sec", None) if hasattr(e, "start_sec") else e.get("start_sec", 0)
        e_end = getattr(e, "end_sec", None) if hasattr(e, "end_sec") else e.get("end_sec", 0)
        overlap_start = max(e_start, win_start)
        overlap_end = min(e_end, win_end)
        overlap = max(0, overlap_end - overlap_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best = e

    if best is None:
        return {"available": False}

    label = getattr(best, "label", None) if hasattr(best, "label") else best.get("label", "Unknown")
    confidence = getattr(best, "confidence", None) if hasattr(best, "confidence") else best.get("confidence", 0.0)
    distribution = getattr(best, "distribution", None) if hasattr(best, "distribution") else best.get("distribution", {})

    return {
        "available": True,
        "label": label,
        "confidence": confidence,
        "distribution": distribution,
    }


def _find_overlapping_features(features_windowed, win_start, win_end):
    """Find acoustic features overlapping this window."""
    if not features_windowed:
        return {}

    best = None
    best_overlap = 0.0
    for f in features_windowed:
        f_start = f.get("start_sec", 0)
        f_end = f.get("end_sec", 0)
        overlap_start = max(f_start, win_start)
        overlap_end = min(f_end, win_end)
        overlap = max(0, overlap_end - overlap_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best = f

    if best is None:
        return {}

    # Extract scalar summary features (skip contours/arrays)
    result = {}
    f0 = best.get("f0", {})
    if f0:
        result["f0_mean"] = f0.get("f0_mean", 0.0)
        result["f0_std"] = f0.get("f0_std", 0.0)
        result["f0_cv"] = f0.get("f0_cv", 0.0)

    energy = best.get("energy", {})
    if energy:
        result["rms_mean_db"] = energy.get("rms_mean_db", 0.0)
        result["dynamic_range_db"] = energy.get("dynamic_range_db", 0.0)

    vq = best.get("voice_quality", {})
    if vq:
        result["jitter_percent"] = vq.get("jitter_percent", 0.0)
        result["shimmer_percent"] = vq.get("shimmer_percent", 0.0)
        result["hnr_db"] = vq.get("hnr_db", 0.0)

    return result


def generate_summary(windows: list[dict], duration_sec: float,
                     transcript: dict) -> dict:
    """Generate a summary dict from assembled windows.

    Args:
        windows: List of per-window dicts from assemble_voice_output().
        duration_sec: Total audio duration in seconds.
        transcript: Dict with "text" and "words".

    Returns:
        Summary dict with aggregate metrics.
    """
    words = transcript.get("words", [])
    word_count = len(words)
    duration_min = duration_sec / 60.0 if duration_sec > 0 else 1.0

    # Speaking rate (words per minute)
    wpm = round(word_count / duration_min, 1) if duration_min > 0 else 0.0

    # Filler stats
    all_fillers = []
    for w in windows:
        all_fillers.extend(w.get("fillers", []))
    # Deduplicate by start time
    seen_fillers = set()
    unique_fillers = []
    for f in all_fillers:
        key = (f["start"], f["end"])
        if key not in seen_fillers:
            seen_fillers.add(key)
            unique_fillers.append(f)
    filler_count = len(unique_fillers)
    fillers_per_minute = round(filler_count / duration_min, 2)

    # Disfluency stats
    all_disfluencies = []
    for w in windows:
        all_disfluencies.extend(w.get("disfluencies", []))
    seen_dis = set()
    unique_dis = []
    for d in all_disfluencies:
        key = (d["type"], d["start"], d["end"])
        if key not in seen_dis:
            seen_dis.add(key)
            unique_dis.append(d)
    disfluency_count = len(unique_dis)
    disfluency_types = dict(Counter(d["type"] for d in unique_dis))

    # Prosody averages
    prosody_windows = [w["prosody"] for w in windows if w["prosody"].get("available")]
    avg_syllable_rate = 0.0
    avg_pitch_cv = 0.0
    avg_pitch_score = 0
    avg_volume_score = 0
    if prosody_windows:
        avg_syllable_rate = round(
            sum(p["syllable_rate"] for p in prosody_windows) / len(prosody_windows), 2)
        avg_pitch_cv = round(
            sum(p["pitch_cv"] for p in prosody_windows) / len(prosody_windows), 4)
        avg_pitch_score = round(
            sum(p["pitch_score"] for p in prosody_windows) / len(prosody_windows))
        avg_volume_score = round(
            sum(p["volume_score"] for p in prosody_windows) / len(prosody_windows))

    # Emotion stats
    emotion_windows = [w["emotion"] for w in windows if w["emotion"].get("available")]
    dominant_emotion = "unavailable"
    emotion_variety = 0.0
    emotion_distribution = {}
    if emotion_windows:
        labels = [e["label"] for e in emotion_windows]
        label_counts = Counter(labels)
        dominant_emotion = label_counts.most_common(1)[0][0]
        unique_emotions = len(label_counts)
        emotion_variety = round(unique_emotions / max(len(labels), 1), 2)
        total = len(labels)
        emotion_distribution = {k: round(v / total, 3) for k, v in label_counts.items()}

    return {
        "duration_sec": round(duration_sec, 2),
        "word_count": word_count,
        "words_per_minute": wpm,
        "filler_count": filler_count,
        "fillers_per_minute": fillers_per_minute,
        "disfluency_count": disfluency_count,
        "disfluency_types": disfluency_types,
        "avg_syllable_rate": avg_syllable_rate,
        "avg_pitch_cv": avg_pitch_cv,
        "avg_pitch_score": avg_pitch_score,
        "avg_volume_score": avg_volume_score,
        "dominant_emotion": dominant_emotion,
        "emotion_variety": emotion_variety,
        "emotion_distribution": emotion_distribution,
    }


def save_output(windows: list[dict], summary: dict, metadata: dict,
                output_path: str) -> str:
    """Save complete analysis output as JSON.

    Args:
        windows: Per-window analysis dicts.
        summary: Summary statistics dict.
        metadata: Pipeline metadata (video path, timestamps, versions, etc.).
        output_path: Path to save the JSON file.

    Returns:
        Path to the saved JSON file.
    """
    output = {
        "metadata": metadata,
        "summary": summary,
        "windows": windows,
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    return str(path)
