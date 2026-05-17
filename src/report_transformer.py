"""Transform raw pipeline output into the SpeechReport JSON format expected by the webapp.

Maps output from SpeechCoachPipeline into the TypeScript SpeechReport interface
used by the Bloomberg-style frontend.
"""

import hashlib
import time


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _safe_get(d: dict | None, *keys, default=None):
    """Safely traverse nested dicts."""
    val = d
    for k in keys:
        if not isinstance(val, dict):
            return default
        val = val.get(k, default)
    return val


def generate_session_id() -> str:
    """Generate a unique session ID like SES_XXXX_Y."""
    h = hashlib.md5(str(time.time()).encode()).hexdigest()[:4].upper()
    suffix = chr(65 + int(h[0], 16) % 26)
    return f"SES_{h}_{suffix}"


def transform_pipeline_output(
    voice_output: dict,
    body_output: dict,
    content_output: dict,
    fusion_output: dict,
    scores: dict,
    coaching: dict,
    session_id: str | None = None,
    title: str = "Speech Analysis",
    video_url: str | None = None,
    processing_time_sec: float = 0.0,
) -> dict:
    """Transform raw pipeline outputs into the webapp's SpeechReport format.

    Returns a dict matching the TypeScript SpeechReport interface.
    """
    session_id = session_id or generate_session_id()
    duration = fusion_output.get("duration_sec", 300.0)
    timeline = fusion_output.get("timeline", [])

    report = {
        "id": session_id,
        "title": title,
        "duration_sec": duration,
        "processing_time_sec": processing_time_sec,
        "video_url": video_url,
        "scores": _transform_scores(scores),
        "content_details": _transform_content_details(content_output),
        "fusion_stats": _transform_fusion_stats(fusion_output),
        "executive_summary": coaching.get("executive_summary", ""),
        "regime_segments": _transform_regime_segments(content_output, fusion_output),
        "regime_transitions": _transform_transitions(fusion_output),
        "disruption_events": _transform_disruptions(fusion_output),
        "coaching_feedback": _transform_coaching(coaching),
        "practice_plan": _transform_practice_plan(coaching),
        "filler_words": _transform_fillers(voice_output),
        "transcript": _extract_transcript(voice_output),
        "timeline": _transform_timeline(timeline),
        "emotion_arc": _transform_emotion_arc(timeline, fusion_output),
        "heatmap_channels": _build_heatmap_channels(timeline),
    }
    return report


def _transform_scores(scores: dict) -> dict:
    return {
        "vocal_clarity": scores.get("vocal_clarity", 0),
        "body_language": scores.get("body_language", 0),
        "content_structure": scores.get("content_structure", 0),
        "audience_engagement": scores.get("audience_engagement", 0),
        "emotional_expressiveness": scores.get("emotional_expressiveness", 0),
        "regime_adaptability": scores.get("regime_adaptability", 0),
        "overall": scores.get("overall", 0),
    }


def _transform_content_details(content_output: dict) -> dict:
    summary = content_output.get("summary", {})
    return {
        "grammar_score": summary.get("grammar_score", 0),
        "fkgl": summary.get("readability", {}).get("fkgl", 0)
        if isinstance(summary.get("readability"), dict)
        else summary.get("fkgl", 0),
        "ttr": summary.get("vocabulary", {}).get("ttr", 0)
        if isinstance(summary.get("vocabulary"), dict)
        else summary.get("ttr", 0),
        "dominant_tone": summary.get("dominant_tone", "Unknown"),
        "dominant_sentiment": summary.get("dominant_sentiment", "neutral"),
        "argument_effectiveness": summary.get("argument_effectiveness", 0),
    }


def _transform_fusion_stats(fusion_output: dict) -> dict:
    recovery = fusion_output.get("recovery", {})
    coherence = fusion_output.get("emotion_coherence", {})
    return {
        "timeline_duration": fusion_output.get("duration_sec", 0),
        "regime_boundaries": len(fusion_output.get("regime_boundaries", [])),
        "total_disruptions": recovery.get("total_disruptions", 0),
        "mean_composure": recovery.get("mean_composure", 0),
        "emotion_coherence": (coherence.get("coherence_score_0_100") or 0) / 100.0
        if coherence.get("coherence_score_0_100") is not None
        else coherence.get("mean_coherence", 0),
    }


def _transform_regime_segments(content_output: dict, fusion_output: dict) -> list:
    """Extract regime segments from content output or fusion boundaries."""
    segments = content_output.get("segments", [])
    if not segments:
        return []

    result = []
    for seg in segments:
        regime_type = seg.get("regime_type") or seg.get("label", "Unknown")
        start = seg.get("start_sec", 0)
        end = seg.get("end_sec", start + seg.get("duration_sec", 0))
        result.append({
            "regime_type": regime_type,
            "start_sec": start,
            "end_sec": end,
            "duration_sec": end - start,
        })
    return result


def _transform_transitions(fusion_output: dict) -> list:
    transitions = fusion_output.get("transitions", {})
    scored = transitions.get("scored_transitions", [])
    result = []
    for t in scored:
        result.append({
            "time_sec": t.get("boundary_sec", 0),
            "from_regime": t.get("from_regime", ""),
            "to_regime": t.get("to_regime", ""),
            "score": t.get("score", 0),
        })
    return result


def _transform_disruptions(fusion_output: dict) -> list:
    recovery = fusion_output.get("recovery", {})
    disruptions = recovery.get("disruptions", [])
    result = []
    for d in disruptions:
        t = d.get("time_sec", 0)
        composure = d.get("composure")
        recovery_sec = d.get("vocal_recovery_sec") or d.get("recovery_sec")
        status = "GOOD"
        if composure is not None and composure < 0.5:
            status = "SLOW"
        if d.get("type") == "emotion_mismatch":
            status = "MISMATCH"
        result.append({
            "time_sec": t,
            "timestamp": _format_timestamp(t),
            "type": d.get("type", "disfluency"),
            "label": d.get("label", d.get("type", "")),
            "composure": composure,
            "recovery_sec": recovery_sec,
            "status": status,
        })
    return result


def _transform_coaching(coaching: dict) -> list:
    feedback_list = coaching.get("segment_feedback", [])
    result = []
    for fb in feedback_list:
        t = fb.get("time_sec", 0)
        result.append({
            "time_sec": t,
            "timestamp": _format_timestamp(t),
            "type": fb.get("type", "general"),
            "label": fb.get("label", ""),
            "feedback": fb.get("feedback", fb.get("text", "")),
        })
    return result


def _transform_practice_plan(coaching: dict) -> list:
    plan = coaching.get("practice_plan", [])
    result = []
    for ex in plan:
        result.append({
            "title": ex.get("title", ""),
            "description": ex.get("description", ""),
            "frequency": ex.get("frequency", ""),
            "target_dimension": ex.get("target_dimension"),
            "target_improvement": ex.get("target_improvement"),
        })
    return result


def _transform_fillers(voice_output: dict) -> list:
    """Extract filler words from voice pipeline output."""
    if isinstance(voice_output, list):
        windows = voice_output
    else:
        windows = voice_output.get("windows", [])

    fillers = []
    for w in windows:
        for f in w.get("fillers", []):
            t = f.get("start", f.get("time_sec", w.get("start_sec", 0)))
            fillers.append({
                "time_sec": t,
                "timestamp": _format_timestamp(t),
                "word": f.get("word", "um"),
                "confidence": f.get("confidence", 0.0),
                "category": f.get("category", ""),
                "context": f.get("context", ""),
            })
    return fillers


def _extract_transcript(voice_output: dict) -> str:
    """Extract full transcript text."""
    if isinstance(voice_output, dict):
        transcript = voice_output.get("transcript")
        if isinstance(transcript, str):
            return transcript
        if isinstance(transcript, dict):
            return transcript.get("text", "")
        # Reconstruct from windows
        windows = voice_output.get("windows", [])
        parts = [w.get("text", "") for w in windows if w.get("text")]
        if parts:
            return " ".join(parts)
    return ""


def _transform_timeline(timeline: list) -> list:
    """Transform fusion timeline to per-second TimelinePoint format."""
    result = []
    for i, tp in enumerate(timeline):
        voice = tp.get("voice") or {}
        body = tp.get("body") or {}
        content = tp.get("content") or {}

        result.append({
            "time_sec": i,
            "voice": {
                "syllable_rate": voice.get("syllable_rate", 0) or voice.get("rate", 0),
                "pitch_cv": voice.get("pitch_cv", 0) or voice.get("f0_cv", 0),
                "emotion_label": voice.get("emotion_label", voice.get("emotion", "neutral")),
                "filler_count": voice.get("filler_count", 0),
                "disfluency_count": voice.get("disfluency_count", 0),
            },
            "body": {
                "posture_score": body.get("posture_score", 50),
                "gesture_dominant": body.get("gesture_dominant", body.get("gesture", "rest")),
                "gaze_engagement": body.get("gaze_engagement", 0.5),
                "facial_emotion": body.get("facial_emotion", body.get("emotion", "neutral")),
            },
            "content": {
                "regime_type": content.get("regime_type", content.get("regime", "Unknown")),
                "grammar_error_count": content.get("grammar_error_count", 0),
            },
        })
    return result


def _transform_emotion_arc(timeline: list, fusion_output: dict) -> list:
    """Build emotion arc from timeline or coherence data."""
    coherence = fusion_output.get("emotion_coherence", {})
    per_second = coherence.get("per_second", [])

    if per_second:
        return [
            {
                "time_sec": p.get("time_sec", i),
                "voice_valence": p.get("voice_valence", 0),
                "face_valence": p.get("face_valence", 0),
                "text_valence": p.get("text_valence", 0),
            }
            for i, p in enumerate(per_second)
        ]

    # Fallback: derive from timeline
    result = []
    for i, tp in enumerate(timeline):
        voice = tp.get("voice") or {}
        body = tp.get("body") or {}
        content = tp.get("content") or {}

        # Map emotion labels to valence
        voice_val = _emotion_to_valence(voice.get("emotion_label", "neutral"))
        face_val = _emotion_to_valence(body.get("facial_emotion", "neutral"))
        text_val = content.get("sentiment_valence", 0.0)

        result.append({
            "time_sec": i,
            "voice_valence": voice_val,
            "face_valence": face_val,
            "text_valence": text_val,
        })
    return result


def _emotion_to_valence(emotion: str) -> float:
    """Convert emotion label to -1..+1 valence."""
    mapping = {
        "happy": 0.8, "joy": 0.8, "surprise": 0.3,
        "neutral": 0.0, "calm": 0.1,
        "sad": -0.6, "angry": -0.7, "fear": -0.5, "disgust": -0.6,
    }
    return mapping.get(emotion.lower(), 0.0) if emotion else 0.0


def _build_heatmap_channels(timeline: list) -> list:
    """Build heatmap channels from timeline data."""
    if not timeline:
        return []

    n = len(timeline)
    channels = [
        {"label": "Speaking Rate", "key": "rate", "values": []},
        {"label": "Pitch Variety", "key": "pitch", "values": []},
        {"label": "Gaze Engagement", "key": "gaze", "values": []},
        {"label": "Filler Density", "key": "fillers", "values": []},
        {"label": "Posture Quality", "key": "posture", "values": []},
        {"label": "Emotion Coherence", "key": "coherence", "values": []},
    ]

    for tp in timeline:
        voice = tp.get("voice") or {}
        body = tp.get("body") or {}

        # Normalize speaking rate to 0-1 (4 syl/s = good)
        rate = voice.get("syllable_rate", 0) or voice.get("rate", 0)
        rate_norm = min(1.0, rate / 6.0) if rate else 0.5
        channels[0]["values"].append(round(rate_norm, 3))

        # Pitch CV to quality (0.15-0.35 = good range)
        pitch_cv = voice.get("pitch_cv", 0) or voice.get("f0_cv", 0)
        pitch_q = min(1.0, pitch_cv / 0.4) if pitch_cv else 0.5
        channels[1]["values"].append(round(pitch_q, 3))

        # Gaze engagement directly 0-1
        gaze = body.get("gaze_engagement", 0.5)
        channels[2]["values"].append(round(gaze, 3))

        # Filler density (inverse — more fillers = lower quality)
        fillers = voice.get("filler_count", 0)
        filler_q = max(0.0, 1.0 - fillers * 0.3)
        channels[3]["values"].append(round(filler_q, 3))

        # Posture score normalized to 0-1
        posture = body.get("posture_score", 50)
        channels[4]["values"].append(round(posture / 100.0, 3))

        # Emotion coherence placeholder
        channels[5]["values"].append(0.5)

    return channels
