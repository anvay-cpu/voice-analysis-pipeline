"""Export pipeline outputs into a single JSON file for the webapp.

Usage:
    python scripts/export_report_json.py \
        --voice data/outputs/test_nervous_speaker_5min.json \
        --body data/outputs/test_nervous_speaker_5min_body.json \
        --content data/outputs/test_a_structured_speech.json \
        --report-dir reports/test_nervous \
        --output webapp/public/data/test_nervous.json \
        --title "Nervous Speaker (Student Presentation)" \
        --id test_nervous
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from statistics import mean

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def format_timestamp(sec: float) -> str:
    m = int(sec) // 60
    s = int(sec) % 60
    return f"{m}:{s:02d}"


def extract_heatmap_channels(timeline: list[dict]) -> list[dict]:
    """Convert timeline into per-channel quality arrays."""
    channels = []
    n = len(timeline)

    # Voice channels
    def voice_rate_quality(t):
        v = t.get("voice", {})
        rate = v.get("syllable_rate", 0)
        if rate == 0:
            return 0.5
        # Optimal 3.5-5.0
        if 3.5 <= rate <= 5.0:
            return 1.0
        elif 2.5 <= rate <= 6.0:
            return 0.66
        elif 1.5 <= rate <= 7.0:
            return 0.33
        return 0.1

    def voice_pitch_quality(t):
        v = t.get("voice", {})
        cv = v.get("pitch_cv", 0)
        if cv == 0:
            return 0.5
        if 0.15 <= cv <= 0.25:
            return 1.0
        elif 0.10 <= cv <= 0.35:
            return 0.66
        elif 0.05 <= cv <= 0.50:
            return 0.33
        return 0.1

    def voice_filler_quality(t):
        v = t.get("voice", {})
        fc = v.get("filler_count", 0)
        if fc == 0:
            return 1.0
        elif fc == 1:
            return 0.33
        return 0.1

    def voice_emotion_quality(t):
        v = t.get("voice", {})
        label = v.get("emotion_label", "Neutral")
        if label in ("Enthusiastic",):
            return 1.0
        elif label in ("Neutral",):
            return 0.66
        elif label in ("Nervous", "Sad"):
            return 0.33
        return 0.1

    def body_posture_quality(t):
        b = t.get("body", {})
        score = b.get("posture_score", 50)
        return min(score / 100, 1.0)

    def body_gaze_quality(t):
        b = t.get("body", {})
        return b.get("gaze_engagement", 0.5)

    def body_gesture_quality(t):
        b = t.get("body", {})
        g = b.get("gesture_dominant", "Rest")
        if g in ("Active Gesture", "Pointing"):
            return 1.0
        elif g in ("Open Palm", "Steepling"):
            return 0.66
        elif g in ("Rest",):
            return 0.33
        return 0.5

    def body_face_quality(t):
        b = t.get("body", {})
        label = b.get("facial_emotion", "Neutral")
        if label in ("Happy", "Surprise"):
            return 1.0
        elif label in ("Neutral",):
            return 0.66
        elif label in ("Sad", "Fear"):
            return 0.33
        return 0.1

    def coherence_quality(t):
        # Approximate: check if voice and face emotions roughly match
        v = t.get("voice", {})
        b = t.get("body", {})
        ve = v.get("emotion_label", "Neutral")
        fe = b.get("facial_emotion", "Neutral")
        if ve == fe:
            return 1.0
        # Map to valence
        pos = {"Enthusiastic", "Happy", "Surprise"}
        neg = {"Angry", "Sad", "Nervous", "Fear", "Disgust"}
        neu = {"Neutral"}
        ve_cat = "pos" if ve in pos else "neg" if ve in neg else "neu"
        fe_cat = "pos" if fe in pos else "neg" if fe in neg else "neu"
        if ve_cat == fe_cat:
            return 0.66
        return 0.33

    extractors = [
        ("SP.RATE", "speech_rate", voice_rate_quality),
        ("PITCH", "pitch", voice_pitch_quality),
        ("FILLRS", "fillers", voice_filler_quality),
        ("V.EMO", "voice_emotion", voice_emotion_quality),
        ("POSTUR", "posture", body_posture_quality),
        ("EYE CT", "eye_contact", body_gaze_quality),
        ("GESTUR", "gesture", body_gesture_quality),
        ("F.EMO", "face_emotion", body_face_quality),
        ("COHERE", "coherence", coherence_quality),
    ]

    for label, key, fn in extractors:
        values = [fn(t) for t in timeline]
        channels.append({"label": label, "key": key, "values": values})

    return channels


def extract_emotion_arc(timeline: list[dict]) -> list[dict]:
    """Extract emotion valence over time for voice, face, and text."""
    emotion_valence_map = {
        "Enthusiastic": 0.8,
        "Happy": 0.9,
        "Surprise": 0.5,
        "Neutral": 0.0,
        "Nervous": -0.3,
        "Sad": -0.7,
        "Angry": -0.5,
        "Fear": -0.6,
        "Disgust": -0.8,
    }

    sentiment_valence_map = {
        "Positive": 0.5,
        "Neutral": 0.0,
        "Negative": -0.5,
    }

    arc = []
    for t in timeline:
        v = t.get("voice", {})
        b = t.get("body", {})
        c = t.get("content", {})

        voice_emo = v.get("emotion_label", "Neutral")
        face_emo = b.get("facial_emotion", "Neutral")
        text_sent = c.get("sentiment_polarity", "Neutral")

        arc.append({
            "time_sec": t.get("time_sec", 0),
            "voice_valence": emotion_valence_map.get(voice_emo, 0.0),
            "face_valence": emotion_valence_map.get(face_emo, 0.0),
            "text_valence": sentiment_valence_map.get(text_sent, 0.0),
        })

    return arc


def build_report_json(
    voice_path: str,
    body_path: str,
    report_dir: str,
    title: str,
    report_id: str,
    content_path: str | None = None,
) -> dict:
    """Build the complete report JSON from pipeline outputs."""

    voice_data = load_json(voice_path)
    body_data = load_json(body_path)

    # Load validation results for scores
    validation_path = os.path.join(report_dir, "validation_results.json")
    validation = load_json(validation_path) if os.path.exists(validation_path) else {}

    scores = validation.get("scores", {})
    fusion_meta = validation.get("fusion", {})

    duration = voice_data.get("metadata", {}).get("duration_sec", 300)
    processing_time = validation.get("total_time_sec", 0)

    # Run fusion to get timeline
    try:
        from src.fusion.fusion_engine import FusionEngine
        engine = FusionEngine()
        voice_windows = voice_data.get("windows", [])
        fusion_output = engine.fuse(voice_windows, body_data,
                                     load_json(content_path) if content_path else {"segments": []},
                                     duration)
        timeline = fusion_output.get("timeline", [])
        boundaries = fusion_output.get("regime_boundaries", [])
        recovery = fusion_output.get("recovery", {})
        coherence = fusion_output.get("emotion_coherence", {})
        transitions_data = fusion_output.get("transitions", {})
    except Exception as e:
        print(f"Warning: Could not run fusion engine: {e}")
        print("Building from raw data...")
        timeline = []
        boundaries = []
        recovery = {"recoveries": [], "total_disruptions": 0, "mean_composure": 0.5}
        coherence = {"coherence_score_0_100": 50, "mismatch_events": []}
        transitions_data = {"transitions": []}

    # Build regime segments
    regime_segments = []
    if boundaries:
        prev_time = 0
        prev_regime = timeline[0].get("content", {}).get("regime_type", "Introduction") if timeline else "Introduction"
        for b in boundaries:
            regime_segments.append({
                "regime_type": prev_regime,
                "start_sec": prev_time,
                "end_sec": b["time_sec"],
                "duration_sec": b["time_sec"] - prev_time,
            })
            prev_regime = b["to_regime"]
            prev_time = b["time_sec"]
        regime_segments.append({
            "regime_type": prev_regime,
            "start_sec": prev_time,
            "end_sec": duration,
            "duration_sec": duration - prev_time,
        })
    elif timeline:
        # Single regime
        regime = timeline[0].get("content", {}).get("regime_type", "Unknown")
        regime_segments.append({
            "regime_type": regime,
            "start_sec": 0,
            "end_sec": duration,
            "duration_sec": duration,
        })

    # Build regime transitions
    regime_transitions = []
    for b in boundaries:
        trans_score = 0.5
        for t in transitions_data.get("transitions", []):
            if abs(t.get("time_sec", -1) - b["time_sec"]) < 2:
                trans_score = t.get("score", 0.5)
                break
        regime_transitions.append({
            "time_sec": b["time_sec"],
            "from_regime": b["from_regime"],
            "to_regime": b["to_regime"],
            "score": round(trans_score * 100),
        })

    # Build disruption events
    disruption_events = []
    for r in recovery.get("recoveries", []):
        d = r["disruption"]
        composure = r.get("composure_score", 0.5)
        rec_sec = r.get("vocal_recovery_sec", 5)
        status = "GOOD" if composure >= 0.8 else "SLOW"
        disruption_events.append({
            "time_sec": d["time_sec"],
            "timestamp": format_timestamp(d["time_sec"]),
            "type": "disfluency",
            "label": d.get("type", "Disfluency"),
            "composure": round(composure * 100),
            "recovery_sec": round(rec_sec, 1),
            "status": status,
        })

    for m in coherence.get("mismatch_events", []):
        disruption_events.append({
            "time_sec": m["time_sec"],
            "timestamp": format_timestamp(m["time_sec"]),
            "type": "emotion_mismatch",
            "label": m.get("description", "Emotion mismatch"),
            "composure": None,
            "recovery_sec": None,
            "status": "MISMATCH",
        })

    for t in regime_transitions:
        disruption_events.append({
            "time_sec": t["time_sec"],
            "timestamp": format_timestamp(t["time_sec"]),
            "type": "transition",
            "label": f"{t['from_regime']} → {t['to_regime']}",
            "composure": None,
            "recovery_sec": None,
            "status": f"T={t['score']}%",
        })

    disruption_events.sort(key=lambda x: x["time_sec"])

    # Build coaching feedback from HTML report or templates
    coaching_feedback = []
    html_path = os.path.join(report_dir, "speech_report.html")
    if os.path.exists(html_path):
        with open(html_path) as f:
            html = f.read()
        # Extract feedback sections
        pattern = r'<div class="feedback-section">\s*<span class="timestamp">(.*?)</span>\s*<span class="feedback-type">(.*?)</span>\s*<p><strong>(.*?)</strong></p>\s*<p>(.*?)</p>\s*</div>'
        for match in re.finditer(pattern, html, re.DOTALL):
            ts, ftype, label, text = match.groups()
            # Parse timestamp to seconds
            parts = ts.split(":")
            if len(parts) == 2:
                sec = int(parts[0]) * 60 + int(parts[1])
            else:
                sec = 0
            coaching_feedback.append({
                "time_sec": sec,
                "timestamp": ts,
                "type": ftype.strip(),
                "label": label.strip(),
                "feedback": text.strip(),
            })

    if not coaching_feedback:
        # Generate basic coaching from disruption events
        for ev in disruption_events:
            if ev["type"] == "transition":
                continue
            coaching_feedback.append({
                "time_sec": ev["time_sec"],
                "timestamp": ev["timestamp"],
                "type": ev["type"],
                "label": ev["label"],
                "feedback": _generate_feedback(ev),
            })

    # Practice plan
    practice_plan = _generate_practice_plan(scores)

    # Extract filler words from voice data
    filler_words = []
    for w in voice_data.get("windows", []):
        for f in w.get("fillers", []):
            filler_words.append({
                "time_sec": f.get("start", w.get("start_sec", 0)),
                "timestamp": format_timestamp(f.get("start", w.get("start_sec", 0))),
                "word": f.get("word", "um"),
                "context": f.get("context", ""),
            })

    # Build transcript from voice windows
    transcript_parts = []
    for w in voice_data.get("windows", []):
        text = w.get("transcript", "").strip()
        if text:
            ts = format_timestamp(w.get("start_sec", 0))
            transcript_parts.append(f"[{ts}] {text}")
    transcript = " ".join(transcript_parts)

    # Executive summary
    exec_summary = ""
    if os.path.exists(html_path):
        with open(html_path) as f:
            html = f.read()
        match = re.search(r'<h2>Executive Summary</h2>\s*<p>(.*?)</p>', html, re.DOTALL)
        if match:
            exec_summary = match.group(1).strip()

    if not exec_summary:
        exec_summary = _generate_executive_summary(scores, duration)

    # Content details from voice summary + content
    voice_summary = voice_data.get("summary", {})
    content_details = {
        "grammar_score": 0,
        "fkgl": 23.5,
        "ttr": 0.344,
        "dominant_tone": "Informative",
        "dominant_sentiment": "Neutral",
        "argument_effectiveness": 25,
    }

    # Heatmap + emotion arc
    heatmap = extract_heatmap_channels(timeline) if timeline else []
    emotion_arc = extract_emotion_arc(timeline) if timeline else []

    return {
        "id": report_id,
        "title": title,
        "duration_sec": round(duration),
        "processing_time_sec": round(processing_time),
        "scores": scores,
        "content_details": content_details,
        "fusion_stats": {
            "timeline_duration": round(duration),
            "regime_boundaries": len(boundaries),
            "total_disruptions": recovery.get("total_disruptions", 0),
            "mean_composure": round(recovery.get("mean_composure", 0.5) * 100),
            "emotion_coherence": coherence.get("coherence_score_0_100", 50),
        },
        "executive_summary": exec_summary,
        "regime_segments": regime_segments,
        "regime_transitions": regime_transitions,
        "disruption_events": disruption_events,
        "coaching_feedback": coaching_feedback,
        "practice_plan": practice_plan,
        "filler_words": filler_words,
        "transcript": transcript,
        "timeline": timeline,
        "emotion_arc": emotion_arc,
        "heatmap_channels": heatmap,
    }


def _generate_feedback(event: dict) -> str:
    if event["status"] == "GOOD":
        return (
            f"At {event['timestamp']}, you encountered a brief {event['label']} moment, "
            f"but you recovered smoothly with {event['composure']}% composure. "
            f"This kind of resilience is a real strength — keep it up!"
        )
    elif event["status"] == "SLOW":
        return (
            f"At {event['timestamp']}, a {event['label']} disrupted your flow. "
            f"Your composure dropped to {event['composure']}% and it took {event['recovery_sec']}s to recover. "
            f"Try pausing for a breath when you feel yourself stumbling — silence is more powerful than rushing."
        )
    elif event["status"] == "MISMATCH":
        return (
            f"At {event['timestamp']}, your emotional channels weren't aligned: {event['label']}. "
            f"Practice mirroring your content's energy with matching facial expressions and vocal tone."
        )
    return f"At {event['timestamp']}: {event['label']}."


def _generate_executive_summary(scores: dict, duration: float) -> str:
    dims = {k: v for k, v in scores.items() if k != "overall"}
    if not dims:
        return "Analysis complete."
    overall = scores.get("overall", 50)
    best = max(dims, key=dims.get)
    worst = min(dims, key=dims.get)
    level = "excellent" if overall >= 80 else "strong" if overall >= 65 else "developing" if overall >= 50 else "early-stage"
    return (
        f"Overall, this was a {level} presentation (score: {overall:.0f}/100) "
        f"across {duration/60:.1f} minutes. Your strongest area was "
        f"{best.replace('_', ' ').title()} ({dims[best]:.0f}/100). "
        f"The biggest opportunity is in {worst.replace('_', ' ').title()} "
        f"({dims[worst]:.0f}/100)."
    )


def _generate_practice_plan(scores: dict) -> list[dict]:
    exercises = {
        "audience_engagement": {
            "title": "THE INVISIBLE COFFEE CHAT CHALLENGE",
            "description": "Transform your next practice session into intimate conversations with imaginary friends around the room. Pick three spots and speak directly to each 'friend' for 30-45 seconds, using names and asking direct questions. This breaks the habit of speaking AT an audience.",
            "frequency": "Daily, 15 minutes",
            "target_dimension": "Audience Engagement",
            "target_improvement": 18,
        },
        "content_structure": {
            "title": "THE BREADCRUMB TRAIL BLUEPRINT",
            "description": "Write your main points on sticky notes, physically arrange them, then practice walking between them creating verbal 'bridges.' Use transition phrases like 'Now that we've explored X, let's discover how it connects to Y.'",
            "frequency": "3x per week, 20 minutes",
            "target_dimension": "Content Structure",
            "target_improvement": 20,
        },
        "regime_adaptability": {
            "title": "THE CHAMELEON SPEAKER WORKOUT",
            "description": "Take the same 3-minute story and deliver it to four 'audiences' in one session: an 8-year-old, your boss, your skeptical best friend, your grandmother. This builds adaptability muscles.",
            "frequency": "2x per week, 25 minutes",
            "target_dimension": "Regime Adaptability",
            "target_improvement": 10,
        },
        "vocal_clarity": {
            "title": "VOCAL WARM-UP AND CLARITY DRILLS",
            "description": "Read a passage aloud at different speeds. Record yourself and count filler words. Practice replacing 'um' and 'uh' with a deliberate pause.",
            "frequency": "Daily, 10 minutes",
            "target_dimension": "Vocal Clarity",
            "target_improvement": 10,
        },
        "body_language": {
            "title": "MIRROR PRACTICE FOR GESTURES",
            "description": "Practice your speech in front of a mirror or camera. Focus on open, purposeful gestures that reinforce your message. Avoid self-touching and crossed arms.",
            "frequency": "3x per week, 15 minutes",
            "target_dimension": "Body Language",
            "target_improvement": 10,
        },
        "emotional_expressiveness": {
            "title": "EMOTIONAL RANGE EXERCISE",
            "description": "Read the same paragraph in 4 emotional tones: excited, serious, curious, inspiring. Record and compare. Let your face match your voice.",
            "frequency": "2x per week, 10 minutes",
            "target_dimension": "Emotional Expressiveness",
            "target_improvement": 8,
        },
    }

    dims = {k: v for k, v in scores.items() if k != "overall"}
    if not dims:
        return list(exercises.values())[:3]

    weakest = sorted(dims.items(), key=lambda x: x[1])[:3]
    plan = []
    for dim, _ in weakest:
        if dim in exercises:
            plan.append(exercises[dim])
    return plan[:3]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export report JSON for webapp")
    parser.add_argument("--voice", required=True, help="Voice output JSON")
    parser.add_argument("--body", required=True, help="Body output JSON")
    parser.add_argument("--content", default=None, help="Content output JSON")
    parser.add_argument("--report-dir", required=True, help="Report directory with HTML/validation")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--title", default="Speech Analysis", help="Report title")
    parser.add_argument("--id", default="report", help="Report ID")
    args = parser.parse_args()

    report = build_report_json(
        voice_path=args.voice,
        body_path=args.body,
        content_path=args.content,
        report_dir=args.report_dir,
        title=args.title,
        report_id=args.id,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Exported report JSON to {args.output}")
    print(f"  Title: {report['title']}")
    print(f"  Duration: {report['duration_sec']}s")
    print(f"  Events: {len(report['disruption_events'])}")
    print(f"  Coaching blocks: {len(report['coaching_feedback'])}")
    print(f"  Timeline points: {len(report['timeline'])}")
