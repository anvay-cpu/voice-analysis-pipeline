#!/usr/bin/env python3
"""Local enrichment script — uses claude -p (Max subscription) to generate
rich coaching feedback from pipeline scores and data.

Called by the webapp's /api/enrich endpoint.
Usage: echo '{"scores": {...}, ...}' | python3 scripts/enrich_report.py
"""

import json
import sys
import subprocess
import logging

logger = logging.getLogger(__name__)

COACHING_SYSTEM = """You are an expert public speaking coach with 20 years of experience.
You give feedback that is:
- WARM: encouraging and supportive, never harsh
- SPECIFIC: reference exact timestamps and metrics
- ACTIONABLE: provide concrete tips the speaker can practice
- INSIGHTFUL: connect observations across voice, body, and content

Write in a professional but warm tone. Address the speaker directly using "you".
Be detailed and thorough — aim for 1500-2000 words total."""


def call_claude(prompt: str, system: str = "", max_tokens: int = 4000) -> str:
    """Call claude -p with Max subscription."""
    cmd = ["claude", "-p", "--model", "claude-sonnet-4-20250514"]
    if system:
        cmd.extend(["--system-prompt", system])
    result = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(f"Claude CLI error: {result.stderr[:300]}")
    return result.stdout.strip()


def enrich_report(report: dict) -> dict:
    """Enrich a SpeechReport with detailed Claude-generated coaching."""
    scores = report.get("scores", {})
    fusion_stats = report.get("fusion_stats", {})
    disruptions = report.get("disruption_events", [])
    transitions = report.get("regime_transitions", [])
    content = report.get("content_details", {})
    transcript = report.get("transcript", "")
    duration = report.get("duration_sec", 0)
    filler_words = report.get("filler_words", [])

    # Build rich context for Claude
    disruption_text = "\n".join(
        f"  - [{d['timestamp']}] {d['type']}: {d['label']} (composure: {d.get('composure', 'N/A')}, recovery: {d.get('recovery_sec', 'N/A')}s)"
        for d in disruptions[:15]
    ) or "  None detected"

    transition_text = "\n".join(
        f"  - [{t.get('time_sec', 0):.0f}s] {t['from_regime']} → {t['to_regime']} (score: {t['score']:.2f})"
        for t in transitions[:10]
    ) or "  None detected"

    filler_text = ""
    if filler_words:
        from collections import Counter
        filler_counts = Counter(f["word"] for f in filler_words)
        filler_text = f"  Total fillers: {len(filler_words)}\n"
        filler_text += "  " + ", ".join(f'"{w}" x{c}' for w, c in filler_counts.most_common(5))

    # --- Generate Executive Summary ---
    summary_prompt = f"""Write a detailed executive summary (6-8 sentences) for this speech coaching report.

SCORES (0-100):
{json.dumps(scores, indent=2)}

SPEECH DETAILS:
- Duration: {duration:.0f} seconds ({duration/60:.1f} minutes)
- Grammar score: {content.get('grammar_score', 'N/A')}
- Readability (FKGL): {content.get('fkgl', 'N/A')}
- Dominant tone: {content.get('dominant_tone', 'Unknown')}
- Dominant sentiment: {content.get('dominant_sentiment', 'neutral')}
- Total disruptions: {fusion_stats.get('total_disruptions', 0)}
- Mean composure: {fusion_stats.get('mean_composure', 0):.2f}
- Emotion coherence: {fusion_stats.get('emotion_coherence', 0):.2f}

DISRUPTIONS:
{disruption_text}

TRANSITIONS:
{transition_text}

FILLER WORDS:
{filler_text or '  None detected'}

Write as if speaking directly to the presenter. Highlight both strengths and growth areas.
Weave the scores into natural language — don't just list numbers."""

    try:
        exec_summary = call_claude(summary_prompt, COACHING_SYSTEM, max_tokens=800)
    except Exception as e:
        logger.warning("Executive summary enrichment failed: %s", e)
        exec_summary = report.get("executive_summary", "")

    # --- Generate Detailed Coaching Feedback ---
    coaching_prompt = f"""Generate 8-12 detailed coaching feedback items for this speech.
Each item should be tied to a specific moment or pattern in the speech.

SCORES:
{json.dumps(scores, indent=2)}

DISRUPTION EVENTS:
{disruption_text}

REGIME TRANSITIONS:
{transition_text}

FILLER WORDS:
{filler_text or 'None detected'}

TRANSCRIPT (first 500 chars):
{transcript[:500]}

For each feedback item, provide:
1. A timestamp or time range
2. What was observed (type: transition, disruption, filler_burst, emotion_mismatch, strength, or improvement)
3. A specific label (e.g., "Pace acceleration", "Strong opening hook")
4. Detailed coaching text (3-5 sentences) with a concrete actionable tip

Format as a JSON array:
[{{"time_sec": 0, "timestamp": "0:00", "type": "...", "label": "...", "feedback": "..."}}]"""

    try:
        coaching_raw = call_claude(coaching_prompt, COACHING_SYSTEM, max_tokens=3000)
        # Parse JSON from response
        start = coaching_raw.index("[")
        end = coaching_raw.rindex("]") + 1
        coaching_items = json.loads(coaching_raw[start:end])
    except Exception as e:
        logger.warning("Coaching enrichment failed: %s", e)
        coaching_items = report.get("coaching_feedback", [])

    # --- Generate Practice Plan ---
    practice_prompt = f"""Create 5 specific, actionable practice exercises for this speaker.

SCORES (weakest areas need the most attention):
{json.dumps(scores, indent=2)}

DISRUPTIONS: {len(disruptions)} total
FILLER WORDS: {len(filler_words)} total

For each exercise provide:
1. title (5-10 words)
2. description (3-4 sentences, very specific and practical)
3. frequency (how often to practice)
4. target_dimension (which score dimension it improves)
5. target_improvement (estimated score improvement, e.g. 10)

Format as JSON array:
[{{"title": "...", "description": "...", "frequency": "...", "target_dimension": "...", "target_improvement": 10}}]"""

    try:
        practice_raw = call_claude(practice_prompt, COACHING_SYSTEM, max_tokens=1500)
        start = practice_raw.index("[")
        end = practice_raw.rindex("]") + 1
        practice_plan = json.loads(practice_raw[start:end])
    except Exception as e:
        logger.warning("Practice plan enrichment failed: %s", e)
        practice_plan = report.get("practice_plan", [])

    # Update report
    report["executive_summary"] = exec_summary
    report["coaching_feedback"] = coaching_items
    report["practice_plan"] = practice_plan
    report["_enriched"] = True

    return report


if __name__ == "__main__":
    report = json.load(sys.stdin)
    enriched = enrich_report(report)
    json.dump(enriched, sys.stdout)
