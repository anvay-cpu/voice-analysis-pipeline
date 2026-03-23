"""Phase 5.2 — Coaching Writer.

Uses Claude API (via CLI wrapper) to generate warm, specific, actionable
coaching feedback from raw multimodal analysis data. Falls back to
template-based feedback if API is unavailable.
"""

import json
import logging
from statistics import mean

logger = logging.getLogger(__name__)

COACHING_SYSTEM_PROMPT = """You are an expert public speaking coach with 20 years of experience.
You give feedback that is:
- WARM: encouraging and supportive, never harsh
- SPECIFIC: reference exact timestamps and metrics
- ACTIONABLE: provide concrete tips the speaker can practice
- INSIGHTFUL: connect observations across voice, body, and content

Keep your language natural and conversational. Avoid jargon."""


class CoachingWriter:
    """Generate coaching feedback from analysis data."""

    def __init__(self, use_api: bool = True):
        self.use_api = use_api
        self._api_available = None

    def generate_full_report(
        self,
        all_segments: list[dict],
        overall_scores: dict,
        fusion_output: dict | None = None,
        speech_metadata: dict | None = None,
    ) -> dict:
        """Generate the complete coaching report.

        Args:
            all_segments: Timeline segments with coaching context.
            overall_scores: 6-dimension scores dict.
            fusion_output: Full fusion output for detailed data.
            speech_metadata: Duration, video path, etc.

        Returns:
            Dict with executive_summary, segment_feedback, practice_plan.
        """
        metadata = speech_metadata or {}

        executive_summary = self._generate_executive_summary(
            overall_scores, metadata
        )

        segment_feedback = self._generate_segment_feedback(
            all_segments, fusion_output
        )

        practice_plan = self._generate_practice_plan(
            overall_scores, fusion_output
        )

        return {
            "executive_summary": executive_summary,
            "segment_feedback": segment_feedback,
            "practice_plan": practice_plan,
        }

    # ── Executive Summary ──────────────────────────────────────────

    def _generate_executive_summary(
        self, scores: dict, metadata: dict
    ) -> str:
        """Generate executive summary via API or template."""
        if self.use_api and self._check_api():
            return self._api_executive_summary(scores, metadata)
        return self._template_executive_summary(scores, metadata)

    def _api_executive_summary(self, scores: dict, metadata: dict) -> str:
        """Generate executive summary via Claude API."""
        try:
            from src.utils.claude_api_wrapper import call_claude

            prompt = f"""Write a 4-6 sentence executive summary for a public speaking coaching report.

Scores (0-100):
{json.dumps({k: v for k, v in scores.items()}, indent=2)}

Speech duration: {metadata.get('duration_sec', 0):.0f} seconds

Write as if speaking directly to the presenter. Be warm, specific, and highlight
both strengths (highest scores) and areas for growth (lowest scores).
Do NOT list the scores numerically — weave them into natural language."""

            return call_claude(prompt, system=COACHING_SYSTEM_PROMPT, max_tokens=500)
        except Exception as e:
            logger.warning("API call failed for executive summary: %s", e)
            return self._template_executive_summary(scores, metadata)

    def _template_executive_summary(self, scores: dict, metadata: dict) -> str:
        """Template-based fallback for executive summary."""
        dims = {k: v for k, v in scores.items() if k != "overall"}
        overall = scores.get("overall", 50.0)

        if not dims:
            return "Analysis complete. See the detailed feedback below for coaching insights."

        best_dim = max(dims, key=dims.get)
        worst_dim = min(dims, key=dims.get)
        duration_min = metadata.get("duration_sec", 0) / 60

        level = "excellent" if overall >= 80 else "strong" if overall >= 65 else "developing" if overall >= 50 else "early-stage"
        best_label = best_dim.replace("_", " ").title()
        worst_label = worst_dim.replace("_", " ").title()

        return (
            f"Overall, this was a {level} presentation (score: {overall:.0f}/100) "
            f"across {duration_min:.1f} minutes of speaking. "
            f"Your strongest area was {best_label} ({dims[best_dim]:.0f}/100), "
            f"which shows real skill in this dimension. "
            f"The biggest opportunity for growth is in {worst_label} "
            f"({dims[worst_dim]:.0f}/100) — focused practice here will "
            f"make the most difference in your next presentation. "
            f"See the detailed feedback below for specific moments and actionable tips."
        )

    # ── Per-Segment Feedback ───────────────────────────────────────

    def _generate_segment_feedback(
        self, segments: list[dict], fusion_output: dict | None
    ) -> list[dict]:
        """Generate coaching feedback for key segments."""
        if not segments and not fusion_output:
            return []

        # Extract key segments from timeline (regime transitions, disruptions)
        key_moments = self._identify_key_moments(fusion_output)

        feedback_list = []
        for moment in key_moments:
            feedback = self._generate_moment_feedback(moment)
            feedback_list.append(feedback)

        return feedback_list

    def _identify_key_moments(self, fusion_output: dict | None) -> list[dict]:
        """Identify key moments worth coaching on."""
        if not fusion_output:
            return []

        moments = []
        timeline = fusion_output.get("timeline", [])

        # Regime boundaries
        boundaries = fusion_output.get("regime_boundaries", [])
        for b in boundaries:
            t = b["time_sec"]
            moments.append({
                "type": "transition",
                "time_sec": t,
                "label": f"{b['from_regime']} → {b['to_regime']}",
                "context": self._get_context_window(timeline, t, 5),
            })

        # Disruption events
        recovery = fusion_output.get("recovery", {})
        for r in recovery.get("recoveries", []):
            d = r["disruption"]
            moments.append({
                "type": "disruption",
                "time_sec": d["time_sec"],
                "label": d["type"],
                "composure": r["composure_score"],
                "recovery_sec": r["vocal_recovery_sec"],
                "context": self._get_context_window(timeline, d["time_sec"], 5),
            })

        # Emotion mismatches
        coherence = fusion_output.get("emotion_coherence", {})
        for m in coherence.get("mismatch_events", []):
            moments.append({
                "type": "emotion_mismatch",
                "time_sec": m["time_sec"],
                "label": m.get("description", "Emotion mismatch"),
                "coherence": m.get("mean_coherence", 0),
            })

        # Sort by time and limit to 12 moments
        moments.sort(key=lambda m: m["time_sec"])
        return moments[:12]

    def _generate_moment_feedback(self, moment: dict) -> dict:
        """Generate feedback for a single key moment."""
        t = moment["time_sec"]
        minutes = t // 60
        seconds = t % 60
        timestamp = f"{minutes}:{seconds:02d}"

        if self.use_api and self._check_api():
            text = self._api_moment_feedback(moment, timestamp)
        else:
            text = self._template_moment_feedback(moment, timestamp)

        return {
            "time_sec": t,
            "timestamp": timestamp,
            "type": moment["type"],
            "label": moment["label"],
            "feedback": text,
        }

    def _api_moment_feedback(self, moment: dict, timestamp: str) -> str:
        """Generate moment feedback via Claude API."""
        try:
            from src.utils.claude_api_wrapper import call_claude

            prompt = f"""Write 3-4 sentences of coaching feedback for this moment in a speech.

Moment type: {moment['type']}
Timestamp: {timestamp}
Label: {moment['label']}
Context data: {json.dumps({k: v for k, v in moment.items() if k != 'context'}, default=str)}

Be specific about what happened and give 1 concrete tip for improvement.
Address the speaker directly using "you"."""

            return call_claude(prompt, system=COACHING_SYSTEM_PROMPT, max_tokens=300)
        except Exception as e:
            logger.warning("API call failed for moment feedback: %s", e)
            return self._template_moment_feedback(moment, timestamp)

    def _template_moment_feedback(self, moment: dict, timestamp: str) -> str:
        """Template-based fallback for moment feedback."""
        mtype = moment["type"]

        if mtype == "transition":
            return (
                f"At {timestamp}, you transitioned from {moment['label']}. "
                f"Consider signposting transitions more clearly — a brief pause "
                f"and a phrase like 'Now let's look at...' helps your audience follow along."
            )
        elif mtype == "disruption":
            composure = moment.get("composure", 0.5)
            if composure >= 0.8:
                return (
                    f"At {timestamp}, there was a brief {moment['label']} moment, "
                    f"but you recovered smoothly (composure: {composure:.0%}). "
                    f"This kind of resilience is a real strength."
                )
            else:
                return (
                    f"At {timestamp}, a {moment['label']} disrupted your flow "
                    f"(composure: {composure:.0%}). Try pausing for a breath "
                    f"when you feel yourself stumbling — silence is more powerful "
                    f"than rushing through."
                )
        elif mtype == "emotion_mismatch":
            return (
                f"At {timestamp}, your emotional channels weren't fully aligned: "
                f"{moment['label']}. Practice mirroring your content's energy "
                f"with matching facial expressions and vocal tone."
            )
        else:
            return f"At {timestamp}: {moment['label']}."

    # ── Practice Plan ──────────────────────────────────────────────

    def _generate_practice_plan(
        self, scores: dict, fusion_output: dict | None
    ) -> list[dict]:
        """Generate 3 prioritized practice exercises."""
        dims = {k: v for k, v in scores.items() if k != "overall"}
        weakest = sorted(dims.items(), key=lambda x: x[1])[:3]

        if self.use_api and self._check_api():
            return self._api_practice_plan(weakest, scores)
        return self._template_practice_plan(weakest)

    def _api_practice_plan(self, weakest: list, scores: dict) -> list[dict]:
        """Generate practice plan via Claude API."""
        try:
            from src.utils.claude_api_wrapper import call_claude

            prompt = f"""Create 3 specific practice exercises for a speaker to improve.

Their weakest dimensions:
{json.dumps([{"dimension": d, "score": s} for d, s in weakest], indent=2)}

All scores: {json.dumps(scores, indent=2)}

For each exercise, provide:
1. A short title (5-8 words)
2. A description (2-3 sentences)
3. How often to practice (e.g., "3x per week, 10 minutes")

Format as JSON array with keys: title, description, frequency"""

            response = call_claude(prompt, system=COACHING_SYSTEM_PROMPT, max_tokens=600)

            # Try to parse JSON from response
            try:
                # Find JSON array in response
                start = response.index("[")
                end = response.rindex("]") + 1
                return json.loads(response[start:end])
            except (ValueError, json.JSONDecodeError):
                return self._template_practice_plan(weakest)

        except Exception as e:
            logger.warning("API call failed for practice plan: %s", e)
            return self._template_practice_plan(weakest)

    def _template_practice_plan(self, weakest: list) -> list[dict]:
        """Template-based fallback for practice plan."""
        exercises = {
            "vocal_clarity": {
                "title": "Vocal Warm-up and Clarity Drills",
                "description": "Read a passage aloud at different speeds. Record yourself and count filler words. Practice replacing 'um' and 'uh' with a brief pause.",
                "frequency": "Daily, 10 minutes",
            },
            "body_language": {
                "title": "Mirror Practice for Gestures",
                "description": "Practice your speech in front of a mirror or camera. Focus on open, purposeful gestures that reinforce your message. Avoid self-touching.",
                "frequency": "3x per week, 15 minutes",
            },
            "content_structure": {
                "title": "Speech Outlining Exercise",
                "description": "Before each practice session, write a 5-point outline: hook, 3 key points, and closing call to action. Time each section.",
                "frequency": "Before every practice run",
            },
            "audience_engagement": {
                "title": "Eye Contact and Questions Drill",
                "description": "Practice with 3 objects placed around the room as 'audience members'. Make eye contact with each for 3-5 seconds. Plant 2 rhetorical questions in your speech.",
                "frequency": "3x per week, 10 minutes",
            },
            "emotional_expressiveness": {
                "title": "Emotional Range Exercise",
                "description": "Read the same paragraph in 4 different emotional tones: excited, serious, curious, inspiring. Record and compare. Let your face match your voice.",
                "frequency": "2x per week, 10 minutes",
            },
            "regime_adaptability": {
                "title": "Transition and Recovery Practice",
                "description": "Practice deliberate transitions between speech sections with a clear verbal bridge. Also practice recovering from intentional mistakes — pause, breathe, continue.",
                "frequency": "2x per week, 15 minutes",
            },
        }

        plan = []
        for dim, score in weakest:
            if dim in exercises:
                plan.append(exercises[dim])

        # Ensure we have at least 3
        while len(plan) < 3:
            for dim in exercises:
                if exercises[dim] not in plan:
                    plan.append(exercises[dim])
                    break
            else:
                break

        return plan[:3]

    # ── Helpers ────────────────────────────────────────────────────

    def _get_context_window(
        self, timeline: list[dict], center: int, radius: int
    ) -> list[dict]:
        """Extract a window of timeline entries around a center point."""
        start = max(0, center - radius)
        end = min(len(timeline), center + radius)
        return timeline[start:end]

    def _check_api(self) -> bool:
        """Check if Claude API is available (cached)."""
        if self._api_available is not None:
            return self._api_available

        try:
            from src.utils.claude_api_wrapper import call_claude
            call_claude("Reply with OK", max_tokens=10, timeout=10)
            self._api_available = True
        except Exception:
            self._api_available = False
            logger.info("Claude API not available, using template fallback")

        return self._api_available
