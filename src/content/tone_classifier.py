"""
Tone Classification — Phase 7 of Modality 3.

Classifies speaking tone using Claude CLI (zero-shot, Option C).
No training needed — uses Max subscription via claude -p.

6 tone classes: Assertive, Persuasive, Informative, Inspirational, Cautionary, Humorous.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

TONE_CLASSES = [
    "Assertive",
    "Persuasive",
    "Informative",
    "Inspirational",
    "Cautionary",
    "Humorous",
]

CLASSIFICATION_PROMPT = """Classify the speaking tone of this speech segment into exactly ONE of these categories:
Assertive, Persuasive, Informative, Inspirational, Cautionary, Humorous.

Definitions:
- Assertive: Confident claims, strong statements, declarations
- Persuasive: Arguments, evidence, trying to convince the audience
- Informative: Facts, explanations, neutral delivery of information
- Inspirational: Motivational, uplifting, emotional appeals
- Cautionary: Warnings, risks, negative consequences
- Humorous: Jokes, wit, self-deprecation, intentional lightness

Segment: {segment_text}

Respond with ONLY the category name, nothing else."""


class ToneClassifier:
    """Classifies speaking tone using Claude CLI (zero-shot).

    Uses the Max subscription via claude -p. No API key or training needed.
    Falls back to keyword heuristics if Claude CLI is unavailable.
    """

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm

    def classify(self, text: str) -> dict:
        """Classify the tone of a text segment.

        Returns:
            Dict with tone, confidence, and method used.
        """
        if not text.strip():
            return {"tone": "Informative", "confidence": 0.0, "method": "empty"}

        if self.use_llm:
            try:
                return self._classify_with_llm(text)
            except Exception as e:
                logger.warning(f"LLM tone classification failed: {e}, using heuristic")

        return self._classify_heuristic(text)

    def classify_transcript(self, transcript: dict) -> dict:
        """Classify tone for all segments in a transcript.

        Args:
            transcript: Output from TranscriptProcessor.

        Returns:
            Dict with per-segment tone classifications.
        """
        segments = transcript.get("segments", [])
        results = []

        for seg in segments:
            result = self.classify(seg["text"])
            result["start_sec"] = seg.get("start_sec", 0.0)
            result["end_sec"] = seg.get("end_sec", 0.0)
            results.append(result)

        # Overall tone: most frequent
        if results:
            tone_counts = {}
            for r in results:
                t = r["tone"]
                tone_counts[t] = tone_counts.get(t, 0) + 1
            dominant = max(tone_counts, key=tone_counts.get)
        else:
            dominant = "Informative"

        return {
            "dominant_tone": dominant,
            "segments": results,
        }

    def _classify_with_llm(self, text: str) -> dict:
        """Use Claude CLI for tone classification."""
        from src.utils.claude_api_wrapper import call_claude

        # Truncate long segments
        words = text.split()
        if len(words) > 150:
            text = " ".join(words[:150]) + "..."

        prompt = CLASSIFICATION_PROMPT.format(segment_text=text)
        response = call_claude(prompt, model="claude-sonnet-4-20250514", timeout=30)
        response = response.strip().strip('"').strip("'")

        tone = self._match_tone(response)
        return {"tone": tone, "confidence": 0.85, "method": "llm"}

    def _classify_heuristic(self, text: str) -> dict:
        """Keyword-based heuristic fallback."""
        text_lower = text.lower()
        scores = {t: 0 for t in TONE_CLASSES}

        # Assertive signals
        for kw in ["must", "clearly", "without doubt", "no question", "is the right",
                    "i believe", "certainly", "definitely", "absolutely"]:
            if kw in text_lower:
                scores["Assertive"] += 1

        # Persuasive signals
        for kw in ["evidence", "data shows", "research", "therefore", "consider",
                    "compelling", "demonstrates", "proves", "should"]:
            if kw in text_lower:
                scores["Persuasive"] += 1

        # Informative signals
        for kw in ["percent", "according to", "study", "report", "statistics",
                    "fact", "defined as", "consists of", "there are"]:
            if kw in text_lower:
                scores["Informative"] += 1

        # Inspirational signals
        for kw in ["imagine", "dream", "together", "vision", "future",
                    "hope", "change the world", "possibility", "inspire"]:
            if kw in text_lower:
                scores["Inspirational"] += 1

        # Cautionary signals
        for kw in ["danger", "risk", "warning", "if we don't", "consequences",
                    "threat", "careful", "beware", "unless we"]:
            if kw in text_lower:
                scores["Cautionary"] += 1

        # Humorous signals
        for kw in ["just kidding", "laugh", "joke", "funny", "seriously though",
                    "i know what you're thinking", "don't worry"]:
            if kw in text_lower:
                scores["Humorous"] += 1

        best = max(scores, key=scores.get)
        if scores[best] == 0:
            best = "Informative"
            confidence = 0.3
        else:
            confidence = min(0.7, 0.3 + scores[best] * 0.15)

        return {"tone": best, "confidence": confidence, "method": "heuristic"}

    def _match_tone(self, response: str) -> str:
        """Match LLM response to valid tone class."""
        response_lower = response.lower().strip()
        for tc in TONE_CLASSES:
            if tc.lower() in response_lower:
                return tc
        return "Informative"
