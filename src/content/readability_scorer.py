"""
Readability Scoring — Phase 4 of Modality 3.

Computes standard readability metrics using textstat library.
Pure computation, no models or training needed.
"""

import logging

import textstat

logger = logging.getLogger(__name__)

# Target ranges for public speaking
TARGETS = {
    "flesch_kincaid_grade": (8, 12),    # US grade level, TED talk range
    "dale_chall": (5.0, 9.0),          # < 5 easy, > 9 college graduate
    "gunning_fog": (10, 14),           # General audience
    "coleman_liau": (8, 12),           # Similar to FKGL
    "words_per_minute": (130, 160),    # Presentation pace
    "avg_sentence_length": (12, 20),   # Shorter in speech than writing
}


class ReadabilityScorer:
    """Computes readability metrics for transcript segments."""

    def score_text(self, text: str, duration_sec: float = 0.0) -> dict:
        """Score a block of text for readability.

        Args:
            text: The text to analyze.
            duration_sec: Duration in seconds (for WPM calculation).

        Returns:
            Dict with all readability metrics and assessment.
        """
        if not text.strip():
            return self._empty_scores()

        word_count = len(text.split())
        sentence_count = textstat.sentence_count(text) or 1

        fkgl = textstat.flesch_kincaid_grade(text)
        dale_chall = textstat.dale_chall_readability_score(text)
        gunning_fog = textstat.gunning_fog(text)
        coleman_liau = textstat.coleman_liau_index(text)

        avg_sentence_length = round(word_count / sentence_count, 1)

        wpm = 0.0
        if duration_sec > 0:
            wpm = round(word_count / (duration_sec / 60), 1)

        assessment = self._assess(fkgl, wpm, avg_sentence_length)

        return {
            "flesch_kincaid_grade": round(fkgl, 1),
            "dale_chall": round(dale_chall, 1),
            "gunning_fog": round(gunning_fog, 1),
            "coleman_liau": round(coleman_liau, 1),
            "words_per_minute": wpm,
            "avg_sentence_length": avg_sentence_length,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "assessment": assessment,
        }

    def score_transcript(self, transcript: dict) -> dict:
        """Score all segments in a processed transcript.

        Args:
            transcript: Output from TranscriptProcessor with 'segments' and 'full_text'.

        Returns:
            Dict with per-segment scores and overall scores.
        """
        segments = transcript.get("segments", [])
        duration = transcript.get("duration_sec", 0.0)
        full_text = transcript.get("full_text", "")

        segment_scores = []
        for seg in segments:
            seg_duration = seg.get("end_sec", 0.0) - seg.get("start_sec", 0.0)
            scores = self.score_text(seg["text"], seg_duration)
            scores["start_sec"] = seg.get("start_sec", 0.0)
            scores["end_sec"] = seg.get("end_sec", 0.0)
            segment_scores.append(scores)

        overall = self.score_text(full_text, duration)

        return {
            "overall": overall,
            "segments": segment_scores,
        }

    def _assess(self, fkgl: float, wpm: float, avg_sent_len: float) -> str:
        """Generate a human-readable assessment."""
        issues = []

        if fkgl > 14:
            issues.append("language is very complex for a general audience")
        elif fkgl > 12:
            issues.append("language may be too complex for some audience members")
        elif fkgl < 6:
            issues.append("language may be too simple for a professional audience")

        if wpm > 0:
            if wpm > 180:
                issues.append("speaking pace is very fast")
            elif wpm > 160:
                issues.append("speaking pace is slightly fast")
            elif wpm < 110:
                issues.append("speaking pace is very slow")
            elif wpm < 130:
                issues.append("speaking pace is slightly slow")

        if avg_sent_len > 25:
            issues.append("sentences are too long for spoken delivery")
        elif avg_sent_len < 8:
            issues.append("sentences are very short and choppy")

        if not issues:
            return "Appropriate for general audience"
        return "Issues: " + "; ".join(issues)

    def _empty_scores(self) -> dict:
        return {
            "flesch_kincaid_grade": 0.0,
            "dale_chall": 0.0,
            "gunning_fog": 0.0,
            "coleman_liau": 0.0,
            "words_per_minute": 0.0,
            "avg_sentence_length": 0.0,
            "word_count": 0,
            "sentence_count": 0,
            "assessment": "No text to analyze",
        }
