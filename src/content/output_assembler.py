"""
Output Assembly — Phase 9 of Modality 3.

Merges all sub-module outputs into a unified per-segment and speech-level JSON.
"""

import json
import logging
import os
from collections import Counter
from statistics import mean, median

logger = logging.getLogger(__name__)


class OutputAssembler:
    """Assembles content analysis results into unified output."""

    def __init__(self, output_dir: str = "data/outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def assemble(
        self,
        transcript: dict,
        grammar_errors: list = None,
        grammar_score: dict = None,
        readability: dict = None,
        vocabulary: dict = None,
        regimes: list = None,
        sentiment: dict = None,
        tone: dict = None,
        argument_results: list = None,
    ) -> dict:
        """Assemble all analysis results into unified output.

        Returns:
            Dict with 'segments' (per-segment detail) and 'summary' (speech-level).
        """
        segments = transcript.get("segments", [])
        regime_list = regimes or []

        assembled_segments = []

        for i, seg in enumerate(segments):
            seg_output = {
                "segment_id": i,
                "timestamp_start": seg.get("start_sec", 0.0),
                "timestamp_end": seg.get("end_sec", 0.0),
                "transcript_excerpt": self._truncate(seg.get("text", ""), 200),
                "word_count": seg.get("word_count", 0),
            }

            # Regime
            if i < len(regime_list):
                r = regime_list[i]
                regime_dict = r.to_dict() if hasattr(r, "to_dict") else r
                seg_output["regime_type"] = regime_dict.get("regime_type", "Unknown")
                seg_output["regime_confidence"] = regime_dict.get("confidence", 0.0)
            else:
                seg_output["regime_type"] = "Unknown"
                seg_output["regime_confidence"] = 0.0

            # Grammar (filter errors for this segment)
            seg_grammar_errors = []
            if grammar_errors:
                indices = set(seg.get("sentence_indices", []))
                seg_grammar_errors = [
                    e.to_dict() if hasattr(e, "to_dict") else e
                    for e in grammar_errors
                    if (e.sentence_idx if hasattr(e, "sentence_idx") else e.get("sentence_idx", -1)) in indices
                ]
            seg_output["grammar"] = {
                "errors": seg_grammar_errors,
                "error_count": len(seg_grammar_errors),
            }

            # Readability
            if readability and i < len(readability.get("segments", [])):
                seg_output["readability"] = readability["segments"][i]
            else:
                seg_output["readability"] = {}

            # Vocabulary
            if vocabulary and i < len(vocabulary.get("segments", [])):
                seg_output["vocabulary"] = vocabulary["segments"][i]
            else:
                seg_output["vocabulary"] = {}

            # Sentiment
            if sentiment and i < len(sentiment.get("segments", [])):
                seg_sent = sentiment["segments"][i]
                seg_output["sentiment"] = {
                    "polarity": seg_sent.get("polarity", "Neutral"),
                    "intensity": seg_sent.get("intensity", 0.0),
                    "valence": seg_sent.get("valence", 0.0),
                }
            else:
                seg_output["sentiment"] = {"polarity": "Neutral", "intensity": 0.0, "valence": 0.0}

            # Tone
            if tone and i < len(tone.get("segments", [])):
                seg_output["tone"] = tone["segments"][i]
            else:
                seg_output["tone"] = {"tone": "Unknown", "confidence": 0.0}

            # Argument
            if argument_results and i < len(argument_results):
                seg_output["argument"] = argument_results[i]
            else:
                seg_output["argument"] = {}

            assembled_segments.append(seg_output)

        summary = self._generate_summary(assembled_segments, grammar_score, transcript)

        return {
            "segments": assembled_segments,
            "summary": summary,
            "metadata": {
                "total_words": transcript.get("total_words", 0),
                "total_sentences": transcript.get("total_sentences", 0),
                "duration_sec": transcript.get("duration_sec", 0.0),
            },
        }

    def save(self, output: dict, filename: str = "content_analysis.json") -> str:
        """Save assembled output to JSON file."""
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        logger.info(f"Saved content analysis to {path}")
        return path

    def _generate_summary(self, segments: list, grammar_score: dict, transcript: dict) -> dict:
        """Aggregate segment-level analysis into speech-level summary."""
        if not segments:
            return {}

        regime_seq = [s.get("regime_type", "Unknown") for s in segments]
        tones = [s.get("tone", {}).get("tone", "Unknown") for s in segments]
        sentiments = [s.get("sentiment", {}).get("polarity", "Neutral") for s in segments]

        # Effectiveness scores (filter None)
        eff_scores = [
            s.get("argument", {}).get("effectiveness_score")
            for s in segments
            if s.get("argument", {}).get("effectiveness_score") is not None
        ]

        # Readability grades
        fkgl_scores = [
            s.get("readability", {}).get("flesch_kincaid_grade", 0)
            for s in segments
            if s.get("readability", {}).get("flesch_kincaid_grade") is not None
        ]

        # Vocabulary TTR
        ttr_scores = [
            s.get("vocabulary", {}).get("ttr", 0)
            for s in segments
            if s.get("vocabulary", {}).get("ttr") is not None
        ]

        unique_tones = len(set(t for t in tones if t != "Unknown"))

        summary = {
            "total_segments": len(segments),
            "regime_sequence": regime_seq,
            "grammar_score_overall": grammar_score.get("score", 0) if grammar_score else 0,
            "readability_level": round(median(fkgl_scores), 1) if fkgl_scores else 0.0,
            "vocabulary_diversity": round(mean(ttr_scores), 3) if ttr_scores else 0.0,
            "dominant_sentiment": Counter(sentiments).most_common(1)[0][0] if sentiments else "Neutral",
            "dominant_tone": Counter(tones).most_common(1)[0][0] if tones else "Unknown",
            "tone_variety_score": round(unique_tones / 6, 2),
            "argument_effectiveness_avg": round(mean(eff_scores), 1) if eff_scores else None,
            "regime_transition_count": len(segments) - 1,
            "content_structure_score": self._evaluate_structure(regime_seq),
        }

        return summary

    def _evaluate_structure(self, regime_seq: list) -> int:
        """Score speech structure 0-100 based on regime sequence."""
        score = 50  # Base score

        if not regime_seq:
            return score

        # Bonus: starts with Introduction
        if regime_seq[0] == "Introduction":
            score += 15

        # Bonus: ends with Conclusion or Call to Action
        if regime_seq[-1] in ("Conclusion", "Call to Action"):
            score += 15

        # Bonus: has variety (not all same type)
        unique = len(set(regime_seq))
        if unique >= 3:
            score += 10
        elif unique >= 2:
            score += 5

        # Bonus: has supporting elements
        if "Data Presentation" in regime_seq or "Anecdote" in regime_seq:
            score += 10

        return min(100, score)

    def _truncate(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."
