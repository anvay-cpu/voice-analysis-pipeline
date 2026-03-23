"""
Vocabulary Analysis — Phase 4 of Modality 3.

Computes vocabulary sophistication metrics: TTR, lexical density,
AWL coverage, word frequency, and repetition detection.
Pure computation with spaCy POS tagging.
"""

import logging
import os
import re
from collections import Counter
from typing import Optional

import spacy

logger = logging.getLogger(__name__)

# Content POS tags (spaCy universal tags)
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}

# Common function words to exclude from repetition detection
FUNCTION_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "must", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "and", "but", "or", "nor", "not", "so",
    "yet", "both", "either", "neither", "each", "every", "all", "any",
    "few", "more", "most", "other", "some", "such", "no", "only", "own",
    "same", "than", "too", "very", "just", "because", "if", "when",
    "where", "how", "what", "which", "who", "whom", "this", "that",
    "these", "those", "i", "me", "my", "we", "us", "our", "you", "your",
    "he", "him", "his", "she", "her", "it", "its", "they", "them", "their",
    "there", "here", "then", "also", "about", "up", "out", "well",
}


class VocabularyAnalyzer:
    """Analyzes vocabulary sophistication in speech transcripts."""

    def __init__(self, awl_path: str = "data/resources/academic_word_list.txt",
                 min_ttr_window: int = 100):
        self.min_ttr_window = min_ttr_window
        self._nlp = None
        self.awl_words = self._load_awl(awl_path)

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        return self._nlp

    def analyze_text(self, text: str) -> dict:
        """Analyze vocabulary for a block of text."""
        if not text.strip():
            return self._empty_result()

        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        if not words:
            return self._empty_result()

        doc = self.nlp(text)

        ttr = self._compute_ttr(words)
        lexical_density = self._compute_lexical_density(doc)
        awl_coverage = self._compute_awl_coverage(words)
        top_repeated = self._find_repetitions(words)
        sophistication = self._assess_sophistication(ttr, lexical_density, awl_coverage)

        return {
            "ttr": round(ttr, 3),
            "lexical_density": round(lexical_density, 3),
            "awl_coverage_pct": round(awl_coverage, 1),
            "unique_words": len(set(words)),
            "total_words": len(words),
            "top_repeated_words": top_repeated,
            "sophistication_level": sophistication,
        }

    def analyze_transcript(self, transcript: dict) -> dict:
        """Analyze vocabulary for all segments and overall."""
        segments = transcript.get("segments", [])
        full_text = transcript.get("full_text", "")

        segment_results = []
        for seg in segments:
            result = self.analyze_text(seg["text"])
            result["start_sec"] = seg.get("start_sec", 0.0)
            result["end_sec"] = seg.get("end_sec", 0.0)
            segment_results.append(result)

        overall = self.analyze_text(full_text)

        return {
            "overall": overall,
            "segments": segment_results,
        }

    def _compute_ttr(self, words: list) -> float:
        """Compute Type-Token Ratio on rolling windows to avoid length bias."""
        if len(words) < self.min_ttr_window:
            return len(set(words)) / len(words) if words else 0.0

        ttrs = []
        for i in range(0, len(words) - self.min_ttr_window + 1, self.min_ttr_window // 2):
            window = words[i:i + self.min_ttr_window]
            ttrs.append(len(set(window)) / len(window))

        return sum(ttrs) / len(ttrs) if ttrs else 0.0

    def _compute_lexical_density(self, doc) -> float:
        """Ratio of content words to total words using spaCy POS tags."""
        tokens = [t for t in doc if t.is_alpha]
        if not tokens:
            return 0.0
        content_count = sum(1 for t in tokens if t.pos_ in CONTENT_POS)
        return content_count / len(tokens)

    def _compute_awl_coverage(self, words: list) -> float:
        """Percentage of words from the Academic Word List."""
        if not words or not self.awl_words:
            return 0.0
        awl_count = sum(1 for w in words if w in self.awl_words)
        return (awl_count / len(words)) * 100

    def _find_repetitions(self, words: list, top_n: int = 10) -> list:
        """Find excessively repeated content words."""
        filtered = [w for w in words if w not in FUNCTION_WORDS and len(w) > 2]
        counts = Counter(filtered)

        if not counts:
            return []

        # Flag words appearing more than 3x expected frequency
        total = len(filtered)
        unique = len(counts)
        if unique == 0:
            return []
        expected_freq = total / unique

        repeated = [(word, count) for word, count in counts.most_common(top_n * 2)
                     if count > max(3, expected_freq * 3)]

        return repeated[:top_n]

    def _assess_sophistication(self, ttr: float, density: float, awl: float) -> str:
        """Assess overall vocabulary sophistication."""
        score = 0
        if ttr > 0.6:
            score += 2
        elif ttr > 0.5:
            score += 1
        if density > 0.55:
            score += 2
        elif density > 0.45:
            score += 1
        if awl > 8:
            score += 2
        elif awl > 4:
            score += 1

        if score >= 5:
            return "High"
        elif score >= 3:
            return "Moderate"
        else:
            return "Low"

    def _load_awl(self, path: str) -> set:
        """Load Academic Word List from file."""
        if not os.path.exists(path):
            logger.warning(f"AWL file not found at {path}, AWL coverage will be 0")
            return set()
        with open(path) as f:
            return {line.strip().lower() for line in f if line.strip()}

    def _empty_result(self) -> dict:
        return {
            "ttr": 0.0,
            "lexical_density": 0.0,
            "awl_coverage_pct": 0.0,
            "unique_words": 0,
            "total_words": 0,
            "top_repeated_words": [],
            "sophistication_level": "N/A",
        }
