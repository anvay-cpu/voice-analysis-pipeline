"""
Grammar Checking — Phase 3 of Modality 3.

Uses LanguageTool (pretrained, no training needed) to detect grammatical errors.
Filters for spoken language context to avoid false positives from speech artifacts.
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

import language_tool_python

logger = logging.getLogger(__name__)

# LanguageTool categories to ignore in spoken language context
SPEECH_IGNORE_CATEGORIES = {
    "COMMA_PARENTHESIS_WHITESPACE",
    "WHITESPACE_RULE",
    "UPPERCASE_SENTENCE_START",
    "MORFOLOGIK_RULE_EN_US",
    "EN_UNPAIRED_BRACKETS",
    "EN_QUOTES",
    "DASH_RULE",
}

# Additional rule IDs to ignore (common false positives in speech)
SPEECH_IGNORE_RULES = {
    "COMMA_COMPOUND_SENTENCE",
    "COMMA_COMPOUND_SENTENCE_2",
    "MISSING_COMMA_AFTER_INTRODUCTORY_PHRASE",
    "OXFORD_COMMA",
    "COMMA_BEFORE_AND",
    "PUNCTUATION_PARAGRAPH_END",
    "SENTENCE_WHITESPACE",
    "DOUBLE_PUNCTUATION",
    "CONSECUTIVE_SPACES",
}

# Map LanguageTool category → severity
SEVERITY_MAP = {
    # Critical: core grammar errors
    "GRAMMAR": "critical",
    "SUBJECT_VERB_AGREEMENT": "critical",
    "TENSE": "critical",
    "NEGATION": "critical",

    # Major: word choice / form errors
    "CONFUSED_WORDS": "major",
    "PREPOSITION": "major",
    "DETERMINERS": "major",
    "ARTICLES": "major",
    "WORD_ORDER": "major",
    "PRONOUN": "major",

    # Minor: style issues
    "STYLE": "minor",
    "REDUNDANCY": "minor",
    "TYPOGRAPHY": "minor",
    "PUNCTUATION": "minor",
    "CASING": "minor",
    "MISC": "minor",
}


@dataclass
class GrammarError:
    sentence_idx: int
    start_sec: float
    end_sec: float
    original: str
    correction: str
    error_type: str
    severity: str
    rule_id: str
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


class GrammarChecker:
    """Detects grammatical errors using LanguageTool, filtered for speech context."""

    def __init__(self, language: str = "en-US", severity_weights: Optional[dict] = None):
        logger.info("Loading LanguageTool (may download ~200MB on first run)...")
        self.tool = language_tool_python.LanguageTool(language)
        self.severity_weights = severity_weights or {
            "critical": 1.0,
            "major": 0.7,
            "minor": 0.3,
        }

    def check_sentence(self, sentence: str) -> list:
        """Check a single sentence for grammar errors.

        Returns list of (match, severity) tuples after filtering.
        """
        matches = self.tool.check(sentence)
        filtered = []
        for m in matches:
            if m.category in SPEECH_IGNORE_CATEGORIES:
                continue
            if m.rule_id in SPEECH_IGNORE_RULES:
                continue
            severity = self._classify_severity(m)
            filtered.append((m, severity))
        return filtered

    def check_transcript(self, sentences: list) -> list:
        """Check all sentences from TranscriptProcessor output.

        Args:
            sentences: List of dicts with keys: text, start_sec, end_sec

        Returns:
            List of GrammarError objects.
        """
        errors = []
        for idx, sent in enumerate(sentences):
            matches = self.check_sentence(sent["text"])
            for match, severity in matches:
                corrections = match.replacements[:3] if match.replacements else []
                original = sent["text"][match.offset:match.offset + match.error_length]
                correction = corrections[0] if corrections else ""

                errors.append(GrammarError(
                    sentence_idx=idx,
                    start_sec=sent.get("start_sec", 0.0),
                    end_sec=sent.get("end_sec", 0.0),
                    original=original,
                    correction=correction,
                    error_type=self._simplify_category(match.category, match.rule_id),
                    severity=severity,
                    rule_id=match.rule_id,
                    message=match.message,
                ))

        logger.info(f"Found {len(errors)} grammar errors across {len(sentences)} sentences")
        return errors

    def score_grammar(self, errors: list, total_words: int) -> dict:
        """Compute grammar score and error statistics.

        Returns:
            dict with score (0-100), error counts, errors_per_100_words.
        """
        if total_words == 0:
            return {"score": 100, "critical": 0, "major": 0, "minor": 0,
                    "total_errors": 0, "errors_per_100_words": 0.0}

        counts = {"critical": 0, "major": 0, "minor": 0}
        for e in errors:
            if e.severity in counts:
                counts[e.severity] += 1

        weighted_penalty = (
            counts["critical"] * 5 +
            counts["major"] * 3 +
            counts["minor"] * 1
        )
        score = max(0, 100 - weighted_penalty)
        errors_per_100 = round(len(errors) / (total_words / 100), 2)

        return {
            "score": score,
            **counts,
            "total_errors": len(errors),
            "errors_per_100_words": errors_per_100,
        }

    def close(self):
        """Shut down the LanguageTool server."""
        self.tool.close()

    # ── Private ──

    def _classify_severity(self, match) -> str:
        """Classify a LanguageTool match into critical/major/minor."""
        # Check specific rule IDs first
        rule_upper = match.rule_id.upper()
        if any(k in rule_upper for k in ["AGREEMENT", "SUBJECT_VERB"]):
            return "critical"
        if any(k in rule_upper for k in ["TENSE", "DOUBLE_NEGATIVE"]):
            return "critical"

        # Fall back to category mapping
        return SEVERITY_MAP.get(match.category, "minor")

    def _simplify_category(self, category: str, rule_id: str) -> str:
        """Convert LanguageTool category/rule to a simple error type string."""
        rule_upper = rule_id.upper()
        if "AGREEMENT" in rule_upper or "SUBJECT_VERB" in rule_upper:
            return "subject_verb"
        if "TENSE" in rule_upper:
            return "tense"
        if "ARTICLE" in rule_upper or "DETERMINER" in rule_upper:
            return "article"
        if "PREPOSITION" in rule_upper:
            return "preposition"
        if "CONFUSED" in rule_upper:
            return "word_choice"
        if "PRONOUN" in rule_upper:
            return "pronoun"
        if "REDUNDAN" in rule_upper:
            return "redundancy"
        if "STYLE" in category.upper():
            return "style"
        return category.lower().replace(" ", "_")
