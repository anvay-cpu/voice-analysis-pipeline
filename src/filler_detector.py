"""Filler word detection from transcription patterns (Stage 1 — regex-based)."""

import re
from dataclasses import dataclass, asdict


@dataclass
class FillerDetection:
    """A detected filler word with metadata."""
    word: str
    start_sec: float
    end_sec: float
    confidence: float
    category: str  # "hesitation" or "discourse"


# Filler patterns keyed by category
FILLER_PATTERNS = {
    "hesitation": [
        r"\bum\b", r"\buh\b", r"\berm\b", r"\bah\b", r"\bmm\b",
        r"\bhmm\b", r"\bumm\b", r"\buhh\b", r"\ber\b",
    ],
    "discourse": [
        r"\byou know\b", r"\bi mean\b", r"\bsort of\b", r"\bkind of\b",
        r"\bokay so\b", r"\bbasically\b", r"\bactually\b",
    ],
}

# Context-dependent words that need surrounding-phrase checks
CONTEXT_WORDS = {"like", "so", "right", "well", "okay"}

# Phrases where the word is NOT a filler
NOT_FILLER_CONTEXTS = {
    "like": [
        r"\bi like\b", r"\blike a\b", r"\blike the\b", r"\blike this\b",
        r"\blike that\b", r"\blooks like\b", r"\bfeel like\b",
        r"\bsounds? like\b", r"\bseems like\b", r"\bwould like\b",
        r"\bjust like\b", r"\blike to\b", r"\blike it\b",
        r"\bsomething like\b", r"\banything like\b", r"\bnothing like\b",
        r"\bmore like\b", r"\blike me\b", r"\blike him\b", r"\blike her\b",
        r"\blike we\b", r"\blike they\b", r"\blike when\b", r"\blike how\b",
    ],
    "so": [
        r"\bso that\b", r"\bso much\b", r"\bso many\b", r"\bso far\b",
        r"\bso long\b", r"\bdo so\b", r"\beven so\b", r"\band so\b",
        r"\bor so\b", r"\bnot so\b",
    ],
    "right": [
        r"\bright now\b", r"\bright there\b", r"\bright here\b",
        r"\bright away\b", r"\ball right\b", r"\bthat's right\b",
        r"\byou're right\b", r"\bright to\b", r"\bright of\b",
    ],
    "well": [
        r"\bas well\b", r"\bvery well\b", r"\bwell known\b",
        r"\bdoing well\b", r"\bquite well\b",
    ],
    "okay": [
        r"\bokay with\b", r"\bokay to\b", r"\bis okay\b", r"\bit's okay\b",
    ],
}


def _build_context_window(words: list[dict], idx: int, window: int = 3) -> str:
    """Build a lowercase context string around the word at idx."""
    start = max(0, idx - window)
    end = min(len(words), idx + window + 1)
    return " ".join(w["word"].lower() for w in words[start:end])


def detect_fillers_from_transcript(whisper_output: dict) -> list[FillerDetection]:
    """Detect filler words from Whisper transcript using regex patterns.

    Args:
        whisper_output: Dict with "text" and "words" keys from transcriber.
            Each word has "word", "start", "end".

    Returns:
        List of FillerDetection objects.
    """
    words = whisper_output.get("words", [])
    if not words:
        return []

    fillers = []
    skip_next = False

    for i, w in enumerate(words):
        if skip_next:
            skip_next = False
            continue

        word_lower = w["word"].strip().lower()
        word_clean = re.sub(r"[^\w\s]", "", word_lower)

        # Check multi-word fillers first (look ahead one word)
        if i < len(words) - 1:
            next_word = re.sub(r"[^\w\s]", "", words[i + 1]["word"].strip().lower())
            two_words = f"{word_clean} {next_word}"
            matched_multi = False
            for pattern in FILLER_PATTERNS["discourse"]:
                if re.fullmatch(pattern.replace(r"\b", ""), two_words):
                    fillers.append(FillerDetection(
                        word=two_words,
                        start_sec=w["start"],
                        end_sec=words[i + 1]["end"],
                        confidence=0.85,
                        category="discourse",
                    ))
                    skip_next = True
                    matched_multi = True
                    break
            if matched_multi:
                continue

        # Check hesitation fillers
        matched = False
        for pattern in FILLER_PATTERNS["hesitation"]:
            if re.fullmatch(pattern.replace(r"\b", ""), word_clean):
                fillers.append(FillerDetection(
                    word=word_clean,
                    start_sec=w["start"],
                    end_sec=w["end"],
                    confidence=0.95,
                    category="hesitation",
                ))
                matched = True
                break

        if matched:
            continue

        # Check context-dependent words
        if word_clean in CONTEXT_WORDS:
            context = _build_context_window(words, i)

            # Skip if it matches a NOT-filler phrase
            is_not_filler = False
            if word_clean in NOT_FILLER_CONTEXTS:
                for nf_pattern in NOT_FILLER_CONTEXTS[word_clean]:
                    if re.search(nf_pattern, context):
                        is_not_filler = True
                        break

            if not is_not_filler:
                # Check for pause before (indicator of filler usage)
                pause_before = 0.0
                if i > 0:
                    pause_before = max(0, w["start"] - words[i - 1]["end"])

                if pause_before > 0.2:
                    fillers.append(FillerDetection(
                        word=word_clean,
                        start_sec=w["start"],
                        end_sec=w["end"],
                        confidence=0.6,
                        category="discourse",
                    ))

        # Check single-word discourse fillers (basically, actually)
        for pattern in FILLER_PATTERNS["discourse"]:
            pat_clean = pattern.replace(r"\b", "")
            if " " not in pat_clean and re.fullmatch(pat_clean, word_clean):
                fillers.append(FillerDetection(
                    word=word_clean,
                    start_sec=w["start"],
                    end_sec=w["end"],
                    confidence=0.7,
                    category="discourse",
                ))
                break

    # Deduplicate by timestamp
    seen = set()
    unique = []
    for f in fillers:
        key = (f.start_sec, f.end_sec)
        if key not in seen:
            seen.add(key)
            unique.append(f)

    return unique


def compute_filler_rate(fillers: list[FillerDetection],
                        duration_sec: float,
                        word_count: int) -> dict:
    """Compute filler rate statistics.

    Args:
        fillers: List of detected fillers.
        duration_sec: Total audio duration in seconds.
        word_count: Total word count in transcript.

    Returns:
        Dict with fillers_per_minute, filler_percentage, counts_by_category.
    """
    duration_min = duration_sec / 60.0 if duration_sec > 0 else 1.0

    counts_by_category = {}
    for f in fillers:
        counts_by_category[f.category] = counts_by_category.get(f.category, 0) + 1

    return {
        "total_fillers": len(fillers),
        "fillers_per_minute": round(len(fillers) / duration_min, 2),
        "filler_percentage": round(100.0 * len(fillers) / max(word_count, 1), 2),
        "counts_by_category": counts_by_category,
    }
