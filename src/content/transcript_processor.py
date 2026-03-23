"""
Transcript Preprocessing — Phase 2 of Modality 3.

Takes raw Whisper transcript (word-level timestamps) and produces clean,
structured text ready for NLP analysis.
"""

import json
import re
import logging
from typing import Optional

import spacy

logger = logging.getLogger(__name__)

# Common ASR contractions to expand for grammar checking
ASR_EXPANSIONS = {
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "kinda": "kind of",
    "sorta": "sort of",
    "dunno": "don't know",
    "lemme": "let me",
    "gimme": "give me",
    "coulda": "could have",
    "shoulda": "should have",
    "woulda": "would have",
    "oughta": "ought to",
    "hafta": "have to",
    "lotsa": "lots of",
    "outta": "out of",
}

# Regex to detect Whisper hallucinations (same phrase repeated 3+ times)
HALLUCINATION_RE = re.compile(r"(\b\w[\w\s]{3,30}?)\1{2,}", re.IGNORECASE)


class TranscriptProcessor:
    """Processes raw Whisper output into structured text for NLP analysis."""

    def __init__(self, sentence_splitter: str = "spacy", min_words_per_segment: int = 10):
        self.min_words = min_words_per_segment
        self.sentence_splitter = sentence_splitter
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = spacy.load("en_core_web_sm")
        return self._nlp

    def load_from_whisper(self, whisper_output: dict) -> dict:
        """Load from Modality 1 Whisper output dict.

        Accepts two formats:
        Format A (segments): {"segments": [{"start": ..., "end": ..., "text": ..., "words": [...]}]}
        Format B (flat words): {"text": "...", "words": [{"word": ..., "start": ..., "end": ...}]}
        """
        words = []

        # Format B: flat words list (from voice pipeline transcriber)
        if "words" in whisper_output and isinstance(whisper_output["words"], list):
            for w in whisper_output["words"]:
                words.append({
                    "word": w.get("word", "").strip(),
                    "start": w.get("start", 0.0),
                    "end": w.get("end", 0.0),
                })
        # Format A: segments with nested words
        else:
            for seg in whisper_output.get("segments", []):
                for w in seg.get("words", []):
                    words.append({
                        "word": w.get("word", "").strip(),
                        "start": w.get("start", 0.0),
                        "end": w.get("end", 0.0),
                    })
                # Fallback: if no word-level data, use segment text
                if not seg.get("words") and seg.get("text"):
                    seg_words = seg["text"].strip().split()
                    n = len(seg_words)
                    if n == 0:
                        continue
                    seg_dur = seg["end"] - seg["start"]
                    for i, sw in enumerate(seg_words):
                        words.append({
                            "word": sw,
                            "start": seg["start"] + (i / n) * seg_dur,
                            "end": seg["start"] + ((i + 1) / n) * seg_dur,
                        })

        if not words:
            logger.warning("No words found in Whisper output")
            return self._empty_result()

        full_text = " ".join(w["word"] for w in words)
        full_text = self._remove_hallucinations(full_text)
        duration = words[-1]["end"] - words[0]["start"]

        sentences = self._split_sentences(full_text, words)
        segments = self._group_into_segments(sentences)

        return {
            "full_text": full_text,
            "sentences": sentences,
            "segments": segments,
            "total_words": len(full_text.split()),
            "total_sentences": len(sentences),
            "duration_sec": round(duration, 2),
        }

    def load_from_json(self, json_path: str) -> dict:
        """Load Whisper output from JSON file."""
        with open(json_path, "r") as f:
            data = json.load(f)
        return self.load_from_whisper(data)

    def load_from_text(self, text: str, duration_sec: float = 0.0) -> dict:
        """Load from plain text (no timestamps). Useful for testing."""
        words_list = text.split()
        n = len(words_list)
        if n == 0:
            return self._empty_result()

        # Synthesize evenly-spaced timestamps
        words = []
        for i, w in enumerate(words_list):
            words.append({
                "word": w,
                "start": (i / n) * duration_sec,
                "end": ((i + 1) / n) * duration_sec,
            })

        text = self._remove_hallucinations(text)
        sentences = self._split_sentences(text, words)
        segments = self._group_into_segments(sentences)

        return {
            "full_text": text,
            "sentences": sentences,
            "segments": segments,
            "total_words": n,
            "total_sentences": len(sentences),
            "duration_sec": round(duration_sec, 2),
        }

    def clean_for_grammar(self, text: str) -> str:
        """Clean text for grammar checking: capitalize, expand ASR contractions."""
        # Expand ASR contractions
        for contraction, expansion in ASR_EXPANSIONS.items():
            text = re.sub(rf"\b{contraction}\b", expansion, text, flags=re.IGNORECASE)

        # Capitalize sentence starts
        text = re.sub(r"(^|[.!?]\s+)(\w)", lambda m: m.group(1) + m.group(2).upper(), text)

        # Capitalize "I" when standalone
        text = re.sub(r"\bi\b", "I", text)

        return text.strip()

    # ── Private methods ──

    def _split_sentences(self, text: str, words: list) -> list:
        """Split text into sentences and map back to timestamps."""
        doc = self.nlp(text)
        sentences = []

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue

            # Force-split very long sentences at commas
            if len(sent_text.split()) > 50:
                sub_sents = self._force_split_long(sent_text)
            else:
                sub_sents = [sent_text]

            for ss in sub_sents:
                word_count = len(ss.split())
                start_sec, end_sec = self._find_timestamps(ss, words)
                sentences.append({
                    "text": ss,
                    "start_sec": round(start_sec, 3),
                    "end_sec": round(end_sec, 3),
                    "word_count": word_count,
                })

        return sentences

    def _force_split_long(self, text: str) -> list:
        """Split long sentences at commas to keep segments manageable."""
        parts = text.split(",")
        result = []
        current = ""
        for part in parts:
            candidate = (current + "," + part).strip(" ,") if current else part.strip()
            if len(candidate.split()) >= 20 and current:
                result.append(current.strip())
                current = part.strip()
            else:
                current = candidate
        if current.strip():
            result.append(current.strip())
        return result if result else [text]

    def _find_timestamps(self, sentence_text: str, words: list) -> tuple:
        """Find start/end timestamps for a sentence by matching words."""
        sent_words = sentence_text.lower().split()
        if not sent_words or not words:
            return (0.0, 0.0)

        # Sliding window match
        best_start_idx = 0
        best_score = -1
        n_sent = len(sent_words)

        for i in range(len(words)):
            score = 0
            for j in range(min(n_sent, len(words) - i)):
                w_clean = re.sub(r"[^\w']", "", words[i + j]["word"].lower())
                s_clean = re.sub(r"[^\w']", "", sent_words[j])
                if w_clean == s_clean:
                    score += 1
            if score > best_score:
                best_score = score
                best_start_idx = i

        end_idx = min(best_start_idx + n_sent - 1, len(words) - 1)
        return (words[best_start_idx]["start"], words[end_idx]["end"])

    def _group_into_segments(self, sentences: list) -> list:
        """Group sentences into segments with at least min_words each."""
        segments = []
        current_texts = []
        current_indices = []
        current_word_count = 0
        current_start = 0.0

        for i, sent in enumerate(sentences):
            if not current_texts:
                current_start = sent["start_sec"]
            current_texts.append(sent["text"])
            current_indices.append(i)
            current_word_count += sent["word_count"]

            if current_word_count >= self.min_words:
                segments.append({
                    "text": " ".join(current_texts),
                    "start_sec": round(current_start, 3),
                    "end_sec": round(sent["end_sec"], 3),
                    "word_count": current_word_count,
                    "sentence_indices": list(current_indices),
                })
                current_texts = []
                current_indices = []
                current_word_count = 0

        # Append remaining sentences
        if current_texts:
            if segments:
                # Merge with last segment if remainder is too small
                last = segments[-1]
                last["text"] += " " + " ".join(current_texts)
                last["end_sec"] = round(sentences[current_indices[-1]]["end_sec"], 3)
                last["word_count"] += current_word_count
                last["sentence_indices"].extend(current_indices)
            else:
                segments.append({
                    "text": " ".join(current_texts),
                    "start_sec": round(current_start, 3),
                    "end_sec": round(sentences[current_indices[-1]]["end_sec"], 3),
                    "word_count": current_word_count,
                    "sentence_indices": list(current_indices),
                })

        return segments

    def _remove_hallucinations(self, text: str) -> str:
        """Remove Whisper hallucinations (repeated phrases)."""
        cleaned = HALLUCINATION_RE.sub(r"\1", text)
        if cleaned != text:
            logger.info("Removed Whisper hallucination artifacts")
        return cleaned

    def _empty_result(self) -> dict:
        return {
            "full_text": "",
            "sentences": [],
            "segments": [],
            "total_words": 0,
            "total_sentences": 0,
            "duration_sec": 0.0,
        }
