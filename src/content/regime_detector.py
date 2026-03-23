"""
Regime Detection / Topic Segmentation — Phase 5 of Modality 3.

Detects topic transition boundaries using embedding distance (unsupervised),
then labels each regime using Claude API via the CLI wrapper.
"""

import logging
import json
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.signal import find_peaks
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)

REGIME_TYPES = [
    "Introduction",
    "Main Argument",
    "Anecdote",
    "Data Presentation",
    "Counterargument",
    "Call to Action",
    "Q&A",
    "Conclusion",
    "Transition",
]

LABELING_PROMPT = """You are analyzing a speech transcript. Classify the following segment into ONE of these categories:
Introduction, Main Argument, Anecdote, Data Presentation, Counterargument, Call to Action, Q&A, Conclusion, Transition.

Previous segment summary: {context}
Current segment: {segment_text}

Respond with ONLY the category name, nothing else."""


@dataclass
class Regime:
    regime_type: str
    start_sentence: int
    end_sentence: int
    text: str
    start_sec: float
    end_sec: float
    sentence_count: int
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


class RegimeDetector:
    """Detects topic regimes in speech transcripts.

    Stage 1: Unsupervised boundary detection via embedding distance peaks.
    Stage 2: Regime type labeling via Claude API (zero-shot).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        half_window: int = 5,
        min_prominence: float = 0.15,
        min_segment_sentences: int = 3,
        use_llm: bool = True,
    ):
        logger.info(f"Loading sentence encoder: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        self.half_window = half_window
        self.min_prominence = min_prominence
        self.min_segment_sentences = min_segment_sentences
        self.use_llm = use_llm

    def detect_boundaries(self, sentences: list) -> list:
        """Detect topic boundary positions using embedding distance.

        Args:
            sentences: List of sentence dicts with 'text' key, or list of strings.

        Returns:
            List of sentence indices where boundaries occur.
        """
        texts = [s["text"] if isinstance(s, dict) else s for s in sentences]
        n = len(texts)
        w = self.half_window

        if n < 2 * w + 1:
            logger.info(f"Too few sentences ({n}) for boundary detection (need {2*w+1}+)")
            return []

        embeddings = self.encoder.encode(texts, show_progress_bar=False)

        # Compute distance signal
        distances = []
        for i in range(w, n - w):
            left_centroid = embeddings[i - w:i].mean(axis=0)
            right_centroid = embeddings[i:i + w].mean(axis=0)
            dist = cosine(left_centroid, right_centroid)
            distances.append(dist)

        if not distances:
            return []

        distances = np.array(distances)

        # Peak detection
        peaks, properties = find_peaks(
            distances,
            prominence=self.min_prominence,
            distance=max(self.min_segment_sentences, w),
        )

        # Convert back to sentence indices
        boundaries = [int(p + w) for p in peaks]

        logger.info(f"Found {len(boundaries)} boundaries in {n} sentences")
        return boundaries

    def segment_transcript(self, sentences: list, boundaries: list) -> list:
        """Split sentences into segments at boundary positions.

        Args:
            sentences: List of sentence dicts from TranscriptProcessor.
            boundaries: List of boundary sentence indices.

        Returns:
            List of segment dicts.
        """
        all_bounds = [0] + sorted(boundaries) + [len(sentences)]
        segments = []

        for i in range(len(all_bounds) - 1):
            start = all_bounds[i]
            end = all_bounds[i + 1]
            seg_sentences = sentences[start:end]

            if not seg_sentences:
                continue

            text = " ".join(s["text"] if isinstance(s, dict) else s for s in seg_sentences)
            start_sec = seg_sentences[0].get("start_sec", 0.0) if isinstance(seg_sentences[0], dict) else 0.0
            end_sec = seg_sentences[-1].get("end_sec", 0.0) if isinstance(seg_sentences[-1], dict) else 0.0

            segments.append({
                "start_sentence": start,
                "end_sentence": end,
                "text": text,
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "sentence_count": len(seg_sentences),
            })

        return segments

    def label_regimes(self, segments: list) -> list:
        """Label each segment with a regime type.

        Uses Claude API if use_llm=True, otherwise uses heuristic fallback.
        """
        regimes = []
        prev_summary = "Beginning of speech"

        for i, seg in enumerate(segments):
            if self.use_llm:
                regime_type, confidence = self._label_with_llm(seg["text"], prev_summary)
            else:
                regime_type, confidence = self._label_heuristic(seg, i, len(segments))

            regime = Regime(
                regime_type=regime_type,
                start_sentence=seg["start_sentence"],
                end_sentence=seg["end_sentence"],
                text=seg["text"],
                start_sec=seg["start_sec"],
                end_sec=seg["end_sec"],
                sentence_count=seg["sentence_count"],
                confidence=confidence,
            )
            regimes.append(regime)

            # Update context for next segment
            words = seg["text"].split()
            prev_summary = " ".join(words[:30]) + ("..." if len(words) > 30 else "")

        return regimes

    def detect_and_label(self, transcript: dict) -> list:
        """Full pipeline: detect boundaries, segment, and label.

        Args:
            transcript: Output from TranscriptProcessor.

        Returns:
            List of Regime objects.
        """
        sentences = transcript.get("sentences", [])
        if not sentences:
            return []

        boundaries = self.detect_boundaries(sentences)
        segments = self.segment_transcript(sentences, boundaries)
        regimes = self.label_regimes(segments)

        return regimes

    # ── Private ──

    def _label_with_llm(self, segment_text: str, context: str) -> tuple:
        """Label a segment using Claude CLI wrapper."""
        try:
            from src.utils.claude_api_wrapper import call_claude

            # Truncate segment to avoid very long prompts
            words = segment_text.split()
            if len(words) > 200:
                segment_text = " ".join(words[:200]) + "..."

            prompt = LABELING_PROMPT.format(
                context=context,
                segment_text=segment_text,
            )

            response = call_claude(prompt, model="claude-sonnet-4-20250514", timeout=30)
            response = response.strip().strip('"').strip("'")

            # Match to closest valid regime type
            matched = self._match_regime_type(response)
            return (matched, 0.9)

        except Exception as e:
            logger.warning(f"LLM labeling failed: {e}, falling back to heuristic")
            return self._label_heuristic_simple(segment_text)

    def _match_regime_type(self, response: str) -> str:
        """Match LLM response to the closest valid regime type."""
        response_lower = response.lower().strip()
        for rt in REGIME_TYPES:
            if rt.lower() in response_lower:
                return rt
        # Fuzzy fallback
        for rt in REGIME_TYPES:
            if any(word in response_lower for word in rt.lower().split()):
                return rt
        return "Main Argument"  # Default fallback

    def _label_heuristic(self, segment: dict, idx: int, total: int) -> tuple:
        """Simple heuristic labeling based on position."""
        if idx == 0:
            return ("Introduction", 0.7)
        if idx == total - 1:
            return ("Conclusion", 0.7)
        return ("Main Argument", 0.4)

    def _label_heuristic_simple(self, text: str) -> tuple:
        """Heuristic based on text content."""
        text_lower = text.lower()
        if any(w in text_lower for w in ["welcome", "hello", "good morning", "today i"]):
            return ("Introduction", 0.6)
        if any(w in text_lower for w in ["in conclusion", "to summarize", "thank you"]):
            return ("Conclusion", 0.6)
        if any(w in text_lower for w in ["percent", "data", "study", "research", "survey"]):
            return ("Data Presentation", 0.5)
        if any(w in text_lower for w in ["imagine", "story", "remember when"]):
            return ("Anecdote", 0.5)
        if any(w in text_lower for w in ["we must", "we need to", "take action", "urge"]):
            return ("Call to Action", 0.5)
        return ("Main Argument", 0.4)
