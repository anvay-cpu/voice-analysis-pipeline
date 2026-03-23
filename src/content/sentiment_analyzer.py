"""
Sentiment Analysis — Phase 6 of Modality 3.

Uses pretrained cardiffnlp/twitter-roberta-base-sentiment-latest.
NO TRAINING NEEDED. Model downloads automatically (~500MB first time).
"""

import logging
from collections import Counter

from transformers import pipeline

logger = logging.getLogger(__name__)

# Map model labels to standard names
LABEL_MAP = {
    "positive": "Positive",
    "negative": "Negative",
    "neutral": "Neutral",
}

# Map polarity to valence for cross-modal coherence
VALENCE_MAP = {
    "Positive": 0.5,
    "Negative": -0.5,
    "Neutral": 0.0,
}


class SentimentAnalyzer:
    """Analyzes sentiment per sentence and per segment using pretrained RoBERTa."""

    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        logger.info(f"Loading sentiment model: {model_name}")
        self.pipe = pipeline(
            "sentiment-analysis",
            model=model_name,
            top_k=3,
            truncation=True,
            max_length=512,
        )

    def analyze_sentence(self, text: str) -> dict:
        """Analyze sentiment of a single sentence.

        Returns:
            Dict with polarity, intensity, valence, and full distribution.
        """
        if not text.strip():
            return {"polarity": "Neutral", "intensity": 0.0, "valence": 0.0,
                    "distribution": {"Positive": 0.0, "Negative": 0.0, "Neutral": 1.0}}

        result = self.pipe(text)[0]

        # Build distribution with standardized labels
        distribution = {}
        for r in result:
            label = LABEL_MAP.get(r["label"].lower(), r["label"].capitalize())
            distribution[label] = round(r["score"], 4)

        # Top prediction
        top = max(distribution, key=distribution.get)
        intensity = distribution[top]
        valence = VALENCE_MAP.get(top, 0.0) * intensity

        return {
            "polarity": top,
            "intensity": round(intensity, 4),
            "valence": round(valence, 4),
            "distribution": distribution,
        }

    def analyze_segment(self, sentences: list) -> dict:
        """Analyze sentiment of a segment by aggregating sentence-level results.

        Args:
            sentences: List of sentence dicts with 'text' key, or list of strings.

        Returns:
            Dict with aggregate polarity, intensity, valence, and per-sentence detail.
        """
        if not sentences:
            return {"polarity": "Neutral", "intensity": 0.0, "valence": 0.0,
                    "sentence_sentiments": [], "polarity_distribution": {}}

        sentence_results = []
        for sent in sentences:
            text = sent["text"] if isinstance(sent, dict) else sent
            result = self.analyze_sentence(text)
            sentence_results.append(result)

        # Aggregate: majority polarity
        polarities = [r["polarity"] for r in sentence_results]
        polarity_counts = Counter(polarities)
        majority_polarity = polarity_counts.most_common(1)[0][0]

        # Mean intensity and valence
        mean_intensity = sum(r["intensity"] for r in sentence_results) / len(sentence_results)
        mean_valence = sum(r["valence"] for r in sentence_results) / len(sentence_results)

        # Polarity distribution across sentences
        polarity_dist = {p: round(c / len(polarities), 3) for p, c in polarity_counts.items()}

        return {
            "polarity": majority_polarity,
            "intensity": round(mean_intensity, 4),
            "valence": round(mean_valence, 4),
            "sentence_sentiments": sentence_results,
            "polarity_distribution": polarity_dist,
        }

    def analyze_transcript(self, transcript: dict) -> dict:
        """Analyze sentiment for all segments in a transcript.

        Args:
            transcript: Output from TranscriptProcessor.

        Returns:
            Dict with overall and per-segment sentiment.
        """
        sentences = transcript.get("sentences", [])
        segments = transcript.get("segments", [])

        # Per-segment analysis
        segment_results = []
        for seg in segments:
            # Get sentences for this segment
            indices = seg.get("sentence_indices", [])
            seg_sentences = [sentences[i] for i in indices if i < len(sentences)]
            if not seg_sentences:
                seg_sentences = [{"text": seg["text"]}]

            result = self.analyze_segment(seg_sentences)
            result["start_sec"] = seg.get("start_sec", 0.0)
            result["end_sec"] = seg.get("end_sec", 0.0)
            segment_results.append(result)

        # Overall analysis
        overall = self.analyze_segment(sentences)

        return {
            "overall": overall,
            "segments": segment_results,
        }
