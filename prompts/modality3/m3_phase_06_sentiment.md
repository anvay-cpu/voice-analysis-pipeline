PHASE 6 — SENTIMENT ANALYSIS
===============================

AGENT ROLE: Sentiment Specialist
DEPENDS ON: Phase 2 (transcript), Phase 5 (regime segments)
DELIVERS TO: Phase 9 (output assembly), Multimodal Fusion Layer
ESTIMATED TIME: 15 minutes (agent), 0 minutes (user)

OBJECTIVE:
Build src/content/sentiment_analyzer.py using a pretrained HuggingFace
model. NO TRAINING NEEDED.

TASKS:
1. Build SentimentAnalyzer class:
   - Load cardiffnlp/twitter-roberta-base-sentiment-latest from HuggingFace
   - Analyze per-sentence AND per-segment sentiment
   - Output: polarity (Positive/Negative/Neutral) + intensity (0-1)

2. Implementation:
```python
from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.pipe = pipeline("sentiment-analysis", model=model_name,
                             top_k=3, truncation=True, max_length=512)

    def analyze_sentence(self, text: str) -> dict:
        result = self.pipe(text)[0]
        # result = [{"label": "positive", "score": 0.85}, ...]
        return {
            "polarity": result[0]["label"].capitalize(),
            "intensity": result[0]["score"],
            "distribution": {r["label"]: r["score"] for r in result}
        }

    def analyze_segment(self, segment_text: str) -> dict:
        """Analyze a full segment by averaging sentence sentiments."""
        sentences = split_sentences(segment_text)
        sentiments = [self.analyze_sentence(s) for s in sentences]
        # Aggregate: majority polarity, mean intensity
        ...
```

3. Map to valence for emotion coherence analysis:
   Positive → valence > 0
   Negative → valence < 0
   Neutral → valence ≈ 0
   This valence feeds into the Multimodal Fusion Layer where it's
   compared with vocal emotion and facial emotion.

4. Handle long segments:
   RoBERTa has a 512 token limit. For segments > 400 words,
   analyze sentence-by-sentence and aggregate.

NO USER INPUT REQUIRED. Model downloads automatically (~500MB first time).
NO TRAINING REQUIRED.

COMPLETION: ✓ when sentiment is computed for all segments with correct polarity.