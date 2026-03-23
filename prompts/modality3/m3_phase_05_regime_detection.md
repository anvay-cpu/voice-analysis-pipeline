PHASE 5 — REGIME DETECTION / TOPIC SEGMENTATION
===================================================

AGENT ROLE: Discourse Analysis Specialist
DEPENDS ON: Phase 2 (transcript with sentences)
DELIVERS TO: Phase 8 (argument analysis), Phase 9 (output assembly)
            Also feeds into the Multimodal Fusion Layer for cross-modal analysis.
ESTIMATED TIME: 30 min (agent code) + 30 min (user data) + 4-6 hrs (M1 training)

OBJECTIVE:
Build src/content/regime_detector.py that segments the speech transcript
into topical "regimes" — contiguous blocks with consistent topic and
communicative intent (Introduction, Main Argument, Anecdote, etc.).

This is one of the most IMPORTANT components because the Multimodal
Fusion Layer uses regime boundaries to evaluate whether the speaker
adapted their voice and body language at each transition.

═══════════════════════════════════════════════════════════════
APPROACH: EMBEDDING DISTANCE + BOUNDARY CLASSIFIER
═══════════════════════════════════════════════════════════════

The algorithm has two stages:

STAGE 1 — Unsupervised boundary detection (no training needed):
1. Compute sentence embeddings using all-MiniLM-L6-v2 (pretrained)
2. For each sentence position, compute cosine distance between
   the left context centroid and right context centroid
3. Peaks in this distance signal = candidate boundaries
4. Apply peak detection with minimum prominence threshold

STAGE 2 — Regime type classification (needs training OR LLM):
Once boundaries are detected, classify each segment into a regime type.
Two options:
  A. Fine-tuned classifier (MiniLM + classification head) — needs labeled data
  B. Zero-shot LLM (Claude API) — no training, but costs per call

RECOMMENDATION: Use Option A for boundary detection (unsupervised, works well)
and Option B for regime labeling (zero-shot with Claude API, since labeling
regime types is a high-level semantic task that small classifiers struggle with).

═══════════════════════════════════════════════════════════════
TASK 1: Build the boundary detector (no training)
═══════════════════════════════════════════════════════════════

```python
class RegimeBoundaryDetector:
    """Detects topic transition points in a speech transcript.

    Algorithm (inspired by TextTiling, Hearst 1997):
    1. Embed each sentence with MiniLM-L6-v2
    2. Sliding window: compute left/right context centroids
    3. Cosine distance between centroids = topic shift signal
    4. Peaks in distance signal = boundaries
    """

    def __init__(self, model_name, half_window=5, min_prominence=0.15):
        self.encoder = SentenceTransformer(model_name)
        self.half_window = half_window      # Sentences each side
        self.min_prominence = min_prominence

    def detect_boundaries(self, sentences: list[str]) -> list[int]:
        """Returns list of sentence indices that are boundaries."""
        embeddings = self.encoder.encode(sentences)  # (N, 384)

        # Compute distance signal
        distances = []
        w = self.half_window
        for i in range(w, len(sentences) - w):
            left_centroid = embeddings[i-w:i].mean(axis=0)
            right_centroid = embeddings[i:i+w].mean(axis=0)
            dist = 1 - cosine_similarity(left_centroid, right_centroid)
            distances.append(dist)

        # Peak detection
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(
            distances,
            prominence=self.min_prominence,
            distance=2 * w  # Minimum distance between boundaries
        )

        # Convert back to sentence indices
        boundaries = [p + w for p in peaks]
        return boundaries
```

FUNCTION SIGNATURES:
  detect_boundaries(sentences) → list[int]  # Boundary sentence indices
  segment_transcript(sentences, boundaries) → list[Segment]
    Each Segment: {"start_sentence", "end_sentence", "text",
                   "start_sec", "end_sec", "sentence_count"}

═══════════════════════════════════════════════════════════════
TASK 2: Build regime type labeler (Claude API or classifier)
═══════════════════════════════════════════════════════════════

Option A — Claude API zero-shot (recommended for v1):
```python
def label_regime_with_llm(segment_text: str, context: str) -> dict:
    """Use Claude API to classify the regime type.

    Prompt template:
    "You are analyzing a speech transcript. Classify the following
     segment into ONE of these categories:
     Introduction, Main Argument, Anecdote, Data Presentation,
     Counterargument, Call to Action, Q&A, Conclusion, Transition.

     Previous segment summary: {context}
     Current segment: {segment_text}

     Respond with ONLY the category name."
    """
```

Option B — Fine-tuned classifier (if no API key):
Train a MiniLM-L6 + linear head on labeled speech segments.
Training data: manually label 50-100 segments from your 10 test speeches.
This takes ~1 hour of manual labeling + 2-3 hours of training on M1.

═══════════════════════════════════════════════════════════════
TASK 3: Build training script (for Option B)
═══════════════════════════════════════════════════════════════

Build training/modality3/train_regime_detector.py:
- Load sentence-transformers/all-MiniLM-L6-v2 as base encoder
- Add classification head: 384 → 128 → 9 classes
- Fine-tune with LoRA (rank=8) on labeled segments
- AdamW lr=2e-5, 20 epochs, early stopping
- Evaluation: WindowDiff (WD) for boundary detection,
  Accuracy for regime type classification

═══════════════════════════════════════════════════════════════
BRIDGE AGENT INTERVENTION
═══════════════════════════════════════════════════════════════

🔔 USER INPUT REQUIRED — Phase 5: Regime Detection

WHAT IS HAPPENING:
The boundary detection works unsupervised (no data needed).
The regime TYPE labeling needs either:
  A. An Anthropic API key (for Claude API zero-shot) — EASIEST
  B. Manually labeled speech segments — FALLBACK

WHAT YOU NEED TO DO:

Option A (recommended — 2 minutes):
  1. Go to: https://console.anthropic.com/settings/keys
  2. Create an API key
  3. Set it in your terminal:
     export ANTHROPIC_API_KEY="sk-ant-your-key-here"
  4. Add to ~/.zshrc so it persists

Option B (if no API key — 1 hour):
  1. Run the transcript processor on your 10 speech files
  2. The agent generates a labeling template with detected segments
  3. For each segment (~50-100 total), type the regime type:
     Introduction, Main Argument, Anecdote, Data, Conclusion, etc.
  4. Save at: data/regime_labels.json

WHAT HAPPENS NEXT:
  Option A: regime types are classified on-the-fly using Claude Sonnet.
            No training needed. Cost: ~$0.01 per speech (very cheap).
  Option B: The regime classifier trains on M1 for 4-6 hours.

COMPLETION: ✓ when detect_boundaries() finds reasonable segment points
            and label_regime() assigns correct types.