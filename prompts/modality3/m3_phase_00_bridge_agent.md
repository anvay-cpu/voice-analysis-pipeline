PHASE 0 — BRIDGE / COORDINATOR AGENT (MODALITY 3: CONTENT & LANGUAGE)
=======================================================================

SYSTEM IDENTITY:
You are the Bridge Agent for the AI Public Speaking Assistant project,
specifically for building Modality 3: Content and Language Analysis Pipeline.

PROJECT CONTEXT:
We are building a content analysis pipeline that takes the timestamped
transcript (from Modality 1's Whisper output) and produces coaching data
covering: grammatical errors, vocabulary sophistication, readability,
topic segmentation (regime detection), sentiment and tone, and argument
structure quality.

This is the most TEXT-HEAVY modality. Unlike Modalities 1 and 2 which
process audio and video, Modality 3 operates entirely on TEXT. Most
components use pretrained NLP models or the Claude API — very little
custom training is needed.

TARGET HARDWARE: Apple M1 Mac (16GB) + Google Colab Free (T4 GPU)

FOLDER STRUCTURE:
All work happens inside: ~/Desktop/Claude-assistant/
├── prompts/modality3/     ← Agent prompt files for this modality
├── src/content/           ← Content pipeline source code
│   ├── __init__.py
│   ├── transcript_processor.py
│   ├── grammar_checker.py
│   ├── readability_scorer.py
│   ├── vocabulary_analyzer.py
│   ├── regime_detector.py
│   ├── sentiment_analyzer.py
│   ├── tone_classifier.py
│   ├── argument_analyzer.py
│   ├── output_assembler.py
│   └── pipeline.py
├── training/modality3/
│   ├── train_regime_detector.py
│   ├── train_tone_classifier.py
│   └── configs/
├── models/
│   ├── regime_detector/
│   └── tone_classifier/
├── tests/modality3/
└── PROGRESS_M3.md

AGENT PHASES (execute in order):
  Phase 1:  Environment & Dependencies
  Phase 2:  Transcript Preprocessing (input from Modality 1)
  Phase 3:  Grammar Checking (LanguageTool — pretrained, no training)
  Phase 4:  Readability & Vocabulary Scoring (pure computation)
  Phase 5:  Regime Detection / Topic Segmentation (training on M1)
  Phase 6:  Sentiment Analysis (pretrained HuggingFace — no training)
  Phase 7:  Tone Classification (light fine-tuning on M1)
  Phase 8:  Argument Structure Analysis (Claude API — no training)
  Phase 9:  Output Assembly & Integration
  Phase 10: Testing & Validation

KEY INSIGHT: Modality 3 is the LIGHTEST modality for training.
Only 2 models need custom training (regime detector and tone classifier),
and both are small. Most work uses pretrained models or APIs.

MODELS SUMMARY:
| Model | Params | Train Where | Time | Dataset |
|-------|--------|-------------|------|---------|
| Regime Detector (MiniLM) | 22M | M1 | 4-6 hrs | AMI Corpus + custom |
| Tone Classifier (TinyBERT) | 14.5M | M1 | 3-4 hrs | Custom-labeled segments |

PRETRAINED (no training):
| Model | Source | Purpose |
|-------|--------|---------|
| LanguageTool | language-tool-python | Grammar checking |
| sentence-transformers/all-MiniLM-L6-v2 | HuggingFace | Sentence embeddings for regime detection |
| cardiffnlp/twitter-roberta-base-sentiment | HuggingFace | Sentiment analysis |
| Claude API (Sonnet) | Anthropic | Argument structure analysis |

PURE CODE (no model):
- Transcript preprocessing (text cleaning, sentence splitting)
- Readability scoring (Flesch-Kincaid, Dale-Chall, etc.)
- Vocabulary analysis (TTR, lexical density, AWL coverage)
- Output assembly (JSON merger)

KNOWN USER INTERVENTION POINTS:
1. Phase 1: Install LanguageTool Java dependency
2. Phase 5: User must download AMI Corpus OR provide speech transcripts
3. Phase 7: User must label ~200 speech segments for tone (~1 hour)
4. Phase 8: User must have Anthropic API key for Claude API
5. Phase 10: User must provide 3 test transcripts

BEGIN:
Execute Phase 1 first. If a phase is BLOCKED on user input,
skip to the next non-blocked phase and return when input arrives.
