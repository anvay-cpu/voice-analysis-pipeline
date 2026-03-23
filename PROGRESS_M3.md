# Modality 3: Content & Language Analysis — Progress

## Status: COMPLETE — 38/38 tests passed

---

## Phase 1: Environment & Dependencies
**Status:** DONE
- Installed: language-tool-python, nltk, textstat, sentence-transformers, spacy
- Downloaded: en_core_web_sm, NLTK punkt/punkt_tab/averaged_perceptron_tagger/cmudict
- Installed Java 17 (openjdk@17 via Homebrew) — required by LanguageTool
- LanguageTool server downloaded (~258MB, v6.8-SNAPSHOT)
- NOTE: Must set JAVA_HOME before running grammar checker:
  `export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home`

## Phase 2: Transcript Preprocessing
**Status:** DONE, TESTED (7/7 tests)
- File: `src/content/transcript_processor.py`
- TranscriptProcessor class: load_from_whisper, load_from_json, load_from_text
- Sentence splitting via spaCy, timestamp mapping, segment grouping
- ASR expansion (gonna→going to, etc.), hallucination removal
- Tested on all 3 scenarios (structured, rambling, technical)

## Phase 3: Grammar Checking
**Status:** DONE, TESTED (2/2 tests)
- File: `src/content/grammar_checker.py`
- GrammarChecker class using LanguageTool (pretrained, no training)
- Filters speech artifacts (capitalization, punctuation, proper nouns)
- Severity classification: critical/major/minor
- Scoring: weighted penalty formula (100 - 5×critical - 3×major - 1×minor)
- FIX APPLIED: LanguageTool Python uses snake_case attributes (rule_id, error_length)

## Phase 4: Readability & Vocabulary Scoring
**Status:** DONE, TESTED (7/7 tests)
- File: `src/content/readability_scorer.py` — 4 tests passed
  - Flesch-Kincaid, Dale-Chall, Gunning Fog, Coleman-Liau, WPM, avg sentence length
  - Structured: FKGL in range 4-18, Rambling: 2-14, Technical: 6-20
- File: `src/content/vocabulary_analyzer.py` — 3 tests passed
  - TTR (rolling window), lexical density (spaCy POS), AWL coverage, repetition detection
- AWL file: `data/resources/academic_word_list.txt` (606 words, generated from Brown corpus + Coxhead headwords)

## Phase 5: Regime Detection
**Status:** DONE, TESTED (via pipeline heavy tests)
- File: `src/content/regime_detector.py`
- Stage 1: Unsupervised boundary detection (MiniLM embeddings + cosine distance peaks)
- Stage 2: Regime labeling via Claude CLI wrapper (zero-shot)
- Heuristic fallback if Claude CLI unavailable
- 9 regime types: Introduction, Main Argument, Anecdote, Data Presentation, etc.
- Pipeline uses regime segments as primary unit for downstream analysis

## Phase 6: Sentiment Analysis
**Status:** DONE, TESTED (via pipeline heavy tests)
- File: `src/content/sentiment_analyzer.py`
- Uses pretrained cardiffnlp/twitter-roberta-base-sentiment-latest
- Per-sentence and per-segment analysis
- Outputs polarity, intensity, valence (for cross-modal coherence)

## Phase 7: Tone Classification
**Status:** DONE, TESTED (6/6 heuristic + pipeline heavy)
- File: `src/content/tone_classifier.py`
- Claude CLI zero-shot at inference time (no training)
- 6 classes: Assertive, Persuasive, Informative, Inspirational, Cautionary, Humorous
- Keyword heuristic fallback — all 4 tone detections pass

## Phase 8: Argument Analysis
**Status:** DONE, TESTED (2/2 fallback + pipeline heavy)
- File: `src/content/argument_analyzer.py`
- Uses Claude CLI wrapper (routes through Max subscription)
- Extracts: claims, evidence, warrants, transitions, fallacies, rhetorical devices
- Graceful fallback if CLI unavailable
- Rate limiting: 1s between API calls

## Phase 9: Output Assembly & Pipeline
**Status:** DONE, TESTED (5/5 assembler + 6/6 pipeline light)
- File: `src/content/output_assembler.py` — merges all sub-module outputs
- File: `src/content/pipeline.py` — master orchestrator
- ContentAnalysisPipeline with regime-based segmentation for downstream modules
- CLI: `python -m src.content.pipeline --transcript FILE --verbose`
- Speech-level summary: structure score, dominant tone, readability, vocabulary diversity

## Phase 10: Testing & Validation
**Status:** DONE — 38/38 tests passed
- File: `tests/modality3/test_phase10_validation.py`
- Test transcripts: `data/test_transcripts/test_a_structured.json`, `test_b_rambling.json`, `test_c_technical.json`
- **Light tests (32):** No Java/models — transcript, readability, vocabulary, assembler, tone heuristic, argument fallback, pipeline light
- **Heavy tests (6):** Grammar checker (2), Full pipeline (3 scenarios)
- Full pipeline processing: ~144s for structured speech (down from 270s after segment alignment fix)

---

## Test Results Summary

| Test Class | Tests | Status |
|---|---|---|
| TranscriptProcessor | 7 | PASS |
| ReadabilityScorer | 4 | PASS |
| VocabularyAnalyzer | 3 | PASS |
| OutputAssembler | 5 | PASS |
| ToneClassifierHeuristic | 6 | PASS |
| ArgumentAnalyzerFallback | 2 | PASS |
| PipelineLight | 6 | PASS |
| GrammarChecker | 2 | PASS |
| PipelineHeavy | 3 | PASS |
| **Total** | **38** | **38/38 PASS** |

---

## Known Requirements

1. **Java 17 PATH** — LanguageTool requires JAVA_HOME set to openjdk@17
   - `export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home`

2. **Claude CLI dependency** — Phases 5, 7, 8 use `claude -p` for LLM calls
   - Works via Max subscription, $0 extra cost
   - Heuristic fallbacks exist for all three if CLI unavailable

---

## API Setup (for Claude CLI calls)
- **Method:** `claude -p` pipe mode (NOT Anthropic SDK, NOT CLIProxyAPI)
- **Auth:** Max subscription (already authenticated via Claude Code)
- **Wrapper:** `src/utils/claude_api_wrapper.py`
- **Config:** `configs/api_config.json`
- **Cost:** $0.00 extra (uses subscription quota)

## Config Files
- `configs/content_pipeline_config.yaml` — all pipeline parameters
- `configs/api_config.json` — Claude API routing config
