PHASE 1 — ENVIRONMENT & DEPENDENCIES (MODALITY 3)
====================================================

AGENT ROLE: Environment Architect
DEPENDS ON: Modality 1 environment (conda env voice-pipeline exists)
DELIVERS TO: All subsequent Modality 3 phases
ESTIMATED TIME: 10 minutes (agent) + 5 minutes (user)

OBJECTIVE:
Extend the existing environment with NLP and text processing dependencies.
Create Modality 3 directory structure and config files.

═══════════════════════════════════════════════════════════════
TASK 1: Install Additional Dependencies
═══════════════════════════════════════════════════════════════

Add to requirements.txt and install:

```
# NLP & Text Processing (Modality 3)
language-tool-python>=2.7.1    # Grammar checking (wraps LanguageTool)
nltk>=3.8.1                    # Tokenization, readability
textstat>=0.7.3                # Readability formulas
sentence-transformers>=2.3.0   # Sentence embeddings for regime detection
spacy>=3.7.0                   # Sentence splitting, POS tagging

# Already installed from Modality 1:
# torch, transformers, numpy, scipy, tqdm, pyyaml
```

Install commands:
```bash
conda activate voice-pipeline
pip install language-tool-python nltk textstat sentence-transformers spacy
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger'); nltk.download('cmudict')"
```

NOTE: language-tool-python downloads a Java-based LanguageTool server
(~200MB) on first use. Requires Java 8+ on the system.

═══════════════════════════════════════════════════════════════
TASK 2: Create Directory Structure
═══════════════════════════════════════════════════════════════

```
src/content/
├── __init__.py
├── transcript_processor.py   # Step 1: Clean and segment transcript
├── grammar_checker.py        # Step 2: Grammar error detection
├── readability_scorer.py     # Step 3: Readability metrics
├── vocabulary_analyzer.py    # Step 4: Vocabulary sophistication
├── regime_detector.py        # Step 5: Topic segmentation
├── sentiment_analyzer.py     # Step 6: Per-segment sentiment
├── tone_classifier.py        # Step 7: Speaking tone classification
├── argument_analyzer.py      # Step 8: Argument structure (Claude API)
├── output_assembler.py       # Step 9: JSON output merger
└── pipeline.py               # Master orchestrator

training/modality3/
├── train_regime_detector.py
├── train_tone_classifier.py
├── create_tone_dataset.py
└── configs/
    ├── regime.yaml
    └── tone.yaml

models/
├── regime_detector/
└── tone_classifier/

tests/modality3/
├── test_grammar.py
├── test_readability.py
├── test_regime.py
├── test_sentiment.py
├── test_tone.py
├── test_argument.py
└── test_integration.py
```

═══════════════════════════════════════════════════════════════
TASK 3: Create Modality 3 Config
═══════════════════════════════════════════════════════════════

Write configs/content_pipeline_config.yaml:

```yaml
content_pipeline:
  name: "content_analysis_v1"
  version: "1.0.0"
  device: "mps"

transcript:
  source: "modality1"          # Gets transcript from voice pipeline
  min_words_per_segment: 10
  sentence_splitter: "spacy"   # "spacy" or "nltk"

grammar:
  engine: "languagetool"       # "languagetool" or "t5-small"
  language: "en-US"
  # Spoken language tolerance: don't flag these in speech context
  ignore_categories:
    - "COMMA_PARENTHESIS_WHITESPACE"
    - "WHITESPACE_RULE"
    - "UPPERCASE_SENTENCE_START"    # Speech has no capitalization
    - "MORFOLOGIK_RULE_EN_US"       # Proper nouns from speech
  severity_weights:
    critical: 1.0               # Subject-verb agreement, tense
    major: 0.7                  # Article errors, preposition
    minor: 0.3                  # Style, comma usage

readability:
  target_audience: "general"    # "general" (FKGL 8-12), "academic" (12+), "simple" (6-8)
  metrics:
    - flesch_kincaid_grade
    - dale_chall
    - gunning_fog
    - coleman_liau

vocabulary:
  awl_path: "data/resources/academic_word_list.txt"
  min_ttr_window: 100          # Words per window for TTR

regime:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  fine_tuned_path: "models/regime_detector/best_model.pt"
  half_window: 5               # Sentences each side for boundary detection
  min_prominence: 0.15         # Peak prominence for boundary detection
  min_segment_sentences: 3     # Minimum sentences per regime
  types:
    - Introduction
    - Main Argument
    - Anecdote
    - Data Presentation
    - Counterargument
    - Call to Action
    - Q&A
    - Conclusion
    - Transition

sentiment:
  model_name: "cardiffnlp/twitter-roberta-base-sentiment-latest"
  # Maps to: Negative, Neutral, Positive

tone:
  model_path: "models/tone_classifier/best_model.pt"
  classes:
    - Assertive
    - Persuasive
    - Informative
    - Inspirational
    - Cautionary
    - Humorous

argument:
  engine: "claude_api"         # "claude_api" or "local_llm"
  model: "claude-sonnet-4-20250514"
  max_tokens: 1000
  # Fallback if no API key: skip argument analysis

output:
  output_dir: "data/outputs"
```

═══════════════════════════════════════════════════════════════
BRIDGE AGENT INTERVENTION
═══════════════════════════════════════════════════════════════

🔔 USER INPUT REQUIRED — Phase 1: Environment

WHAT IS HAPPENING:
The agent has created all directories and config files.
You need to install NLP packages and ensure Java is available.

WHAT YOU NEED TO DO:
```bash
conda activate voice-pipeline

# Install NLP packages
pip install language-tool-python nltk textstat sentence-transformers spacy
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger'); nltk.download('cmudict')"

# Verify Java (needed for LanguageTool)
java -version
# If "command not found": brew install openjdk@17

# Verify everything works
python -c "
import language_tool_python
import nltk
import textstat
import spacy
from sentence_transformers import SentenceTransformer
print('All Modality 3 dependencies OK')
"
```

NOTE: The first import of language_tool_python downloads ~200MB.
This is a one-time download.

WHAT HAPPENS NEXT:
Phases 2, 3, 4 run with no user input (all pretrained/code-only).

COMPLETION: ✓ when all imports succeed and Java is available.
