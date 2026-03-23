PHASE 3 — GRAMMAR CHECKING
==============================

AGENT ROLE: Grammar Specialist
DEPENDS ON: Phase 2 (clean transcript with sentences)
DELIVERS TO: Phase 9 (output assembly)
ESTIMATED TIME: 20 minutes (agent), 0 minutes (user)

OBJECTIVE:
Build src/content/grammar_checker.py using LanguageTool (pretrained, no training).
Detect grammatical errors and classify their severity.

TASKS:
1. Build GrammarChecker class:
   - __init__(language="en-US") → loads LanguageTool
   - check_sentence(sentence: str) → list of GrammarError
   - check_transcript(sentences: list) → list of errors with timestamps
   - score_grammar(errors, total_words) → grammar_score (0-100)

2. GrammarError dataclass:
```python
@dataclass
class GrammarError:
    sentence_idx: int
    start_sec: float
    end_sec: float
    original: str          # The erroneous text
    correction: str        # Suggested fix
    error_type: str        # "subject_verb", "tense", "article", "preposition", etc.
    severity: str          # "critical", "major", "minor"
    rule_id: str           # LanguageTool rule identifier
    message: str           # Human-readable explanation
```

3. CRITICAL: Filter for spoken language context.
   LanguageTool is designed for written text. Many "errors" it flags
   are perfectly normal in speech. The config lists categories to ignore:
   - Missing commas (speakers don't dictate punctuation)
   - Capitalization (Whisper output is lowercase)
   - Spelling of proper nouns (names, places)
   - Informal contractions ("gonna", "wanna" — already cleaned in Phase 2)

4. Severity classification:
   - CRITICAL: Subject-verb agreement ("the data shows" → "the data show"),
     tense inconsistency, double negatives
   - MAJOR: Article errors ("a information"), preposition errors, wrong word form
   - MINOR: Style issues, comma placement, passive voice overuse

5. Grammar score formula:
   score = max(0, 100 - (critical_count * 5 + major_count * 3 + minor_count * 1))
   Normalized per 100 words: errors_per_100 = total_errors / (total_words / 100)

NO USER INPUT REQUIRED. LanguageTool is pretrained (Java-based server).
Downloads ~200MB on first use, then runs locally. No API key needed.

COMPLETION: ✓ when grammar checker detects real errors and filters speech artifacts.