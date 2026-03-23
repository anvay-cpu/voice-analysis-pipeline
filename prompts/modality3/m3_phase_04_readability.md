PHASE 4 — READABILITY & VOCABULARY SCORING
=============================================

AGENT ROLE: Linguistics Analyst
DEPENDS ON: Phase 2 (transcript)
DELIVERS TO: Phase 9 (output assembly)
ESTIMATED TIME: 15 minutes (agent), 0 minutes (user)

OBJECTIVE:
Build src/content/readability_scorer.py and src/content/vocabulary_analyzer.py.
Pure computation — no models, no training.

═══════════════════════════════════════════════════════════════
TASK 1: Readability Scorer
═══════════════════════════════════════════════════════════════

Implement these standard readability formulas using the textstat library:

1. FLESCH-KINCAID GRADE LEVEL:
   FKGL = 0.39 × (words/sentences) + 11.8 × (syllables/words) - 15.59
   Interpretation: US school grade level. TED talk target: 8-12.

2. DALE-CHALL READABILITY:
   Percentage of "difficult" words (not in the Dale-Chall 3,000-word list).
   Score < 5.0 = easily understood. Score > 9.0 = college graduate level.

3. GUNNING FOG INDEX:
   0.4 × ((words/sentences) + 100 × (complex_words/words))
   "Complex" = 3+ syllables. Target: 10-14 for general audience.

4. COLEMAN-LIAU INDEX:
   0.0588 × L - 0.296 × S - 15.8
   where L = avg letters per 100 words, S = avg sentences per 100 words.

5. SPEAKING-SPECIFIC METRICS:
   - Words per minute (from total_words / duration_sec × 60)
     Target: 130-160 wpm for presentations
   - Average sentence length in words
     Target: 15-20 words (shorter in speech than writing)

Compute ALL metrics per segment AND for the full transcript.

═══════════════════════════════════════════════════════════════
TASK 2: Vocabulary Analyzer
═══════════════════════════════════════════════════════════════

1. TYPE-TOKEN RATIO (TTR):
   TTR = unique_words / total_words
   Compute on rolling 100-word windows to avoid length bias.
   TTR > 0.6 = diverse vocabulary. TTR < 0.4 = repetitive.

2. LEXICAL DENSITY:
   Ratio of content words (nouns, verbs, adjectives, adverbs) to total words.
   Use spaCy POS tagging: content = {NOUN, VERB, ADJ, ADV}
   Density > 0.55 = dense, information-rich.
   Density < 0.40 = filler-heavy, vague.

3. ACADEMIC WORD LIST (AWL) COVERAGE:
   Percentage of words from Coxhead's 570 Academic Word List families.
   Higher = more formal/sophisticated. TED talk target: 5-10%.

4. WORD FREQUENCY ANALYSIS:
   Compute the average word frequency rank using a word frequency list.
   Lower average rank = more common words (simpler speech).
   Flag rare/sophisticated words for the report.

5. REPETITION DETECTION:
   Find words/phrases repeated excessively (more than 3× expected frequency).
   Useful coaching feedback: "You used the word 'actually' 23 times."

Build a static resource file data/resources/academic_word_list.txt
containing the 570 AWL word families.

═══════════════════════════════════════════════════════════════
OUTPUT per segment:
```python
{
    "readability": {
        "flesch_kincaid_grade": 10.2,
        "dale_chall": 8.1,
        "gunning_fog": 12.3,
        "coleman_liau": 11.0,
        "words_per_minute": 148,
        "avg_sentence_length": 16.3,
        "assessment": "Appropriate for general audience"
    },
    "vocabulary": {
        "ttr": 0.62,
        "lexical_density": 0.48,
        "awl_coverage_pct": 7.2,
        "avg_word_frequency_rank": 2340,
        "top_repeated_words": [("actually", 23), ("important", 15)],
        "sophistication_level": "Moderate"
    }
}
```

NO USER INPUT REQUIRED. NO TRAINING REQUIRED. Pure computation.

COMPLETION: ✓ when readability and vocabulary metrics are computed correctly
            on a sample transcript.