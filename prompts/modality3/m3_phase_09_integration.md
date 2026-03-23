PHASE 9 — OUTPUT ASSEMBLY & INTEGRATION
==========================================

AGENT ROLE: Systems Integrator
DEPENDS ON: ALL phases 2-8
DELIVERS TO: Phase 10 (testing), Multimodal Fusion Layer
ESTIMATED TIME: 25 minutes (agent), 0 minutes (user)

OBJECTIVE:
Build the output assembler and master pipeline that chains all
content analysis steps together.

═══════════════════════════════════════════════════════════════
TASK 1: Build src/content/output_assembler.py
═══════════════════════════════════════════════════════════════

Merge all sub-module outputs per regime segment:

```json
{
    "segment_id": 3,
    "timestamp_start": 45.2,
    "timestamp_end": 92.8,
    "regime_type": "Main Argument",
    "regime_confidence": 0.91,
    "transcript_excerpt": "The data clearly shows that...",
    "grammar": {
        "errors": [
            {"type": "subject_verb", "original": "data shows",
             "correction": "data show", "severity": "minor"}
        ],
        "error_count": 1,
        "errors_per_100_words": 0.8,
        "grammar_score": 92
    },
    "readability": {
        "flesch_kincaid_grade": 10.2,
        "dale_chall": 8.1,
        "gunning_fog": 12.3,
        "words_per_minute": 148,
        "assessment": "Appropriate for general audience"
    },
    "vocabulary": {
        "ttr": 0.62,
        "lexical_density": 0.48,
        "awl_coverage_pct": 7.2,
        "sophistication": "Moderate"
    },
    "sentiment": {
        "polarity": "Positive",
        "intensity": 0.72,
        "valence": 0.44
    },
    "tone": {
        "label": "Persuasive",
        "confidence": 0.78,
        "distribution": {"Persuasive": 0.78, "Assertive": 0.12, ...}
    },
    "argument": {
        "claims": ["Renewable energy reduces costs by 30%"],
        "evidence_quality": "Strong",
        "transitions": "Smooth signposting",
        "effectiveness_score": 82,
        "coaching_suggestion": "Add a specific example to strengthen the claim."
    }
}
```

═══════════════════════════════════════════════════════════════
TASK 2: Build src/content/pipeline.py
═══════════════════════════════════════════════════════════════

ContentAnalysisPipeline class:
- __init__(config_path) → load config, load all models/tools
- process(transcript: dict) → complete content analysis
  Chains: preprocess → grammar → readability → vocabulary →
          regime detection → sentiment → tone → argument → assemble
- Also accepts: process_from_video(video_path) → runs Whisper first
- Graceful degradation:
  No API key → skip argument analysis
  No tone model → skip tone (or use sentiment as proxy)
  No regime model → use equal-length segments as fallback
- CLI: python -m src.content.pipeline --transcript transcript.json

═══════════════════════════════════════════════════════════════
TASK 3: Generate speech-level summary
═══════════════════════════════════════════════════════════════

```python
def generate_summary(segments_analysis: list) -> dict:
    """Aggregate segment-level analysis into speech-level summary."""
    return {
        "total_segments": len(segments_analysis),
        "regime_sequence": [s["regime_type"] for s in segments_analysis],
        "grammar_score_overall": mean([s["grammar"]["grammar_score"]
                                       for s in segments_analysis]),
        "readability_level": median([s["readability"]["flesch_kincaid_grade"]
                                     for s in segments_analysis]),
        "vocabulary_diversity": mean([s["vocabulary"]["ttr"]
                                      for s in segments_analysis]),
        "dominant_sentiment": most_common([s["sentiment"]["polarity"]
                                           for s in segments_analysis]),
        "dominant_tone": most_common([s["tone"]["label"]
                                      for s in segments_analysis]),
        "tone_variety_score": count_unique_tones / 6,
        "argument_effectiveness_avg": mean([s["argument"]["effectiveness_score"]
                                            for s in segments_analysis
                                            if s["argument"]["effectiveness_score"]]),
        "regime_transition_count": len(segments_analysis) - 1,
        "content_structure_score": evaluate_structure(segments_analysis),
    }
```

NO USER INPUT REQUIRED.

COMPLETION: ✓ when pipeline produces valid JSON for a test transcript.