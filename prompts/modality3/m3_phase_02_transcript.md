PHASE 2 — TRANSCRIPT PREPROCESSING
======================================

AGENT ROLE: Text Processing Specialist
DEPENDS ON: Phase 1 (environment) + Modality 1 Whisper output
DELIVERS TO: All phases 3-8
ESTIMATED TIME: 15 minutes (agent), 0 minutes (user)

OBJECTIVE:
Build src/content/transcript_processor.py that takes the raw Whisper
transcript (word-level timestamps) and produces clean, structured text
ready for NLP analysis.

TASKS:
1. Build TranscriptProcessor class:
   - load_from_whisper(whisper_output: dict) → structured transcript
   - load_from_json(json_path: str) → structured transcript
   - split_sentences(text, method="spacy") → list of sentences with timestamps
     Use spaCy en_core_web_sm for sentence boundary detection.
     Map each sentence back to start/end timestamps from word-level data.
   - group_into_segments(sentences, min_words=10) → segments
     Group very short sentences together until segment has ≥ min_words.
   - clean_for_grammar(text) → text
     Add capitalization at sentence starts (Whisper outputs lowercase).
     Fix common ASR artifacts ("gonna" → "going to", "wanna" → "want to").
     Preserve filler words (they're handled by Modality 1, not here).

2. Output structure:
```python
{
    "full_text": "Good morning everyone. I am delighted to be here...",
    "sentences": [
        {"text": "Good morning everyone.",
         "start_sec": 0.0, "end_sec": 1.34, "word_count": 3},
        {"text": "I am delighted to be here today.",
         "start_sec": 1.52, "end_sec": 3.21, "word_count": 7},
    ],
    "segments": [
        {"text": "Good morning everyone. I am delighted to be here today.",
         "start_sec": 0.0, "end_sec": 3.21, "word_count": 10,
         "sentence_indices": [0, 1]},
    ],
    "total_words": 1847,
    "total_sentences": 142,
    "duration_sec": 612.4
}
```

3. Handle edge cases:
   - Whisper hallucinations (repeated phrases) → detect and remove
   - Very long sentences (> 50 words with no period) → force split at commas
   - Empty segments → skip

NO USER INPUT REQUIRED. Pure text processing code.

COMPLETION: ✓ when transcript processor outputs correct sentences with timestamps.