PHASE 3 — SPEECH-TO-TEXT AGENT (WHISPER)
==========================================

AGENT ROLE: Transcription Specialist
DEPENDS ON: Phase 2 (clean audio + VAD segments)
DELIVERS TO: Phase 5 (filler detection), Phase 7 (prosody), Phase 9 (content pipeline)
ESTIMATED TIME: 15 minutes (agent), 0 minutes (user)

OBJECTIVE:
Build src/transcriber.py that loads Whisper Small (pretrained, no training)
and produces word-level timestamped transcripts.

TASKS:
1. Build src/transcriber.py with:
   - load_whisper(model_id, device) → pipeline
   - transcribe(pipe, audio_path, language) → dict with word timestamps
   - filter_by_vad(transcript, speech_segments) → filtered transcript
     (removes any Whisper hallucinations outside VAD speech regions)
   - Format output as: {"text": str, "words": [{"word", "start", "end"}, ...]}

2. Key implementation details:
   - Use transformers pipeline with return_timestamps="word"
   - Set chunk_length_s=30, stride_length_s=5 for long audio
   - batch_size=1 on M1 (memory constraint)
   - Use float32 (MPS doesn't handle float16 well)
   - Filter Whisper output: discard any word whose timestamp falls
     outside a VAD speech segment (catches hallucinated text in silence)

3. Write tests/test_transcriber.py:
   - Test on a short known audio clip
   - Verify word timestamps are monotonically increasing
   - Verify VAD filtering removes phantom words

NO USER INPUT REQUIRED.
NO TRAINING REQUIRED. Download pretrained model on first use.

COMPLETION: ✓ when transcribe() returns correct word-level timestamps
            and VAD filtering works on test audio.
