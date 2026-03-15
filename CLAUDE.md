cat > ~/Desktop/Claude-assistant/CLAUDE.md << 'EOF'
# AI Public Speaking Assistant — Voice Pipeline (Modality 1)

## Project
Building a voice analysis pipeline that takes a speech video and produces
timestamped coaching data: transcription, filler words, disfluencies,
prosody, and vocal emotion.

## Agent Prompts
The prompts/ folder contains 11 phase-based agent prompts (phase_00 through phase_10).
Execute them IN ORDER starting from Phase 1.
Read each prompt file fully before executing its tasks.

## Rules
- Write all source code to src/
- Write all training scripts to training/
- Write all tests to tests/
- Save trained models to models/
- Use configs/pipeline_config.yaml for all configuration
- Target hardware: Apple M1 Mac 16GB (MPS GPU)
- Python environment: conda env "voice-pipeline"
- When a phase needs user input, STOP and clearly explain what's needed
- Update PROGRESS.md after completing each phase

## Commands
- Dev: conda activate voice-pipeline
- Test: python -m pytest tests/ -v
- Run pipeline: python -m src.pipeline --video INPUT.mp4

## Tech Stack
- PyTorch (MPS backend), transformers, librosa, parselmouth, speechbrain
- Whisper small (pretrained), Silero VAD (pretrained)
- Models to train: Filler MLP, Wav2Vec2 disfluency, ECAPA-TDNN emotion
EOF