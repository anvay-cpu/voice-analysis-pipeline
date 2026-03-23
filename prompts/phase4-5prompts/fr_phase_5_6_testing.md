PHASE 5.6 — END-TO-END TESTING
=================================

AGENT ROLE: QA Engineer
DEPENDS ON: Phase 5.5 (master pipeline)
DELIVERS TO: FINAL OUTPUT — the project is complete
RUNS ON: COLAB T4 for full pipeline runs (20-60 min per video)
         Local M1 for unit tests and report inspection

OBJECTIVE:
Run the complete system on 3 real speech videos and validate
every component produces correct, useful output.

═══════════════════════════════════════════════════════════════
COLAB SETUP (VS Code extension)
═══════════════════════════════════════════════════════════════

The user runs VS Code with Colab extension. For heavy processing:

1. Connect to Colab T4 runtime from VS Code
2. Mount Google Drive for data/model access
3. Upload or sync the project code
4. Run the master pipeline on test videos
5. Download reports to local machine

```python
# In Colab notebook:
from google.colab import drive
drive.mount('/content/drive')

# Copy project from Drive (or upload)
!cp -r "/content/drive/MyDrive/Claude-assistant" /content/
%cd /content/Claude-assistant

# Install dependencies
!pip install -q torch torchaudio torchvision transformers
!pip install -q mediapipe ultralytics opencv-python
!pip install -q librosa praat-parselmouth noisereduce soundfile
!pip install -q speechbrain sentence-transformers spacy
!pip install -q language-tool-python nltk textstat
!pip install -q matplotlib weasyprint anthropic
!python -m spacy download en_core_web_sm

# Run on test video
!python -m src.master_pipeline \
    --video data/raw/videos/test_good_speaker.mp4 \
    --output reports/test_good/
```

═══════════════════════════════════════════════════════════════
TEST SCENARIOS
═══════════════════════════════════════════════════════════════

Test A: GOOD SPEAKER (polished TED talk, 5-10 min)
  Expected scores: most dimensions 70-90/100
  Expected report: mostly positive feedback with minor tips
  Expected time: 3-5 minutes processing

Test B: NERVOUS SPEAKER (student presentation, 5-10 min)
  Expected scores: some dimensions 40-60/100
  Expected report: constructive feedback on nervousness signals,
  specific recovery moments identified, practice plan focused
  on confidence-building

Test C: MONOTONE SPEAKER (dry lecture, 5-10 min)
  Expected scores: Emotional Expressiveness low (<50),
  Content Structure potentially high (organized content)
  Expected report: feedback on vocal variety, gesture use,
  audience engagement

═══════════════════════════════════════════════════════════════
BRIDGE AGENT INTERVENTION
═══════════════════════════════════════════════════════════════

🔔 USER INPUT REQUIRED — Phase 5.6: Testing

WHAT YOU NEED TO DO:

Step 1: Provide 3 test videos (if not already from earlier testing):
  yt-dlp -f "bestvideo[height<=720]+bestaudio" --merge-output-format mp4 \
      -o "data/raw/videos/test_good.mp4" "GOOD_SPEAKER_URL"
  yt-dlp -f "bestvideo[height<=720]+bestaudio" --merge-output-format mp4 \
      -o "data/raw/videos/test_nervous.mp4" "NERVOUS_SPEAKER_URL"
  yt-dlp -f "bestvideo[height<=720]+bestaudio" --merge-output-format mp4 \
      -o "data/raw/videos/test_monotone.mp4" "MONOTONE_SPEAKER_URL"

Step 2: Connect VS Code to Colab T4 runtime

Step 3: Upload videos to Colab or Google Drive

Step 4: Run the master pipeline on each video

Step 5: Inspect the generated HTML reports

═══════════════════════════════════════════════════════════════
VALIDATION CHECKLIST
═══════════════════════════════════════════════════════════════

For each test video, verify:

MODALITY 1 (Voice):
  [ ] Whisper produced word-level timestamps
  [ ] Filler words detected with timestamps
  [ ] Prosody metrics in reasonable ranges
  [ ] Vocal emotion varies across speech

MODALITY 2 (Body):
  [ ] Person detected in all frames
  [ ] Pose keypoints extracted
  [ ] Gestures classified
  [ ] Gaze zones computed
  [ ] Facial emotion varies

MODALITY 3 (Content):
  [ ] Grammar errors detected (not false positives)
  [ ] Readability in expected range (FKGL 8-14)
  [ ] Regime boundaries at logical transition points
  [ ] Sentiment matches content

FUSION:
  [ ] Timeline aligned at 1-second resolution
  [ ] Regime transitions scored
  [ ] Recovery events detected (if disruptions exist)
  [ ] Emotion coherence computed

REPORT:
  [ ] Executive summary reads like a human coach
  [ ] Radar chart shows 6 scores
  [ ] Timeline heatmap renders correctly
  [ ] Per-segment coaching is specific with timestamps
  [ ] Practice plan has 3 actionable items
  [ ] HTML opens correctly in browser
  [ ] PDF generates (if weasyprint installed)
  [ ] Total report is 2000-3000 words, 8-12 pages

PERFORMANCE:
  [ ] Full pipeline completes in < 10 minutes on Colab T4
  [ ] No OOM errors
  [ ] All API calls succeed through proxy

═══════════════════════════════════════════════════════════════
FINAL OUTPUT
═══════════════════════════════════════════════════════════════

After all 3 tests pass, generate:
  docs/final_validation_report.md

Then print:

```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   AI PUBLIC SPEAKING ASSISTANT — BUILD COMPLETE              ║
║                                                              ║
║   All 3 modalities ✓                                         ║
║   Multimodal fusion ✓                                        ║
║   6-dimension scoring ✓                                      ║
║   Coaching report generation ✓                               ║
║   HTML/PDF report output ✓                                   ║
║                                                              ║
║   Usage:                                                     ║
║   python -m src.master_pipeline --video YOUR_SPEECH.mp4      ║
║                                                              ║
║   Report: reports/speech_report.html                         ║
║   Cost per speech: $0.00 (via Max subscription proxy)        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

THE PROJECT IS COMPLETE.