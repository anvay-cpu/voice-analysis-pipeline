PHASE 7 — TONE CLASSIFICATION
================================

AGENT ROLE: Tone Analysis Specialist
DEPENDS ON: Phase 2 (transcript segments)
DELIVERS TO: Phase 9 (output assembly)
ESTIMATED TIME: 20 min (agent code) + 1 hr (user labeling) + 3-4 hrs (M1 training)

OBJECTIVE:
Build src/content/tone_classifier.py that classifies the speaking TONE
of each segment. Tone is different from sentiment: a segment can be
Positive sentiment with Cautionary tone, or Neutral sentiment with
Humorous tone.

═══════════════════════════════════════════════════════════════
6 TONE CLASSES
═══════════════════════════════════════════════════════════════

1. ASSERTIVE — Confident claims, strong statements, declarations
   "This approach IS the right one. There is no doubt."

2. PERSUASIVE — Arguments, evidence, trying to convince
   "The data clearly shows that investing in education pays off."

3. INFORMATIVE — Facts, explanations, neutral delivery of information
   "The GDP grew by 3.2% in the last quarter."

4. INSPIRATIONAL — Motivational, uplifting, emotional appeals
   "Imagine a world where every child has access to clean water."

5. CAUTIONARY — Warnings, risks, negative consequences
   "If we don't act now, the consequences will be severe."

6. HUMOROUS — Jokes, wit, self-deprecation, intentional lightness
   "I know what you're thinking — another PowerPoint? I promise this one has GIFs."

═══════════════════════════════════════════════════════════════
APPROACH: TinyBERT FINE-TUNED ON CUSTOM DATA
═══════════════════════════════════════════════════════════════

Model: TinyBERT-4L (14.5M params) — small enough for M1
Base: huggingface.co/huawei-noah/TinyBERT_General_4L_312D
Head: 312 → 128 → 6 classes
Training time: 3-4 hours on M1

ALTERNATIVE (no training): Use Claude API for zero-shot tone
classification, same as regime detection. Costs ~$0.01 per speech.

═══════════════════════════════════════════════════════════════
TRAINING DATA: SEMI-AUTOMATED LABELING
═══════════════════════════════════════════════════════════════

Similar to the filler word auto-labeler, we use heuristics + review:

1. AUTO-LABEL with keyword/pattern heuristics:
   - Contains "imagine", "dream", "together we can" → Inspirational
   - Contains numbers, percentages, "data shows" → Informative
   - Contains "must", "should", "I believe" → Assertive
   - Contains "danger", "risk", "warning", "if we don't" → Cautionary
   - Contains rhetorical questions + short sentences → Persuasive
   - Marked with laughter markers or setup-punchline structure → Humorous

2. USER REVIEWS uncertain cases (~200 segments, ~1 hour)

3. Alternatively: USE CLAUDE API to auto-label with high accuracy
   (Claude is very good at tone classification), then train TinyBERT
   to reproduce Claude's labels. This is DISTILLATION — training a
   small model to mimic a large one.

═══════════════════════════════════════════════════════════════
BRIDGE AGENT INTERVENTION
═══════════════════════════════════════════════════════════════

🔔 USER INPUT REQUIRED — Phase 7: Tone Classification

WHAT IS HAPPENING:
The Tone Classifier needs labeled training data. There are 3 options.

OPTION A — Claude API distillation (10 minutes, recommended):
  If you have ANTHROPIC_API_KEY from Phase 5:
  1. The agent sends each segment to Claude with a classification prompt
  2. Claude labels all ~200 segments automatically (~$0.02 total)
  3. TinyBERT trains on Claude's labels (distillation)
  → No manual labeling needed. Just confirm you have the API key.

OPTION B — Manual labeling (1 hour):
  1. Agent generates data/tone_labeling_template.json
  2. For each of ~200 segments, assign one of 6 tone labels
  3. Save at data/tone_labels.json

OPTION C — Skip and use Claude API at inference time (0 minutes):
  Don't train TinyBERT. Use Claude API for every speech processed.
  Downside: requires API key at runtime, costs ~$0.01 per speech.
  Upside: most accurate and zero training.

RECOMMENDED: Option A if you have an API key. Option C if you want
to skip training entirely. Option B as fallback.

TRAINING (if Option A or B):
  python training/modality3/train_tone_classifier.py --device mps
  Time: 3-4 hours on M1
  Target: F1 ≥ 0.65

COMPLETION: ✓ when tone is classified for test segments with reasonable accuracy.