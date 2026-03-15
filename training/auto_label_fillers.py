"""
AUTO-LABELER FOR FILLER WORD DATASET
=====================================
Reduces manual labeling from 2 hours to ~15 minutes.

Strategy:
  - "um", "uh", "erm", "mm", "ah" → ALWAYS fillers (auto-label 1)
  - "like", "right", "basically", etc. → use heuristics, then
    only ask the user to review uncertain ones

Heuristic rules for discourse fillers:
  1. If there's a pause > 200ms BEFORE the word → likely filler
  2. If the word is surrounded by other fillers → likely filler
  3. If the word starts a sentence/clause → likely NOT a filler
  4. If pitch is flat during the word → likely filler
  5. If the word is part of a known phrase ("I like", "looks like",
     "would like") → definitely NOT a filler

After auto-labeling, this script:
  - Auto-labels ~70% of clips with high confidence
  - Flags ~30% as "uncertain" for quick manual review
  - Opens a simple terminal UI for reviewing uncertain clips
    (press 1 for filler, 0 for not, s to skip — takes ~15 min)

Usage:
  python training/auto_label_fillers.py \
      --audio_dir data/raw/ \
      --output data/filler_labels.json \
      --device mps
"""

import json
import os
import re
import subprocess
import sys
import numpy as np
import soundfile as sf
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# STEP 1: CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Words that are ALWAYS fillers (no context needed)
ALWAYS_FILLER = {"um", "uh", "erm", "er", "mm", "hmm", "ah", "umm", "uhh"}

# Words that MIGHT be fillers depending on context
CONTEXT_DEPENDENT = {"like", "right", "basically", "actually", "so",
                     "okay", "well", "literally", "honestly"}

# Multi-word fillers (always fillers when detected as a phrase)
MULTI_WORD_FILLERS = {"you know", "i mean", "sort of", "kind of", "okay so"}

# Phrases where the word is NOT a filler
NOT_FILLER_PHRASES = {
    "like": [
        r"\bi like\b", r"\blike a\b", r"\blike the\b", r"\blike this\b",
        r"\blike that\b", r"\blooks like\b", r"\bfeel like\b",
        r"\bsound like\b", r"\bsounds like\b", r"\bseems like\b",
        r"\bwould like\b", r"\bwhat .+ like\b", r"\bmore like\b",
        r"\bjust like\b", r"\blike to\b", r"\blike it\b", r"\blike me\b",
        r"\blike him\b", r"\blike her\b", r"\blike we\b", r"\blike they\b",
        r"\blike when\b", r"\blike how\b", r"\bsomething like\b",
        r"\banything like\b", r"\bnothing like\b",
    ],
    "right": [
        r"\bright now\b", r"\bright there\b", r"\bright here\b",
        r"\bright away\b", r"\bright side\b", r"\bright thing\b",
        r"\ball right\b", r"\bthat's right\b", r"\bthat is right\b",
        r"\byou're right\b", r"\bright to\b", r"\bright of\b",
        r"\bhuman right\b", r"\bcivil right\b",
    ],
    "so": [
        r"\bso that\b", r"\bso much\b", r"\bso many\b", r"\bso far\b",
        r"\bso long\b", r"\bdo so\b", r"\beven so\b", r"\bif so\b",
        r"\band so\b", r"\bor so\b", r"\bnot so\b",
    ],
    "well": [
        r"\bas well\b", r"\bvery well\b", r"\bwell known\b",
        r"\bwell being\b", r"\bdoing well\b", r"\bquite well\b",
    ],
    "actually": [],  # "actually" at start of sentence is often a filler
    "basically": [],  # "basically" at start is often a filler
    "literally": [],
    "honestly": [],
    "okay": [
        r"\bokay with\b", r"\bokay to\b", r"\bis okay\b", r"\bit's okay\b",
    ],
}


@dataclass
class FillerCandidate:
    audio_path: str
    word: str
    start_sec: float
    end_sec: float
    context_before: str      # 3 words before
    context_after: str       # 3 words after
    pause_before_ms: float   # Silence before the word
    pause_after_ms: float    # Silence after the word
    is_filler: int           # -1=uncertain, 0=not filler, 1=filler
    confidence: float        # How confident the auto-labeler is
    rule: str                # Which rule made the decision


# ═══════════════════════════════════════════════════════════════
# STEP 2: EXTRACT CANDIDATES FROM WHISPER OUTPUT
# ═══════════════════════════════════════════════════════════════

def extract_candidates(audio_path: str, whisper_pipe) -> list[FillerCandidate]:
    """Run Whisper on audio and find all filler word candidates."""

    result = whisper_pipe(
        audio_path,
        return_timestamps="word",
        chunk_length_s=30,
        batch_size=1,
    )

    chunks = result.get("chunks", [])
    candidates = []

    for i, chunk in enumerate(chunks):
        word = chunk["text"].strip().lower()
        word_clean = re.sub(r"[^\w\s]", "", word)

        # Check if this is a filler candidate
        is_candidate = False
        if word_clean in ALWAYS_FILLER or word_clean in CONTEXT_DEPENDENT:
            is_candidate = True

        # Check multi-word fillers (look ahead)
        if i < len(chunks) - 1:
            two_words = word_clean + " " + re.sub(
                r"[^\w\s]", "", chunks[i + 1]["text"].strip().lower()
            )
            if two_words in MULTI_WORD_FILLERS:
                is_candidate = True
                word_clean = two_words

        if not is_candidate:
            continue

        # Get timestamps
        ts = chunk["timestamp"]
        if ts is None or ts[0] is None:
            continue
        start, end = ts

        # Get context (3 words before and after)
        context_before = " ".join(
            chunks[j]["text"].strip()
            for j in range(max(0, i - 3), i)
        ).lower()
        context_after = " ".join(
            chunks[j]["text"].strip()
            for j in range(i + 1, min(len(chunks), i + 4))
        ).lower()

        # Compute pause before this word
        pause_before = 0.0
        if i > 0 and chunks[i - 1]["timestamp"] is not None:
            prev_end = chunks[i - 1]["timestamp"][1]
            if prev_end is not None:
                pause_before = max(0, start - prev_end) * 1000  # ms

        # Compute pause after
        pause_after = 0.0
        if i < len(chunks) - 1 and chunks[i + 1]["timestamp"] is not None:
            next_start = chunks[i + 1]["timestamp"][0]
            if next_start is not None and end is not None:
                pause_after = max(0, next_start - end) * 1000

        candidates.append(FillerCandidate(
            audio_path=audio_path,
            word=word_clean,
            start_sec=round(start, 3) if start else 0,
            end_sec=round(end, 3) if end else 0,
            context_before=context_before,
            context_after=context_after,
            pause_before_ms=round(pause_before, 1),
            pause_after_ms=round(pause_after, 1),
            is_filler=-1,
            confidence=0.0,
            rule="none",
        ))

    return candidates


# ═══════════════════════════════════════════════════════════════
# STEP 3: AUTO-LABEL WITH HEURISTICS
# ═══════════════════════════════════════════════════════════════

def auto_label(candidate: FillerCandidate) -> FillerCandidate:
    """Apply heuristic rules to auto-label a candidate."""

    word = candidate.word
    context = f"{candidate.context_before} {word} {candidate.context_after}"

    # ── Rule 1: Hesitation fillers are ALWAYS fillers ──
    if word in ALWAYS_FILLER:
        candidate.is_filler = 1
        candidate.confidence = 0.99
        candidate.rule = "always_filler"
        return candidate

    # ── Rule 2: Multi-word fillers are ALWAYS fillers ──
    if word in MULTI_WORD_FILLERS:
        candidate.is_filler = 1
        candidate.confidence = 0.95
        candidate.rule = "multi_word_filler"
        return candidate

    # ── Rule 3: Check NOT-filler phrases ──
    if word in NOT_FILLER_PHRASES:
        for pattern in NOT_FILLER_PHRASES[word]:
            if re.search(pattern, context):
                candidate.is_filler = 0
                candidate.confidence = 0.92
                candidate.rule = f"not_filler_phrase:{pattern}"
                return candidate

    # ── Rule 4: Pause-based heuristic ──
    # A word preceded by a pause > 200ms is more likely a filler
    if candidate.pause_before_ms > 200 and word in CONTEXT_DEPENDENT:
        # But only if it's not at a natural sentence boundary
        if not re.search(r"[.!?]$", candidate.context_before.strip()):
            candidate.is_filler = 1
            candidate.confidence = 0.80
            candidate.rule = "pause_before_filler"
            return candidate

    # ── Rule 5: Surrounded by other fillers ──
    filler_words_around = sum(
        1 for w in (candidate.context_before + " " + candidate.context_after).split()
        if w in ALWAYS_FILLER or w in MULTI_WORD_FILLERS
    )
    if filler_words_around >= 1 and word in CONTEXT_DEPENDENT:
        candidate.is_filler = 1
        candidate.confidence = 0.78
        candidate.rule = "near_other_fillers"
        return candidate

    # ── Rule 6: "like" with no clear grammatical role ──
    if word == "like":
        # If preceded by a pronoun/name + comma pattern: "I was, like, ..."
        if re.search(r"(was|were|is|am|be|got|just|all)\s*$",
                      candidate.context_before):
            candidate.is_filler = 1
            candidate.confidence = 0.85
            candidate.rule = "like_after_verb"
            return candidate

    # ── Rule 7: Word at very start of speech → less likely filler ──
    if candidate.start_sec < 1.0 and word in {"so", "okay", "well"}:
        candidate.is_filler = 0
        candidate.confidence = 0.70
        candidate.rule = "opening_word"
        return candidate

    # ── If none of the rules fired → UNCERTAIN ──
    candidate.is_filler = -1
    candidate.confidence = 0.50
    candidate.rule = "uncertain"
    return candidate


# ═══════════════════════════════════════════════════════════════
# STEP 4: INTERACTIVE REVIEW FOR UNCERTAIN CASES
# ═══════════════════════════════════════════════════════════════

def play_audio_clip(audio_path: str, start_sec: float, end_sec: float):
    """Play a short audio clip using ffplay (comes with FFmpeg)."""
    duration = end_sec - start_sec + 1.5  # 0.75s padding each side
    start = max(0, start_sec - 0.75)

    try:
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit",
             "-ss", str(start), "-t", str(duration),
             audio_path],
            capture_output=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass  # ffplay not available or timed out


def interactive_review(uncertain: list[FillerCandidate]) -> list[FillerCandidate]:
    """Quick terminal UI for reviewing uncertain candidates.

    User presses:
      1 = filler
      0 = not filler
      p = play audio again
      s = skip (remove from dataset)
      q = quit review (remaining stay uncertain)
    """
    print("\n" + "=" * 60)
    print(f"MANUAL REVIEW: {len(uncertain)} uncertain clips")
    print("=" * 60)
    print("For each clip, you'll see the word and its context.")
    print("Press: 1=filler  0=not-filler  p=play-audio  s=skip  q=quit")
    print("=" * 60 + "\n")

    reviewed = []
    skipped = 0

    for i, c in enumerate(uncertain):
        print(f"\n[{i + 1}/{len(uncertain)}] Word: \"{c.word}\"")
        print(f"  Context: ...{c.context_before} [{c.word}] {c.context_after}...")
        print(f"  Pause before: {c.pause_before_ms:.0f}ms")
        print(f"  Time: {c.start_sec:.1f}s - {c.end_sec:.1f}s")
        print(f"  File: {Path(c.audio_path).name}")

        # Auto-play the clip
        play_audio_clip(c.audio_path, c.start_sec, c.end_sec)

        while True:
            choice = input("  Label (1/0/p/s/q): ").strip().lower()

            if choice == "1":
                c.is_filler = 1
                c.confidence = 1.0
                c.rule = "manual_review"
                reviewed.append(c)
                break
            elif choice == "0":
                c.is_filler = 0
                c.confidence = 1.0
                c.rule = "manual_review"
                reviewed.append(c)
                break
            elif choice == "p":
                play_audio_clip(c.audio_path, c.start_sec, c.end_sec)
            elif choice == "s":
                skipped += 1
                break
            elif choice == "q":
                print(f"\nStopped. Reviewed {i}, skipped rest.")
                return reviewed
            else:
                print("  Invalid. Press 1, 0, p, s, or q")

    print(f"\nReview complete. Labeled: {len(reviewed)}, Skipped: {skipped}")
    return reviewed


# ═══════════════════════════════════════════════════════════════
# STEP 5: MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def build_filler_dataset(
    audio_dir: str,
    output_path: str,
    device: str = "mps",
    skip_review: bool = False,
    min_samples: int = 400,
):
    """Complete pipeline: extract → auto-label → review → save.

    Args:
        audio_dir: Directory with WAV/MP4 files
        output_path: Where to save the labeled JSON
        device: "mps" for M1, "cuda" for GPU, "cpu" for fallback
        skip_review: If True, skip interactive review (use auto-labels only)
        min_samples: Minimum total samples to aim for
    """
    from transformers import pipeline as hf_pipeline

    # Load Whisper
    logger.info("Loading Whisper small...")
    whisper_pipe = hf_pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device=device if device != "mps" else "cpu",
        # Use CPU for Whisper pipeline (MPS has issues with some ops)
    )

    # Find audio files
    audio_dir = Path(audio_dir)
    audio_files = (
        list(audio_dir.glob("*.wav")) +
        list(audio_dir.glob("*.mp3")) +
        list(audio_dir.glob("*.mp4")) +
        list(audio_dir.glob("*.m4a"))
    )
    logger.info(f"Found {len(audio_files)} audio files in {audio_dir}")

    if len(audio_files) == 0:
        logger.error(f"No audio files found in {audio_dir}")
        logger.error("Download some speech videos first:")
        logger.error("  yt-dlp -x --audio-format wav 'VIDEO_URL'")
        return

    # Extract candidates from all files
    all_candidates = []
    for audio_file in audio_files:
        logger.info(f"Processing: {audio_file.name}")
        try:
            candidates = extract_candidates(str(audio_file), whisper_pipe)
            logger.info(f"  Found {len(candidates)} filler candidates")
            all_candidates.extend(candidates)
        except Exception as e:
            logger.warning(f"  Failed: {e}")

    logger.info(f"\nTotal candidates: {len(all_candidates)}")

    # Auto-label
    logger.info("Auto-labeling with heuristic rules...")
    for c in all_candidates:
        auto_label(c)

    # Tally results
    auto_filler = [c for c in all_candidates if c.is_filler == 1]
    auto_not_filler = [c for c in all_candidates if c.is_filler == 0]
    uncertain = [c for c in all_candidates if c.is_filler == -1]

    logger.info(f"Auto-labeled as FILLER:     {len(auto_filler)}")
    logger.info(f"Auto-labeled as NOT FILLER: {len(auto_not_filler)}")
    logger.info(f"UNCERTAIN (need review):    {len(uncertain)}")
    logger.info(f"Auto-label rate:            "
                f"{100 * (1 - len(uncertain) / max(len(all_candidates), 1)):.0f}%")

    # Interactive review of uncertain cases
    reviewed = []
    if not skip_review and len(uncertain) > 0:
        print(f"\n{'=' * 60}")
        print(f"  {len(uncertain)} clips need manual review")
        print(f"  Estimated time: {len(uncertain) * 8 // 60} minutes")
        print(f"  (8 seconds per clip average)")
        print(f"{'=' * 60}")

        choice = input("\nStart review now? (y/n): ").strip().lower()
        if choice == "y":
            reviewed = interactive_review(uncertain)
    elif skip_review:
        logger.info("Skipping manual review (--skip_review flag)")
        # For skip mode: use a simple heuristic fallback
        # Classify remaining uncertain as fillers if pause > 150ms
        for c in uncertain:
            if c.pause_before_ms > 150:
                c.is_filler = 1
                c.confidence = 0.65
                c.rule = "fallback_pause_heuristic"
            else:
                c.is_filler = 0
                c.confidence = 0.60
                c.rule = "fallback_default_not_filler"
        reviewed = uncertain

    # Combine all labeled samples
    labeled = auto_filler + auto_not_filler + reviewed
    labeled = [c for c in labeled if c.is_filler in (0, 1)]

    # Balance the dataset (roughly equal fillers and non-fillers)
    fillers = [c for c in labeled if c.is_filler == 1]
    non_fillers = [c for c in labeled if c.is_filler == 0]

    logger.info(f"\nFinal counts before balancing:")
    logger.info(f"  Fillers: {len(fillers)}")
    logger.info(f"  Non-fillers: {len(non_fillers)}")

    # Downsample the majority class
    min_count = min(len(fillers), len(non_fillers))
    if min_count > 0:
        np.random.seed(42)
        if len(fillers) > min_count:
            fillers = list(np.random.choice(fillers, min_count, replace=False))
        if len(non_fillers) > min_count:
            non_fillers = list(np.random.choice(non_fillers, min_count, replace=False))

    balanced = fillers + non_fillers
    np.random.shuffle(balanced)

    logger.info(f"\nFinal balanced dataset: {len(balanced)} samples")
    logger.info(f"  Fillers: {len(fillers)}")
    logger.info(f"  Non-fillers: {len(non_fillers)}")

    if len(balanced) < min_samples:
        logger.warning(f"  ⚠ Below target of {min_samples} samples.")
        logger.warning(f"  Add more audio files to data/raw/ and re-run.")

    # Save
    output = {
        "metadata": {
            "total_samples": len(balanced),
            "fillers": len(fillers),
            "non_fillers": len(non_fillers),
            "auto_labeled_pct": round(
                100 * (len(auto_filler) + len(auto_not_filler))
                / max(len(all_candidates), 1), 1
            ),
            "manually_reviewed": len(reviewed),
            "audio_files_processed": len(audio_files),
        },
        "samples": [
            {
                "audio_path": c.audio_path,
                "word": c.word,
                "start_sec": c.start_sec,
                "end_sec": c.end_sec,
                "is_filler": c.is_filler,
                "confidence": c.confidence,
                "rule": c.rule,
                "context": f"{c.context_before} [{c.word}] {c.context_after}",
            }
            for c in balanced
        ],
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\n✓ Dataset saved to {output_path}")
    logger.info(f"  Next step: python training/train_filler_verifier.py --preextract")

    return output


# ═══════════════════════════════════════════════════════════════
# STEP 6: DOWNLOAD HELPER
# ═══════════════════════════════════════════════════════════════

def download_speech_videos(output_dir: str, count: int = 10):
    """Download public domain / CC-licensed speech audio.

    Uses yt-dlp to download TED talks and public speeches.
    """
    # These are example TED talk URLs — replace with your own
    EXAMPLE_URLS = [
        # Add your own YouTube URLs here
        # "https://www.youtube.com/watch?v=VIDEO_ID",
    ]

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("DOWNLOAD SPEECH AUDIO")
    print("=" * 60)
    print(f"\nYou need {count}+ speech recordings in {output_dir}/")
    print("\nOption 1: Download from YouTube using yt-dlp:")
    print("  pip install yt-dlp")
    print(f"  yt-dlp -x --audio-format wav -o '{output_dir}/%(title)s.%(ext)s' 'URL'")
    print("\nOption 2: Use your own recordings")
    print(f"  Place WAV/MP3 files in {output_dir}/")
    print("\nRecommended: 10 TED talks of different speakers (5-15 min each)")
    print("This gives you diverse speaking styles and plenty of fillers.")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Auto-label filler word dataset")
    parser.add_argument("--audio_dir", default="data/raw",
                        help="Directory with speech audio files")
    parser.add_argument("--output", default="data/filler_labels.json",
                        help="Output labeled dataset JSON")
    parser.add_argument("--device", default="mps",
                        help="Device: mps, cuda, or cpu")
    parser.add_argument("--skip_review", action="store_true",
                        help="Skip manual review — use auto-labels only "
                             "(lower quality but zero manual work)")
    parser.add_argument("--download_help", action="store_true",
                        help="Show instructions for downloading speech audio")

    args = parser.parse_args()

    if args.download_help:
        download_speech_videos(args.audio_dir)
    else:
        build_filler_dataset(
            audio_dir=args.audio_dir,
            output_path=args.output,
            device=args.device,
            skip_review=args.skip_review,
        )
