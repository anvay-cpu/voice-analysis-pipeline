#!/usr/bin/env python3
"""Phase 10 — Run voice pipeline on all 3 test videos with progress tracking.

Uses actual src/ module imports. Run with unbuffered output:
    /opt/anaconda3/envs/voice-pipeline/bin/python -u run_phase10.py
"""

import sys
import os
import time
import json
import traceback
from pathlib import Path

# Force unbuffered
os.environ["PYTHONUNBUFFERED"] = "1"

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_memory_mb():
    if HAS_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    return 0.0


def progress_bar(current, total, prefix="", width=40):
    pct = current / total if total > 0 else 1.0
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    print(f"  {prefix} [{bar}] {pct*100:5.1f}%", flush=True)


TEST_VIDEOS = [
    ("Good Speaker (TED talk)", "data/raw/test_good_speaker_5min.mp4"),
    ("Nervous Speaker (student)", "data/raw/test_nervous_speaker_5min.mp4"),
    ("Monotone Speaker (lecture)", "data/raw/test_monotone_speaker_5min.mp4"),
]


def load_all_models(config):
    """Load all models upfront and return them in a dict."""
    device = config.get("pipeline", {}).get("device", "cpu")
    models = {}
    model_steps = [
        ("Whisper (transcription, ~244M params)", "whisper"),
        ("Filler Verifier MLP (~200K params)", "filler"),
        ("Wav2Vec2 Disfluency (~95M params)", "disfluency"),
        ("ECAPA-TDNN Emotion (~15M params)", "emotion"),
    ]

    print("=" * 60, flush=True)
    print("  MODEL LOADING", flush=True)
    print("=" * 60, flush=True)
    total_start = time.time()

    for i, (name, key) in enumerate(model_steps):
        progress_bar(i, len(model_steps), prefix="Overall")
        log(f"[{i+1}/{len(model_steps)}] Loading {name}...")
        step_start = time.time()

        try:
            if key == "whisper":
                from src.transcriber import load_whisper
                whisper_cfg = config.get("whisper", {})
                pipe = load_whisper(
                    model_id=whisper_cfg.get("model_id", "openai/whisper-small"),
                    device=device,
                )
                models["whisper_pipe"] = pipe
                models["whisper_model"] = pipe.model
                models["whisper_processor"] = pipe.tokenizer

            elif key == "filler":
                filler_path = Path("models/filler/best_model.pt")
                if filler_path.exists():
                    from src.filler_verifier import load_filler_verifier
                    models["filler_verifier"] = load_filler_verifier(
                        str(filler_path), device=device)
                else:
                    log(f"  [SKIP] Filler model not found — using regex-only")

            elif key == "disfluency":
                model_path = config.get("disfluency", {}).get(
                    "model_path", "models/disfluency/best_model.pt")
                if Path(model_path).exists():
                    from src.disfluency_model import load_disfluency_model
                    model, processor = load_disfluency_model(model_path, device=device)
                    models["disfluency_model"] = model
                    models["disfluency_processor"] = processor
                else:
                    log(f"  [SKIP] Disfluency model not found")

            elif key == "emotion":
                model_path = config.get("emotion", {}).get(
                    "model_path", "models/vocal_emotion/best_model.pt")
                if Path(model_path).exists():
                    from src.vocal_emotion import load_emotion_model
                    models["emotion_classifier"] = load_emotion_model(
                        model_path, device=device)
                else:
                    log(f"  [SKIP] Emotion model not found")

            elapsed = time.time() - step_start
            log(f"  → OK ({elapsed:.1f}s) | Memory: {get_memory_mb():.0f} MB")

        except Exception as e:
            elapsed = time.time() - step_start
            log(f"  → FAILED ({elapsed:.1f}s): {e}")
            traceback.print_exc()

    progress_bar(len(model_steps), len(model_steps), prefix="Overall")

    total_elapsed = time.time() - total_start
    log(f"All models loaded in {total_elapsed:.1f}s | Memory: {get_memory_mb():.0f} MB")

    # Summary table
    print(flush=True)
    print("  ┌─────────────────────────────┬──────────┐", flush=True)
    print("  │ Model                       │ Status   │", flush=True)
    print("  ├─────────────────────────────┼──────────┤", flush=True)
    checks = [
        ("Whisper", "whisper_pipe" in models),
        ("Filler Verifier", "filler_verifier" in models),
        ("Disfluency (Wav2Vec2)", "disfluency_model" in models),
        ("Emotion (ECAPA-TDNN)", "emotion_classifier" in models),
    ]
    for name, loaded in checks:
        status = "✓ loaded" if loaded else "- skip"
        print(f"  │ {name:27s} │ {status:8s} │", flush=True)
    print("  └─────────────────────────────┴──────────┘", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)

    return models


def process_video(video_path, config, models, video_num, total_videos):
    """Process a single video using pre-loaded models."""
    import librosa
    from datetime import datetime

    device = config.get("pipeline", {}).get("device", "cpu")
    sr = config.get("audio", {}).get("sample_rate", 16000)
    video_label = Path(video_path).stem

    steps = ["preprocessing", "transcription", "features", "fillers",
             "disfluency", "prosody", "emotion", "assembly"]
    total_steps = len(steps)
    timings = {}

    print(flush=True)
    print("=" * 60, flush=True)
    print(f"  VIDEO {video_num+1}/{total_videos}: {video_label}", flush=True)
    print(f"  Input: {video_path}", flush=True)
    print("=" * 60, flush=True)

    total_start = time.time()

    # ── Step 1-3: Preprocessing ──
    log(f"Step 1/{total_steps}: preprocessing...")
    progress_bar(0, total_steps, prefix=f"Video {video_num+1}")
    step_start = time.time()

    from src.noise_reduction import preprocess_audio
    output_dir = config.get("output", {}).get("output_dir", "data/outputs")
    audio_dir = str(Path(output_dir).parent / "audio")
    preprocess_result = preprocess_audio(video_path, audio_dir, config)

    timings["preprocessing"] = round(time.time() - step_start, 2)
    clean_audio_path = preprocess_result["clean_audio_path"]
    speech_segments = preprocess_result["speech_segments"]
    duration_sec = preprocess_result["duration_sec"]

    log(f"  Duration: {duration_sec:.1f}s | SNR: {preprocess_result['snr_db']:.1f}dB | "
        f"Segments: {len(speech_segments)} | {timings['preprocessing']:.1f}s")

    audio, _ = librosa.load(clean_audio_path, sr=sr, mono=True)

    # ── Step 4: Transcription ──
    log(f"Step 2/{total_steps}: transcription...")
    progress_bar(1, total_steps, prefix=f"Video {video_num+1}")
    step_start = time.time()

    from src.transcriber import transcribe, filter_by_vad
    whisper_cfg = config.get("whisper", {})
    transcript = transcribe(
        models["whisper_pipe"], clean_audio_path,
        language=whisper_cfg.get("language", "en"),
        chunk_length_s=whisper_cfg.get("chunk_length_s", 30),
        batch_size=whisper_cfg.get("batch_size", 1),
    )
    transcript = filter_by_vad(transcript, speech_segments)

    timings["transcription"] = round(time.time() - step_start, 2)
    log(f"  Words: {len(transcript['words'])} | {timings['transcription']:.1f}s | "
        f"Memory: {get_memory_mb():.0f} MB")

    # ── Step 5: Features ──
    log(f"Step 3/{total_steps}: features...")
    progress_bar(2, total_steps, prefix=f"Video {video_num+1}")
    step_start = time.time()

    from src.features import AcousticFeatureExtractor
    extractor = AcousticFeatureExtractor(config)
    features_windowed = extractor.extract_windowed(audio, window_sec=2.0, hop_sec=1.0)

    timings["features"] = round(time.time() - step_start, 2)
    log(f"  Windows: {len(features_windowed)} | {timings['features']:.1f}s")

    # ── Step 6: Fillers ──
    log(f"Step 4/{total_steps}: fillers...")
    progress_bar(3, total_steps, prefix=f"Video {video_num+1}")
    step_start = time.time()

    from src.filler_detector import detect_fillers_from_transcript
    fillers = detect_fillers_from_transcript(transcript)

    if "filler_verifier" in models:
        from src.filler_verifier import verify_fillers
        from transformers import WhisperProcessor
        processor = WhisperProcessor.from_pretrained(
            whisper_cfg.get("model_id", "openai/whisper-small"))
        fillers = verify_fillers(
            models["filler_verifier"], fillers,
            models["whisper_model"], processor,
            device=device, audio_path=clean_audio_path,
        )

    timings["fillers"] = round(time.time() - step_start, 2)
    log(f"  Fillers: {len(fillers)} | {timings['fillers']:.1f}s")

    # ── Step 7: Disfluency ──
    log(f"Step 5/{total_steps}: disfluency...")
    progress_bar(4, total_steps, prefix=f"Video {video_num+1}")
    step_start = time.time()

    disfluencies = []
    if "disfluency_model" in models:
        from src.disfluency_model import predict_disfluency
        dis_cfg = config.get("disfluency", {})
        disfluencies = predict_disfluency(
            models["disfluency_model"], models["disfluency_processor"], audio,
            sr=sr, device=device,
            window_sec=dis_cfg.get("window_sec", 3.0),
            hop_sec=dis_cfg.get("hop_sec", 1.5),
        )

    timings["disfluency"] = round(time.time() - step_start, 2)
    log(f"  Disfluencies: {len(disfluencies)} | {timings['disfluency']:.1f}s | "
        f"Memory: {get_memory_mb():.0f} MB")

    # ── Step 8: Prosody ──
    log(f"Step 6/{total_steps}: prosody...")
    progress_bar(5, total_steps, prefix=f"Video {video_num+1}")
    step_start = time.time()

    from src.prosody import analyze_prosody
    prosody_results = analyze_prosody(audio, None, config)

    timings["prosody"] = round(time.time() - step_start, 2)
    log(f"  Windows: {len(prosody_results)} | {timings['prosody']:.1f}s")

    # ── Step 9: Emotion ──
    log(f"Step 7/{total_steps}: emotion...")
    progress_bar(6, total_steps, prefix=f"Video {video_num+1}")
    step_start = time.time()

    emotion_results = []
    if "emotion_classifier" in models:
        emotion_cfg = config.get("emotion", {})
        emotion_results = models["emotion_classifier"].predict_windowed(
            audio, sr=sr,
            window_sec=emotion_cfg.get("window_sec", 2.0),
        )

    timings["emotion"] = round(time.time() - step_start, 2)
    log(f"  Windows: {len(emotion_results)} | {timings['emotion']:.1f}s | "
        f"Memory: {get_memory_mb():.0f} MB")

    # ── Step 10: Assembly ──
    log(f"Step 8/{total_steps}: assembly...")
    progress_bar(7, total_steps, prefix=f"Video {video_num+1}")
    step_start = time.time()

    from src.output_assembler import (
        assemble_voice_output, generate_summary, save_output,
    )
    windows = assemble_voice_output(
        duration_sec=duration_sec,
        transcript=transcript,
        speech_segments=speech_segments,
        fillers=fillers,
        disfluencies=disfluencies,
        prosody_results=prosody_results,
        emotion_results=emotion_results,
        features_windowed=features_windowed,
        config=config,
    )
    summary = generate_summary(windows, duration_sec, transcript)

    metadata = {
        "pipeline_version": config.get("pipeline", {}).get("version", "1.0.0"),
        "video_path": str(video_path),
        "processed_at": datetime.now().isoformat(),
        "device": device,
        "duration_sec": round(duration_sec, 2),
        "snr_db": preprocess_result["snr_db"],
        "noise_reduced": preprocess_result["noise_reduced"],
        "models": {
            "whisper": whisper_cfg.get("model_id", "openai/whisper-small"),
            "disfluency": "available" if "disfluency_model" in models else "unavailable",
            "emotion": "available" if "emotion_classifier" in models else "unavailable",
            "filler_verifier": "available" if "filler_verifier" in models else "unavailable",
        },
        "timings": timings,
        "peak_memory_mb": round(get_memory_mb(), 1),
    }

    out_dir = config.get("output", {}).get("output_dir", "data/outputs")
    output_path = str(Path(out_dir) / f"{Path(video_path).stem}.json")
    saved_path = save_output(windows, summary, metadata, output_path)

    timings["assembly"] = round(time.time() - step_start, 2)

    progress_bar(total_steps, total_steps, prefix=f"Video {video_num+1}")

    total_elapsed = round(time.time() - total_start, 1)
    log(f"Video complete in {total_elapsed}s → {saved_path}")

    # Print step timings
    print(flush=True)
    print("  Step Timings:", flush=True)
    for step, secs in timings.items():
        print(f"    {step:20s} {secs:6.1f}s", flush=True)
    print(f"    {'TOTAL':20s} {sum(timings.values()):6.1f}s", flush=True)
    print(flush=True)

    return {
        "metadata": metadata,
        "summary": summary,
        "windows": windows,
        "output_path": saved_path,
        "total_time": total_elapsed,
    }


def main():
    import yaml

    print(flush=True)
    print("╔" + "═" * 58 + "╗", flush=True)
    print("║  PHASE 10: TESTING & VALIDATION                          ║", flush=True)
    print("║  Voice Analysis Pipeline — 3 Test Videos                 ║", flush=True)
    print("╚" + "═" * 58 + "╝", flush=True)
    print(flush=True)

    # Verify test videos
    missing = []
    for label, path in TEST_VIDEOS:
        if not Path(path).exists():
            missing.append((label, path))
    if missing:
        log("ERROR: Missing test videos:")
        for label, path in missing:
            print(f"  - {label}: {path}", flush=True)
        sys.exit(1)

    log(f"Found {len(TEST_VIDEOS)} test videos:")
    for i, (label, path) in enumerate(TEST_VIDEOS):
        size_mb = Path(path).stat().st_size / (1024 * 1024)
        print(f"  [{i+1}] {label}: {path} ({size_mb:.1f} MB)", flush=True)
    print(flush=True)

    # Load config
    with open("configs/pipeline_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # ── Step 1: Load all models upfront ──
    models = load_all_models(config)

    # ── Step 2: Process each video ──
    results = []
    video_timings = []
    batch_start = time.time()
    total_videos = len(TEST_VIDEOS)

    for i, (label, video_path) in enumerate(TEST_VIDEOS):
        # ETA for remaining videos
        if video_timings:
            avg_time = sum(video_timings) / len(video_timings)
            remaining = (total_videos - i) * avg_time
            log(f">>> Estimated remaining: ~{int(remaining // 60)}m{int(remaining % 60):02d}s "
                f"({total_videos - i} videos left)")

        result = process_video(video_path, config, models,
                               video_num=i, total_videos=total_videos)
        results.append((label, result))
        video_timings.append(result["total_time"])

    batch_elapsed = time.time() - batch_start

    # ── Final Summary ──
    print(flush=True)
    print("╔" + "═" * 58 + "╗", flush=True)
    print("║  PHASE 10 COMPLETE — RESULTS SUMMARY                    ║", flush=True)
    print("╚" + "═" * 58 + "╝", flush=True)
    print(flush=True)

    for i, (label, result) in enumerate(results):
        summary = result["summary"]
        meta = result["metadata"]
        print(f"  [{i+1}] {label}", flush=True)
        print(f"      Duration:     {meta['duration_sec']:.0f}s", flush=True)
        print(f"      Words:        {summary['word_count']} ({summary['words_per_minute']:.0f} WPM)", flush=True)
        print(f"      Fillers:      {summary['filler_count']} ({summary['fillers_per_minute']:.1f}/min)", flush=True)
        print(f"      Disfluencies: {summary['disfluency_count']}", flush=True)
        print(f"      Pitch CV:     {summary['avg_pitch_cv']:.4f}", flush=True)
        print(f"      Emotion:      {summary['dominant_emotion']} "
              f"(variety: {summary['emotion_variety']:.2f})", flush=True)
        print(f"      Proc time:    {result['total_time']:.1f}s", flush=True)
        print(f"      Output:       {result['output_path']}", flush=True)
        print(flush=True)

    log(f"Total batch time: {batch_elapsed:.1f}s")
    log("Output files saved to data/outputs/")
    print(flush=True)


if __name__ == "__main__":
    main()
