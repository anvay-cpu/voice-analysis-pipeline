"""Main pipeline orchestrator for voice analysis."""

import argparse
import sys
import time
import yaml
import numpy as np
import librosa
from datetime import datetime
from pathlib import Path


def load_config(config_path: str = "configs/pipeline_config.yaml") -> dict:
    """Load pipeline configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _progress_bar(current, total, prefix="", width=40):
    """Render a text progress bar to stdout."""
    pct = current / total if total > 0 else 1.0
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    sys.stdout.write(f"\r  {prefix} [{bar}] {pct*100:5.1f}%")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


class VoiceAnalysisPipeline:
    """End-to-end voice analysis pipeline.

    Chains all 10 steps: extract audio → VAD → denoise → transcribe →
    features → fillers → disfluencies → prosody → emotion → assemble output.

    Handles missing models gracefully — skips unavailable modules.
    """

    def __init__(self, config_path: str = "configs/pipeline_config.yaml"):
        self.config = load_config(config_path)
        self.device = self._resolve_device(
            self.config.get("pipeline", {}).get("device", "cpu")
        )
        self.sr = self.config.get("audio", {}).get("sample_rate", 16000)
        self.timings = {}
        self._models_loaded = False

        # Models loaded lazily in process() or upfront via load_all_models()
        self._whisper_pipe = None
        self._whisper_model = None
        self._whisper_processor = None
        self._filler_verifier = None
        self._disfluency_model = None
        self._disfluency_processor = None
        self._emotion_classifier = None

        # Timing history for ETA estimation (step_name → list of durations)
        self._step_history = {}

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve device string, with 'auto' picking best available."""
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    def _timer(self, name: str):
        """Context-manager-like timer. Call start/stop manually."""
        return _StepTimer(name, self.timings)

    def _load_whisper(self):
        """Load Whisper model (for transcription + filler verification)."""
        if self._whisper_pipe is not None:
            return
        from src.transcriber import load_whisper
        whisper_cfg = self.config.get("whisper", {})
        self._whisper_pipe = load_whisper(
            model_id=whisper_cfg.get("model_id", "openai/whisper-small"),
            device=self.device,
        )
        # Also get raw model + processor for filler verification
        self._whisper_model = self._whisper_pipe.model
        self._whisper_processor = self._whisper_pipe.tokenizer

    def _load_filler_verifier(self):
        """Load filler verification MLP. Skips if model not found."""
        if self._filler_verifier is not None:
            return
        filler_model_path = Path("models/filler_verifier/best_model.pt")
        if not filler_model_path.exists():
            print("  [SKIP] Filler verifier model not found — using regex-only")
            return
        try:
            from src.filler_verifier import load_filler_verifier
            self._filler_verifier = load_filler_verifier(
                str(filler_model_path), device=self.device)
        except Exception as e:
            print(f"  [SKIP] Filler verifier failed to load: {e} — using regex-only")

    def _load_disfluency(self):
        """Load disfluency model. Skips if model not found."""
        if self._disfluency_model is not None:
            return
        model_path = self.config.get("disfluency", {}).get(
            "model_path", "models/disfluency/best_model.pt")
        if not Path(model_path).exists():
            print("  [SKIP] Disfluency model not found — skipping disfluency detection")
            return
        try:
            from src.disfluency_model import load_disfluency_model
            self._disfluency_model, self._disfluency_processor = load_disfluency_model(
                model_path, device=self.device)
        except Exception as e:
            print(f"  [SKIP] Disfluency model failed to load: {e}")

    def _load_emotion(self):
        """Load vocal emotion classifier. Skips if model not found."""
        if self._emotion_classifier is not None:
            return
        model_path = self.config.get("emotion", {}).get(
            "model_path", "models/vocal_emotion/best_model.pt")
        if not Path(model_path).exists():
            print("  [SKIP] Emotion model not found — skipping emotion classification")
            return
        try:
            from src.vocal_emotion import load_emotion_model
            self._emotion_classifier = load_emotion_model(model_path, device=self.device)
        except Exception as e:
            print(f"  [SKIP] Emotion model failed to load: {e}")

    def load_all_models(self):
        """Load all models upfront with progress tracking.

        Call this before process() to avoid per-video model loading delays.
        Models that are already loaded are skipped.
        """
        models = [
            ("Whisper (transcription)", self._load_whisper, "~244M params"),
            ("Filler Verifier MLP", self._load_filler_verifier, "~200K params"),
            ("Wav2Vec2 Disfluency", self._load_disfluency, "~95M params"),
            ("ECAPA-TDNN Emotion", self._load_emotion, "~15M params"),
        ]

        total = len(models)
        print("=" * 60)
        print("  MODEL LOADING")
        print("=" * 60)

        load_timings = {}
        total_start = time.time()

        for i, (name, loader, size_hint) in enumerate(models):
            _progress_bar(i, total, prefix="Overall")
            print(f"\n  [{i+1}/{total}] Loading {name} ({size_hint})...")
            step_start = time.time()
            try:
                loader()
                elapsed = time.time() - step_start
                load_timings[name] = elapsed
                status = "OK"
            except Exception as e:
                elapsed = time.time() - step_start
                load_timings[name] = elapsed
                status = f"SKIP ({e})"
            print(f"         → {status} ({elapsed:.1f}s)")

        _progress_bar(total, total, prefix="Overall")

        total_elapsed = time.time() - total_start
        print(f"\n  All models loaded in {total_elapsed:.1f}s")

        # Print summary table
        print("\n  ┌─────────────────────────────┬──────────┬──────────┐")
        print("  │ Model                       │ Time     │ Status   │")
        print("  ├─────────────────────────────┼──────────┼──────────┤")
        whisper_status = "✓ loaded" if self._whisper_pipe else "✗ missing"
        filler_status = "✓ loaded" if self._filler_verifier else "- regex"
        disf_status = "✓ loaded" if self._disfluency_model else "✗ missing"
        emo_status = "✓ loaded" if self._emotion_classifier else "✗ missing"
        statuses = [whisper_status, filler_status, disf_status, emo_status]
        for (name, _, _), status in zip(models, statuses):
            t = load_timings.get(name, 0)
            print(f"  │ {name:27s} │ {t:6.1f}s  │ {status:8s} │")
        print("  └─────────────────────────────┴──────────┴──────────┘")
        print("=" * 60)
        print()

        self._models_loaded = True
        return load_timings

    def _fmt_eta(self, remaining_sec):
        """Format seconds as mm:ss string."""
        m, s = divmod(int(remaining_sec), 60)
        return f"{m}m{s:02d}s"

    def _estimate_step_eta(self, step_name, duration_sec):
        """Estimate step duration based on history and audio duration."""
        if step_name in self._step_history and self._step_history[step_name]:
            # Use average seconds-per-audio-second from history
            rates = self._step_history[step_name]
            avg_rate = sum(r for _, r in rates) / len(rates)
            return avg_rate * duration_sec
        return None

    def _record_step_timing(self, step_name, step_time, duration_sec):
        """Record step timing for future ETA estimation."""
        rate = step_time / duration_sec if duration_sec > 0 else 0
        if step_name not in self._step_history:
            self._step_history[step_name] = []
        self._step_history[step_name].append((step_time, rate))

    def process(self, video_path: str, output_path: str = None,
                video_num: int = 0, total_videos: int = 1) -> dict:
        """Run the full voice analysis pipeline on a video file.

        Args:
            video_path: Path to input video file.
            output_path: Optional output JSON path. Defaults to data/outputs/<stem>.json.
            video_num: Current video index (0-based) for multi-video progress.
            total_videos: Total number of videos being processed.

        Returns:
            Complete output dict with metadata, summary, and windows.
        """
        self.timings = {}  # Reset per-video timings
        total_start = time.time()
        video_label = Path(video_path).stem

        steps = ["preprocessing", "transcription", "features", "fillers",
                 "disfluency", "prosody", "emotion", "assembly"]
        total_steps = len(steps)

        print()
        print("=" * 60)
        if total_videos > 1:
            print(f"  VIDEO {video_num+1}/{total_videos}: {video_label}")
        else:
            print(f"  VIDEO: {video_label}")
        print(f"  Input: {video_path}")
        print("=" * 60)

        def _step_header(step_idx, step_name, extra=""):
            """Print step header with overall progress."""
            _progress_bar(step_idx, total_steps, prefix=f"Video {video_num+1}")
            eta_str = ""
            if step_name in self._step_history and self._step_history[step_name]:
                # We don't know duration yet for preprocessing, skip ETA
                pass
            print(f"\n  Step {step_idx+1}/{total_steps}: {step_name}{extra}")

        # ── Step 1-3: Audio extraction, VAD, noise reduction ──
        _step_header(0, "preprocessing")
        t = self._timer("preprocessing")
        t.start()
        from src.noise_reduction import preprocess_audio
        output_dir = self.config.get("output", {}).get("output_dir", "data/outputs")
        audio_dir = str(Path(output_dir).parent / "audio")
        preprocess_result = preprocess_audio(video_path, audio_dir, self.config)
        t.stop()

        clean_audio_path = preprocess_result["clean_audio_path"]
        speech_segments = preprocess_result["speech_segments"]
        duration_sec = preprocess_result["duration_sec"]
        self._record_step_timing("preprocessing", self.timings["preprocessing"], duration_sec)

        print(f"    Duration: {duration_sec:.1f}s | SNR: {preprocess_result['snr_db']:.1f}dB | "
              f"Speech segments: {len(speech_segments)}")

        # Estimate remaining time after we know audio duration
        remaining_steps = steps[1:]
        eta_total = 0
        has_eta = False
        for s in remaining_steps:
            est = self._estimate_step_eta(s, duration_sec)
            if est is not None:
                eta_total += est
                has_eta = True
        if has_eta:
            print(f"    Estimated remaining: ~{self._fmt_eta(eta_total)}")

        # Load clean audio as numpy array
        audio, _ = librosa.load(clean_audio_path, sr=self.sr, mono=True)

        # ── Step 4: Transcription ──
        eta = self._estimate_step_eta("transcription", duration_sec)
        eta_hint = f" (est. ~{self._fmt_eta(eta)})" if eta else ""
        _step_header(1, "transcription", eta_hint)
        t = self._timer("transcription")
        t.start()
        self._load_whisper()
        from src.transcriber import transcribe, filter_by_vad
        whisper_cfg = self.config.get("whisper", {})
        transcript = transcribe(
            self._whisper_pipe, clean_audio_path,
            language=whisper_cfg.get("language", "en"),
            chunk_length_s=whisper_cfg.get("chunk_length_s", 30),
            batch_size=whisper_cfg.get("batch_size", 1),
        )
        transcript = filter_by_vad(transcript, speech_segments)
        t.stop()
        self._record_step_timing("transcription", self.timings["transcription"], duration_sec)
        print(f"    Words: {len(transcript['words'])} | "
              f"Text: \"{transcript['text'][:80]}{'...' if len(transcript['text']) > 80 else ''}\"")

        # ── Step 5: Feature extraction ──
        eta = self._estimate_step_eta("features", duration_sec)
        eta_hint = f" (est. ~{self._fmt_eta(eta)})" if eta else ""
        _step_header(2, "features", eta_hint)
        t = self._timer("features")
        t.start()
        from src.features import AcousticFeatureExtractor
        extractor = AcousticFeatureExtractor(self.config)
        features_windowed = extractor.extract_windowed(audio, window_sec=2.0, hop_sec=1.0)
        t.stop()
        self._record_step_timing("features", self.timings["features"], duration_sec)
        print(f"    Feature windows: {len(features_windowed)}")

        # ── Step 6: Filler detection ──
        eta = self._estimate_step_eta("fillers", duration_sec)
        eta_hint = f" (est. ~{self._fmt_eta(eta)})" if eta else ""
        _step_header(3, "fillers", eta_hint)
        t = self._timer("fillers")
        t.start()
        from src.filler_detector import detect_fillers_from_transcript
        fillers = detect_fillers_from_transcript(transcript)

        # Try acoustic verification (model already loaded if available)
        self._load_filler_verifier()
        if self._filler_verifier is not None:
            from src.filler_verifier import verify_fillers
            from transformers import WhisperProcessor
            processor = WhisperProcessor.from_pretrained(
                whisper_cfg.get("model_id", "openai/whisper-small"))
            fillers = verify_fillers(
                self._filler_verifier, fillers,
                self._whisper_pipe.model, processor,
                device=self.device, audio_path=clean_audio_path,
            )
        t.stop()
        self._record_step_timing("fillers", self.timings["fillers"], duration_sec)
        print(f"    Fillers detected: {len(fillers)}")

        # ── Step 7: Disfluency detection ──
        eta = self._estimate_step_eta("disfluency", duration_sec)
        eta_hint = f" (est. ~{self._fmt_eta(eta)})" if eta else ""
        _step_header(4, "disfluency", eta_hint)
        t = self._timer("disfluency")
        t.start()
        disfluencies = []
        self._load_disfluency()
        if self._disfluency_model is not None:
            from src.disfluency_model import predict_disfluency
            dis_cfg = self.config.get("disfluency", {})
            disfluencies = predict_disfluency(
                self._disfluency_model, self._disfluency_processor, audio,
                sr=self.sr, device=self.device,
                window_sec=dis_cfg.get("window_sec", 3.0),
                hop_sec=dis_cfg.get("hop_sec", 1.5),
            )
        t.stop()
        self._record_step_timing("disfluency", self.timings["disfluency"], duration_sec)
        print(f"    Disfluencies detected: {len(disfluencies)}")

        # ── Step 8: Prosody analysis ──
        eta = self._estimate_step_eta("prosody", duration_sec)
        eta_hint = f" (est. ~{self._fmt_eta(eta)})" if eta else ""
        _step_header(5, "prosody", eta_hint)
        t = self._timer("prosody")
        t.start()
        from src.prosody import analyze_prosody
        prosody_results = analyze_prosody(audio, None, self.config)
        t.stop()
        self._record_step_timing("prosody", self.timings["prosody"], duration_sec)
        print(f"    Prosody windows: {len(prosody_results)}")

        # ── Step 9: Vocal emotion ──
        eta = self._estimate_step_eta("emotion", duration_sec)
        eta_hint = f" (est. ~{self._fmt_eta(eta)})" if eta else ""
        _step_header(6, "emotion", eta_hint)
        t = self._timer("emotion")
        t.start()
        emotion_results = []
        self._load_emotion()
        if self._emotion_classifier is not None:
            emotion_cfg = self.config.get("emotion", {})
            emotion_results = self._emotion_classifier.predict_windowed(
                audio, sr=self.sr,
                window_sec=emotion_cfg.get("window_sec", 2.0),
            )
        t.stop()
        self._record_step_timing("emotion", self.timings["emotion"], duration_sec)
        print(f"    Emotion windows: {len(emotion_results)}")

        # ── Step 10: Assemble output ──
        _step_header(7, "assembly")
        t = self._timer("assembly")
        t.start()
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
            config=self.config,
        )
        summary = generate_summary(windows, duration_sec, transcript)

        metadata = {
            "pipeline_version": self.config.get("pipeline", {}).get("version", "1.0.0"),
            "video_path": str(video_path),
            "processed_at": datetime.now().isoformat(),
            "device": self.device,
            "duration_sec": round(duration_sec, 2),
            "snr_db": preprocess_result["snr_db"],
            "noise_reduced": preprocess_result["noise_reduced"],
            "models": {
                "whisper": whisper_cfg.get("model_id", "openai/whisper-small"),
                "disfluency": "available" if self._disfluency_model else "unavailable",
                "emotion": "available" if self._emotion_classifier else "unavailable",
                "filler_verifier": "available" if self._filler_verifier else "unavailable",
            },
            "timings": self.timings,
        }

        # Determine output path
        if output_path is None:
            out_dir = self.config.get("output", {}).get("output_dir", "data/outputs")
            output_path = str(Path(out_dir) / f"{Path(video_path).stem}.json")

        saved_path = save_output(windows, summary, metadata, output_path)
        t.stop()
        self._record_step_timing("assembly", self.timings["assembly"], duration_sec)

        # Final progress bar
        _progress_bar(total_steps, total_steps, prefix=f"Video {video_num+1}")

        total_elapsed = round(time.time() - total_start, 1)
        print(f"\n  Video complete in {total_elapsed}s → {saved_path}")
        print()
        _print_timings(self.timings)

        return {
            "metadata": metadata,
            "summary": summary,
            "windows": windows,
            "transcript": transcript,
            "output_path": saved_path,
        }


class _StepTimer:
    """Simple timer for pipeline steps."""

    def __init__(self, name: str, timings: dict):
        self.name = name
        self.timings = timings
        self._start = None

    def start(self):
        self._start = time.time()
        print(f"[{self.name}] Running...")

    def stop(self):
        elapsed = round(time.time() - self._start, 2)
        self.timings[self.name] = elapsed
        print(f"[{self.name}] Done ({elapsed}s)")


def _print_timings(timings: dict):
    """Print a summary of step timings."""
    print("Step Timings:")
    for step, secs in timings.items():
        print(f"  {step:20s} {secs:6.1f}s")
    total = sum(timings.values())
    print(f"  {'TOTAL':20s} {total:6.1f}s")


def run_pipeline(video_path: str, config: dict, output_path: str = None) -> dict:
    """Run the full voice analysis pipeline on a video file.

    Args:
        video_path: Path to input video file.
        config: Pipeline configuration dictionary.
        output_path: Optional output JSON path.

    Returns:
        Dictionary containing all analysis results.
    """
    pipeline = VoiceAnalysisPipeline.__new__(VoiceAnalysisPipeline)
    pipeline.config = config
    pipeline.device = config.get("pipeline", {}).get("device", "cpu")
    pipeline.sr = config.get("audio", {}).get("sample_rate", 16000)
    pipeline.timings = {}
    pipeline._models_loaded = False
    pipeline._step_history = {}
    pipeline._whisper_pipe = None
    pipeline._whisper_model = None
    pipeline._whisper_processor = None
    pipeline._filler_verifier = None
    pipeline._disfluency_model = None
    pipeline._disfluency_processor = None
    pipeline._emotion_classifier = None
    return pipeline.process(video_path, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="AI Public Speaking Assistant — Voice Analysis Pipeline"
    )
    parser.add_argument("--video", nargs="+", required=True,
                        help="Path(s) to input video file(s)")
    parser.add_argument(
        "--config",
        default="configs/pipeline_config.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--output", default=None, help="Output JSON path (default: data/outputs/)"
    )
    args = parser.parse_args()

    pipeline = VoiceAnalysisPipeline(args.config)

    videos = args.video
    total = len(videos)

    # Load all models upfront if processing multiple videos
    if total > 1 or True:  # Always preload for progress visibility
        pipeline.load_all_models()

    results = []
    batch_start = time.time()

    for i, video_path in enumerate(videos):
        result = pipeline.process(video_path, args.output,
                                  video_num=i, total_videos=total)
        results.append(result)

    # Print batch summary if multiple videos
    if total > 1:
        batch_elapsed = time.time() - batch_start
        print()
        print("=" * 60)
        print("  BATCH SUMMARY")
        print("=" * 60)
        for i, (vpath, res) in enumerate(zip(videos, results)):
            name = Path(vpath).stem
            dur = res["metadata"]["duration_sec"]
            proc_time = sum(res["metadata"]["timings"].values())
            print(f"  [{i+1}] {name}")
            print(f"      Audio: {dur:.0f}s | Processed in: {proc_time:.1f}s")
            print(f"      Output: {res['output_path']}")
        print(f"\n  Total batch time: {batch_elapsed:.1f}s")
        print("=" * 60)

    return results[0] if total == 1 else results


if __name__ == "__main__":
    main()
