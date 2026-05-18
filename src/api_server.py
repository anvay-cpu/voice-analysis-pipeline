"""FastAPI backend for the Speech Coach webapp.

Handles video uploads, pipeline execution, job tracking, and report serving.

Run: uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import logging
import os
import shutil
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
SESSIONS_DIR = BASE_DIR / "data" / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR = BASE_DIR / "data" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ── App ─────────────────────────────────────────────────────────────

app = FastAPI(title="Speech Coach API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded videos
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

# In-memory job tracker
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

# ── Helpers ─────────────────────────────────────────────────────────


def _load_session_index() -> list[dict]:
    """Load the session index file."""
    index_path = SESSIONS_DIR / "index.json"
    if index_path.exists():
        with open(index_path) as f:
            return json.load(f)
    return []


def _save_session_index(sessions: list[dict]):
    """Save the session index file."""
    index_path = SESSIONS_DIR / "index.json"
    with open(index_path, "w") as f:
        json.dump(sessions, f, indent=2)


def _get_session_dir(session_id: str) -> Path:
    return SESSIONS_DIR / session_id


def _update_job(job_id: str, **kwargs):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


# Global pre-loaded pipeline (set by Colab Cell 5 or on first use)
_pipeline_instance: "SpeechCoachPipeline | None" = None
_pipeline_lock = threading.Lock()


def get_pipeline():
    """Get or create the shared pipeline instance."""
    global _pipeline_instance
    with _pipeline_lock:
        if _pipeline_instance is None:
            from src.master_pipeline import SpeechCoachPipeline
            _pipeline_instance = SpeechCoachPipeline()
        return _pipeline_instance


def set_pipeline(pipeline):
    """Set a pre-loaded pipeline instance (called from Colab notebook)."""
    global _pipeline_instance
    with _pipeline_lock:
        _pipeline_instance = pipeline


def _run_pipeline(job_id: str, session_id: str, video_path: str, title: str):
    """Run the full pipeline in a background thread."""
    session_dir = _get_session_dir(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        _update_job(job_id, status="running", step="PREPROCESSING",
                     step_index=0, total_steps=8, progress=2.0)

        start = time.time()

        # Re-encode to H.264 if needed (AV1, HEVC etc. not supported by OpenCV)
        ext = os.path.splitext(video_path)[1].lower()
        if ext in {".mp4", ".webm", ".avi", ".mov", ".mkv"}:
            try:
                import subprocess as _sp
                probe = _sp.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=codec_name", "-of", "csv=p=0", video_path],
                    capture_output=True, text=True, timeout=10,
                )
                codec = probe.stdout.strip().lower()
                if codec and codec not in ("h264", ""):
                    logger.info("Re-encoding video from %s to H.264", codec)
                    fixed = video_path + ".tmp.mp4"
                    _sp.run(
                        ["ffmpeg", "-y", "-i", video_path, "-c:v", "libx264", "-c:a", "aac", fixed],
                        capture_output=True, timeout=600,
                    )
                    if os.path.exists(fixed) and os.path.getsize(fixed) > 0:
                        os.replace(fixed, video_path)
                        logger.info("Re-encoded to H.264: %.1fMB", os.path.getsize(video_path) / 1e6)
            except Exception as e:
                logger.warning("Re-encoding failed: %s (proceeding with original)", e)

        from src.report_transformer import transform_pipeline_output

        pipeline = get_pipeline()

        # Step 1: Voice
        _update_job(job_id, step="VOICE_ANALYSIS", step_index=1, progress=8.0)
        voice_output = pipeline._run_voice_pipeline(video_path)
        _update_job(job_id, step="VOICE_ANALYSIS", step_index=1, progress=20.0,
                     step_status="complete")

        # Step 2: Body language
        _update_job(job_id, step="BODY_LANGUAGE_ANALYSIS", step_index=2, progress=25.0)
        body_output = pipeline._run_body_pipeline(video_path)
        _update_job(job_id, step="BODY_LANGUAGE_ANALYSIS", step_index=2, progress=45.0,
                     step_status="complete")

        # Step 3: Content
        _update_job(job_id, step="CONTENT_ANALYSIS", step_index=3, progress=48.0)
        content_output = pipeline._run_content_pipeline(voice_output)
        _update_job(job_id, step="CONTENT_ANALYSIS", step_index=3, progress=60.0,
                     step_status="complete")

        # Step 4: Fusion
        _update_job(job_id, step="MULTIMODAL_FUSION", step_index=4, progress=62.0)
        duration = pipeline._get_duration(voice_output, body_output)
        voice_windows = pipeline._extract_voice_windows(voice_output)
        fusion_output = pipeline.fusion_engine.fuse(
            voice_windows, body_output, content_output, duration
        )
        _update_job(job_id, step="MULTIMODAL_FUSION", step_index=4, progress=72.0,
                     step_status="complete")

        # Step 5: Scoring
        _update_job(job_id, step="DIMENSION_SCORING", step_index=5, progress=75.0)
        scores = pipeline.scorer.score_all(fusion_output)
        _update_job(job_id, step="DIMENSION_SCORING", step_index=5, progress=78.0,
                     step_status="complete")

        # Step 6: Coaching
        _update_job(job_id, step="COACHING_GENERATION", step_index=6, progress=80.0)
        coaching = pipeline.coaching_writer.generate_full_report(
            all_segments=[],
            overall_scores=scores,
            fusion_output=fusion_output,
            speech_metadata={"duration_sec": duration, "video_path": video_path},
        )
        _update_job(job_id, step="COACHING_GENERATION", step_index=6, progress=88.0,
                     step_status="complete")

        # Step 7: Charts
        _update_job(job_id, step="CHART_GENERATION", step_index=7, progress=90.0)
        chart_dir = str(session_dir / "charts")
        os.makedirs(chart_dir, exist_ok=True)
        pipeline.chart_generator.output_dir = chart_dir
        charts = pipeline.chart_generator.generate_all(
            scores, fusion_output["timeline"], fusion_output
        )
        _update_job(job_id, step="CHART_GENERATION", step_index=7, progress=95.0,
                     step_status="complete")

        # Step 8: Transform to SpeechReport format
        _update_job(job_id, step="REPORT_ASSEMBLY", step_index=8, progress=96.0)
        processing_time = time.time() - start
        video_url = f"/uploads/{os.path.basename(video_path)}"

        speech_report = transform_pipeline_output(
            voice_output=voice_output,
            body_output=body_output,
            content_output=content_output,
            fusion_output=fusion_output,
            scores=scores,
            coaching=coaching,
            session_id=session_id,
            title=title,
            video_url=video_url,
            processing_time_sec=processing_time,
        )

        # Save report
        report_path = session_dir / "report.json"
        with open(report_path, "w") as f:
            json.dump(speech_report, f, indent=2)

        # Also save raw outputs for debugging
        raw_path = session_dir / "raw_outputs.json"
        with open(raw_path, "w") as f:
            json.dump({
                "scores": scores,
                "coaching_keys": list(coaching.keys()),
                "fusion_keys": list(fusion_output.keys()),
                "duration_sec": duration,
            }, f, indent=2)

        # Update session index
        sessions = _load_session_index()
        for s in sessions:
            if s["id"] == session_id:
                s["status"] = "COMPLETE"
                s["score"] = scores.get("overall", 0)
                s["dominant_tone"] = speech_report["content_details"]["dominant_tone"]
                s["processing_time_sec"] = processing_time
                break
        _save_session_index(sessions)

        _update_job(job_id, status="complete", progress=100.0,
                     step="COMPLETE", step_status="complete",
                     report_id=session_id, processing_time_sec=processing_time)

        logger.info("Pipeline complete for %s in %.0fs", session_id, processing_time)

    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Pipeline failed for %s: %s", session_id, tb)
        # Include last step info in error for easier debugging
        with _jobs_lock:
            current_step = _jobs.get(job_id, {}).get("step", "unknown")
        error_msg = f"[{current_step}] {type(e).__name__}: {e}"
        _update_job(job_id, status="failed", error=error_msg)

        # Update session index
        sessions = _load_session_index()
        for s in sessions:
            if s["id"] == session_id:
                s["status"] = "FAILED"
                break
        _save_session_index(sessions)


# ── Routes ──────────────────────────────────────────────────────────


@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.post("/api/upload")
async def upload_video(
    file: UploadFile = File(...),
    title: str = Form("Speech Analysis"),
    profile: str = Form("EXECUTIVE_PRESENTATION_V2"),
):
    """Upload a video and start pipeline processing."""
    if not file.filename:
        raise HTTPException(400, "No file provided")

    # Validate file type
    allowed = {".mp4", ".webm", ".avi", ".mov", ".mkv", ".wav", ".mp3", ".flac", ".ogg"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    # Generate IDs
    from src.report_transformer import generate_session_id
    session_id = generate_session_id()
    job_id = f"job_{session_id}"

    # Save uploaded file
    safe_name = f"{session_id}{ext}"
    file_path = UPLOADS_DIR / safe_name
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    file_size = file_path.stat().st_size

    # Create session entry
    sessions = _load_session_index()
    sessions.insert(0, {
        "id": session_id,
        "title": title,
        "profile": profile,
        "filename": file.filename,
        "file_size": file_size,
        "status": "PROCESSING",
        "score": None,
        "dominant_tone": None,
        "created_at": datetime.now().isoformat(),
        "duration": None,
        "processing_time_sec": None,
    })
    _save_session_index(sessions)

    # Initialize job
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "session_id": session_id,
            "status": "queued",
            "step": "QUEUED",
            "step_index": 0,
            "total_steps": 8,
            "progress": 0.0,
            "created_at": datetime.now().isoformat(),
        }

    # Launch pipeline in background thread
    t = threading.Thread(
        target=_run_pipeline,
        args=(job_id, session_id, str(file_path), title),
        daemon=True,
    )
    t.start()

    return {
        "job_id": job_id,
        "session_id": session_id,
        "status": "queued",
        "message": f"Processing started for '{file.filename}'",
    }


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a processing job."""
    with _jobs_lock:
        job = _jobs.get(job_id)

    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found")

    return job


@app.get("/api/reports/{session_id}")
async def get_report(session_id: str):
    """Get the completed SpeechReport for a session."""
    report_path = _get_session_dir(session_id) / "report.json"
    if not report_path.exists():
        raise HTTPException(404, f"Report '{session_id}' not found")

    with open(report_path) as f:
        return json.load(f)


@app.get("/api/sessions")
async def list_sessions():
    """List all sessions."""
    sessions = _load_session_index()
    return {"sessions": sessions}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its data."""
    session_dir = _get_session_dir(session_id)
    if session_dir.exists():
        shutil.rmtree(session_dir)

    sessions = _load_session_index()
    sessions = [s for s in sessions if s["id"] != session_id]
    _save_session_index(sessions)

    return {"status": "deleted", "session_id": session_id}


@app.get("/api/speeches/{session_id}/report")
async def get_speech_report(session_id: str):
    """Backwards-compatible endpoint matching the existing webapp API route."""
    return await get_report(session_id)
