from __future__ import annotations

import uuid
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from main import main as run_pipeline  


app = FastAPI(title="Soccer Analytics API")

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

JOBS: Dict[str, Dict[str, Any]] = {}


class JobStatus(BaseModel):
    job_id: str
    status: str  
    error: Optional[str] = None
    artifacts: Optional[Dict[str, str]] = None


def _is_mp4(upload: UploadFile) -> bool:
    name_ok = (upload.filename or "").lower().endswith(".mp4")
    type_ok = (upload.content_type or "").lower() in ("video/mp4", "application/octet-stream")
    return name_ok and type_ok


def _run_job(job_id: str, video_path: Path, enable_team: bool):
    try:
        JOBS[job_id]["status"] = "running"

        out_dir = OUTPUTS_DIR / job_id
        out_dir.mkdir(parents=True, exist_ok=True)

        run_pipeline(str(video_path), out_dir=str(out_dir), enable_team=enable_team)

        annotated = out_dir / "annotated.mp4"
        csv_path = out_dir / "per_frame_tracks.csv"

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["artifacts"] = {
            "annotated_video": f"/jobs/{job_id}/artifacts/annotated.mp4",
            "csv": f"/jobs/{job_id}/artifacts/per_frame_tracks.csv",
        }

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)


@app.post("/analyze-video", response_model=JobStatus)
async def analyze_video(
    bg: BackgroundTasks,
    file: UploadFile = File(...),
    enable_team: bool = True,
):
    # Validate
    if not _is_mp4(file):
        raise HTTPException(status_code=400, detail="Please upload an .mp4 video file.")

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"job_id": job_id, "status": "queued", "error": None, "artifacts": None}

    
    video_path = UPLOADS_DIR / f"{job_id}.mp4"
    with video_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run pipeline in background
    bg.add_task(_run_job, job_id, video_path, enable_team)

    return JobStatus(job_id=job_id, status="queued")


@app.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(**job)


@app.get("/jobs/{job_id}/artifacts/{filename}")
def get_artifact(job_id: str, filename: str):
    out_dir = OUTPUTS_DIR / job_id
    file_path = out_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(str(file_path))
