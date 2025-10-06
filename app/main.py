# app/main.py
import os
import time
import uuid
import subprocess
import shutil
from pathlib import Path
import json
from fastapi import Header, Query, HTTPException
from fastapi.responses import FileResponse

from fastapi import FastAPI, Request, UploadFile, Form, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from redis import Redis
from rq import Queue, Retry

# Import here to avoid circular imports
from .worker_tasks import process_job

load_dotenv()

APP_ROOT = Path(__file__).resolve().parents[1]
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", "/tmp/uploads"))  # Render-fast tmp
MEDIA_DIR = APP_ROOT / "media"
TEMPLATES = Jinja2Templates(directory=str(APP_ROOT / "app" / "templates"))

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Redis key configuration
QUEUE_NAME = os.environ.get("QUEUE_NAME", "video-jobs")
KEY_PREFIX = os.environ.get("KEY_PREFIX", "")
QUEUE_KEY = f"{KEY_PREFIX}{QUEUE_NAME}"

MEDIA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = APP_ROOT / "app" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

redis = Redis.from_url(REDIS_URL)
queue = Queue(QUEUE_NAME, connection=redis, default_timeout=5400)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return TEMPLATES.TemplateResponse("upload.html", {"request": request})


def _safe_name(name: str) -> str:
    keep = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    cleaned = "".join(ch if ch in keep else "_" for ch in (name or "upload"))
    if not cleaned or cleaned.startswith("."):
        cleaned = f"file_{uuid.uuid4().hex}"
    return cleaned[:128]


_ALLOWED_STYLES = {"kid_friendly_cartoon", "japanese_kawaii", "storybook_illustrated", "watercolor_storybook", "paper_cutout", "cutout_collage", "realistic_3d", "claymation", "needle_felted", "stop_motion_felt_clay", "hybrid_mix", "japanese_anime", "pixel_art", "van_gogh", "impressionism", "art_deco", "cubism", "graphic_novel", "motion_comic", "comic_book", "gothic", "silhouette", "fantasy_magic_glow", "surrealism_hybrid", "ink_parchment", "japanese_woodblock", "ink_wash", "japanese_gold_screen", "japanese_scroll", "japanese_court"}


@app.post("/submit", response_class=HTMLResponse)
async def submit(
    request: Request,
    email: str = Form(...),
    audio: UploadFile = File(...),
    portrait: UploadFile | None = File(default=None),
    style: str = Form("kid"),
):
    # validate style
    style = (style or "kid").lower().strip()
    if style not in _ALLOWED_STYLES:
        raise HTTPException(status_code=400, detail=f"Invalid style '{style}'. Choose one of: {sorted(_ALLOWED_STYLES)}")

    allowed_mimes = {
        "audio/mpeg", "audio/wav", "audio/x-wav",
        "audio/m4a", "audio/x-m4a", "audio/mp4",
        "audio/aac", "audio/ogg",
    }
    if audio.content_type not in allowed_mimes:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {audio.content_type}")

    data = await audio.read()
    if len(data) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")

    job_id = str(uuid.uuid4())
    user_dir = UPLOADS_DIR / job_id

    # Ensure uploads directory is writable
    try:
        test_file = user_dir / ".write_test"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_bytes(b"test")
        test_file.unlink()  # Clean up test file
        print(f"[DEBUG] Uploads directory is writable: {user_dir}", flush=True)
    except Exception as e:
        print(f"[ERROR] Uploads directory not writable: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"Upload directory not writable: {e}")

    user_dir.mkdir(parents=True, exist_ok=True)

    # Save audio with original (sanitized) name
    orig_audio_name = _safe_name(audio.filename or "audio_upload")
    audio_path = user_dir / orig_audio_name
    print(f"[DEBUG] Saving audio file as: {audio_path}", flush=True)
    print(f"[DEBUG] Original filename: {audio.filename}", flush=True)
    print(f"[DEBUG] Sanitized filename: {orig_audio_name}", flush=True)
    print(f"[DEBUG] Audio file size to save: {len(data)} bytes", flush=True)
    print(f"[DEBUG] Job directory before save: {user_dir}", flush=True)
    print(f"[DEBUG] Job directory exists: {user_dir.exists()}", flush=True)

    with open(audio_path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())  # Force write to disk

    print(f"[DEBUG] File write completed for: {audio_path}", flush=True)

    # Additional verification that file was written correctly
    if not audio_path.exists():
        raise HTTPException(status_code=500, detail=f"Failed to create audio file: {audio_path}")

    actual_size = audio_path.stat().st_size
    if actual_size != len(data):
        audio_path.unlink()  # Remove corrupted file
        raise HTTPException(status_code=500, detail=f"Audio file size mismatch: expected {len(data)} bytes, got {actual_size} bytes")

    # Optional portrait
    if portrait and portrait.filename:
        img = await portrait.read()
        if img:
            portrait_name = _safe_name(portrait.filename)
            (user_dir / portrait_name).write_bytes(img)

    # verify write with more robust checking
    print(f"[DEBUG] Verifying audio file write to {audio_path}", flush=True)
    deadline = time.time() + 10
    verification_attempts = 0
    while time.time() < deadline:
        try:
            verification_attempts += 1
            if audio_path.exists():
                saved_size = audio_path.stat().st_size
                print(f"[DEBUG] Verification attempt {verification_attempts}: file exists, size={saved_size}, expected={len(data)}", flush=True)
                if saved_size == len(data) and saved_size > 0:
                    print(f"[DEBUG] Audio file verification successful", flush=True)
                    break
            else:
                print(f"[DEBUG] Verification attempt {verification_attempts}: file does not exist yet", flush=True)
        except FileNotFoundError as e:
            print(f"[DEBUG] Verification attempt {verification_attempts}: FileNotFoundError: {e}", flush=True)
        except Exception as e:
            print(f"[DEBUG] Verification attempt {verification_attempts}: Unexpected error: {e}", flush=True)
        time.sleep(0.1)

    # Final verification
    if not audio_path.exists():
        raise HTTPException(status_code=500, detail=f"Audio file was not created: {audio_path}")

    final_size = audio_path.stat().st_size
    if final_size != len(data):
        raise HTTPException(status_code=500, detail=f"Audio file size mismatch: expected {len(data)} bytes, got {final_size} bytes")

    if final_size == 0:
        raise HTTPException(status_code=500, detail="Audio file was created but is empty (0 bytes)")

    print(f"[web] saved upload -> {audio_path} ({final_size} bytes)", flush=True)
    try:
        dir_contents = [p.name for p in user_dir.glob('*')]
        print(f"[web] dir listing {user_dir}: {dir_contents}", flush=True)
        print(f"[web] audio file in directory: {audio_path.name in dir_contents}", flush=True)
        print(f"[web] audio path exists: {audio_path.exists()}", flush=True)
        print(f"[web] audio path size: {audio_path.stat().st_size if audio_path.exists() else 'N/A'}", flush=True)

        # Additional verification - check if file is actually readable
        try:
            with open(audio_path, 'rb') as f:
                chunk = f.read(1024)  # Read first 1KB
                print(f"[web] audio file readable, first chunk size: {len(chunk)} bytes", flush=True)
        except Exception as e:
            print(f"[web] ERROR: Cannot read audio file: {e}", flush=True)
            raise HTTPException(status_code=500, detail=f"Audio file not readable after save: {e}")
    except Exception as e:
        print(f"[web] ERROR: Cannot list directory: {e}", flush=True)

    # Enhanced debugging for audio file issues
    print(f"[DEBUG] ===== AUDIO FILE DEBUGGING =====", flush=True)
    print(f"[DEBUG] Original uploaded filename: {audio.filename}", flush=True)
    print(f"[DEBUG] Sanitized filename: {orig_audio_name}", flush=True)
    print(f"[DEBUG] Final audio path: {audio_path}", flush=True)
    print(f"[DEBUG] Audio file exists: {audio_path.exists()}", flush=True)

    if audio_path.exists():
        audio_size = audio_path.stat().st_size
        print(f"[DEBUG] Audio file size: {audio_size} bytes", flush=True)
        print(f"[DEBUG] Audio file path: {audio_path}", flush=True)
        print(f"[DEBUG] Audio file absolute path: {audio_path.absolute()}", flush=True)
        print(f"[DEBUG] Audio file parent directory: {audio_path.parent}", flush=True)
        print(f"[DEBUG] Audio file parent exists: {audio_path.parent.exists()}", flush=True)

        # Check if file is readable
        try:
            with open(audio_path, 'rb') as f:
                header = f.read(64)  # Read first 64 bytes
                print(f"[DEBUG] Audio file header (first 64 bytes): {header[:32]}...", flush=True)
                print(f"[DEBUG] Audio file is readable: True", flush=True)
        except Exception as e:
            print(f"[ERROR] Audio file not readable: {e}", flush=True)
    else:
        print(f"[ERROR] Audio file does not exist at path: {audio_path}", flush=True)

    print(f"[DEBUG] Job directory contents before enqueue: {[p.name for p in user_dir.glob('*')]}", flush=True)
    print(f"[DEBUG] Job directory path: {user_dir}", flush=True)
    print(f"[DEBUG] Upload path string that will be passed to worker: {str(audio_path)}", flush=True)

    # Check for hardcoded filename issue
    if 'input_cut.mp3' in str(audio_path):
        print(f"[WARNING] DETECTED HARDCODED FILENAME in upload path: {audio_path}", flush=True)
    else:
        print(f"[DEBUG] Upload path does not contain hardcoded filename", flush=True)

    print(f"[DEBUG] ================================", flush=True)

    rq_job = queue.enqueue(
        process_job,      # Function to execute
        job_id,           # First argument to process_job
        email,            # Second argument to process_job
        str(audio_path),  # Third argument to process_job
        style,            # Fourth argument to process_job
        retry=Retry(max=3, interval=[15, 30, 60]),
        job_timeout=5400,
    )

    print(f"[DEBUG] Job enqueued successfully with ID: {rq_job.id}", flush=True)

    # Store the RQ job ID mapping for status checks
    rq_job_mapping = job_dir / ".rq_job_id"
    rq_job_mapping.write_text(rq_job.id)
    print(f"[DEBUG] Stored RQ job ID mapping: {job_id} -> {rq_job.id}", flush=True)

    return TEMPLATES.TemplateResponse("upload.html", {"request": request, "job_id": job_id})


@app.get("/status/{job_id}")
async def status(job_id: str):
    """Get job status from file-based tracking (primary) and Redis (fallback for active jobs)"""
    from rq.job import Job

    # First, check if job directory exists
    job_dir = UPLOADS_DIR / job_id
    if not job_dir.exists():
        return {"state": "unknown", "error": f"Job directory not found: {job_id}"}

    # Check for completion file first (most reliable method)
    completion_file = job_dir / "completion_status.json"
    if completion_file.exists():
        try:
            with open(completion_file, 'r') as f:
                completion_data = json.load(f)
            # Map old state names to new ones for backward compatibility
            if completion_data.get("state") == "finished":
                completion_data["state"] = "done"
            return completion_data
        except Exception as file_error:
            print(f"[WARNING] Failed to read completion file for job {job_id}: {file_error}", flush=True)

    # If no completion file, try Redis for active jobs
    rq_job_id = None

    # First, try to get the actual RQ job ID from the mapping file
    rq_job_mapping = job_dir / ".rq_job_id"
    if rq_job_mapping.exists():
        try:
            rq_job_id = rq_job_mapping.read_text().strip()
            print(f"[DEBUG] Found RQ job ID mapping: {job_id} -> {rq_job_id}", flush=True)
        except Exception as e:
            print(f"[WARNING] Failed to read RQ job ID mapping: {e}", flush=True)

    # Try to fetch job using RQ job ID if available, otherwise use directory job_id
    lookup_job_id = rq_job_id if rq_job_id else job_id

    try:
        job = Job.fetch(lookup_job_id, connection=redis)
        meta = job.meta or {}

        if job.is_finished:
            # Job completed successfully - save to file for persistence
            _save_job_completion(job_id, "done", job.result)
            return {"state": "done", "result": job.result}
        elif job.is_failed:
            # Job failed - save to file for persistence
            _save_job_completion(job_id, "failed", {"error": str(job.exc_info)[:2000]})
            return {"state": "failed", "exc": str(job.exc_info)[:2000]}
        else:
            # Job still in progress - map RQ states to desired states
            rq_state = job.get_status(refresh=True)
            mapped_state = _map_rq_state_to_custom(rq_state)
            return {"state": mapped_state, "meta": meta}

    except Exception as e:
        # Job not found in Redis - this is normal for completed jobs that have been cleaned up
        print(f"[DEBUG] Job {lookup_job_id} not found in Redis (may be completed): {str(e)}", flush=True)

    # No completion file and no active job - job is likely queued or processing
    # Check if job directory has any content (indicates job was created)
    try:
        dir_contents = list(job_dir.iterdir())
        if dir_contents:
            # Job directory exists with content - job is likely queued or processing
            return {"state": "queue", "message": "Job is queued or processing"}
        else:
            # Empty job directory - job may have been cleaned up
            return {"state": "unknown", "error": f"Job directory exists but is empty: {job_id}"}
    except Exception as e:
        return {"state": "unknown", "error": f"Error checking job directory: {str(e)}"}


def _map_rq_state_to_custom(rq_state: str) -> str:
    """Map RQ job states to custom frontend states"""
    state_mapping = {
        "queued": "queue",
        "started": "active",
        "deferred": "queue",  # Treat deferred jobs as queued
        "processing": "processing",  # Custom state from worker
    }
    return state_mapping.get(rq_state, rq_state)  # Return original state if no mapping found


def _save_job_completion(job_id: str, state: str, result: dict) -> None:
    """Save job completion status to a file for persistence beyond Redis cleanup"""
    try:
        job_dir = UPLOADS_DIR / job_id
        completion_file = job_dir / "completion_status.json"

        completion_data = {
            "state": state,
            "result": result,
            "completed_at": time.time()
        }

        # Ensure directory exists
        completion_file.parent.mkdir(parents=True, exist_ok=True)

        # Write completion status
        with open(completion_file, 'w') as f:
            json.dump(completion_data, f, indent=2)

    except Exception as e:
        # Log error but don't fail the job
        print(f"[WARNING] Failed to save completion status for job {job_id}: {e}", flush=True)


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/health")
async def detailed_health():
    """Comprehensive health check for all dependencies"""
    import shutil

    health_status = {
        "status": "healthy",
        "checks": {},
        "timestamp": time.time()
    }

    # Check Redis connection
    try:
        redis.ping()
        health_status["checks"]["redis"] = {"status": "ok", "message": "Connected successfully"}
    except Exception as e:
        health_status["checks"]["redis"] = {"status": "error", "message": str(e)}
        health_status["status"] = "degraded"

    # Check required environment variables
    required_env_vars = ["OPENAI_API_KEY", "REPLICATE_API_TOKEN"]
    for var in required_env_vars:
        if os.getenv(var):
            health_status["checks"][var] = {"status": "ok", "message": "Set"}
        else:
            health_status["checks"][var] = {"status": "error", "message": "Missing"}
            health_status["status"] = "unhealthy"

    # Check B2 credentials
    b2_vars = ["B2_BUCKET_NAME", "B2_KEY_ID", "B2_APPLICATION_KEY"]
    b2_configured = all(os.getenv(var) for var in b2_vars)
    if b2_configured:
        health_status["checks"]["b2_storage"] = {"status": "ok", "message": "Configured"}
    else:
        health_status["checks"]["b2_storage"] = {"status": "warning", "message": "Not configured (optional)"}

    # Check disk space
    try:
        disk_usage = shutil.disk_usage(UPLOADS_DIR)
        free_gb = disk_usage.free / (1024**3)
        if free_gb < 1:  # Less than 1GB free
            health_status["checks"]["disk_space"] = {"status": "warning", "message": f"Low space: {free_gb:.1f}GB free"}
            health_status["status"] = "degraded"
        else:
            health_status["checks"]["disk_space"] = {"status": "ok", "message": f"{free_gb:.1f}GB free"}
    except Exception as e:
        health_status["checks"]["disk_space"] = {"status": "error", "message": str(e)}
        health_status["status"] = "degraded"

    # Check ffmpeg availability
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            health_status["checks"]["ffmpeg"] = {"status": "ok", "message": "Available"}
        else:
            health_status["checks"]["ffmpeg"] = {"status": "error", "message": "Not available"}
            health_status["status"] = "unhealthy"
    except FileNotFoundError:
        health_status["checks"]["ffmpeg"] = {"status": "error", "message": "Not found"}
        health_status["status"] = "unhealthy"

    return health_status


ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()


@app.get("/admin/prompts/{job_id}")
async def get_prompt_json(
    job_id: str,
    x_admin_key: str | None = Header(default=None),
    key: str | None = Query(default=None),
):
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=500, detail="ADMIN_TOKEN not set on server")
    supplied = x_admin_key or key
    if supplied != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    p = UPLOADS_DIR / job_id / "pipeline" / "scenes" / "prompt.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="prompt.json not found")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read prompt.json: {e}")
    return JSONResponse(data)


@app.get("/admin/script/{job_id}")
async def admin_download_script(
    job_id: str,
    key: str | None = Query(default=None),
    x_admin_key: str | None = Header(default=None),
):
    admin_token = os.getenv("ADMIN_TOKEN", "")
    if not admin_token or (key != admin_token and x_admin_key != admin_token):
        raise HTTPException(status_code=403, detail="Forbidden")

    path = UPLOADS_DIR / job_id / "pipeline" / "scripts" / "input_script.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="script not found")

    return FileResponse(path, media_type="application/json", filename=f"{job_id}_script.json")
