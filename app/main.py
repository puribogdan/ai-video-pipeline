# app/main.py
import os
import time
import uuid
from pathlib import Path
import json
from fastapi import Header

from fastapi import FastAPI, Request, UploadFile, Form, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from redis import Redis
from rq import Queue, Retry

load_dotenv()

APP_ROOT = Path(__file__).resolve().parents[1]
# Store uploads in /tmp on Render (fast + writable)
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", "/tmp/uploads"))
MEDIA_DIR = APP_ROOT / "media"
TEMPLATES = Jinja2Templates(directory=str(APP_ROOT / "app" / "templates"))

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Ensure required dirs exist
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = APP_ROOT / "app" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Redis + RQ (30 min default timeout)
redis = Redis.from_url(REDIS_URL)
queue = Queue("video-jobs", connection=redis, default_timeout=1800)

# simple in-memory status (MVP)
JOB_STATUS = {}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return TEMPLATES.TemplateResponse("upload.html", {"request": request})


@app.post("/submit", response_class=HTMLResponse)
async def submit(
    request: Request,
    email: str = Form(...),
    audio: UploadFile = File(...),
):
    # Validate type (worker will convert non-MP3 to MP3)
    allowed_types = {
        "audio/mpeg",
        "audio/wav", "audio/x-wav",
        "audio/m4a", "audio/x-m4a",
        "audio/mp4",
        "audio/aac",
        "audio/ogg",
    }
    if audio.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {audio.content_type}")

    # Read file and size-check (50 MB)
    data = await audio.read()
    if len(data) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")

    # Per-job folder
    job_id = str(uuid.uuid4())
    user_dir = UPLOADS_DIR / job_id
    user_dir.mkdir(parents=True, exist_ok=True)

    # Choose extension from original filename; default to .mp3
    orig_name = (audio.filename or "").lower()
    ext = Path(orig_name).suffix or ".mp3"
    if ext not in {".mp3", ".wav", ".m4a", ".aac", ".ogg"}:
        ext = ".mp3"

    upload_path = user_dir / f"input{ext}"

    # Write + flush + fsync and verify
    with open(upload_path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            if upload_path.exists() and upload_path.stat().st_size == len(data):
                break
        except FileNotFoundError:
            pass
        time.sleep(0.1)

    if not (upload_path.exists() and upload_path.stat().st_size == len(data)):
        raise HTTPException(status_code=500, detail="Upload write verification failed")

    print(f"[web] saved upload -> {upload_path} ({len(data)} bytes)", flush=True)
    try:
        print(f"[web] dir listing {user_dir}: {[p.name for p in user_dir.glob('*')]}", flush=True)
    except Exception:
        pass

    # Enqueue with retries and longer timeout
    from app.worker_tasks import process_job  # import here so RQ can pickle
    rq_job = queue.enqueue(
        process_job,
        job_id,
        email,
        str(upload_path),
        retry=Retry(max=3, interval=[15, 30, 60]),
        job_timeout=1800,  # allow up to 30 minutes
    )
    JOB_STATUS[job_id] = {"state": "queued", "rq_id": rq_job.get_id()}

    return TEMPLATES.TemplateResponse("upload.html", {"request": request, "job_id": job_id})


@app.get("/status/{job_id}")
async def status(job_id: str):
    info = JOB_STATUS.get(job_id)
    if not info:
        return JSONResponse({"error": "unknown job id"}, status_code=404)

    from rq.job import Job
    try:
        job = Job.fetch(info["rq_id"], connection=redis)
        meta = job.meta or {}
        if job.is_finished:
            return {"state": "finished", "result": job.result}
        if job.is_failed:
            return {"state": "failed", "exc": str(job.exc_info)[:2000]}
        return {"state": job.get_status(refresh=True), "meta": meta}
    except Exception as e:
        return {"state": info.get("state", "unknown"), "error": str(e)}


@app.get("/healthz")
async def healthz():
    return {"ok": True}


ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()

@app.get("/admin/prompts/{job_id}")
async def get_prompt_json(
    job_id: str,
    x_admin_key: str | None = Header(default=None),  # send header: X-Admin-Key: <token>
    key: str | None = Query(default=None),           # OR use ?key=<token> in the URL
):
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=500, detail="ADMIN_TOKEN not set on server")

    supplied = x_admin_key or key
    if supplied != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    p = UPLOADS_DIR / job_id / "pipeline" / "scenes" / "prompt.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="prompt.json not found (job may not have run yet, or the instance restarted)")

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read prompt.json: {e}")

    return JSONResponse(data)