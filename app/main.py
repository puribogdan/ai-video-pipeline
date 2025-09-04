# app/main.py
import os
import time
import uuid
import json
from pathlib import Path
from typing import Optional

from fastapi import (
    FastAPI,
    Request,
    UploadFile,
    Form,
    File,
    HTTPException,
    Header,
    Query,
)
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from dotenv import load_dotenv
from redis import Redis
from rq import Queue, Retry

load_dotenv()

# --- Paths & setup ---
APP_ROOT = Path(__file__).resolve().parents[1]
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", "/tmp/uploads"))  # Render-safe, writable
MEDIA_DIR = APP_ROOT / "media"
TEMPLATES = Jinja2Templates(directory=str(APP_ROOT / "app" / "templates"))
STATIC_DIR = APP_ROOT / "app" / "static"

MEDIA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# --- Redis / RQ ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis = Redis.from_url(REDIS_URL)
queue = Queue("video-jobs", connection=redis, default_timeout=1800)  # 30 min

# Simple in-memory tracker (MVP)
JOB_STATUS: dict[str, dict] = {}

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return TEMPLATES.TemplateResponse("upload.html", {"request": request})


@app.post("/submit", response_class=HTMLResponse)
async def submit(
    request: Request,
    email: str = Form(...),
    audio: UploadFile = File(...),
    portrait: Optional[UploadFile] = File(None),  # optional user portrait
):
    # Validate audio type
    allowed_audio = {
        "audio/mpeg", "audio/wav", "audio/x-wav",
        "audio/m4a", "audio/x-m4a", "audio/mp4",
        "audio/aac", "audio/ogg",
    }
    if audio.content_type not in allowed_audio:
        raise HTTPException(status_code=400, detail=f"Unsupported audio type: {audio.content_type}")

    audio_bytes = await audio.read()
    if len(audio_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Audio too large (max 50MB)")

    # Validate portrait (optional)
    portrait_bytes: Optional[bytes] = None
    portrait_path: Optional[Path] = None
    if portrait is not None and portrait.filename:
        allowed_img = {"image/jpeg", "image/png"}
        if portrait.content_type not in allowed_img:
            raise HTTPException(status_code=400, detail=f"Unsupported portrait type: {portrait.content_type}")
        portrait_bytes = await portrait.read()
        if len(portrait_bytes) > 8 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Portrait too large (max 8MB)")

    # Per-job folder
    job_id = str(uuid.uuid4())
    user_dir = UPLOADS_DIR / job_id
    user_dir.mkdir(parents=True, exist_ok=True)

    # Pick extension from original audio; default to .mp3
    ext = (Path(audio.filename or "").suffix or ".mp3").lower()
    if ext not in {".mp3", ".wav", ".m4a", ".aac", ".ogg"}:
        ext = ".mp3"
    upload_path = user_dir / f"input{ext}"

    # Write audio to disk + verify
    with open(upload_path, "wb") as f:
        f.write(audio_bytes)
        f.flush()
        os.fsync(f.fileno())

    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            if upload_path.exists() and upload_path.stat().st_size == len(audio_bytes):
                break
        except FileNotFoundError:
            pass
        time.sleep(0.1)
    if not (upload_path.exists() and upload_path.stat().st_size == len(audio_bytes)):
        raise HTTPException(status_code=500, detail="Upload write verification failed (audio)")

    # Write portrait if present
    if portrait_bytes:
        pext = (Path(portrait.filename or "").suffix or ".jpg").lower()
        if pext not in {".jpg", ".jpeg", ".png"}:
            pext = ".jpg"
        portrait_path = user_dir / f"portrait{pext}"
        with open(portrait_path, "wb") as f:
            f.write(portrait_bytes)
            f.flush()
            os.fsync(f.fileno())

    print(f"[web] saved audio -> {upload_path} ({len(audio_bytes)} bytes)", flush=True)
    if portrait_path:
        print(f"[web] saved portrait -> {portrait_path} ({len(portrait_bytes or b'') } bytes)", flush=True)
    try:
        print(f"[web] dir listing {user_dir}: {[p.name for p in user_dir.glob('*')]}", flush=True)
    except Exception:
        pass

    # Enqueue job (with retries + longer timeout)
    from app.worker_tasks import process_job  # late import to avoid pickle/import issues
    rq_job = queue.enqueue(
        process_job,
        job_id,
        email,
        str(upload_path),
        str(portrait_path) if portrait_path else None,  # pass portrait or None
        retry=Retry(max=3, interval=[15, 30, 60]),
        job_timeout=1800,
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

# --- Admin-only endpoints for prompt & script ---

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()

@app.get("/admin/prompts/{job_id}")
async def admin_get_prompt_json(
    job_id: str,
    x_admin_key: Optional[str] = Header(default=None),  # header: X-Admin-Key: <token>
    key: Optional[str] = Query(default=None),           # or query: ?key=<token>
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


@app.get("/admin/script/{job_id}")
async def admin_download_script(
    job_id: str,
    key: Optional[str] = Query(default=None),          # ?key=ADMIN_TOKEN
    x_admin_key: Optional[str] = Header(default=None), # or header: X-Admin-Key: ADMIN_TOKEN
):
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=500, detail="ADMIN_TOKEN not set on server")
    supplied = x_admin_key or key
    if supplied != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    path = UPLOADS_DIR / job_id / "pipeline" / "scripts" / "input_script.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="script not found")

    # force a download with a nice filename
    return FileResponse(
        path,
        media_type="application/json",
        filename=f"{job_id}_script.json",
    )
