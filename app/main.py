# app/main.py
import os
import time
import uuid
import json
from pathlib import Path

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
from rq.job import Job
from rq.exceptions import NoSuchJobError

load_dotenv()

APP_ROOT = Path(__file__).resolve().parents[1]
# On Render, write to /tmp (ephemeral but fast/writable)
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

# simple in-memory status (MVP); note: this is volatile across restarts
JOB_STATUS = {}

# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return TEMPLATES.TemplateResponse("upload.html", {"request": request})


@app.post("/submit", response_class=HTMLResponse)
async def submit(
    request: Request,
    email: str = Form(...),
    audio: UploadFile = File(...),
    portrait: UploadFile | None = File(default=None),
):
    # Validate audio type (worker will convert non-MP3 to MP3)
    allowed_audio_types = {
        "audio/mpeg",
        "audio/wav", "audio/x-wav",
        "audio/m4a", "audio/x-m4a",
        "audio/mp4",
        "audio/aac",
        "audio/ogg",
    }
    if audio.content_type not in allowed_audio_types:
        raise HTTPException(status_code=400, detail=f"Unsupported audio type: {audio.content_type}")

    # Read + size guard (50 MB)
    audio_bytes = await audio.read()
    if len(audio_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Audio too large (max 50MB)")

    job_id = str(uuid.uuid4())
    user_dir = UPLOADS_DIR / job_id
    user_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Save audio
    # -----------------------------
    orig_audio_name = (audio.filename or "").lower()
    audio_ext = Path(orig_audio_name).suffix or ".mp3"
    if audio_ext not in {".mp3", ".wav", ".m4a", ".aac", ".ogg"}:
        audio_ext = ".mp3"
    audio_path = user_dir / f"input{audio_ext}"

    with open(audio_path, "wb") as f:
        f.write(audio_bytes)
        f.flush()
        os.fsync(f.fileno())

    # Verify write
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            if audio_path.exists() and audio_path.stat().st_size == len(audio_bytes):
                break
        except FileNotFoundError:
            pass
        time.sleep(0.1)

    if not (audio_path.exists() and audio_path.stat().st_size == len(audio_bytes)):
        raise HTTPException(status_code=500, detail="Upload write verification failed (audio)")

    # -----------------------------
    # Save optional portrait (if provided)
    # -----------------------------
    if portrait and portrait.filename:
        portrait_bytes = await portrait.read()
        if len(portrait_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Portrait too large (max 10MB)")
        # Accept common image types; we'll keep original ext if present
        portrait_name = (portrait.filename or "").lower()
        p_ext = Path(portrait_name).suffix or ".jpg"
        if p_ext not in {".jpg", ".jpeg", ".png", ".webp"}:
            p_ext = ".jpg"
        portrait_path = user_dir / f"portrait{p_ext}"
        with open(portrait_path, "wb") as f:
            f.write(portrait_bytes)
            f.flush()
            os.fsync(f.fileno())

    # Debug
    try:
        print(f"[web] saved upload -> {audio_path} ({len(audio_bytes)} bytes)", flush=True)
        print(f"[web] dir listing {user_dir}: {[p.name for p in user_dir.glob('*')]}", flush=True)
    except Exception:
        pass

    # -----------------------------
    # Enqueue job (with retries + longer timeout)
    # -----------------------------
    from app.worker_tasks import process_job  # import here so RQ can pickle
    rq_job = queue.enqueue(
        process_job,
        job_id,
        email,
        str(audio_path),
        retry=Retry(max=3, interval=[15, 30, 60]),
        job_timeout=1800,  # 30 minutes
    )
    JOB_STATUS[job_id] = {"state": "queued", "rq_id": rq_job.get_id()}

    # NEW: persist the mapping in Redis so /status works after restarts
    try:
        redis.set(f"jobmap:{job_id}", rq_job.get_id(), ex=24 * 3600)  # 24h TTL
    except Exception:
        pass

    return TEMPLATES.TemplateResponse("upload.html", {"request": request, "job_id": job_id})


@app.get("/status/{job_id}")
async def status(job_id: str):
    """
    Robust status lookup:
      1) Try in-memory JOB_STATUS (works before restart).
      2) Fall back to Redis mapping jobmap:{job_id} -> rq_id (survives restart).
      3) As last resort, treat job_id as the RQ id itself.
    """
    # 1) In-memory
    rq_id = JOB_STATUS.get(job_id, {}).get("rq_id")

    # 2) Redis mapping
    if not rq_id:
        try:
            val = redis.get(f"jobmap:{job_id}")
            if val:
                rq_id = val.decode("utf-8")
        except Exception:
            rq_id = None

    # 3) Last resort: caller passed actual RQ job id
    if not rq_id:
        rq_id = job_id

    # 4) Query RQ
    try:
        job = Job.fetch(rq_id, connection=redis)
    except NoSuchJobError:
        return JSONResponse({"error": "unknown job id"}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": f"status lookup failed: {e}"}, status_code=500)

    status = job.get_status(refresh=True)
    if job.is_finished:
        return {"state": "finished", "result": job.result}
    if job.is_failed:
        return {"state": "failed", "exc": (job.exc_info or "")[:2000]}
    return {"state": status, "meta": job.meta or {}}


@app.get("/healthz")
async def healthz():
    return {"ok": True}


# -----------------------------
# Admin endpoints (protected)
# -----------------------------
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()


@app.get("/admin/prompts/{job_id}")
async def admin_get_prompt_json(
    job_id: str,
    x_admin_key: str | None = Header(default=None),  # header: X-Admin-Key: <token>
    key: str | None = Query(default=None),           # or query: ?key=<token>
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
    key: str | None = Query(default=None),          # ?key=ADMIN_TOKEN
    x_admin_key: str | None = Header(default=None), # or header: X-Admin-Key: ADMIN_TOKEN
):
    if not ADMIN_TOKEN or (key != ADMIN_TOKEN and x_admin_key != ADMIN_TOKEN):
        raise HTTPException(status_code=403, detail="Forbidden")

    path = UPLOADS_DIR / job_id / "pipeline" / "scripts" / "input_script.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="script not found")

    # Force a download with a nice filename
    return FileResponse(
        path,
        media_type="application/json",
        filename=f"{job_id}_script.json",
    )
