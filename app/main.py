# app/main.py
import os
import time
import uuid
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
queue = Queue(QUEUE_NAME, connection=redis, default_timeout=1800)

JOB_STATUS = {}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return TEMPLATES.TemplateResponse("upload.html", {"request": request})


def _safe_name(name: str) -> str:
    keep = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    cleaned = "".join(ch if ch in keep else "_" for ch in (name or "upload"))
    if not cleaned or cleaned.startswith("."):
        cleaned = f"file_{uuid.uuid4().hex}"
    return cleaned[:128]


_ALLOWED_STYLES = {"anime", "3d", "kid", "storybook", "fantasy", "japanese_kawaii", "claymation", "watercolor", "pixel_art", "paper_cutout", "van_gogh", "felt_needle", "stop_motion_felt_clay"}


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
    user_dir.mkdir(parents=True, exist_ok=True)

    # Save audio with original (sanitized) name
    orig_audio_name = _safe_name(audio.filename or "audio_upload")
    audio_path = user_dir / orig_audio_name
    with open(audio_path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

    # Optional portrait
    if portrait and portrait.filename:
        img = await portrait.read()
        if img:
            portrait_name = _safe_name(portrait.filename)
            (user_dir / portrait_name).write_bytes(img)

    # verify write
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            if audio_path.exists() and audio_path.stat().st_size == len(data):
                break
        except FileNotFoundError:
            pass
        time.sleep(0.1)
    if not (audio_path.exists() and audio_path.stat().st_size == len(data)):
        raise HTTPException(status_code=500, detail="Upload write verification failed")

    print(f"[web] saved upload -> {audio_path} ({len(data)} bytes)", flush=True)
    try:
        print(f"[web] dir listing {user_dir}: {[p.name for p in user_dir.glob('*')]}", flush=True)
    except Exception:
        pass

    # Enqueue with style forwarded to worker
    from app.worker_tasks import process_job
    rq_job = queue.enqueue(
        process_job,
        job_id,
        email,
        str(audio_path),
        style,  # <-- pass style through
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
