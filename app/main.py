
import os
import uuid
from pathlib import Path


from fastapi import FastAPI, Request, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from redis import Redis
from rq import Queue


load_dotenv()


APP_ROOT = Path(__file__).resolve().parents[1]
UPLOADS_DIR = APP_ROOT / "uploads"
MEDIA_DIR = APP_ROOT / "media"
TEMPLATES = Jinja2Templates(directory=str(APP_ROOT / "app" / "templates"))


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


app = FastAPI()
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")
app.mount("/static", StaticFiles(directory=str(APP_ROOT / "app" / "static")), name="static")


# RQ setup
redis = Redis.from_url(REDIS_URL)
queue = Queue("video-jobs", connection=redis)


# Track simple in-memory status (for MVP). In prod, use RQ's job meta or a DB.
JOB_STATUS = {}




@app.get("/", response_class=HTMLResponse)
def index(request: Request):
return TEMPLATES.TemplateResponse("upload.html", {"request": request})




@app.post("/submit", response_class=HTMLResponse)
def submit(request: Request, email: str = Form(...), audio: UploadFile = Form(...)):
if audio.content_type not in {"audio/mpeg", "audio/wav"}:
raise HTTPException(status_code=400, detail="Unsupported file type. Use MP3 or WAV.")
# Max ~50MB (simple check)
if audio.size and audio.size > 50 * 1024 * 1024:
raise HTTPException(status_code=400, detail="File too large (max 50MB)")


job_id = str(uuid.uuid4())
user_dir = UPLOADS_DIR / job_id
user_dir.mkdir(parents=True, exist_ok=True)


# Save uploaded file
ext = ".mp3" if audio.content_type == "audio/mpeg" else ".wav"
upload_path = user_dir / f"input{ext}"


with open(upload_path, "wb") as f:
f.write(audio.file.read())


# Enqueue job
from app.worker_tasks import process_job # local import for RQ pickling
rq_job = queue.enqueue(process_job, job_id, email, str(upload_path))


JOB_STATUS[job_id] = {"state": "queued", "rq_id": rq_job.get_id()}


return TEMPLATES.TemplateResponse(
"upload.html",
{"request": request, "job_id": job_id},
)




@app.get("/status/{job_id}")
def status(job_id: str):
info = JOB_STATUS.get(job_id)
if not info:
return JSONResponse({"error": "unknown job id"}, status_code=404)


# Try to read RQ job state
from rq.job import Job
try:
job = Job.fetch(info["rq_id"], connection=redis)
meta = job.meta or {}
if job.is_finished:
return {"state": "finished", "result": job.result}
if job.is_failed:
return {"state": "failed", "exc": str(job.exc_info)[:500]}
return {"state": job.get_status(refresh=True), "meta": meta}
except Exception:
return info