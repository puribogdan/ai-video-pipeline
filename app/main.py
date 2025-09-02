# ... inside submit(...)
data = await audio.read()
if len(data) > 50 * 1024 * 1024:
    raise HTTPException(status_code=400, detail="File too large (max 50MB)")

job_id = str(uuid.uuid4())
user_dir = UPLOADS_DIR / job_id
user_dir.mkdir(parents=True, exist_ok=True)

# Pick extension from the real filename; fall back to .mp3
orig_name = (audio.filename or "").lower()
from pathlib import Path as _P
ext = _P(orig_name).suffix or ""
if ext not in {".mp3", ".wav", ".m4a", ".aac", ".ogg"}:
    ext = ".mp3"

upload_path = user_dir / f"input{ext}"
with open(upload_path, "wb") as f:
    f.write(data)

# Log for debugging
print(f"[web] saved upload -> {upload_path} ({len(data)} bytes)", flush=True)

from rq import Retry
from app.worker_tasks import process_job  # import here so RQ can pickle

rq_job = queue.enqueue(
    process_job,
    job_id,
    email,
    str(upload_path),
    retry=Retry(max=3, interval=[15, 30, 60])  # auto-retry on transient failures
)
JOB_STATUS[job_id] = {"state": "queued", "rq_id": rq_job.get_id()}

return TEMPLATES.TemplateResponse("upload.html", {"request": request, "job_id": job_id})
