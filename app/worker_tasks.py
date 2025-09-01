# app/worker_tasks.py
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from .email_utils import send_link_email

load_dotenv()

APP_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SRC = APP_ROOT / "pipeline"   # must contain make_video.py
MEDIA_DIR = APP_ROOT / "media"         # served publicly by the web app
UPLOADS_DIR = APP_ROOT / "uploads"
BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8000")

REQUIRED_ENV = ["OPENAI_API_KEY", "REPLICATE_API_TOKEN"]


def log(msg: str) -> None:
    print(f"[worker] {msg}", flush=True)


def _copy_pipeline_to(job_dir: Path) -> None:
    """Copy the pipeline folder into the per-job workspace."""
    if not PIPELINE_SRC.exists():
        raise RuntimeError(f"Pipeline folder not found: {PIPELINE_SRC}")
    target = job_dir / "pipeline"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(PIPELINE_SRC, target)


def _run_make_video(job_dir: Path, audio_src: Path) -> Path:
    """
    Run make_video.py inside the job workspace.
    Returns the path to final_video.mp4.
    """
    pipe_dir = job_dir / "pipeline"
    audio_input_dir = pipe_dir / "audio_input"
    audio_input_dir.mkdir(parents=True, exist_ok=True)

    # Put the uploaded audio where the pipeline expects it.
    # NOTE: Pipeline expects input.mp3. For now we assume the upload is MP3.
    target_audio = audio_input_dir / "input.mp3"
    shutil.copy2(audio_src, target_audio)

    # Ensure required env vars exist for the pipeline
    env = os.environ.copy()
    for key in REQUIRED_ENV:
        if not env.get(key):
            raise RuntimeError(f"Missing required env: {key}")

    # Run the pipeline
    cmd = [sys.executable, "make_video.py", "--job-id", job_dir.name]
    log(f"RUN (cwd={pipe_dir}): {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(pipe_dir), check=True, env=env)

    final_video = pipe_dir / "final_video.mp4"
    if not final_video.exists():
        raise RuntimeError("final_video.mp4 not produced by pipeline")
    return final_video


def process_job(job_id: str, email: str, upload_path: str) -> Dict[str, str]:
    """
    Background task executed by RQ worker (or by the combined web+worker service).
    """
    # Quick fake mode for testing without the heavy pipeline:
    if os.getenv("DEV_FAKE_PIPELINE", "0") == "1":
        MEDIA_DIR.mkdir(parents=True, exist_ok=True)
        placeholder = APP_ROOT / "app" / "static" / "sample.mp4"
        if not placeholder.exists():
            # create a tiny placeholder file if missing
            placeholder.parent.mkdir(parents=True, exist_ok=True)
            with open(placeholder, "wb") as f:
                f.write(b"\x00")
        out = MEDIA_DIR / f"{job_id}.mp4"
        shutil.copy2(placeholder, out)
        video_url = f"{BASE_URL}/media/{job_id}.mp4"
        send_link_email(email, video_url, job_id)
        return {"status": "done", "video_url": video_url}

    # Real pipeline
    job_dir = UPLOADS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    src_audio = Path(upload_path)
    if not src_audio.exists():
        raise RuntimeError("Uploaded audio missing")

    _copy_pipeline_to(job_dir)
    final_video = _run_make_video(job_dir, src_audio)

    # Move final video to public /media and email the link
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    public_path = MEDIA_DIR / f"{job_id}.mp4"
    shutil.copy2(final_video, public_path)

    video_url = f"{BASE_URL}/media/{job_id}.mp4"
    send_link_email(email, video_url, job_id)

    return {"status": "done", "video_url": video_url}
