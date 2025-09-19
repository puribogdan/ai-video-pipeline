# app/worker_tasks.py
import os
import shutil
import subprocess
import sys
import mimetypes
import time
from pathlib import Path
from typing import Dict, Optional
from collections import deque

from dotenv import load_dotenv
from .email_utils import send_link_email

# Google Drive imports
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from googleapiclient.http import MediaFileUpload

import json
import tempfile

load_dotenv()

APP_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SRC = APP_ROOT / "pipeline"
MEDIA_DIR = APP_ROOT / "media"
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", "/tmp/uploads"))
BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8000")

REQUIRED_ENV = ["OPENAI_API_KEY", "REPLICATE_API_TOKEN"]


def log(msg: str) -> None:
    print(f"[worker] {msg}", flush=True)


def _copy_pipeline_to(job_dir: Path) -> None:
    if not PIPELINE_SRC.exists():
        raise RuntimeError(f"Pipeline folder not found: {PIPELINE_SRC}")
    target = job_dir / "pipeline"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(PIPELINE_SRC, target)


def ensure_mp3(src_path: Path) -> Path:
    mt, _ = mimetypes.guess_type(str(src_path))
    is_mp3 = (src_path.suffix.lower() == ".mp3") or (mt == "audio/mpeg")
    if is_mp3:
        return src_path
    out_path = src_path.with_suffix(".mp3")
    cmd = ["ffmpeg", "-y", "-i", str(src_path), "-vn", "-acodec", "libmp3lame", "-b:a", "192k", "-ar", "44100", "-ac", "2", str(out_path)]
    log("Converting to MP3: " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_path


def run_with_live_output(cmd: list[str], cwd: Path, env: dict) -> None:
    log(f"RUN (cwd={cwd}): {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, cwd=str(cwd), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    assert proc.stdout is not None
    tail = deque(maxlen=200)
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip())
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError("make_video.py failed (exit %d)\n--- tail ---\n%s" % (ret, "\n".join(tail)))


def _listdir_safe(p: Path) -> list[str]:
    try:
        return sorted([x.name for x in p.iterdir()])
    except Exception:
        return []


def _find_any_audio(job_dir: Path) -> Optional[Path]:
    audio_exts = (".mp3", ".wav", ".m4a", ".aac", ".ogg")
    for ext in audio_exts:
        for f in sorted(job_dir.glob(f"*{ext}")):
            if f.is_file() and f.stat().st_size > 0:
                return f
    for f in sorted(job_dir.glob("*")):
        if f.is_file():
            mt, _ = mimetypes.guess_type(str(f))
            if mt and mt.startswith("audio/") and f.stat().st_size > 0:
                return f
    return None


def _wait_for_any_audio(job_id: str, hint_path: Optional[Path], timeout_s: float = 180.0) -> Path:
    t0 = time.time()
    job_dir = UPLOADS_DIR / job_id
    log(f"wait_for_audio: job_dir={job_dir}, hint={hint_path}")
    while time.time() - t0 < timeout_s:
        if hint_path and hint_path.exists() and hint_path.stat().st_size > 0:
            log(f"wait_for_audio: found hint {hint_path} ({hint_path.stat().st_size} bytes)")
            return hint_path
        found = _find_any_audio(job_dir) if job_dir.exists() else None
        if found:
            log(f"wait_for_audio: discovered audio {found} ({found.stat().st_size} bytes)")
            return found
        if int(time.time() - t0) % 3 == 0:
            log(f"wait_for_audio: waiting… listing={_listdir_safe(job_dir)}")
        time.sleep(0.25)
    raise RuntimeError(f"Audio not found in {job_dir}. Listing: {_listdir_safe(job_dir)}")


def _find_portrait_file(job_dir: Path) -> Optional[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    for p in sorted(job_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts and p.stat().st_size > 0:
            log(f"portrait pick (ext): {p.resolve()} ({p.stat().st_size} bytes)")
            return p
    for p in sorted(job_dir.iterdir()):
        if not p.is_file():
            continue
        mt, _ = mimetypes.guess_type(str(p))
        if mt and mt.startswith("image/") and p.stat().st_size > 0:
            log(f"portrait pick (mime): {p.resolve()} ({p.stat().st_size} bytes)")
            return p
    return None


def _run_make_video(job_dir: Path, hint_audio: Optional[Path], style: str) -> Path:
    audio_src = _wait_for_any_audio(job_id=job_dir.name, hint_path=hint_audio)

    _copy_pipeline_to(job_dir)
    pipe_dir = job_dir / "pipeline"
    audio_input_dir = pipe_dir / "audio_input"
    audio_input_dir.mkdir(parents=True, exist_ok=True)

    audio_for_pipeline = ensure_mp3(audio_src)
    target_audio = audio_input_dir / "input.mp3"
    shutil.copy2(audio_for_pipeline, target_audio)
    log(f"Staged audio -> {target_audio} ({target_audio.stat().st_size} bytes)")

    env = os.environ.copy()
    for key in REQUIRED_ENV:
        if not env.get(key):
            raise RuntimeError(f"Missing required env: {key}")

    portrait = _find_portrait_file(job_dir)
    if portrait:
        env["PORTRAIT_PATH"] = str(portrait)
        log(f"Staging portrait via PORTRAIT_PATH={portrait}")

    # Forward selected style
    env["STYLE_CHOICE"] = style  # anime | 3d | kid | storybook | fantasy

    trim_extra = os.getenv("TRIM_EXTRA_ARGS", "").strip()
    if trim_extra:
        env["TRIM_EXTRA_ARGS"] = trim_extra

    cmd = [sys.executable, "make_video.py", "--job-id", job_dir.name]
    run_with_live_output(cmd, cwd=pipe_dir, env=env)

    final_video = pipe_dir / "final_video.mp4"
    if not final_video.exists():
        raise RuntimeError("final_video.mp4 not produced by pipeline")
    return final_video


def upload_to_drive(job_id: str, video_path: Path) -> Optional[str]:
    """Upload video to Google Drive and return shareable URL, or None on failure."""
    folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    if not folder_id:
        log("GOOGLE_DRIVE_FOLDER_ID env var not set; skipping upload.")
        return None

    key_path = APP_ROOT / "service-account-key.json"
    if not key_path.exists():
        log(f"Service account key not found at {key_path}; skipping upload.")
        return None

    try:
        creds = Credentials.from_service_account_file(str(key_path))
        service = build("drive", "v3", credentials=creds)

        file_metadata = {
            "name": f"{job_id}.mp4",
            "parents": [folder_id],
        }
        media = MediaFileUpload(str(video_path), mimetype="video/mp4", resumable=True)
        uploaded_file = (
            service.files()
            .create(body=file_metadata, media_body=media, fields="id")
            .execute()
        )

        # Get shareable link
        file_id = uploaded_file["id"]
        video_url = f"https://drive.google.com/file/d/{file_id}/view"

        # Set public view permission
        permission = {"type": "anyone", "role": "reader"}
        service.permissions().create(
            fileId=file_id, body=permission
        ).execute()

        log(f"Uploaded to Drive: {video_url}")
        return video_url

    except Exception as e:
        log(f"Drive upload failed: {e}; falling back to local URL.")
        # Clean up if temp file exists
        if 'tmp_key_path' in locals():
            try:
                os.unlink(tmp_key_path)
            except:
                pass
        return None


def process_job(job_id: str, email: str, upload_path: str, style: str) -> Dict[str, str]:
    if os.getenv("DEV_FAKE_PIPELINE", "0") == "1":
        MEDIA_DIR.mkdir(parents=True, exist_ok=True)
        placeholder = APP_ROOT / "app" / "static" / "sample.mp4"
        if not placeholder.exists():
            placeholder.parent.mkdir(parents=True, exist_ok=True)
            with open(placeholder, "wb") as f:
                f.write(b"\x00")
        out = MEDIA_DIR / f"{job_id}.mp4"
        shutil.copy2(placeholder, out)
        video_url = f"{BASE_URL}/media/{job_id}.mp4"
        send_link_email(email, video_url, job_id)
        return {"status": "done", "video_url": video_url}

    job_dir = UPLOADS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    hint_audio = Path(upload_path) if upload_path else None
    log(f"job dir initial listing {job_dir}: {sorted([p.name for p in job_dir.iterdir()]) if job_dir.exists() else []}")

    final_video = _run_make_video(job_dir, hint_audio, style)

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    public_path = MEDIA_DIR / f"{job_id}.mp4"
    shutil.copy2(final_video, public_path)

    # Upload to Google Drive (fallback to local if fails)
    drive_url = upload_to_drive(job_id, public_path)
    video_url = drive_url or f"{BASE_URL}/media/{job_id}.mp4"
    send_link_email(email, video_url, job_id)
    return {"status": "done", "video_url": video_url}
