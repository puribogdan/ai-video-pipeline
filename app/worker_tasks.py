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

load_dotenv()

APP_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SRC = APP_ROOT / "pipeline"
MEDIA_DIR = APP_ROOT / "media"
# ⚠️ Read uploads from /tmp (matches main.py)
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
    """If src is MP3, return it. Otherwise convert to MP3 with ffmpeg and return new path."""
    mt, _ = mimetypes.guess_type(str(src_path))
    is_mp3 = (src_path.suffix.lower() == ".mp3") or (mt == "audio/mpeg")
    if is_mp3:
        return src_path

    out_path = src_path.with_suffix(".mp3")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src_path),
        "-vn",
        "-acodec", "libmp3lame",
        "-b:a", "192k",
        "-ar", "44100",
        "-ac", "2",
        str(out_path),
    ]
    log("Converting to MP3: " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_path


def run_with_live_output(cmd: list[str], cwd: Path, env: dict) -> None:
    """Run a subprocess, stream stdout to logs, and keep a tail buffer for errors."""
    log(f"RUN (cwd={cwd}): {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    tail = deque(maxlen=200)  # keep last 200 lines
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip())
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(
            "make_video.py failed (exit %d)\n--- tail ---\n%s" %
            (ret, "\n".join(tail))
        )


def _wait_for_upload(job_id: str, upload_path: Path, timeout_s: float = 30.0) -> Path:
    """Wait for the uploaded file to be visible; fall back to globbing input.* in job dir."""
    t0 = time.time()
    job_dir = UPLOADS_DIR / job_id
    last_listing: Optional[list[str]] = None

    while time.time() - t0 < timeout_s:
        if upload_path.exists():
            return upload_path
        if job_dir.exists():
            candidates = sorted(job_dir.glob("input.*"))
            if candidates:
                return candidates[0]
            last_listing = [p.name for p in job_dir.glob("*")]
        time.sleep(0.2)

    listing = last_listing if last_listing is not None else (
        [p.name for p in job_dir.glob("*")] if job_dir.exists() else []
    )
    raise RuntimeError(f"Uploaded audio missing. Expected {upload_path}. Job dir listing: {listing}")


def _find_portrait_file(job_dir: Path) -> Optional[Path]:
    """
    Return the first plausible portrait image in the job dir, regardless of filename.
    We skip the audio file and the copied pipeline folder.
    """
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    for p in sorted(job_dir.iterdir()):
        if not p.is_file():
            continue
        if p.name.startswith("input."):   # skip the uploaded audio (input.mp3 / input.wav etc)
            continue
        if p.suffix.lower() in exts and p.stat().st_size > 0:
            return p

    portrait = _find_portrait_file(job_dir)
    if portrait:
        log(f"✅ Found portrait: {portrait.resolve()} (size={portrait.stat().st_size} bytes)")



    # Fallback: check MIME if extensions are odd
    import mimetypes
    for p in sorted(job_dir.iterdir()):
        if not p.is_file():
            continue
        if p.name.startswith("input."):
            continue
        mt, _ = mimetypes.guess_type(str(p))
        if mt and mt.startswith("image/") and p.stat().st_size > 0:
            return p

    return None



def _run_make_video(job_dir: Path, audio_src: Path) -> Path:
    pipe_dir = job_dir / "pipeline"
    audio_input_dir = pipe_dir / "audio_input"
    audio_input_dir.mkdir(parents=True, exist_ok=True)

    # Ensure the uploaded audio exists (wait + fallback)
    audio_src = _wait_for_upload(job_id=job_dir.name, upload_path=audio_src)

    # Ensure MP3 for the pipeline
    audio_for_pipeline = ensure_mp3(audio_src)

    target_audio = audio_input_dir / "input.mp3"
    shutil.copy2(audio_for_pipeline, target_audio)

    # Base env (validate required)
    env = os.environ.copy()
    for key in REQUIRED_ENV:
        if not env.get(key):
            raise RuntimeError(f"Missing required env: {key}")

    # If a portrait exists, pass it through as PORTRAIT_PATH so generate_images_flux_schnell.py can use it.
    portrait = _find_portrait_file(job_dir)
    if portrait:
        env["PORTRAIT_PATH"] = str(portrait)
        log(f"Staging portrait for pipeline: {portrait}")

    # Optional: pass any trim extra args if you use them
    trim_extra = os.getenv("TRIM_EXTRA_ARGS", "").strip()
    if trim_extra:
        env["TRIM_EXTRA_ARGS"] = trim_extra

    # Run the pipeline
    cmd = [sys.executable, "make_video.py", "--job-id", job_dir.name]
    run_with_live_output(cmd, cwd=pipe_dir, env=env)

    final_video = pipe_dir / "final_video.mp4"
    if not final_video.exists():
        raise RuntimeError("final_video.mp4 not produced by pipeline")
    return final_video


def process_job(job_id: str, email: str, upload_path: str) -> Dict[str, str]:
    # quick fake mode for fast testing
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

    src_audio = Path(upload_path)

    _copy_pipeline_to(job_dir)
    # Debug listing before waiting
    try:
        log(f"job dir pre-wait listing {job_dir}: {[p.name for p in job_dir.glob('*')]}")
    except Exception:
        pass

    final_video = _run_make_video(job_dir, src_audio)

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    public_path = MEDIA_DIR / f"{job_id}.mp4"
    shutil.copy2(final_video, public_path)

    video_url = f"{BASE_URL}/media/{job_id}.mp4"
    send_link_email(email, video_url, job_id)
    return {"status": "done", "video_url": video_url}
