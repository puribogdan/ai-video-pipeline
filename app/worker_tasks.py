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
import logging

from dotenv import load_dotenv
from .email_utils import send_link_email

import boto3
from botocore.exceptions import ClientError

import json
import tempfile
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=5),
    retry=retry_if_exception_type((subprocess.CalledProcessError, FileNotFoundError, Exception)),
    reraise=True
)
def ensure_mp3(src_path: Path) -> Path:
    try:
        mt, _ = mimetypes.guess_type(str(src_path))
        is_mp3 = (src_path.suffix.lower() == ".mp3") or (mt == "audio/mpeg")
        if is_mp3:
            return src_path
        out_path = src_path.with_suffix(".mp3")
        cmd = ["ffmpeg", "-y", "-i", str(src_path), "-vn", "-acodec", "libmp3lame", "-b:a", "192k", "-ar", "44100", "-ac", "2", str(out_path)]
        log(f"[LOG] ensure_mp3 input: {src_path} (ext: {src_path.suffix}, size: {src_path.stat().st_size}), cmd: {' '.join(cmd)}")

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.returncode != 0:
            log(f"[ERROR] ffmpeg failed with return code {result.returncode}: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)

        out_size = out_path.stat().st_size if out_path.exists() else 0
        log(f"[LOG] ensure_mp3 output: {out_path} (size: {out_size})")
        if out_size == 0:
            raise RuntimeError("Conversion produced empty file")
        return out_path
    except Exception as e:
        log(f"[ERROR] ensure_mp3 failed for {src_path}: {e}")
        raise


def run_with_live_output(cmd: list[str], cwd: Path, env: dict, timeout: int = 1800) -> None:
    log(f"RUN (cwd={cwd}, timeout={timeout}s): {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, cwd=str(cwd), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    assert proc.stdout is not None
    tail = deque(maxlen=200)

    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            tail.append(line.rstrip())

        ret = proc.wait(timeout=timeout)
        if ret != 0:
            raise RuntimeError("make_video.py failed (exit %d)\n--- tail ---\n%s" % (ret, "\n".join(tail)))
    except subprocess.TimeoutExpired:
        log(f"Process timed out after {timeout} seconds, terminating...")
        proc.kill()
        proc.wait()
        raise RuntimeError(f"make_video.py timed out after {timeout} seconds")


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

    # Timeout reached - provide better error message
    job_listing = _listdir_safe(job_dir)
    if not job_dir.exists():
        raise RuntimeError(f"Job directory does not exist: {job_dir}")
    elif not job_listing:
        raise RuntimeError(f"Job directory is empty: {job_dir}")
    else:
        raise RuntimeError(f"Audio not found in {job_dir} after {timeout_s}s. Listing: {job_listing}")


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
    run_with_live_output(cmd, cwd=pipe_dir, env=env, timeout=1800)  # 30 minute timeout

    final_video = pipe_dir / "final_video.mp4"
    if not final_video.exists():
        raise RuntimeError("final_video.mp4 not produced by pipeline")
    return final_video



@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((ClientError, Exception)),
    reraise=True
)
def upload_to_b2(job_id: str, video_path: Path, job_dir: Optional[Path] = None) -> Optional[str]:
    """Upload video, scripts, and portrait images to Backblaze B2 (S3-compatible) and return public URL, or None on failure."""
    bucket_name = os.getenv("B2_BUCKET_NAME")
    key_id = os.getenv("B2_KEY_ID")
    app_key = os.getenv("B2_APPLICATION_KEY")
    if bucket_name is None or key_id is None or app_key is None:
        log("B2 env vars not set; skipping upload.")
        return None

    # Safe logging
    log(f"[DEBUG] B2_KEY_ID length: {len(key_id)}")
    log(f"[DEBUG] B2_APP_KEY length: {len(app_key)}")
    log(f"[DEBUG] B2_KEY_ID prefix: {key_id[:5]}...")
    log(f"[DEBUG] B2_APP_KEY prefix: {app_key[:5]}...")

    local_file = video_path.absolute()
    file_name = f"exports/{job_id}/final_video.mp4"
    log(f"[DEBUG] local_file absolute path: {local_file}")
    log(f"[DEBUG] file_name in B2: {file_name}")
    log(f"[DEBUG] file_size: {video_path.stat().st_size} bytes")

    region = os.getenv("B2_REGION", "eu-central-003")
    endpoint = f'https://s3.{region}.backblazeb2.com'
    log(f"[DEBUG] Using endpoint: {endpoint}, region: {region}")

    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=key_id,
            aws_secret_access_key=app_key,
            endpoint_url=endpoint,
            region_name=region
        )
        log("Using S3-compatible boto3 client for Backblaze B2.")
        log("[DEBUG] S3 client created successfully.")

        # Upload using S3 client (S3-compatible)
        log(f"[DEBUG] Starting upload...")
        s3.upload_file(
            Filename=str(local_file),
            Bucket=bucket_name,
            Key=file_name,
            ExtraArgs={'ContentType': 'video/mp4'}
        )
        log("[DEBUG] Upload call completed.")

        # Upload input_script.json and prompt.json if job_dir provided
        if job_dir:
            # Upload input_script.json
            script_path = job_dir / "pipeline" / "scripts" / "input_script.json"
            if script_path.exists():
                script_key = f"exports/{job_id}/input_script.json"
                s3.upload_file(
                    str(script_path),
                    bucket_name,
                    script_key,
                    ExtraArgs={'ContentType': 'application/json'}
                )
                log(f"Uploaded input_script.json to: {script_key}")

            # Upload existing prompt.json
            prompt_path = job_dir / "pipeline" / "scenes" / "prompt.json"
            if prompt_path.exists():
                prompt_key = f"exports/{job_id}/prompt.json"
                s3.upload_file(
                    str(prompt_path),
                    bucket_name,
                    prompt_key,
                    ExtraArgs={'ContentType': 'application/json'}
                )
                log(f"Uploaded prompt.json to: {prompt_key}")

            # Upload video_prompts.json
            video_prompts_path = job_dir / "pipeline" / "scenes" / "video_prompts.json"
            if video_prompts_path.exists():
                video_prompts_key = f"exports/{job_id}/video_prompts.json"
                s3.upload_file(
                    str(video_prompts_path),
                    bucket_name,
                    video_prompts_key,
                    ExtraArgs={'ContentType': 'application/json'}
                )
                log(f"Uploaded video_prompts.json to: {video_prompts_key}")

            # Upload portrait image if it exists
            portrait_path = job_dir / "pipeline" / "scenes" / "portrait_ref.png"
            portrait_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp']
            portrait_uploaded = False

            for ext in portrait_extensions:
                potential_portrait = job_dir / "pipeline" / "scenes" / f"portrait_ref{ext}"
                if potential_portrait.exists():
                    portrait_path = potential_portrait
                    break

            if portrait_path.exists():
                # Determine content type based on file extension
                _, ext = os.path.splitext(str(portrait_path))
                content_type = 'image/png'  # default
                if ext.lower() == '.jpg' or ext.lower() == '.jpeg':
                    content_type = 'image/jpeg'
                elif ext.lower() == '.webp':
                    content_type = 'image/webp'
                elif ext.lower() == '.bmp':
                    content_type = 'image/bmp'

                portrait_key = f"exports/{job_id}/portrait_ref{ext}"
                s3.upload_file(
                    str(portrait_path),
                    bucket_name,
                    portrait_key,
                    ExtraArgs={'ContentType': content_type}
                )
                log(f"Uploaded portrait image to: {portrait_key}")
                portrait_uploaded = True

            # Upload audio files if job_dir provided
            if job_dir:
                # Upload original input audio
                audio_input_dir = job_dir / "pipeline" / "audio_input"
                original_audio_path = audio_input_dir / "input.mp3"
                if original_audio_path.exists():
                    audio_key = f"exports/{job_id}/audio_input.mp3"
                    s3.upload_file(
                        str(original_audio_path),
                        bucket_name,
                        audio_key,
                        ExtraArgs={'ContentType': 'audio/mpeg'}
                    )
                    log(f"Uploaded original input audio to: {audio_key}")

                # Upload trimmed audio if it exists
                trimmed_audio_path = audio_input_dir / "input_trimmed.mp3"
                if trimmed_audio_path.exists():
                    trimmed_audio_key = f"exports/{job_id}/audio_input_trimmed.mp3"
                    s3.upload_file(
                        str(trimmed_audio_path),
                        bucket_name,
                        trimmed_audio_key,
                        ExtraArgs={'ContentType': 'audio/mpeg'}
                    )
                    log(f"Uploaded trimmed input audio to: {trimmed_audio_key}")

        # Get ETag from last response (or head for verification)
        try:
            head = s3.head_object(Bucket=bucket_name, Key=file_name)
            etag = head.get('ETag', 'unknown')
            content_type = head.get('ContentType', 'unknown')
            size = head['ContentLength']
            log(f"[DEBUG] Upload verified: ETag={etag}, size={size} bytes, content_type={content_type}")
        except ClientError as head_e:
            log(f"[DEBUG] Head verification failed: {head_e.response['Error']['Message']}")

        log("Upload successful.")

        # Construct public URL (assuming public bucket)
        video_url = f"https://{bucket_name}.s3.{region}.backblazeb2.com/exports/{job_id}/final_video.mp4"
        log(f"Uploaded to B2: {video_url}")
        return video_url

    except Exception as e:
        import traceback
        err_msg = str(e)
        if isinstance(e, ClientError):
            err_code = e.response['Error']['Code']
            err_message = e.response['Error']['Message']
            err_msg += f" | Server: {err_code} - {err_message}"
        log(f"B2 upload failed: {type(e).__name__}: {err_msg}")
        log(f"Full traceback:\n{traceback.format_exc()}")
        return None


def process_job(job_id: str, email: str, upload_path: str, style: str) -> Dict[str, str]:
    """Process a video job with comprehensive error handling and retry logic"""
    try:
        log(f"Starting job processing: {job_id} (email: {email}, style: {style})")

        if os.getenv("DEV_FAKE_PIPELINE", "0") == "1":
            log("Using fake pipeline mode")
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

        # Create job directory
        job_dir = UPLOADS_DIR / job_id
        try:
            job_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            log(f"[ERROR] Failed to create job directory {job_dir}: {e}")
            raise RuntimeError(f"Failed to create job directory: {e}")

        hint_audio = Path(upload_path) if upload_path else None
        log(f"Job dir initial listing: {sorted([p.name for p in job_dir.iterdir()]) if job_dir.exists() else []}")

        # Run the main video processing pipeline
        try:
            final_video = _run_make_video(job_dir, hint_audio, style)
        except Exception as e:
            log(f"[ERROR] Video processing failed for job {job_id}: {e}")
            import traceback
            log(f"[ERROR] Full traceback:\n{traceback.format_exc()}")
            raise RuntimeError(f"Video processing failed: {e}")

        # Copy to public media directory
        try:
            MEDIA_DIR.mkdir(parents=True, exist_ok=True)
            public_path = MEDIA_DIR / f"{job_id}.mp4"
            shutil.copy2(final_video, public_path)
            log(f"Video copied to public path: {public_path}")
        except Exception as e:
            log(f"[ERROR] Failed to copy video to public directory: {e}")
            raise RuntimeError(f"Failed to copy video to public directory: {e}")

        # Upload to Backblaze B2 (with fallback to local)
        try:
            b2_url = upload_to_b2(job_id, public_path, job_dir)
            video_url = b2_url or f"{BASE_URL}/media/{job_id}.mp4"
            log(f"Using video URL: {video_url}")
        except Exception as e:
            log(f"[WARNING] B2 upload failed, falling back to local URL: {e}")
            video_url = f"{BASE_URL}/media/{job_id}.mp4"

        # Send email notification
        try:
            send_link_email(email, video_url, job_id)
            log(f"Email notification sent to {email}")
        except Exception as e:
            log(f"[WARNING] Failed to send email notification: {e}")
            # Don't fail the job if email fails

        log(f"Job {job_id} completed successfully")
        return {"status": "done", "video_url": video_url}

    except Exception as e:
        log(f"[ERROR] Job {job_id} failed completely: {e}")
        import traceback
        log(f"[ERROR] Full traceback:\n{traceback.format_exc()}")
        raise
