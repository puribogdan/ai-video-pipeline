# make_video.py
# Orchestrates the full pipeline end-to-end using fixed audio paths under ./audio_input:
#   audio_input/input.mp3  -> (optional trim) -> audio_input/input_trimmed.mp3
#   IMPORTANT: We do NOT replace input.mp3. We keep both files and use input_trimmed.mp3 if present.
#
# Steps:
#   0) trim_silence.py                 -> audio_input/input_trimmed.mp3
#   1) get_subtitles.py                -> subtitles/input_subtitles.json
#   2) generate_script.py              -> scripts/input_script.json
#   3) generate_images_flux_schnell.py -> scenes/scene_*.png
#   4) generate_video_chunks_seedance.py -> video_chunks/chunk_*.mp4
#   5) merge_and_add_audio.py          -> final_video.mp4
#
# Usage:
#   python make_video.py
#   python make_video.py --trim-args "--target_silence_ms 800 --keep_head_ms 600 --min_silence_ms 700"
#
# Notes:
# - Expects to run from the same folder where the pipeline scripts live.
# - Works primarily with ./audio_input/input.mp3; if trimming runs, we set AUDIO_INPUT_FILE to input_trimmed.mp3
#   and keep input.mp3 intact.
# - Checks for required env vars (OpenAI, Replicate). Exits early if missing.
# - Prints a JSON summary on success; exits non-zero on failure.

from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from config import settings  # loads .env via pydantic-settings
from app.worker_tasks import upload_to_b2

# ---------- Paths & constants ----------
ROOT = Path(__file__).resolve().parent

AUDIO_DIR = ROOT / "audio_input"
PIPELINE_AUDIO = AUDIO_DIR / "input.mp3"
TRIMMED_AUDIO = AUDIO_DIR / "input_trimmed.mp3"

SUBS_FILE = ROOT / "subtitles" / "input_subtitles.json"
SCRIPT_FILE = ROOT / "scripts" / "input_script.json"
SCENES_DIR = ROOT / "scenes"
CHUNKS_DIR = ROOT / "video_chunks"
FINAL_VIDEO = ROOT / "final_video.mp4"

REQUIRED_ENV = ["OPENAI_API_KEY", "REPLICATE_API_TOKEN"]

# ---------- Logging ----------
def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ---------- Subprocess helpers ----------
def run(cmd: str, cwd: Optional[Path] = None, extra_env: Optional[Dict[str, str]] = None) -> None:
    log(f"RUN: {cmd}")
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if extra_env:
        env.update(extra_env)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Command failed ({ret}): {cmd}")

def run_args(args_list: list[str], cwd: Optional[Path] = None, extra_env: Optional[Dict[str, str]] = None) -> None:
    log(f"RUN: {' '.join(args_list)}")
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if extra_env:
        env.update(extra_env)
    proc = subprocess.Popen(
        args_list,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        shell=False,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Command failed ({ret}): {' '.join(args_list)}")

# ---------- Env & file utils ----------
def ensure_env() -> None:
    missing = []
    if not settings.OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not settings.REPLICATE_API_TOKEN:
        missing.append("REPLICATE_API_TOKEN")
    if missing:
        raise SystemExit(f"Missing required environment variables: {', '.join(missing)}")

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="End-to-end video generation orchestrator (keeps input.mp3 intact).")
    ap.add_argument("--job-id", default=None, help="Optional job id for logs/metadata.")
    ap.add_argument("--limit-scenes", type=int, default=None, help="Optional cap for scenes (dev/testing).")
    ap.add_argument("--resolution", default="480p", help="Seedance resolution: 540p, 720p, 1080p.")
    ap.add_argument("--fps", type=int, default=24, help="Output FPS.")
    ap.add_argument("--force", action="store_true", help="Force downstream scripts to overwrite outputs where supported.")
    # Silence trimming
    ap.add_argument("--skip-trim", action="store_true", help="Skip trim_silence.py.")
    ap.add_argument("--trim-args", default="", help="Extra args passed to trim_silence.py.")
    # Optional skips
    ap.add_argument("--skip-subtitles", action="store_true")
    ap.add_argument("--skip-images", action="store_true")
    ap.add_argument("--skip-i2v", action="store_true")
    ap.add_argument("--skip-merge", action="store_true")
    args = ap.parse_args()

    ensure_env()

    # Ensure folders
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "subtitles").mkdir(parents=True, exist_ok=True)
    (ROOT / "scripts").mkdir(parents=True, exist_ok=True)
    SCENES_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    if not PIPELINE_AUDIO.exists():
        raise SystemExit(f"Audio file not found: {PIPELINE_AUDIO}")

    job_id = args.job_id or f"job_{int(time.time())}"
    log(f"Starting make_video for job_id={job_id}")

    # 0) Trim silence
    trim_script = ROOT / "trim_silence.py"
    if not args.skip_trim:
        if trim_script.exists():
            log("Trimming silence …")
            argv = [
                sys.executable,
                str(trim_script),
                "--input", str(PIPELINE_AUDIO),
                "--output", str(TRIMMED_AUDIO),
            ]
            if args.trim_args.strip():
                import shlex
                argv += shlex.split(args.trim_args.strip())

            extra = os.getenv("TRIM_EXTRA_ARGS", "").strip()
            if extra:
                import shlex
                argv += shlex.split(extra)

            run_args(argv, cwd=ROOT)
        else:
            log("trim_silence.py not found, skipping trimming step.")
    else:
        log("Skipping trim_silence.py (flag set).")

    # choose audio
    if TRIMMED_AUDIO.exists() and TRIMMED_AUDIO.stat().st_size > 0:
        audio_for_pipeline = TRIMMED_AUDIO
        log(f"Using trimmed audio: {audio_for_pipeline}")
    else:
        audio_for_pipeline = PIPELINE_AUDIO
        log(f"Using original audio: {audio_for_pipeline}")

    # environment for children
    child_env = {
        "AUDIO_INPUT_FILE": str(audio_for_pipeline),
        "AUDIO_INPUT_DIR": str(AUDIO_DIR),
    }

    # 1) Subtitles
    if args.skip_subtitles:
        log("Skipping get_subtitles.py.")
        if not SUBS_FILE.exists():
            raise SystemExit("skip-subtitles set, but no subtitles file.")
    else:
        if args.force or not SUBS_FILE.exists():
            run("python get_subtitles.py", cwd=ROOT, extra_env=child_env)
        else:
            log("Subtitles already exist; skipping.")

    # 2) Script
    run("python generate_script.py", cwd=ROOT, extra_env=child_env)

    # 3) Images
    if not args.skip_images:
        cmd_images = "python generate_images_flux_schnell.py"
        if args.limit_scenes is not None:
            cmd_images += f" --limit {int(args.limit_scenes)}"
        run(cmd_images, cwd=ROOT, extra_env=child_env)
    else:
        log("Skipping image generation.")

    # 4) Video chunks
    if not args.skip_i2v:
        cmd_i2v = f"python generate_video_chunks_seedance.py --resolution {args.resolution} --fps {int(args.fps)}"
        if args.limit_scenes is not None:
            cmd_i2v += f" --limit {int(args.limit_scenes)}"
        if args.force:
            cmd_i2v += " --force"
        run(cmd_i2v, cwd=ROOT, extra_env=child_env)
    else:
        log("Skipping i2v.")

    # 5) Merge
    if not args.skip_merge:
        run("python merge_and_add_audio.py", cwd=ROOT, extra_env=child_env)
    else:
        log("Skipping merge.")

    if not FINAL_VIDEO.exists():
        raise SystemExit("final_video.mp4 not found.")

    log("Uploading to B2...")
    b2_url = upload_to_b2(job_id, FINAL_VIDEO)
    if b2_url:
        log(f"Uploaded to B2: {b2_url}")
    else:
        log("B2 upload failed or skipped; using local file.")

    meta: Dict[str, object] = {
        "job_id": job_id,
        "audio_selected": str(audio_for_pipeline),
        "final_video": str(FINAL_VIDEO),
        "final_size_bytes": FINAL_VIDEO.stat().st_size,
        "b2_url": b2_url,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "resolution": args.resolution,
        "fps": args.fps,
        "limit_scenes": args.limit_scenes,
        "trim_applied": trim_script.exists() and not args.skip_trim,
        "trim_args": args.trim_args,
    }
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted.")
        sys.exit(130)
    except Exception as e:
        log(f"ERROR: {e}")
        sys.exit(1)
