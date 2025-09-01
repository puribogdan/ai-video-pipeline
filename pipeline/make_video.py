# make_video.py
# Orchestrates the full pipeline end-to-end using fixed audio paths under ./audio_input:
#   audio_input/input.mp3  -> (optional trim) -> audio_input/input_trimmed.mp3
#   IMPORTANT: We do NOT replace input.mp3. We keep both files and use input_trimmed.mp3 if present.
#
# Steps:
#   0) trim_silence.py                 -> audio_input/input_trimmed.mp3  (we DO NOT overwrite input.mp3)
#   1) get_subtitles.py                -> subtitles/input_subtitles.json
#   2) generate_script.py              -> scripts/input_script.json
#   3) generate_images_flux_schnell.py -> scenes/scene_*.png
#   4) generate_video_chunks_seedance.py -> video_chunks/chunk_*.mp4
#   5) merge_and_add_audio.py          -> final_video.mp4
#
# Usage:
#   # Place your audio at ./audio_input/input.mp3
#   python make_video.py
#   # Optional:
#   python make_video.py --trim-args "--target_silence_ms 800 --keep_head_ms 600 --min_silence_ms 700"
#
# Notes:
# - Expects to run from the same folder where the pipeline scripts live.
# - Works primarily with ./audio_input/input.mp3; if trimming runs, we set AUDIO_INPUT_FILE to input_trimmed.mp3
#   and keep input.mp3 intact.
# - Checks for required env vars (OpenAI, Replicate). Exits early if missing.
# - Streams child process output as UTF-8 to avoid Windows console issues.
# - Prints a small JSON summary on success; exits non-zero on failure.

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
    """
    Run a shell command, stream output, raise on non-zero.
    Force UTF-8 decoding to avoid Windows 'charmap' issues.
    Accepts extra_env to pass per-process environment variables (e.g., AUDIO_INPUT_FILE).
    """
    log(f"RUN: {cmd}")
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")  # tell child to emit UTF-8
    if extra_env:
        env.update(extra_env)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",   # decode as UTF-8 in parent
        errors="replace",   # never crash on odd bytes
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Command failed ({ret}): {cmd}")

def run_args(args_list: list[str], cwd: Optional[Path] = None, extra_env: Optional[Dict[str, str]] = None) -> None:
    """
    Run a process with explicit argv list (Windows-safe, no shell quoting).
    Force UTF-8 decoding for stdout/stderr.
    Accepts extra_env to pass per-process environment variables (e.g., AUDIO_INPUT_FILE).
    """
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
    ap.add_argument("--resolution", default="720p", help="Seedance resolution: 540p, 720p, 1080p.")
    ap.add_argument("--fps", type=int, default=24, help="Output FPS for i2v and final video.")
    ap.add_argument("--force", action="store_true", help="Force downstream scripts to overwrite outputs where supported.")
    # Silence trimming
    ap.add_argument("--skip-trim", action="store_true", help="Skip trim_silence.py (use raw audio_input/input.mp3).")
    ap.add_argument("--trim-args", default="", help="Extra args passed to trim_silence.py (e.g. \"--target_silence_ms 800 --keep_head_ms 600 --min_silence_ms 700\").")
    # Optional skips
    ap.add_argument("--skip-subtitles", action="store_true", help="Skip get_subtitles.py if subtitles/input_subtitles.json already exists.")
    ap.add_argument("--skip-images", action="store_true", help="Skip image generation.")
    ap.add_argument("--skip-i2v", action="store_true", help="Skip image->video chunks.")
    ap.add_argument("--skip-merge", action="store_true", help="Skip final merge/add-audio.")
    args = ap.parse_args()

    ensure_env()

    # Ensure required folders exist
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "subtitles").mkdir(parents=True, exist_ok=True)
    (ROOT / "scripts").mkdir(parents=True, exist_ok=True)
    SCENES_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    if not PIPELINE_AUDIO.exists():
        raise SystemExit(f"Audio file not found: {PIPELINE_AUDIO} (place your audio at ./audio_input/input.mp3)")

    job_id = args.job_id or f"job_{int(time.time())}"
    log(f"Starting make_video for job_id={job_id}")

    # 0) Trim silence (optional; only if not skipped and script exists)
    trim_script = ROOT / "trim_silence.py"
    if not args.skip_trim:
        if trim_script.exists():
            log("Trimming silence …")
            argv = [
                sys.executable,             # same interpreter
                str(trim_script),
                "--input", str(PIPELINE_AUDIO),
                "--output", str(TRIMMED_AUDIO),
            ]
            if args.trim_args.strip():
                import shlex
                argv += shlex.split(args.trim_args.strip())
            run_args(argv, cwd=ROOT)
        else:
            log("trim_silence.py not found, skipping trimming step.")
    else:
        log("Skipping trim_silence.py (flag set).")

    # Decide which audio to use for the rest of the pipeline
    if TRIMMED_AUDIO.exists() and TRIMMED_AUDIO.stat().st_size > 0:
        audio_for_pipeline = TRIMMED_AUDIO
        log(f"Using trimmed audio: {audio_for_pipeline}")
    else:
        audio_for_pipeline = PIPELINE_AUDIO
        log(f"Using original audio: {audio_for_pipeline}")

    # Environment to pass the chosen audio path to child scripts
    child_env = {
        "AUDIO_INPUT_FILE": str(audio_for_pipeline),      # downstream scripts can read this if supported
        "AUDIO_INPUT_DIR": str(AUDIO_DIR),
    }

    # 1) get_subtitles (unless skipped)
    if args.skip_subtitles:
        log("Skipping get_subtitles.py (flag set).")
        if not SUBS_FILE.exists():
            raise SystemExit("skip-subtitles set, but subtitles/input_subtitles.json is missing.")
    else:
        if args.force or not SUBS_FILE.exists():
            run("python get_subtitles.py", cwd=ROOT, extra_env=child_env)
        else:
            log("Subtitles already exist; skipping get_subtitles.py.")

    # 2) generate_script (always run; ensures alignment)
    run("python generate_script.py", cwd=ROOT, extra_env=child_env)

    # 3) generate images
    if args.skip_images:
        log("Skipping generate_images_flux_schnell.py (flag set).")
    else:
        cmd_images = "python generate_images_flux_schnell.py"
        if args.limit_scenes is not None:
            cmd_images += f" --limit {int(args.limit_scenes)}"
        if args.force:
            log("Note: images script overwrites by default (DEFAULT_FORCE=True).")
        run(cmd_images, cwd=ROOT, extra_env=child_env)

    # 4) image->video (Seedance)
    if args.skip_i2v:
        log("Skipping generate_video_chunks_seedance.py (flag set).")
    else:
        cmd_i2v = f"python generate_video_chunks_seedance.py --resolution {args.resolution} --fps {int(args.fps)}"
        if args.limit_scenes is not None:
            cmd_i2v += f" --limit {int(args.limit_scenes)}"
        if args.force:
            cmd_i2v += " --force"
        run(cmd_i2v, cwd=ROOT, extra_env=child_env)

    # 5) merge + add audio (FFmpeg variant recommended)
    if args.skip_merge:
        log("Skipping merge_and_add_audio.py (flag set).")
    else:
        run("python merge_and_add_audio.py", cwd=ROOT, extra_env=child_env)

    if not FINAL_VIDEO.exists():
        raise SystemExit("final_video.mp4 not found after merge step.")

    # Prepare result metadata
    meta: Dict[str, object] = {
        "job_id": job_id,
        "audio_selected": str(audio_for_pipeline),
        "final_video": str(FINAL_VIDEO),
        "final_size_bytes": FINAL_VIDEO.stat().st_size,
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
