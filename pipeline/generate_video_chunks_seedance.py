# generate_video_chunks_seedance.py — Image→Video per scene (Seedance), over-generate by +1s and hard-trim
from __future__ import annotations
import os, sys, json, math, tempfile, uuid, io, time, logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"):  # Pillow >=10
    PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS

import requests
from tqdm import tqdm
from dotenv import load_dotenv
import replicate

# Only trimming now (no padding / Ken Burns)
from moviepy.editor import VideoFileClip

ROOT = Path(__file__).parent
SCRIPT_PATH = ROOT / "scripts" / "input_script.json"
SCENES_DIR  = ROOT / "scenes"
PROMPT_JSON_PATH = SCENES_DIR / "prompt.json"
VIDEO_PROMPTS_JSON_PATH = SCENES_DIR / "video_prompts.json"
OUT_DIR     = ROOT / "video_chunks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Seedance-1-Lite on Replicate (high-quality I2V)
DEFAULT_MODEL = "bytedance/seedance-1-lite"
DEFAULT_RESOLUTION = "480p"
DEFAULT_FPS = 24

# Configure logging for video chunk creation
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "video_chunk_creation.log"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ---- Windows-safe FS ops -----------------------------------------------------

def _retry_windows_io(fn, *args, tries: int = 6, delay: float = 0.15, **kwargs):
    last_err = None
    for attempt in range(tries):
        try:
            return fn(*args, **kwargs)
        except PermissionError as e:
            last_err = e
            time.sleep(delay * (1.5 ** attempt))
        except OSError as e:
            if getattr(e, "winerror", None) == 32:
                last_err = e
                time.sleep(delay * (1.5 ** attempt))
            else:
                raise
    if last_err:
        raise last_err

def safe_unlink(p: Path):
    if p.exists():
        _retry_windows_io(p.unlink)

def safe_replace(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        safe_unlink(dst)
    _retry_windows_io(os.replace, str(src), str(dst))

# -----------------------------------------------------------------------------

def load_scenes() -> List[Dict[str, Any]]:
    if not SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Missing {SCRIPT_PATH}. Run generate_script.py first.")
    data = json.loads(SCRIPT_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("scripts/input_script.json must be a non-empty list.")
    for i, s in enumerate(data, 1):
        if "start_time" not in s or "end_time" not in s:
            raise ValueError(f"Scene {i} missing timing (need start_time and end_time).")
    return data

def load_image_prompts() -> Dict[str, str]:
    """Load image prompts from prompt.json file."""
    if not PROMPT_JSON_PATH.exists():
        raise FileNotFoundError(f"Missing {PROMPT_JSON_PATH}. Run generate_images_flux_schnell.py first.")
    data = json.loads(PROMPT_JSON_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("scenes/prompt.json must be a dictionary.")
    # Extract just the prompt text for each scene
    prompts = {}
    for scene_id, scene_data in data.items():
        if isinstance(scene_data, dict) and "prompt" in scene_data:
            prompts[scene_id] = scene_data["prompt"]
    return prompts

def clean_video_prompt(image_prompt: str) -> str:
    """Remove protagonist identity instructions from image prompt for video generation."""
    # Remove the protagonist identity line that comes from image generation
    lines = image_prompt.split('\n')
    cleaned_lines = []

    for line in lines:
        # Skip the protagonist identity instruction
        if "Maintain the protagonist's identity from the reference image" in line:
            continue
        if "match face & hair" in line:
            continue
        # Skip empty lines that might be left after removing the protagonist line
        if line.strip() == "":
            continue
        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines).strip()

def sec_len(s: Dict[str, Any]) -> float:
    return max(0.1, float(s["end_time"]) - float(s["start_time"]))

# ---------- Output handling ----------
def _download_bytes(url: str) -> bytes:
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    return r.content

def _fileoutput_to_bytes(obj: Any) -> Optional[bytes]:
    for attr in ("url", "uri", "href"):
        val = getattr(obj, attr, None)
        if isinstance(val, str) and val.startswith(("http://", "https://")):
            try:
                return _download_bytes(val)
            except Exception:
                pass
    if hasattr(obj, "open"):
        try:
            with obj.open() as f:  # type: ignore[attr-defined]
                return f.read()
        except Exception:
            pass
    if hasattr(obj, "read"):
        try:
            return obj.read()  # type: ignore[attr-defined]
        except Exception:
            pass
    for meth in ("save", "download", "write_to"):
        if hasattr(obj, meth):
            tmp = Path(tempfile.gettempdir()) / f"replicate_{uuid.uuid4().hex}.bin"
            try:
                getattr(obj, meth)(str(tmp))  # type: ignore[misc]
                data = tmp.read_bytes()
                try: tmp.unlink()
                except Exception: pass
                return data
            except Exception:
                try:
                    if tmp.exists():
                        data = tmp.read_bytes()
                        try: tmp.unlink()
                        except Exception: pass
                        return data
                except Exception:
                    pass
    return None

def outputs_to_video_bytes(out: Any) -> Optional[bytes]:
    if isinstance(out, str) and out.startswith(("http://", "https://")):
        try:
            return _download_bytes(out)
        except Exception:
            pass
    if isinstance(out, (list, tuple)):
        for item in out:
            if isinstance(item, str) and item.startswith(("http://", "https://")):
                try:
                    return _download_bytes(item)
                except Exception:
                    continue
            if isinstance(item, (list, tuple, dict)):
                b = outputs_to_video_bytes(item)
                if b:
                    return b
            b = _fileoutput_to_bytes(item)
            if b:
                return b
    if isinstance(out, dict):
        for v in out.values():
            b = outputs_to_video_bytes(v)
            if b:
                return b
    return _fileoutput_to_bytes(out)


def write_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if path.stat().st_size < 1024:
        raise RuntimeError("Downloaded video is too small; generation likely failed.")

# ---------- Seedance call ----------
def try_seedance(model: str, image_path: Path, prompt: str, duration_s: int, resolution: str, fps: int):
    """
    Call Seedance with desired duration. If API insists on 5s/10s, we CEIL to the next allowed
    value so the result is never shorter than requested (no retries/padding later).
    """
    def _run(dur: int):
        with open(image_path, "rb") as f:
            return replicate.run(
                model,
                input={
                    "prompt": prompt,
                    "image": f,
                    "duration": int(dur),
                    "resolution": resolution,
                    "fps": int(fps),
                },
            )
    try:
        return _run(duration_s)
    except replicate.exceptions.ReplicateError as e:
        # If API rejects the duration, try without the +1 second buffer
        # This preserves the exact script timing instead of forcing to 5/10 seconds
        if duration_s > 1:
            fallback_duration = int(duration_s - 1)
            print(f"⚠️  API rejected {duration_s}s duration, trying with exact script duration ({fallback_duration}s)")
            return _run(fallback_duration)
        raise

# ---------- Trim to target (no padding, no zoom) ----------
def trim_to_target(src_path: Path, dst_path: Path, target_sec: float, fps: int) -> float:
    """
    Open the raw clip and trim to exactly target_sec. If the raw clip is (unexpectedly) shorter,
    we just write it through as-is (no padding, per your requirement).
    Returns the raw duration.
    """
    clip = VideoFileClip(str(src_path), audio=False)
    try:
        cur = float(clip.duration or 0.0)
        if cur >= target_sec:
            fixed = clip.subclip(0, target_sec)
            try:
                fixed.write_videofile(
                    str(dst_path),
                    fps=int(clip.fps or fps),
                    codec="libx264",
                    audio=False,
                    preset="medium",
                    threads=2,
                    verbose=False,
                    logger=None,
                    ffmpeg_params=["-movflags", "+faststart"],
                )
            finally:
                try: fixed.close()
                except Exception: pass
        else:
            # You said this will never happen; if it does, we just pass it through.
            # (No Ken Burns, no retries.)
            clip.write_videofile(
                str(dst_path),
                fps=int(clip.fps or fps),
                codec="libx264",
                audio=False,
                preset="medium",
                threads=2,
                verbose=False,
                logger=None,
                ffmpeg_params=["-movflags", "+faststart"],
            )
        return cur
    finally:
        try: clip.close()
        except Exception: pass

# ---------- Effective target computation ----------
def effective_target_for_scene(i: int, scenes: List[Dict[str, Any]]) -> float:
    """
    Scene i (0-based): target duration = (next_start - virtual_start)
    - virtual_start = 0.0 for the first scene; else scenes[i]["start_time"]
    - next_start = scenes[i+1]["start_time"] if exists, else scenes[i]["end_time"]
    """
    n = len(scenes)
    s = scenes[i]
    virtual_start = 0.0 if i == 0 else float(s["start_time"])
    if i + 1 < n:
        next_start = float(scenes[i + 1]["start_time"])
    else:
        next_start = float(s["end_time"])
    target = max(0.1, next_start - virtual_start)
    return target

def main():
    import argparse
    load_dotenv()
    if not os.getenv("REPLICATE_API_TOKEN"):
        print("ERROR: Missing REPLICATE_API_TOKEN in .env", file=sys.stderr)
        sys.exit(1)

    ap = argparse.ArgumentParser(description="Image→Video per scene via Seedance-1-Lite (Replicate), over-generate +1s and trim.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--resolution", default=DEFAULT_RESOLUTION, help="540p, 720p, 1080p")
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    scenes = load_scenes()
    if args.limit:
        scenes = scenes[:max(1, args.limit)]

    # Load image prompts from prompt.json
    try:
        image_prompts = load_image_prompts()
        print(f"✅ Loaded {len(image_prompts)} image prompts from {PROMPT_JSON_PATH}")
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
        image_prompts = {}

    # Dictionary to store video prompts for logging
    video_prompts_log = {}

    print("  Strategy: Over-generate by +1s, then hard-trim to each scene's exact target (no padding).")
    print(f"🎬 Seedance I2V | res={args.resolution} | fps={args.fps} | scenes={len(scenes)}")

    for i, s in enumerate(tqdm(scenes, desc="I2V"), start=1):
        idx = i - 1
        scene_id = f"{i:03d}"
        img_path = SCENES_DIR / f"scene_{scene_id}.png"
        raw_path = OUT_DIR / f"chunk_{scene_id}.raw.mp4"
        out_path = OUT_DIR / f"chunk_{scene_id}.mp4"

        if out_path.exists() and out_path.stat().st_size > 1024 and not args.force:
            continue
        if not img_path.exists():
            raise FileNotFoundError(f"Missing start frame image: {img_path}")

        # Exact target duration for the scene
        target = effective_target_for_scene(idx, scenes)

        # Always ask Seedance for +1s longer than needed (ceil to avoid too short),
        # e.g. target=4.2 -> desired=6 (ceil to 5 then +1 -> 6; but API might force 5 or 10)
        desired = int(math.ceil(target)) + 1
        request_int = max(2, desired)

        scene_text = s.get("scene_description") or s.get("narration") or ""

        # Get the specific image prompt for this scene
        raw_image_prompt = image_prompts.get(scene_id, "")
        image_prompt = ""  # Initialize for logging

        if raw_image_prompt:
            # Clean the image prompt by removing protagonist identity instructions
            image_prompt = clean_video_prompt(raw_image_prompt)
            # Use the cleaned image prompt as the base, then add animation instructions
            prompt = (
                f"{image_prompt}\n"
                "Animate gently with subtle parallax and small camera moves. "
                "Preserve the exact style and subject from the start frame. "

                "Ensure realistic physics: characters must respect solid objects, no clipping through surfaces, consistent body structure throughout, and maintain spatial continuity in the scene."
            )
        else:
            # Fallback to original behavior if no image prompt available
            prompt = (
                "Animate gently with subtle parallax and small camera moves. "
                "Preserve the exact style and subject from the start frame. "

                "Ensure realistic physics: characters must respect solid objects, no clipping through surfaces, consistent body structure throughout, and maintain spatial continuity in the scene."
                f"Animate this moment: {scene_text}"
            )

        # Log the video prompt for this scene
        video_prompts_log[scene_id] = {
            "scene_description": scene_text,
            "raw_image_prompt": raw_image_prompt,
            "cleaned_image_prompt": image_prompt,
            "video_prompt": prompt,
            "duration_seconds": target,
            "model": args.model,
            "resolution": args.resolution,
            "fps": args.fps
        }

        try:
            out = try_seedance(args.model, img_path, prompt, request_int, args.resolution, args.fps)
            data = outputs_to_video_bytes(out)
            if not data:
                print(f"DEBUG output type: {type(out)}; sample: {str(out)[:200]}")
                raise RuntimeError(f"No usable video bytes in model output (scene {scene_id}).")

            write_bytes(raw_path, data)
            raw_len = trim_to_target(raw_path, out_path, target, args.fps)
            safe_unlink(raw_path)

            # Log successful video chunk creation
            timestamp = datetime.now().isoformat()
            logger.info(f"VIDEO_CHUNK_CREATED - Scene: {scene_id}, Timestamp: {timestamp}, "
                       f"Input: {img_path.name}, Output: {out_path.name}, "
                       f"Target Duration: {target:.3f}s, Raw Duration: {raw_len:.3f}s, "
                       f"Requested Duration: {request_int}s, Model: {args.model}, "
                       f"Resolution: {args.resolution}, FPS: {args.fps}, "
                       f"Scene Description: {scene_text[:100]}{'...' if len(scene_text) > 100 else ''}")

            print(f"✅ {out_path.name} (target {target:.3f}s, raw {raw_len:.3f}s, requested {request_int}s)")
        except Exception as e:
            # Log failed video chunk creation
            timestamp = datetime.now().isoformat()
            logger.error(f"VIDEO_CHUNK_FAILED - Scene: {scene_id}, Timestamp: {timestamp}, "
                        f"Input: {img_path.name}, Target Duration: {target:.3f}s, "
                        f"Requested Duration: {request_int}s, Model: {args.model}, "
                        f"Resolution: {args.resolution}, FPS: {args.fps}, "
                        f"Error: {str(e)}, Scene Description: {scene_text[:100]}{'...' if len(scene_text) > 100 else ''}")

            print(f"⚠️  Scene {scene_id} failed: {e}")
            try:
                safe_unlink(raw_path)
            except Exception:
                pass

    print(f"✅ All chunks are in: {OUT_DIR}")

    # Save video prompts log to JSON file
    if video_prompts_log:
        VIDEO_PROMPTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        VIDEO_PROMPTS_JSON_PATH.write_text(json.dumps(video_prompts_log, indent=2), encoding="utf-8")
        print(f"📝 Video prompts logged to: {VIDEO_PROMPTS_JSON_PATH}")

if __name__ == "__main__":
    main()
