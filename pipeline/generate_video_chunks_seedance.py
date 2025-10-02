# generate_video_chunks_seedance.py — Scene Description→Video per scene (Seedance), fixed 5-second videos (no trimming)
from __future__ import annotations
import os, sys, json, math, tempfile, uuid, io, time, logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import PIL.Image
# Handle Pillow version compatibility for ANTIALIAS
try:
    # Try to access ANTIALIAS (older Pillow versions)
    PIL.Image.ANTIALIAS
except AttributeError:
    try:
        # Pillow >=10 compatibility
        PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS
    except AttributeError:
        # If Resampling doesn't exist either, create a fallback
        PIL.Image.ANTIALIAS = 1  # Use a default integer value

import requests
from tqdm import tqdm
from dotenv import load_dotenv
import replicate
from replicate.exceptions import ReplicateError

# Only trimming now (no padding / Ken Burns)
from moviepy.editor import VideoFileClip

# LLM provider for scene enhancement
from providers.factory import get_llm_provider

ROOT = Path(__file__).parent
SCRIPT_PATH = ROOT / "scripts" / "input_script.json"
SCENES_DIR  = ROOT / "scenes"
PROMPT_JSON_PATH = SCENES_DIR / "prompt.json"
VIDEO_PROMPTS_JSON_PATH = SCENES_DIR / "video_prompts.json"
OUT_DIR     = ROOT / "video_chunks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Kling v2.5 Turbo Pro on Replicate (high-quality I2V)
DEFAULT_MODEL = "kwaivgi/kling-v2.5-turbo-pro"
DEFAULT_RESOLUTION = "480p"
DEFAULT_FPS = 24

# -------------------- Video Animation Style Library --------------------
VIDEO_STYLE_LIBRARY: Dict[str, str] = {
  "kid_friendly_cartoon": "Animate in a playful cartoon style with gentle, bouncy motion. Keep characters stable and interacting naturally with their environment.",
  
  "japanese_kawaii": "Animate in a cute chibi style with soft, cheerful movements. Keep proportions stable and motion gentle.",
  
  "storybook_illustrated": "Animate like a storybook illustration with simple, smooth pans and natural character motion. Keep everything stable and grounded.",
  
  "watercolor_storybook": "Animate with soft watercolor textures and calm, flowing motion. Keep proportions consistent and characters steady.",
  
  "paper_cutout": "Animate in a flat paper cut-out style with simple, minimal movement. Keep characters stable and grounded.",
  
  "cutout_collage": "Animate in a collage style with simple cut-out movements. Keep motion steady and avoid chaotic shifts.",
  
  "realistic_3d": "Animate with natural 3D-style motion and smooth camera pans. Keep body proportions stable and grounded.",
  
  "claymation": "Animate in a claymation style with gentle, stable movements. Keep proportions consistent and characters steady.",
  
  "needle_felted": "Animate in a soft felt style with simple, playful motions. Keep everything stable and grounded.",
  
  "stop_motion_felt_clay": "Animate in a handmade stop-motion style with gentle, stable pacing. Keep proportions consistent and grounded.",
  
  "hybrid_mix": "Animate with a mix of 2D and 3D style using simple, stable motion. Keep characters consistent and grounded.",
  
  "japanese_anime": "Animate in an anime style with smooth, stable character motions. Avoid extreme action; keep proportions steady.",
  
  "pixel_art": "Animate in pixel art style with simple, blocky motion. Keep characters stable and proportions consistent.",
  
  "van_gogh": "Animate in a Van Gogh painting style with gentle brushstroke motion. Keep characters steady and grounded.",
  
  "impressionism": "Animate in an impressionist painting style with calm, flowing movement. Keep proportions stable.",
  
  "art_deco": "Animate in an art deco style with smooth geometric motion. Keep characters steady and proportions consistent.",
  
  "cubism": "Animate in a cubist style with simple shifting shapes. Keep motion steady and proportions consistent.",
  
  "graphic_novel": "Animate like a graphic novel with simple panel-like motion and stable characters.",
  
  "motion_comic": "Animate like a motion comic with subtle character movements and smooth panel shifts. Keep everything stable.",
  
  "comic_book": "Animate in a comic book style with simple, bold motions. Keep proportions stable and grounded.",
  
  "gothic": "Animate in a gothic style with subtle, moody motion. Keep characters steady and grounded.",
  
  "silhouette": "Animate in a silhouette style with simple, puppet-like movement. Keep proportions consistent and grounded.",
  
  "fantasy_magic_glow": "Animate with soft glowing magical effects and calm, stable motion. Keep characters steady.",
  
  "surrealism_hybrid": "Animate with a light surreal style using gentle, playful motion. Keep everything stable.",
  
  "ink_parchment": "Animate like ink on parchment with simple flowing motion. Keep proportions stable and grounded.",
  
  "japanese_woodblock": "Animate in a woodblock print style with calm, sliding motion. Keep characters steady and grounded.",
  
  "ink_wash": "Animate in an ink wash style with gentle brushstroke motion. Keep proportions stable and consistent.",
  
  "japanese_gold_screen": "Animate in a gold screen style with smooth, ceremonial motion. Keep characters steady.",
  
  "japanese_scroll": "Animate like a Japanese scroll with continuous side-scrolling motion. Keep proportions stable.",
  
  "japanese_court": "Animate in a Japanese court art style with gentle pans and subtle gestures. Keep everything grounded."
}


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


def sec_len(s: Dict[str, Any]) -> float:
    return max(0.1, float(s["end_time"]) - float(s["start_time"]))

def enhance_scene_descriptions_with_claude(scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Send all scene descriptions to Claude with the specified prompt and update them with enhanced versions.
    """
    if not scenes:
        print("[DEBUG] No scenes provided for enhancement")
        return scenes

    print(f"[DEBUG] Starting enhancement for {len(scenes)} scenes")

    # Prepare the scene descriptions for Claude
    scene_descriptions = []
    for i, scene in enumerate(scenes):
        original_desc = scene.get("scene_description", "")
        narration = scene.get("narration", "")
        scene_descriptions.append({
            "scene_index": i,
            "original_description": original_desc,
            "narration": narration
        })
        print(f"[DEBUG] Scene {i}: '{original_desc[:50]}{'...' if len(original_desc) > 50 else ''}' | Narration: '{narration[:50]}{'...' if len(narration) > 50 else ''}'")

    # Create the prompt with all scene descriptions - just the scenes, no duplicate instructions
    scene_prompts = []
    for i, scene in enumerate(scene_descriptions):
        scene_prompts.append(f"Scene {i}: {scene['original_description']}")

    # Join all scene prompts
    full_prompt = "\n".join(scene_prompts)
    print(f"[DEBUG] Full prompt length: {len(full_prompt)} characters")

    prompt = f"""
Rewrite the following scene descriptions into concise prompts for an AI image-to-video generator. Each rewritten prompt should:

- Focus on one clear action or state per clip (no multiple events)
- Keep characters' proportions and style stable
- Ensure natural, believable contact with the environment (no floating, no glitches)
- Add gentle, intentional motion only (like slight gestures, breeze, or smooth camera movement)
- Remove dialogue or complex multi-step actions
- Use clear, simple language suitable for 5–10 second video clips

Scene descriptions to rewrite:
{full_prompt}

Return a JSON object with the rewritten prompts in this exact format:
{{
    "enhanced_scenes": [
        {{
            "scene_index": 0,
            "rewritten_prompt": "concise rewritten prompt here"
        }},
        {{
            "scene_index": 1,
            "rewritten_prompt": "concise rewritten prompt here"
        }}
    ]
}}
"""

    try:
        print("[DEBUG] Getting LLM provider...")
        provider = get_llm_provider()
        print(f"[DEBUG] Using provider: {type(provider).__name__}")

        # Check if provider has the required method
        if not hasattr(provider, 'chat_json'):
            raise AttributeError(f"Provider {type(provider).__name__} doesn't have chat_json method")

        print("[DEBUG] Calling provider.chat_json()...")
        response_data = provider.chat_json(
            model="claude-opus-4-1-20250805",
            messages=[
                {"role": "system", "content": "You are a creative video scene enhancer. You must output ONLY valid JSON, no other text."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )

        print(f"[DEBUG] Received response data: {type(response_data)}")
        if response_data:
            print(f"[DEBUG] Response keys: {list(response_data.keys())}")

        enhanced_scenes = response_data.get("enhanced_scenes", [])
        print(f"[DEBUG] Enhanced scenes count: {len(enhanced_scenes)}")

        if not enhanced_scenes:
            print("[WARNING] No enhanced scenes returned from provider, keeping original descriptions")
            print(f"[DEBUG] Response data was: {response_data}")
            return scenes

        # Update the scenes with rewritten prompts
        updated_count = 0
        for enhanced_scene in enhanced_scenes:
            scene_index = enhanced_scene.get("scene_index")
            rewritten_prompt = enhanced_scene.get("rewritten_prompt", "")

            print(f"[DEBUG] Processing enhanced scene - index: {scene_index}, prompt length: {len(rewritten_prompt)}")

            if scene_index is not None and scene_index < len(scenes):
                if rewritten_prompt:
                    original_desc = scenes[scene_index].get("scene_description", "")
                    scenes[scene_index]["scene_description"] = rewritten_prompt
                    if original_desc != rewritten_prompt:
                        updated_count += 1
                        print(f"[DEBUG] Updated Scene {scene_index}:")
                        print(f"  Original: '{original_desc[:100]}{'...' if len(original_desc) > 100 else ''}'")
                        print(f"  Enhanced: '{rewritten_prompt[:100]}{'...' if len(rewritten_prompt) > 100 else ''}'")
                else:
                    print(f"[WARNING] Empty rewritten prompt for scene {scene_index}")
            else:
                print(f"[WARNING] Invalid scene index: {scene_index}")

        print(f"[DEBUG] Successfully updated {updated_count}/{len(enhanced_scenes)} scene descriptions")
        if updated_count == 0:
            print("[WARNING] No scenes were actually updated - possible provider issue")
        return scenes

    except Exception as e:
        print(f"[ERROR] Failed to enhance scene descriptions: {e}")
        print(f"[ERROR] Exception type: {type(e).__name__}")
        import traceback
        print(f"[ERROR] Full traceback: {traceback.format_exc()}")
        print("[WARNING] Keeping original scene descriptions")
        return scenes

# ---------- Output handling ----------
def _download_bytes(url: str) -> bytes:
    r = requests.get(url, stream=True, timeout=600)  # Increased timeout for Render deployment
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
    Call Seedance with desired duration. API accepts 5-12 seconds, so clamp to this range.
    If API rejects the duration, try with a fallback within the valid range.
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

    # Clamp duration to API's valid range (5-12 seconds)
    clamped_duration = max(5, min(12, duration_s))

    # Retry logic for network/API errors
    max_retries = 3
    base_delay = 2.0

    for attempt in range(max_retries):
        try:
            return _run(clamped_duration)
        except ReplicateError as e:
            # If API rejects the clamped duration, try with a fallback within the valid range
            if clamped_duration > 5 and "duration" in str(e).lower():
                fallback_duration = clamped_duration - 1
                print(f"⚠️  API rejected {clamped_duration}s duration, trying with fallback ({fallback_duration}s)")
                try:
                    return _run(fallback_duration)
                except ReplicateError:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"⚠️  Fallback duration also failed, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    raise
            else:
                # Network or other API errors - retry with exponential backoff
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"⚠️  API error (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"⚠️  Retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"❌ All {max_retries} attempts failed")
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
            # Raw clip is shorter than target - just pass it through
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

    ap = argparse.ArgumentParser(description="Image→Video per scene via Kling v2.5 Turbo Pro (Replicate), with scene-based duration logic.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--resolution", default=DEFAULT_RESOLUTION, help="540p, 720p, 1080p")
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    scenes = load_scenes()
    if args.limit:
        scenes = scenes[:max(1, args.limit)]

    # Enhance scene descriptions for video generation (after image generation)
    print("[DEBUG] Enhancing scene descriptions for video generation...")
    # Store original descriptions before enhancement for video_prompts.json
    for i, scene in enumerate(scenes):
        scene['original_scene_description'] = scene.get('scene_description', '')

    scenes = enhance_scene_descriptions_with_claude(scenes)
    print(f"[DEBUG] Enhanced {len(scenes)} scene descriptions for video generation")

    # Note: Image prompts are no longer loaded - using scene descriptions directly

    print("  Strategy: Generate 10-second videos for all scenes, then trim to scene timing (except last scene).")
    print(f"🎬 Seedance I2V | res={args.resolution} | fps={args.fps} | scenes={len(scenes)}")

    # Dictionary to store video prompts for logging
    video_prompts_log = {}

    # Always ensure video_prompts_log exists, even if no scenes are processed
    if not scenes:
        print("[WARNING] No scenes to process")

    for i, s in enumerate(tqdm(scenes, desc="I2V"), start=1):
        idx = i - 1
        scene_id = f"{i:03d}"
        img_path = SCENES_DIR / f"scene_{scene_id}.png"
        raw_path = OUT_DIR / f"chunk_{scene_id}.raw.mp4"
        out_path = OUT_DIR / f"chunk_{scene_id}.mp4"

        if out_path.exists() and out_path.stat().st_size > 1024 and not args.force:
            # Log skipped scenes with original descriptions
            original_scene_text = s.get("original_scene_description") or s.get("scene_description") or s.get("narration") or ""
            scene_text = s.get("scene_description") or s.get("narration") or ""
            style_key = (os.getenv("STYLE_CHOICE", "kid_friendly_cartoon") or "kid_friendly_cartoon").lower().strip()
            animation_style = VIDEO_STYLE_LIBRARY.get(style_key, VIDEO_STYLE_LIBRARY["kid_friendly_cartoon"])
            target = effective_target_for_scene(idx, scenes)

            video_prompts_log[scene_id] = {
                "scene_description": original_scene_text,
                "enhanced_scene_description": scene_text if scene_text != original_scene_text else "",
                "video_prompt": f"Scene {scene_id} was skipped (already exists)",
                "duration_seconds": target,
                "model": args.model,
                "resolution": args.resolution,
                "fps": args.fps,
                "style_key": style_key,
                "animation_style": animation_style,
                "status": "skipped"
            }
            continue
        if not img_path.exists():
            raise FileNotFoundError(f"Missing start frame image: {img_path}")

        # Exact target duration for the scene
        target = effective_target_for_scene(idx, scenes)

        # Always generate 10 seconds for all scenes (over-generation for trimming)
        request_int = 10

        scene_text = s.get("scene_description") or s.get("narration") or ""

        # Get style from environment variable (same as image generation)
        style_key = (os.getenv("STYLE_CHOICE", "kid_friendly_cartoon") or "kid_friendly_cartoon").lower().strip()
        animation_style = VIDEO_STYLE_LIBRARY.get(style_key, VIDEO_STYLE_LIBRARY["kid_friendly_cartoon"])

        # Use only the scene description as the base for video generation
        if scene_text:
            # Use the scene description as the base, then add style-specific animation instructions and generic prompt
            prompt = (
                f"{scene_text}\n"
                f"{animation_style}\n"
                
                "Do not add or duplicate characters unless specified. "
                "If any text appears in the image render it in English language "
                "Respect accurate perspective and depth, keeping realistic distance, proportions, and scale between foreground and background objects."
               
            )
        else:
            # Fallback if no scene description available
            prompt = (
                f"{animation_style}\n"
                
                "Do not add or duplicate characters unless specified. "                
                "If any text appears in the image render it in English language "
                "Respect accurate perspective and depth, keeping realistic distance, proportions, and scale between foreground and background objects."
            )

        # Log the video prompt for this scene (use original scene description)
        original_scene_text = s.get("original_scene_description") or s.get("scene_description") or s.get("narration") or ""
        video_prompts_log[scene_id] = {
            "scene_description": original_scene_text,
            "enhanced_scene_description": scene_text if scene_text != original_scene_text else "",
            "video_prompt": prompt,
            "duration_seconds": target,
            "model": args.model,
            "resolution": args.resolution,
            "fps": args.fps,
            "style_key": style_key,
            "animation_style": animation_style
        }

        try:
            out = try_seedance(args.model, img_path, prompt, request_int, args.resolution, args.fps)
            data = outputs_to_video_bytes(out)
            if not data:
                print(f"DEBUG output type: {type(out)}; sample: {str(out)[:200]}")
                raise RuntimeError(f"No usable video bytes in model output (scene {scene_id}).")

            write_bytes(raw_path, data)

            # Apply trimming for all scenes except the last one
            if idx < len(scenes) - 1:  # Not the last scene
                # Trim to the exact scene duration
                trim_to_target(raw_path, out_path, target, args.fps)
            else:
                # Last scene - keep the full 10-second video (no trimming)
                safe_replace(raw_path, out_path)

            # Clean up the raw file after processing
            try:
                safe_unlink(raw_path)
            except Exception:
                pass

            # Log successful video chunk creation
            timestamp = datetime.now().isoformat()
            logger.info(f"VIDEO_CHUNK_CREATED - Scene: {scene_id}, Timestamp: {timestamp}, "
                        f"Input: {img_path.name}, Output: {out_path.name}, "
                        f"Target Duration: {target:.3f}s, "
                        f"Generated Duration: 10s, Model: {args.model}, "
                        f"Resolution: {args.resolution}, FPS: {args.fps}, Style: {style_key}, "
                        f"Trimmed: {'Yes' if idx < len(scenes) - 1 else 'No (last scene)'}, "
                        f"Scene Description: {scene_text[:100]}{'...' if len(scene_text) > 100 else ''}")

            print(f"✅ {out_path.name} (target {target:.3f}s, generated 10s{' trimmed' if idx < len(scenes) - 1 else ' kept full'})")
        except Exception as e:
            # Log failed video chunk creation
            timestamp = datetime.now().isoformat()
            logger.error(f"VIDEO_CHUNK_FAILED - Scene: {scene_id}, Timestamp: {timestamp}, "
                        f"Input: {img_path.name}, Target Duration: {target:.3f}s, "
                        f"Generated Duration: 10s, Model: {args.model}, "
                        f"Resolution: {args.resolution}, FPS: {args.fps}, "
                        f"Style: {style_key}, Error: {str(e)}, "
                        f"Scene Description: {scene_text[:100]}{'...' if len(scene_text) > 100 else ''}")

            print(f"⚠️  Scene {scene_id} failed: {e}")
            # Clean up the raw file in case of error
            try:
                safe_unlink(raw_path)
            except Exception:
                pass

    print(f"✅ All chunks are in: {OUT_DIR}")

    # Save video prompts log to JSON file (now includes both original and enhanced descriptions)
    if 'video_prompts_log' in locals() and video_prompts_log:
        VIDEO_PROMPTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        VIDEO_PROMPTS_JSON_PATH.write_text(json.dumps(video_prompts_log, indent=2), encoding="utf-8")
        print(f"📝 Video prompts logged to: {VIDEO_PROMPTS_JSON_PATH} (with original scene descriptions)")

def test_enhancement():
    """Test function to verify scene enhancement works"""
    print("=== Testing Scene Enhancement ===")

    # Create test scenes (keep originals separate for comparison)
    original_scenes = [
        {
            "start_time": 0.0,
            "end_time": 5.0,
            "scene_description": "A young boy is playing with a red ball in a green park on a sunny day",
            "narration": "The boy plays with his ball"
        },
        {
            "start_time": 5.0,
            "end_time": 10.0,
            "scene_description": "A little girl with pigtails is reading a colorful storybook under a large oak tree",
            "narration": "She reads her favorite book"
        }
    ]

    # Create a copy for enhancement (to avoid modifying originals)
    test_scenes = [scene.copy() for scene in original_scenes]

    print(f"Original scenes: {len(test_scenes)}")
    for i, scene in enumerate(original_scenes):
        print(f"  Scene {i}: {scene['scene_description']}")

    try:
        enhanced_scenes = enhance_scene_descriptions_with_claude(test_scenes)
        print(f"\nEnhanced scenes: {len(enhanced_scenes)}")
        for i, scene in enumerate(enhanced_scenes):
            print(f"  Scene {i}: {scene['scene_description']}")

        # Check if enhancement actually happened
        changes = 0
        for i, (original, enhanced) in enumerate(zip(original_scenes, enhanced_scenes)):
            if original['scene_description'] != enhanced['scene_description']:
                changes += 1
                print(f"  [OK] Scene {i} was enhanced")
                print(f"    Original: '{original['scene_description']}'")
                print(f"    Enhanced: '{enhanced['scene_description']}'")
            else:
                print(f"  [FAIL] Scene {i} was not enhanced")

        if changes > 0:
            print(f"\n[SUCCESS] Enhancement test PASSED: {changes}/{len(test_scenes)} scenes were enhanced")
        else:
            print(f"\n[ERROR] Enhancement test FAILED: No scenes were enhanced")

    except Exception as e:
        print(f"[ERROR] Enhancement test FAILED with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test-enhancement":
        test_enhancement()
    else:
        main()
