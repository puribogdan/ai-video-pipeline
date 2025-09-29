# generate_video_chunks_seedance.py — Scene Description→Video per scene (Seedance), fixed 5-second videos (no trimming)
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
    "kid_friendly_cartoon": "Animate with bouncy squash-and-stretch, oversized reactions, and playful comedic timing. Use star-burst wipes, balloon transitions, and quick zooms to keep energy high. Preserve the exact style and subject from the input frames. Keep body proportions stable across frames. Characters interact believably with their environment: when on solid ground, feet align naturally and do not drift; when jumping, flying, floating, or moving across non-solid surfaces (like water or clouds), motion arcs and contact look intentional and consistent with the story. Maintain clear spatial continuity shot-to-shot.",
    "japanese_kawaii": "Animate with chibi-style bouncing motions, sparkly blushes, and looping giggles. Sprinkle heart and star particle effects, and use gentle push-in camera moves for extra cuteness. Preserve the exact style and subject from the input frames. Keep body proportions stable across frames. Characters interact believably with their environment: when on solid ground, feet align naturally and do not drift; when jumping, flying, floating, or moving across non-solid surfaces (like water or clouds), motion arcs and contact look intentional and consistent with the story. Maintain clear spatial continuity shot-to-shot.",
    "storybook_illustrated": "Animate with hinged puppet-like movements, sliding parallax layers, and textured page-turn wipes. Characters should pivot as if jointed from cut paper, with camera pans revealing layered collages. Preserve the exact style and subject from the input frames. Keep body proportions stable across frames. Characters interact believably with their environment: when on solid ground, feet align naturally and do not drift; when jumping, flying, floating, or moving across non-solid surfaces (like water or clouds), motion arcs and contact look intentional and consistent with the story. Maintain clear spatial continuity shot-to-shot.",
    "watercolor_storybook": "Animate with watercolor washes blooming outward, characters drifting gently as if painted on flowing paper. Use soft dissolves, brushstroke morphs, and calm pans across pastel textures. Preserve the exact style and subject from the input frames. Keep body proportions stable across frames. Characters interact believably with their environment: when on solid ground, feet align naturally and do not drift; when jumping, flying, floating, or moving across non-solid surfaces (like water or clouds), motion arcs and contact look intentional and consistent with the story. Maintain clear spatial continuity shot-to-shot.",
    "paper_cutout": "Animate with stiff, jerky cut-out motions, simple head tilts, and minimal arm flaps. Dialogue uses basic mouth flaps while eyes exaggerate reactions. Camera is flat, zooming only for comedic emphasis. Preserve the exact style and subject from the input frames. Keep body proportions stable across frames. Characters interact believably with their environment: when on solid ground, feet align naturally and do not drift; when jumping, flying, floating, or moving across non-solid surfaces (like water or clouds), motion arcs and contact look intentional and consistent with the story. Maintain clear spatial continuity shot-to-shot.",
    "cutout_collage": "Animate with surreal paper collage energy: rotating limbs, bouncing cut-outs, and absurd scale shifts. Use chaotic zooms, skewed pans, and fast collage transitions for comedic surrealism. Preserve the exact style and subject from the input frames. Keep body proportions stable across frames. Characters interact believably with their environment: when on solid ground, feet align naturally and do not drift; when jumping, flying, floating, or moving across non-solid surfaces (like water or clouds), motion arcs and contact look intentional and consistent with the story. Maintain clear spatial continuity shot-to-shot.",
    "realistic_3d": "Animate with lifelike motion capture pacing, cinematic sweeps, and expressive eye movements. Use dolly-ins, crane sweeps, and depth-of-field pulls for a film-quality look. Preserve the exact style and subject from the input frames. Keep body proportions stable across frames. Characters interact believably with their environment: when on solid ground, feet align naturally and do not drift; when jumping, flying, floating, or moving across non-solid surfaces (like water or clouds), motion arcs and contact look intentional and consistent with the story. Maintain clear spatial continuity shot-to-shot.",
    "claymation": "Animate with tactile stop-motion pacing, slight handmade jitter, and squashy hops. Pan across clay-built sets, revealing hand-molded imperfections as part of the charm. Preserve the exact style and subject from the input frames. Keep body proportions stable across frames. Characters interact believably with their environment: when on solid ground, feet align naturally and do not drift; when jumping, flying, floating, or moving across non-solid surfaces (like water or clouds), motion arcs and contact look intentional and consistent with the story. Maintain clear spatial continuity shot-to-shot.",
    "needle_felted": "Animate with plush rolling motions, jittery felt hops, and soft waddles. Use close-up pans to highlight fuzzy textures and toy-like charm. Preserve the exact style and subject from the input frames. Keep body proportions stable across frames. Characters interact believably with their environment: when on solid ground, feet align naturally and do not drift; when jumping, flying, floating, or moving across non-solid surfaces (like water or clouds), motion arcs and contact look intentional and consistent with the story. Maintain clear spatial continuity shot-to-shot.",
    "stop_motion_felt_clay": "Animate with smooth handcrafted pacing, cozy environmental details, and tactile lighting shifts. Use warm dolly moves across miniature sets and soft fade transitions. Preserve the exact style and subject from the input frames. Keep body proportions stable across frames. Characters interact believably with their environment: when on solid ground, feet align naturally and do not drift; when jumping, flying, floating, or moving across non-solid surfaces (like water or clouds), motion arcs and contact look intentional and consistent with the story. Maintain clear spatial continuity shot-to-shot.",
    "hybrid_mix": "Animate with exaggerated 2D character motions against dynamic 3D camera sweeps. Contrast flat motion with dimensional lighting and parallax to emphasize the clash of worlds. Preserve the exact style and subject from the input frames. Keep body proportions stable across frames. Characters interact believably with their environment: when on solid ground, feet align naturally and do not drift; when jumping, flying, floating, or moving across non-solid surfaces (like water or clouds), motion arcs and contact look intentional and consistent with the story. Maintain clear spatial continuity shot-to-shot.",
    "japanese_anime": "Animate with dramatic whip pans, speed-line bursts, glowing aura effects, and cinematic fight choreography. Use close-up snap zooms and stylized impact frames to match anime pacing. Preserve the exact style and subject from the input frames. Keep body proportions stable across frames. Characters interact believably with their environment: when on solid ground, feet align naturally and do not drift; when jumping, flying, floating, or moving across non-solid surfaces (like water or clouds), motion arcs and contact look intentional and consistent with the story. Maintain clear spatial continuity shot-to-shot.",
    "pixel_art": "Animate with looping sprite walk cycles, jump arcs, and blocky motion shifts. Use retro game-style pans, wipes, and choppy transitions to mimic pixel animation. Preserve the exact style and subject from the input frames. Keep body proportions stable across frames. Characters interact believably with their environment: when on solid ground, feet align naturally and do not drift; when jumping, flying, floating, or moving across non-solid surfaces (like water or clouds), motion arcs and contact look intentional and consistent with the story. Maintain clear spatial continuity shot-to-shot.",
    "van_gogh": "Animate with swirling brushstrokes that ripple across skies, morphing into shifting landscapes. Camera sweeps should feel like diving into a living painting with strokes that animate dynamically. Preserve the exact style and subject from the input frames. Keep body proportions stable across frames. Characters interact believably with their environment: when on solid ground, feet align naturally and do not drift; when jumping, flying, floating, or moving across non-solid surfaces (like water or clouds), motion arcs and contact look intentional and consistent with the story. Maintain clear spatial continuity shot-to-shot.",
    "impressionism": "Animate with dappled sunlight flickering across scenes, pastel brushstrokes appearing dynamically, and characters moving as if bathed in shifting light. Camera pans should glide dreamily through painted textures. Preserve the exact style and subject from the input frames. Keep body proportions stable across frames. Characters interact believably with their environment: when on solid ground, feet align naturally and do not drift; when jumping, flying, floating, or moving across non-solid surfaces (like water or clouds), motion arcs and contact look intentional and consistent with the story. Maintain clear spatial continuity shot-to-shot.",
    "art_deco": "Animate with symmetrical sweeps, rotating geometric forms, and shimmering metallic reveals. Camera glides should mimic poster displays unfolding into elegant movement. Preserve the exact style and subject from the input frames. Keep body proportions stable across frames. Characters interact believably with their environment: when on solid ground, feet align naturally and do not drift; when jumping, flying, floating, or moving across non-solid surfaces (like water or clouds), motion arcs and contact look intentional and consistent with the story. Maintain clear spatial continuity shot-to-shot.",
    "cubism": "Animate with fractured planes shifting, rotating, and reassembling into abstract figures. Camera sweeps should mimic perspective shifts, sliding across fragmented geometry. Preserve the exact style and subject from the input frames. Keep body proportions stable across frames. Characters interact believably with their environment: when on solid ground, feet align naturally and do not drift; when jumping, flying, floating, or moving across non-solid surfaces (like water or clouds), motion arcs and contact look intentional and consistent with the story. Maintain clear spatial continuity shot-to-shot.",
    "graphic_novel": "Animate with bold panel slides, chiaroscuro light flickers, and slow-motion reveals. Use angled pans and dramatic speech bubble pop-ins for cinematic graphic novel pacing. Preserve the exact style and subject from the input frames. Keep body proportions stable across frames. Characters interact believably with their environment: when on solid ground, feet align naturally and do not drift; when jumping, flying, floating, or moving across non-solid surfaces (like water or clouds), motion arcs and contact look intentional and consistent with the story. Maintain clear spatial continuity shot-to-shot.",
    "motion_comic": "Animate with manga-style speed lines, panel wipes, and subtle character motions like blinks or hair sways. Use snap zooms and angled pans to mimic manga drama. Preserve the exact style and subject from the input frames. Keep body proportions stable across frames. Characters interact believably with their environment: when on solid ground, feet align naturally and do not drift; when jumping, flying, floating, or moving across non-solid surfaces (like water or clouds), motion arcs and contact look intentional and consistent with the story. Maintain clear spatial continuity shot-to-shot.",
    "comic_book": "Animate with exaggerated punches, dynamic action poses, and bold onomatopoeia bursts like 'BAM!' and 'POW!'. Use comic panel smash-cuts and sweeping zooms for superhero energy. Preserve the exact style and subject from the input frames. Keep body proportions stable across frames. Characters interact believably with their environment: when on solid ground, feet align naturally and do not drift; when jumping, flying, floating, or moving across non-solid surfaces (like water or clouds), motion arcs and contact look intentional and consistent with the story. Maintain clear spatial continuity shot-to-shot.",
    "gothic": "Animate with shadowy flickers, elongated character jerks, and swooping camera tilts. Use eerie stop-motion jolts blended with playful gothic charm. Preserve the exact style and subject from the input frames. Keep body proportions stable across frames. Characters interact believably with their environment: when on solid ground, feet align naturally and do not drift; when jumping, flying, floating, or moving across non-solid surfaces (like water or clouds), motion arcs and contact look intentional and consistent with the story. Maintain clear spatial continuity shot-to-shot.",
    "silhouette": "Animate with articulated cut-out puppet motions, pivoting limbs at jointed points, and smooth sliding across a glowing backdrop. Use parallax shadow layers and shifting light beams to mimic live shadow theater. Preserve the exact style and subject from the input frames. Keep body proportions stable across frames. Characters interact believably with their environment: when on solid ground, feet align naturally and do not drift; when jumping, flying, floating, or moving across non-solid surfaces (like water or clouds), motion arcs and contact look intentional and consistent with the story. Maintain clear spatial continuity shot-to-shot.",
    "fantasy_magic_glow": "Animate with floating spark particles, luminous aura blooms, and enchanted sweeps through glowing landscapes. Use smooth crane pans and spark-trail transitions for magical atmosphere. Preserve the exact style and subject from the input frames. Keep body proportions stable across frames. Characters interact believably with their environment: when on solid ground, feet align naturally and do not drift; when jumping, flying, floating, or moving across non-solid surfaces (like water or clouds), motion arcs and contact look intentional and consistent with the story. Maintain clear spatial continuity shot-to-shot.",
    "surrealism_hybrid": "Animate with whimsical stretching forms, rhyming bounce loops, and dreamlike transitions. Use distorted perspective pans and playful camera swings for surreal energy. Preserve the exact style and subject from the input frames. Keep body proportions stable across frames. Characters interact believably with their environment: when on solid ground, feet align naturally and do not drift; when jumping, flying, floating, or moving across non-solid surfaces (like water or clouds), motion arcs and contact look intentional and consistent with the story. Maintain clear spatial continuity shot-to-shot.",
    "ink_parchment": "Animate with sliding parchment reveals, bold ink strokes appearing dynamically, and martial arts poses flipping and swiping across the scroll. Camera moves mimic shadow puppetry with dramatic pans along textured paper. Preserve the exact style and subject from the input frames. Keep body proportions stable across frames. Characters interact believably with their environment: when on solid ground, feet align naturally and do not drift; when jumping, flying, floating, or moving across non-solid surfaces (like water or clouds), motion arcs and contact look intentional and consistent with the story. Maintain clear spatial continuity shot-to-shot.",
    "japanese_woodblock": "Animate with flowing waves, layered clouds, and sliding parallax mountains like a woodblock print in motion. Characters should move in flat yet powerful stylized gestures, with camera pans following traditional composition rhythms. Preserve the exact style and subject from the input frames. Keep body proportions stable across frames. Characters interact believably with their environment: when on solid ground, feet align naturally and do not drift; when jumping, flying, floating, or moving across non-solid surfaces (like water or clouds), motion arcs and contact look intentional and consistent with the story. Maintain clear spatial continuity shot-to-shot.",
    "ink_wash": "Animate with ink spreading naturally across paper, brushstrokes forming dynamically, and scenes dissolving like fresh ink soaking into parchment. Use slow pans and minimalistic motion for meditative pacing. Preserve the exact style and subject from the input frames. Keep body proportions stable across frames. Characters interact believably with their environment: when on solid ground, feet align naturally and do not drift; when jumping, flying, floating, or moving across non-solid surfaces (like water or clouds), motion arcs and contact look intentional and consistent with the story. Maintain clear spatial continuity shot-to-shot.",
    "japanese_gold_screen": "Animate with folding-screen reveals, shimmering gold textures catching light, and stylized cranes or pines sliding in layered parallax. Camera should glide ceremonially, echoing screen panels unfolding. Preserve the exact style and subject from the input frames. Keep body proportions stable across frames. Characters interact believably with their environment: when on solid ground, feet align naturally and do not drift; when jumping, flying, floating, or moving across non-solid surfaces (like water or clouds), motion arcs and contact look intentional and consistent with the story. Maintain clear spatial continuity shot-to-shot.",
    "japanese_scroll": "Animate with continuous side-scrolling motion across unfolding narrative scenes. Characters glide across the frame while layered parallax reveals landscapes gradually. Preserve the exact style and subject from the input frames. Keep body proportions stable across frames. Characters interact believably with their environment: when on solid ground, feet align naturally and do not drift; when jumping, flying, floating, or moving across non-solid surfaces (like water or clouds), motion arcs and contact look intentional and consistent with the story. Maintain clear spatial continuity shot-to-shot.",
    "japanese_court": "Animate with sliding panel reveals, gentle pans across seasonal landscapes, and subtle character gestures. Camera pacing should echo the elegance of Heian-era narrative scrolls. Preserve the exact style and subject from the input frames. Keep body proportions stable across frames. Characters interact believably with their environment: when on solid ground, feet align naturally and do not drift; when jumping, flying, floating, or moving across non-solid surfaces (like water or clouds), motion arcs and contact look intentional and consistent with the story. Maintain clear spatial continuity shot-to-shot.",
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
        return scenes

    # Prepare the scene descriptions for Claude
    scene_descriptions = []
    for i, scene in enumerate(scenes):
        scene_descriptions.append({
            "scene_index": i,
            "original_description": scene.get("scene_description", ""),
            "narration": scene.get("narration", "")
        })

    # Create the prompt with all scene descriptions - just the scenes, no duplicate instructions
    scene_prompts = []
    for i, scene in enumerate(scene_descriptions):
        scene_prompts.append(f"Scene {i}: {scene['original_description']}")

    # Join all scene prompts
    full_prompt = "\n".join(scene_prompts)

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
        provider = get_llm_provider()
        response_data = provider.chat_json(
            model="claude-opus-4-1-20250805",
            messages=[
                {"role": "system", "content": "You are a creative video scene enhancer. You must output ONLY valid JSON, no other text."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )

        enhanced_scenes = response_data.get("enhanced_scenes", [])
        if not enhanced_scenes:
            print("[WARNING] No enhanced scenes returned from Claude, keeping original descriptions")
            return scenes

        # Update the scenes with rewritten prompts
        for enhanced_scene in enhanced_scenes:
            scene_index = enhanced_scene.get("scene_index")
            if scene_index is not None and scene_index < len(scenes):
                rewritten_prompt = enhanced_scene.get("rewritten_prompt", "")
                if rewritten_prompt:
                    scenes[scene_index]["scene_description"] = rewritten_prompt

        print(f"[DEBUG] Successfully rewritten {len(enhanced_scenes)} scene descriptions into video prompts")
        return scenes

    except Exception as e:
        print(f"[ERROR] Failed to enhance scene descriptions: {e}")
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
# COMMENTED OUT: This function was creating 1+s chunks by over-generating and trimming
# def trim_to_target(src_path: Path, dst_path: Path, target_sec: float, fps: int) -> float:
#     """
#     Open the raw clip and trim to exactly target_sec. If the raw clip is (unexpectedly) shorter,
#     we just write it through as-is (no padding, per your requirement).
#     Returns the raw duration.
#     """
#     clip = VideoFileClip(str(src_path), audio=False)
#     try:
#         cur = float(clip.duration or 0.0)
#         if cur >= target_sec:
#             fixed = clip.subclip(0, target_sec)
#             try:
#                 fixed.write_videofile(
#                     str(dst_path),
#                     fps=int(clip.fps or fps),
#                     codec="libx264",
#                     audio=False,
#                     preset="medium",
#                     threads=2,
#                     verbose=False,
#                     logger=None,
#                     ffmpeg_params=["-movflags", "+faststart"],
#                 )
#             finally:
#                 try: fixed.close()
#                 except Exception: pass
#         else:
#             # You said this will never happen; if it does, we just pass it through.
#             # (No Ken Burns, no retries.)
#             clip.write_videofile(
#                 str(dst_path),
#                 fps=int(clip.fps or fps),
#                 codec="libx264",
#                 audio=False,
#                 preset="medium",
#                 threads=2,
#                 verbose=False,
#                 logger=None,
#                 ffmpeg_params=["-movflags", "+faststart"],
#             )
#         return cur
#     finally:
#         try: clip.close()
#         except Exception: pass

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
    scenes = enhance_scene_descriptions_with_claude(scenes)
    print(f"[DEBUG] Enhanced {len(scenes)} scene descriptions for video generation")

    # Note: Image prompts are no longer loaded - using scene descriptions directly

    # Dictionary to store video prompts for logging
    video_prompts_log = {}

    print("  Strategy: Generate 5-second videos for all scenes (no trimming to exact scene timing).")
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

        # Always generate 5 seconds for all scenes (no over-generation, just consistent timing)
        request_int = 5

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
                "If any text appears in the image render it in English language"
                "Respect accurate perspective and depth, keeping realistic distance, proportions, and scale between foreground and background objects."
               
            )
        else:
            # Fallback if no scene description available
            prompt = (
                f"{animation_style}\n"
                
                "Do not add or duplicate characters unless specified. "                
                "If any text appears in the image render it in English language"
                "Respect accurate perspective and depth, keeping realistic distance, proportions, and scale between foreground and background objects."
            )

        # Log the video prompt for this scene
        video_prompts_log[scene_id] = {
            "scene_description": scene_text,
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

            # No trimming needed - use the raw video directly
            # Just move/rename the file from raw to final
            safe_replace(raw_path, out_path)

            # Log successful video chunk creation
            timestamp = datetime.now().isoformat()
            logger.info(f"VIDEO_CHUNK_CREATED - Scene: {scene_id}, Timestamp: {timestamp}, "
                        f"Input: {img_path.name}, Output: {out_path.name}, "
                        f"Target Duration: {target:.3f}s, "
                        f"Requested Duration: {request_int}s, Model: {args.model}, "
                        f"Resolution: {args.resolution}, FPS: {args.fps}, Style: {style_key}, "
                        f"Scene Description: {scene_text[:100]}{'...' if len(scene_text) > 100 else ''}")

            print(f"✅ {out_path.name} (target {target:.3f}s, requested {request_int}s)")
        except Exception as e:
            # Log failed video chunk creation
            timestamp = datetime.now().isoformat()
            logger.error(f"VIDEO_CHUNK_FAILED - Scene: {scene_id}, Timestamp: {timestamp}, "
                        f"Input: {img_path.name}, Target Duration: {target:.3f}s, "
                        f"Requested Duration: {request_int}s, Model: {args.model}, "
                        f"Resolution: {args.resolution}, FPS: {args.fps}, "
                        f"Style: {style_key}, Error: {str(e)}, "
                        f"Scene Description: {scene_text[:100]}{'...' if len(scene_text) > 100 else ''}")

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
