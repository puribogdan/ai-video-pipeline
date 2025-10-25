# generate_images_flux_schnell.py
# Portrait-aware minimal pipeline with prompt logging.
# - If PORTRAIT_PATH is set and exists:
#     • Preprocess once: portrait_cutout = uploaded portrait with background removed (transparent PNG).
#     • Scene 1 = EDIT using [portrait_cutout]
#     • Scene 2 = EDIT using [portrait_cutout, scene_001]
#     • Scene 3+ = EDIT using [portrait_cutout, previous]
#     • Add the "portrait identity" prompt block to every scene (identity = image[0]; single instance; always included; can be protagonist unless the main subject is non-human, then supporting).
# - Else (no portrait):
#     • Scene 1 = T2I
#     • Scene 2 = EDIT using [scene_001]
#     • Scene 3+ = EDIT using [previous]
# - Only the *style* line is variable (chosen by user). Everything else in prompts stays the same.
# - Saves native PNGs, logs prompts to scenes/prompt.json

from __future__ import annotations
import os, io, sys, json, argparse, contextlib, time, hashlib, logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from shutil import copy2
from datetime import datetime

import requests
from dotenv import load_dotenv
from PIL import Image
import replicate
import rembg

# Import B2 upload function
try:
    from app.worker_tasks import upload_images_to_b2, log as worker_log
except ImportError:
    # Fallback if not available
    upload_images_to_b2 = None
    def worker_log(msg): print(f"[worker] {msg}", flush=True)

# -------------------- Model --------------------
NANO_BANANA_MODEL = "google/nano-banana"

# -------------------- Paths --------------------
ROOT         = Path(__file__).parent
SCRIPT_PATH  = ROOT / "scripts" / "input_script.json"
SCENES_DIR   = ROOT / "scenes"
SCENES_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR   = ROOT / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST     = SCENES_DIR / "manifest.json"
PROMPT_JSON  = SCENES_DIR / "prompt.json"

# -------------------- Logging Setup --------------------
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "image_generation.log"

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

# -------------------- Style Library --------------------
STYLE_LIBRARY: Dict[str, str] = {
    "kid_friendly_cartoon": "A cartoon style inspired by Nickelodeon and Cartoon Network classics. Preserve bright colors, rounded character designs, and exaggerated playful expressions. Preserve the exact style and subject from the start frame.",
    "japanese_kawaii": "A kawaii chibi style inspired by Sanrio characters like Hello Kitty and Pompompurin. Preserve pastel palettes, big sparkly eyes, soft rounded shapes, and cozy cute details. Preserve the exact style and subject from the start frame.",
    "storybook_illustrated": "A cut-out collage style inspired by Eric Carle and Ezra Jack Keats. Preserve layered textures, hand-painted paper patterns, and playful handmade charm. Preserve the exact style and subject from the start frame.",
    "watercolor_storybook": "A watercolor illustration style inspired by Eric Carle and Ezra Jack Keats. Preserve soft pastel washes, hand-painted brush textures, and visible paper grain. Preserve the exact style and subject from the start frame.",
    "paper_cutout": "A paper cut-out style inspired by South Park's early seasons. Preserve flat 2D shapes, bold outlines, lo-fi paper textures, and simple collage charm. Preserve the exact style and subject from the start frame.",
    "cutout_collage": "A surreal cut-out collage style inspired by Terry Gilliam's Monty Python animations. Preserve mismatched magazine textures, bold cut edges, and layered paper compositions. Preserve the exact style and subject from the start frame.",
    "realistic_3d": "A cinematic 3D style inspired by Pixar and DreamWorks. Preserve high-quality rendering, expressive character eyes, detailed textures, and polished lighting. Preserve the exact style and subject from the start frame.",
    "claymation": "A claymation style inspired by Aardman Studios and Tumble Leaf. Preserve hand-molded clay textures, colorful sets, soft lighting, and visible handcrafted imperfections. Preserve the exact style and subject from the start frame.",
    "needle_felted": "A felted stop-motion style inspired by Pui Pui Molcar. Preserve fuzzy wool textures, pastel color palettes, and plush toy-like rounded characters. Preserve the exact style and subject from the start frame.",
    "stop_motion_felt_clay": "A handcrafted style inspired by Pokémon Concierge. Preserve miniature sets, felt and clay textures, soft natural lighting, and cozy handmade charm. Preserve the exact style and subject from the start frame.",
    "hybrid_mix": "A hybrid cartoon style inspired by The Amazing World of Gumball. Preserve flat 2D characters placed in semi-realistic 3D environments, maintaining playful contrast. Preserve the exact style and subject from the start frame.",
    "japanese_anime": "An anime style inspired by Kyoto Animation and Toei. Preserve sharp linework, expressive eyes, stylized hair and costumes, and bold dramatic angles. Preserve the exact style and subject from the start frame.",
    "pixel_art": "A pixel art style inspired by Minecraft and retro 8-bit games. Preserve blocky voxel characters, chunky textures, and colorful simplified environments. Preserve the exact style and subject from the start frame.",
    "van_gogh": "A painterly style inspired by Vincent van Gogh. Preserve swirling brushstrokes, thick textured oils, vivid contrasting colors, and bold impasto effects. Preserve the exact style and subject from the start frame.",
    "impressionism": "An Impressionist style inspired by Claude Monet and Pierre-Auguste Renoir. Preserve soft brushstrokes, dappled light, pastel colors, and dreamy open-air compositions. Preserve the exact style and subject from the start frame.",
    "art_deco": "An Art Deco style inspired by Tamara de Lempicka and 1920s poster art. Preserve sleek geometry, metallic tones, streamlined symmetry, and glamorous forms. Preserve the exact style and subject from the start frame.",
    "motion_comic": "A manga style inspired by Japanese comics. Preserve bold linework, screentone textures, expressive facial features, and iconic manga aesthetics. Preserve the exact style and subject from the start frame.",
    "comic_book": "A comic book style inspired by Jack Kirby and classic superhero comics. Preserve bold outlines, halftone dots, vibrant colors, and exaggerated action poses. Preserve the exact style and subject from the start frame.",
    "gothic": "A gothic whimsical style inspired by Tim Burton and Edward Gorey. Preserve elongated characters, moody lighting, and playful gothic charm. Preserve the exact style and subject from the start frame.",
    "fantasy_magic_glow": "A glowing fantasy style inspired by Studio Ghibli and Disney's Fantasia. Preserve luminous glows, soft gradients, spark-like particles, and magical enchanted settings. Preserve the exact style and subject from the start frame.",
    "surrealism_hybrid": "A surrealist whimsical style blending Dr. Seuss and Salvador Dalí. Preserve Seussian curves, surreal dreamscapes, and imaginative distorted forms. Preserve the exact style and subject from the start frame.",
    "japanese_woodblock": "A woodblock style inspired by Hokusai and Hiroshige. Preserve bold outlines, flat layered colors, and dramatic wave and mountain compositions. Preserve the exact style and subject from the start frame.",
}

DEFAULT_STYLE_KEY = "kid_friendly_cartoon"  # fallback if env not set

# -------------------- Camera Angle Variation Prompt --------------------
CAMERA_ANGLE_PROMPT = """Add variation:
Use a different camera angle or framing for this image — for example, a close-up, wide shot, low angle, high angle, or over-the-shoulder view — while keeping the same style and subject consistency."""

# -------------------- Helpers --------------------
def load_scenes() -> List[Dict[str, Any]]:
    if not SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Missing {SCRIPT_PATH}. Run generate_script.py first.")
    data = json.loads(SCRIPT_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("scripts/input_script.json must be a non-empty list.")
    for i, s in enumerate(data, 1):
        if "scene_description" not in s and "narration" not in s:
            raise ValueError(f"Scene {i} missing 'scene_description' or 'narration'.")
    return data

def _download_bytes(url: str) -> bytes:
    r = requests.get(url, stream=True, timeout=180)
    r.raise_for_status()
    return r.content

def _fileoutput_to_bytes(obj: Any) -> Optional[bytes]:
    for attr in ("url", "uri", "href"):
        val = getattr(obj, attr, None)
        if isinstance(val, str) and val.startswith(("http://", "https://")):
            with contextlib.suppress(Exception):
                return _download_bytes(val)
    if hasattr(obj, "open"):
        try:
            with obj.open() as f:  # type: ignore[attr-defined]
                return f.read()
        except Exception:
            pass
    if hasattr(obj, "read"):
        with contextlib.suppress(Exception):
            return obj.read()  # type: ignore[attr-defined]
    for meth in ("save", "download", "write_to"):
        if hasattr(obj, meth):
            tmp = SCENES_DIR / f"rep_tmp_download"
            with contextlib.suppress(Exception):
                getattr(obj, meth)(str(tmp))  # type: ignore[misc]
                data = tmp.read_bytes()
                with contextlib.suppress(Exception): tmp.unlink()
                return data
    return None

def outputs_to_image_bytes(out: Any) -> Optional[bytes]:
    if isinstance(out, str) and out.startswith(("http://", "https://")):
        with contextlib.suppress(Exception):
            return _download_bytes(out)
    if isinstance(out, (list, tuple)):
        for item in out:
            if isinstance(item, str) and item.startswith(("http://", "https://")):
                with contextlib.suppress(Exception):
                    return _download_bytes(item)
            if isinstance(item, (list, tuple, dict)):
                b = outputs_to_image_bytes(item)
                if b:
                    return b
            b = _fileoutput_to_bytes(item)
            if b:
                return b
    if isinstance(out, dict):
        for v in out.values():
            b = outputs_to_image_bytes(v)
            if b:
                return b
    b = _fileoutput_to_bytes(out)
    if b:
        return b
    return None

def save_png_no_resize(path: Path, img_bytes: bytes) -> None:
    im = Image.open(io.BytesIO(img_bytes))
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, format="PNG")

def save_image_with_b2_backup(scene_path: Path, img_bytes: bytes, job_id: Optional[str] = None) -> None:
    """Save image to both scenes directory and images directory, and upload to B2 if job_id provided."""
    # Save to scenes directory (original location)
    save_png_no_resize(scene_path, img_bytes)

    # Also save to images directory for B2 upload
    if job_id:
        image_filename = scene_path.name
        image_path = IMAGES_DIR / image_filename
        save_png_no_resize(image_path, img_bytes)
        worker_log(f"Saved image to images directory: {image_path}")

def _sha12(path: Path) -> str:
    import hashlib
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]

# -------------------- Prompt builders (style line and portrait instructions vary) --------------------
def build_scene1_prompt(desc: str, style_line: str, has_portrait: bool) -> str:
    if has_portrait:
        portrait_prompt = """image[0] shows a recurring human character.
Keep the same face, hairstyle, body type, and outfit as in image[0] across all scenes.
Always include this person somewhere in the frame — they should never disappear or be off-screen.
Only one instance of this person should appear (no duplicates, clones, or twins).
Other people or animals may appear naturally as required by the story.

If the story primarily features human characters, the person from image[0] should be the main protagonist and central focus of the scene.
If the story's main subject is non-human (for example, an animal, vehicle, or landscape),
include the person from image[0] as a supporting or background character interacting naturally with the scene,
but do not make them the main focus.


Match the overall scene layout, mood, and direction to image[1].
Ensure the person from image[0] integrates naturally into the environment without duplication."""
        return f"{desc}\n{portrait_prompt}\n{CAMERA_ANGLE_PROMPT}\n{style_line}"
    else:
        return f"{desc}\n{CAMERA_ANGLE_PROMPT}\n{style_line}"

def build_edit_prompt(desc: str, style_line: str, has_portrait: bool) -> str:
    if has_portrait:
        portrait_prompt = """image[0] shows a recurring human character.
Keep the same face, hairstyle, body type, and outfit as in image[0] across all scenes.
Always include this person somewhere in the frame — they should never disappear or be off-screen.
Only one instance of this person should appear (no duplicates, clones, or twins).
Other people or animals may appear naturally as required by the story.

If the story primarily features human characters, the person from image[0] should be the main protagonist and central focus of the scene.
If the story's main subject is non-human (for example, an animal, vehicle, or landscape),
include the person from image[0] as a supporting or background character interacting naturally with the scene,
but do not make them the main focus.


Match the overall scene layout, mood, and direction to image[1].
Ensure the person from image[0] integrates naturally into the environment without duplication."""
        return f"SCENE BRIEF: {desc}\n{portrait_prompt}\n{CAMERA_ANGLE_PROMPT}\n{style_line}"
    else:
        return f"SCENE BRIEF: {desc}\n{CAMERA_ANGLE_PROMPT}\n{style_line}"

# -------------------- Model calls (with enhanced retry) --------------------
def _run_with_retry(fn, *args, retries: int = 3, delay: float = 1.5, **kwargs):
    last = None
    for attempt in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last = e
            if attempt < retries:
                time.sleep(delay * (attempt + 1))
            else:
                raise last

def log_image_generation_event(scene_id, event_type, message, **kwargs):
    """Log detailed image generation events for debugging"""
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "scene_id": scene_id,
        "event_type": event_type,
        "message": message,
        **kwargs
    }
    logger.info(f"IMAGE_GENERATION - {json.dumps(log_entry)}")

def generate_scene_with_retry(scene_data, scene_id, out_path, job_id, max_scene_retries=2):
    """Generate a single scene with visible retry logging"""

    for scene_attempt in range(max_scene_retries):
        try:
            log_image_generation_event(scene_id, "generation_start", f"Starting generation attempt {scene_attempt + 1}/{max_scene_retries}",
                                     mode=scene_data["mode"], prompt_length=len(scene_data["prompt"]),
                                     refs_count=len(scene_data["refs"]) if scene_data["refs"] else 0)
            print(f"🖼️  Scene {scene_id}: Starting generation (attempt {scene_attempt + 1}/{max_scene_retries})")

            # Generate the image
            if scene_data["mode"] == "t2i":
                out = run_nano_banana_t2i(scene_data["prompt"])
            else:  # edit mode
                out = run_nano_banana_edit(scene_data["prompt"], scene_data["refs"])

            print(f"📸 Scene {scene_id}: Generation submitted, downloading...")

            # Download and process the image
            data = outputs_to_image_bytes(out)

            if data and len(data) > 1024:
                log_image_generation_event(scene_id, "generation_success", f"Succeeded on attempt {scene_attempt + 1}",
                                         data_size=len(data), attempt=scene_attempt + 1)
                print(f"✅ Scene {scene_id}: Succeeded on attempt {scene_attempt + 1}")
                save_image_with_b2_backup(out_path, data, job_id)
                return data
            else:
                log_image_generation_event(scene_id, "generation_failure", f"Generated data too small: {len(data) if data else 0} bytes",
                                         attempt=scene_attempt + 1, data_size=len(data) if data else 0)
                print(f"⚠️  Scene {scene_id}: Generated data too small ({len(data) if data else 0} bytes)")
                raise RuntimeError("Generated image too small")

        except Exception as e:
            error_msg = str(e)
            log_image_generation_event(scene_id, "generation_failure", f"Attempt {scene_attempt + 1} failed: {error_msg}",
                                     attempt=scene_attempt + 1, error_type=type(e).__name__)
            print(f"❌ Scene {scene_id}: Attempt {scene_attempt + 1} failed: {e}")

            if scene_attempt < max_scene_retries - 1:
                delay = 10 * (2 ** scene_attempt)  # 10s, 20s
                print(f"⏳ Scene {scene_id}: Retrying in {delay}s...")
                time.sleep(delay)
                continue
            else:
                log_image_generation_event(scene_id, "generation_final_failure", f"All {max_scene_retries} attempts failed",
                                         total_attempts=max_scene_retries, final_error=error_msg)
                print(f"💀 Scene {scene_id}: All {max_scene_retries} attempts failed")
                raise e

    return None

def run_nano_banana_t2i(prompt: str):
    return _run_with_retry(replicate.run, NANO_BANANA_MODEL, input={"prompt": prompt, "aspect_ratio": "16:9"})

def run_nano_banana_edit(prompt: str, refs: List[Path]):
    files = []
    for p in refs:
        f = open(p, "rb")
        files.append(f)
    try:
        return _run_with_retry(
            replicate.run,
            NANO_BANANA_MODEL,
            input={"prompt": prompt, "image_input": files, "aspect_ratio": "16:9"},
        )
    finally:
        for f in files:
            with contextlib.suppress(Exception):
                f.close()

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description="Generate scene images with google/nano-banana (portrait-aware, style-selectable).")
    parser.add_argument("--limit", type=int, default=None, help="Number of scenes to generate (default: all).")
    parser.add_argument("--job-id", type=str, default=None, help="Job ID for B2 upload (optional).")
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("REPLICATE_API_TOKEN"):
        print("ERROR: Missing REPLICATE_API_TOKEN in .env", file=sys.stderr)
        sys.exit(1)

    # Resolve style from env
    style_key = (os.getenv("STYLE_CHOICE", DEFAULT_STYLE_KEY) or DEFAULT_STYLE_KEY).lower().strip()
    style_line = STYLE_LIBRARY.get(style_key, STYLE_LIBRARY[DEFAULT_STYLE_KEY])
    print(f"🎨 Style: key={style_key} line=\"{style_line}\"")

    # Portrait support (via env set by worker)
    portrait_env = os.getenv("PORTRAIT_PATH", "").strip()
    portrait_path: Optional[Path] = None
    if portrait_env:
        src = Path(portrait_env)
        if src.exists() and src.is_file():
            SCENES_DIR.mkdir(parents=True, exist_ok=True)
            portrait_local = SCENES_DIR / ("portrait_ref" + src.suffix.lower())
            copy2(src, portrait_local)

            # Remove background from portrait image using rembg
            try:
                with Image.open(portrait_local) as img:
                    # Convert image to bytes before passing to rembg
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format='PNG')
                    img_bytes = img_buffer.getvalue()

                    # Process image bytes with rembg to remove background
                    output_bytes = rembg.remove(img_bytes)

                    # Save the processed image (without background)
                    processed_img = Image.open(io.BytesIO(output_bytes))
                    if processed_img.mode not in ("RGB", "RGBA"):
                        processed_img = processed_img.convert("RGBA")

                    # Create new filename for processed image
                    portrait_no_bg = SCENES_DIR / ("portrait_ref_no_bg.png")
                    processed_img.save(portrait_no_bg, format="PNG")

                    portrait_path = portrait_no_bg
                    print(f"👤 Using portrait reference (background removed): {portrait_path} (size={portrait_path.stat().st_size} sha1={_sha12(portrait_path)})")

                    # Also save the original image for database storage
                    portrait_original = SCENES_DIR / ("portrait_ref_original.png")
                    # Convert original to PNG if it's not already
                    if portrait_local.suffix.lower() != '.png':
                        original_img = Image.open(portrait_local)
                        if original_img.mode not in ("RGB", "RGBA"):
                            original_img = original_img.convert("RGB")
                        original_img.save(portrait_original, format="PNG")
                    else:
                        copy2(portrait_local, portrait_original)

                    print(f"💾 Saved both original and background-removed portrait images for database storage")

            except Exception as e:
                # Fallback to original image if rembg processing fails
                print(f"⚠️ Background removal failed: {e}, using original image")
                portrait_path = portrait_local

                # Still save original for database storage
                portrait_original = SCENES_DIR / ("portrait_ref_original.png")
                if portrait_local.suffix.lower() != '.png':
                    original_img = Image.open(portrait_local)
                    if original_img.mode not in ("RGB", "RGBA"):
                        original_img = original_img.convert("RGB")
                    original_img.save(portrait_original, format="PNG")
                else:
                    copy2(portrait_local, portrait_original)

                print(f"💾 Saved original portrait image for database storage")
                print(f"👤 Using portrait reference (original): {portrait_path} (size={portrait_path.stat().st_size} sha1={_sha12(portrait_path)})")
        else:
            print(f"⚠️ PORTRAIT_PATH set but file not found: {src}")

    scenes = load_scenes()
    if args.limit is not None:
        scenes = scenes[:max(1, args.limit)]

    job_id = args.job_id or os.getenv("JOB_ID")
    if job_id:
        print(f"📁 Job ID: {job_id}")

    print("🖼️  model=google/nano-banana")
    if portrait_path:
        print("   strategy: scene_001 = EDIT [portrait_cutout], scene_002 = EDIT [portrait_cutout, scene_001], scene_003+ = EDIT [portrait_cutout, previous] (portrait identity prompt in all scenes)")
    else:
        print("   strategy: scene_001 = T2I, scene_002 = EDIT [scene_001], scene_003+ = EDIT [previous]")
    print(f"   frames: {len(scenes)} (limit={'all' if args.limit is None else args.limit})")

    manifest: Dict[str, Dict[str, Any]] = {}
    prompt_log: Dict[str, Dict[str, str]] = {}

    ref_png: Optional[Path] = None
    prev_png: Optional[Path] = None

    for i, s in enumerate(scenes, start=1):
        sid = f"{i:03d}"
        out_path = SCENES_DIR / f"scene_{sid}.png"

        if out_path.exists() and out_path.stat().st_size > 1024:
            # DEFAULT_FORCE behavior retained (overwrite); if you want skip, add a flag
            pass

        desc = (s.get("scene_description") or s.get("narration") or "Simple scene.").strip()
        refs = []  # Initialize refs to avoid unbound variable

        try:
            # Prepare scene data for retry function
            scene_data = {}

            if portrait_path:
                # Use portrait_cutout for all scenes when portrait is present
                if i == 1:
                    # Scene 1: only needs portrait_path
                    refs = [portrait_path]
                elif i == 2:
                    # Scene 2: needs both portrait and scene_001
                    if ref_png is None or not ref_png.exists():
                        raise RuntimeError("scene_001.png not found; cannot perform edit for scene 002.")
                    refs = [portrait_path, ref_png]
                else:
                    # Scene 3+: needs portrait and previous scene
                    if prev_png is None or not prev_png.exists():
                        raise RuntimeError(f"Missing references for scene {sid}.")
                    refs = [portrait_path, prev_png]

                # Use portrait identity prompt block for every scene
                prompt = build_edit_prompt(desc=desc, style_line=style_line, has_portrait=True)
                mode = "edit"
                scene_data = {
                    "mode": mode,
                    "prompt": prompt,
                    "refs": refs
                }
                prompt_log[sid] = {
                    "mode": mode,
                    "model": NANO_BANANA_MODEL,
                    "prompt": prompt,
                    "references": ", ".join([Path(r).name for r in refs]),
                    "references_abs": ", ".join([str(r) for r in refs]),
                    "style_key": style_key,
                    "style_line": style_line,
                }

            else:
                # No portrait
                if i == 1:
                    prompt = build_scene1_prompt(desc=desc, style_line=style_line, has_portrait=False)
                    mode = "t2i"
                    scene_data = {
                        "mode": mode,
                        "prompt": prompt,
                        "refs": []
                    }
                    prompt_log[sid] = {
                        "mode": mode,
                        "model": NANO_BANANA_MODEL,
                        "prompt": prompt,
                        "style_key": style_key,
                        "style_line": style_line,
                    }
                else:
                    # Scene 2: Edit using scene_001
                    if i == 2:
                        if ref_png is None or not ref_png.exists():
                            raise RuntimeError("scene_001.png not found; cannot perform edit for scene 002.")
                        refs = [ref_png]
                    # Scene 3+: Edit using only the previous scene (not scene_001)
                    else:
                        if prev_png is None or not prev_png.exists():
                            raise RuntimeError(f"Missing previous scene for scene {sid}.")
                        refs = [prev_png]

                    prompt = build_edit_prompt(desc=desc, style_line=style_line, has_portrait=False)
                    mode = "edit"
                    scene_data = {
                        "mode": mode,
                        "prompt": prompt,
                        "refs": refs
                    }
                    prompt_log[sid] = {
                        "mode": mode,
                        "model": NANO_BANANA_MODEL,
                        "prompt": prompt,
                        "references": ", ".join([Path(r).name for r in refs]),
                        "references_abs": ", ".join([str(r) for r in refs]),
                        "style_key": style_key,
                        "style_line": style_line,
                    }

            # 🎯 Use scene-level retry function
            scene_bytes = generate_scene_with_retry(scene_data, sid, out_path, job_id, max_scene_retries=2)

            if scene_bytes:
                if i == 1 and ref_png is None:
                    ref_png = out_path
                prev_png = out_path

                entry: Dict[str, Any] = {"image_path": str(out_path), "mode": mode}
                if mode == "edit" and refs:
                    entry["edit_refs"] = [str(p) for p in refs]
                manifest[sid] = entry
            else:
                print(f"⚠️  Scene {sid}: Skipped - all retries failed")
                # Create placeholder for failed scene
                placeholder = Image.new("RGB", (64, 64), (220, 220, 220))
                out_path.parent.mkdir(parents=True, exist_ok=True)
                placeholder.save(out_path, "PNG")
                if i == 1 and ref_png is None:
                    ref_png = out_path
                manifest[sid] = {"image_path": str(out_path), "error": "All retry attempts failed"}

        except Exception as e:
            print(f"⚠️  Scene {sid} failed: {e}")
            placeholder = Image.new("RGB", (64, 64), (220, 220, 220))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            placeholder.save(out_path, "PNG")
            if i == 1 and ref_png is None:
                ref_png = out_path
            manifest[sid] = {"image_path": str(out_path), "error": str(e)}

    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    PROMPT_JSON.write_text(json.dumps(prompt_log, indent=2), encoding="utf-8")

    # Note: B2 upload is now handled by the worker process after video completion
    # This ensures all images are uploaded together with proper error handling

    print(f"✅ Done. Images in {SCENES_DIR} | Manifest: {MANIFEST}")
    print(f"📝 Prompts logged to: {PROMPT_JSON}")
    if job_id:
        print(f"🖼️ Images also saved to: {IMAGES_DIR}")

if __name__ == "__main__":
    main()
