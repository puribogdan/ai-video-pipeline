# generate_images_flux_schnell.py
# Minimal pipeline with prompt logging + robust retries:
#   • Scene 1: google/nano-banana (text → image)
#   • Scenes 2..N: google/nano-banana (image edit)
#       - Scene 2 edits FROM scene_001
#       - Scene 3+ edit FROM [scene_001, previous scene]  (dual refs)
#   • Only fields used: prompt (always), image_input (for edits). Nothing else.
#   • No ref preprocessing. No cropping on save (save native size as PNG).
#   • Prompts logged to scenes/prompt.json

from __future__ import annotations
import os, io, sys, json, argparse, contextlib, time, random
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from PIL import Image
import replicate

# -------------------- Model --------------------
NANO_BANANA_MODEL = "google/nano-banana"

# -------------------- Paths --------------------
ROOT         = Path(__file__).parent
SCRIPT_PATH  = ROOT / "scripts" / "input_script.json"
SCENES_DIR   = ROOT / "scenes"
SCENES_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST     = SCENES_DIR / "manifest.json"
PROMPT_JSON  = SCENES_DIR / "prompt.json"

# -------------------- Defaults --------------------
DEFAULT_FORCE = True  # overwrite existing frames if present
DEFAULT_STYLE = "Bright, kid-friendly, simple shapes, soft lighting"

# Retry controls (env-tunable)
MAX_ATTEMPTS    = int(os.getenv("IMG_MAX_ATTEMPTS", "5"))
BASE_DELAY_S    = float(os.getenv("IMG_RETRY_BASE_DELAY", "1.5"))
MAX_DELAY_S     = 20.0

# -------------------- Helpers --------------------
def _sleep_backoff(attempt: int) -> None:
    """Exponential backoff with a little jitter."""
    delay = min(BASE_DELAY_S * (2 ** (attempt - 1)), MAX_DELAY_S)
    delay += random.uniform(0.0, 0.25 * delay)
    time.sleep(delay)

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
    # Replicate may return a URL or a file-like object; try both.
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
    # Handles lists, dicts, and strings returned by Replicate.
    if isinstance(out, str) and out.startswith(("http://", "https://")):
        with contextlib.suppress(Exception):
            return _download_bytes(out)
    if isinstance(out, (list, tuple)):
        for item in out:
            # URL case
            if isinstance(item, str) and item.startswith(("http://", "https://")):
                with contextlib.suppress(Exception):
                    return _download_bytes(item)
            # Nested structures
            if isinstance(item, (list, tuple, dict)):
                b = outputs_to_image_bytes(item)
                if b:
                    return b
            # FileOutput-ish
            b = _fileoutput_to_bytes(item)
            if b:
                return b
    if isinstance(out, dict):
        for v in out.values():
            b = outputs_to_image_bytes(v)
            if b:
                return b
    # Last chance
    b = _fileoutput_to_bytes(out)
    if b:
        return b
    return None

def save_png_no_resize(path: Path, img_bytes: bytes) -> None:
    """
    Save the model output as PNG WITHOUT resizing/cropping.
    We decode with PIL (to normalize format) and save directly as PNG.
    """
    im = Image.open(io.BytesIO(img_bytes))
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, format="PNG")

# -------------------- Prompt builders --------------------
def build_t2i_prompt(desc: str, style: str) -> str:
    # Keep it simple to mimic UI behavior.
    return f"{desc}\n{style}"

def build_edit_prompt(desc: str) -> str:
    # Minimal edit prompt guided by refs.
    return f"SCENE BRIEF: {desc}\nMatch the main character(s) identity and outfit from the reference image(s)."

# -------------------- Model calls (single-attempt primitives) --------------------
def _run_nano_banana_t2i_once(prompt: str):
    # ONLY 'prompt' (no extra params).
    return replicate.run(
        NANO_BANANA_MODEL,
        input={"prompt": prompt},
    )

def _run_nano_banana_edit_once(prompt: str, refs: List[Path]):
    # ONLY 'prompt' + 'image_input' (list).
    files = [open(p, "rb") for p in refs]
    try:
        return replicate.run(
            NANO_BANANA_MODEL,
            input={
                "prompt": prompt,
                "image_input": files,   # dual ref supported
            },
        )
    finally:
        for f in files:
            with contextlib.suppress(Exception):
                f.close()

# -------------------- Robust wrappers with retry --------------------
def generate_t2i_with_retry(prompt: str) -> bytes:
    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            out = _run_nano_banana_t2i_once(prompt)
            data = outputs_to_image_bytes(out)
            if data:
                if attempt > 1:
                    print(f"✅ T2I succeeded on attempt {attempt}/{MAX_ATTEMPTS}", flush=True)
                return data
            raise RuntimeError("Empty output from model")
        except Exception as e:
            last_err = e
            print(f"⚠️  T2I attempt {attempt}/{MAX_ATTEMPTS} failed: {e}", flush=True)
            if attempt < MAX_ATTEMPTS:
                _sleep_backoff(attempt)
    raise RuntimeError(f"T2I failed after {MAX_ATTEMPTS} attempts: {last_err}")

def generate_edit_with_retry(prompt: str, refs: List[Path]) -> bytes:
    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            out = _run_nano_banana_edit_once(prompt, refs)
            data = outputs_to_image_bytes(out)
            if data:
                if attempt > 1:
                    print(f"✅ EDIT succeeded on attempt {attempt}/{MAX_ATTEMPTS}", flush=True)
                return data
            raise RuntimeError("Empty output from model")
        except Exception as e:
            last_err = e
            print(f"⚠️  EDIT attempt {attempt}/{MAX_ATTEMPTS} failed: {e}", flush=True)
            if attempt < MAX_ATTEMPTS:
                _sleep_backoff(attempt)
    raise RuntimeError(f"EDIT failed after {MAX_ATTEMPTS} attempts: {last_err}")

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description="Generate scene images with google/nano-banana (text & edit).")
    parser.add_argument("--limit", type=int, default=None, help="Number of scenes to generate (default: all).")
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("REPLICATE_API_TOKEN"):
        print("ERROR: Missing REPLICATE_API_TOKEN in .env", file=sys.stderr)
        sys.exit(1)

    scenes = load_scenes()
    if args.limit is not None:
        scenes = scenes[:max(1, args.limit)]

    print(f"🖼️  model={NANO_BANANA_MODEL}")
    print(f"   frames: {len(scenes)} (limit={'all' if args.limit is None else args.limit}), force={DEFAULT_FORCE}")
    print("   edit references: scene_001 for #2; [scene_001, previous] for #3+")

    manifest: Dict[str, Dict[str, Any]] = {}
    prompt_log: Dict[str, Dict[str, str]] = {}

    ref_png: Optional[Path] = None   # path to scene_001.png
    prev_png: Optional[Path] = None  # path to most recently generated frame

    for i, s in enumerate(scenes, start=1):
        sid = f"{i:03d}"
        out_path = SCENES_DIR / f"scene_{sid}.png"

        # Skip existing unless force
        if out_path.exists() and out_path.stat().st_size > 1024 and not DEFAULT_FORCE:
            manifest[sid] = {"image_path": str(out_path), "mode": "skip-existing"}
            if i == 1 and ref_png is None:
                ref_png = out_path
            prev_png = out_path
            continue

        desc = (s.get("scene_description") or s.get("narration") or "Simple scene.").strip()

        try:
            if i == 1:
                # TEXT → IMAGE (with retries)
                prompt = build_t2i_prompt(desc=desc, style=DEFAULT_STYLE)
                img_bytes = generate_t2i_with_retry(prompt)
                mode = "t2i"
                prompt_log[sid] = {
                    "mode": mode,
                    "model": NANO_BANANA_MODEL,
                    "prompt": prompt,
                }
            else:
                # EDIT (with retries)
                if i == 2:
                    if ref_png is None or not ref_png.exists():
                        raise RuntimeError("scene_001.png not found; cannot perform edit for scene 002.")
                    refs = [ref_png]
                else:
                    if ref_png is None or not ref_png.exists() or prev_png is None or not prev_png.exists():
                        raise RuntimeError(f"Missing references for scene {sid}.")
                    refs = [ref_png, prev_png]

                edit_prompt = build_edit_prompt(desc=desc)
                img_bytes = generate_edit_with_retry(edit_prompt, refs)
                mode = "edit"
                prompt_log[sid] = {
                    "mode": mode,
                    "model": NANO_BANANA_MODEL,
                    "prompt": edit_prompt,
                    "references": ", ".join([p.name for p in refs]),
                }

            # Save AS-IS size (no crop/resize)
            save_png_no_resize(out_path, img_bytes)

            if i == 1 and ref_png is None:
                ref_png = out_path
            prev_png = out_path

            manifest[sid] = {
                "image_path": str(out_path),
                "mode": mode,
                **({"edit_refs": [str(p) for p in ([ref_png] if i == 2 else ([ref_png, prev_png] if i > 2 else []))]} if mode == "edit" else {}),
            }

        except Exception as e:
            # All retries failed; write placeholder but DO NOT silently move on without a frame.
            print(f"❌ Scene {sid} failed after {MAX_ATTEMPTS} attempts: {e}", flush=True)
            placeholder = Image.new("RGB", (64, 64), (220, 220, 220))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            placeholder.save(out_path, "PNG")
            if i == 1 and ref_png is None:
                ref_png = out_path
            prev_png = out_path
            manifest[sid] = {"image_path": str(out_path), "error": str(e), "retries": MAX_ATTEMPTS}

    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    PROMPT_JSON.write_text(json.dumps(prompt_log, indent=2), encoding="utf-8")
    print(f"✅ Done. Images in {SCENES_DIR} | Manifest: {MANIFEST}")
    print(f"📝 Prompts logged to: {PROMPT_JSON}")

if __name__ == "__main__":
    main()
