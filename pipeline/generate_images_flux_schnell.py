# generate_images_flux_schnell.py
# Scenes:
#   • Scene 1: t2i normally, OR if --portrait/ENV provided -> edit with portrait ref
#   • Scene 2: edit FROM [scene_001, portrait?]
#   • Scene 3+: edit FROM [scene_001, previous, portrait?]
#   • Prompts logged to scenes/prompt.json
#   • Retries each scene (3x) on transient failures

from __future__ import annotations
import os, io, sys, json, argparse, contextlib, time
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
DEFAULT_FORCE = True  # overwrite frames if present
DEFAULT_STYLE = "Bright, kid-friendly, simple shapes, soft lighting"
MAX_TRIES     = 3
RETRY_SLEEP_S = 3.0

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

# -------------------- Prompt builders --------------------
IDENTITY_GUIDE = (
    "Cast the person from the reference photo as the main character. "
    "Match face structure, age, skin tone, and hair shape (not the background). "
    "Keep a friendly, appealing look suitable for a kids' story. "
)

def build_t2i_prompt(desc: str, style: str, has_portrait: bool) -> str:
    # If portrait present, steer the model to adopt that identity even for scene 1.
    id_line = (IDENTITY_GUIDE + "Use the reference person as the protagonist.") if has_portrait else ""
    return f"{desc}\n{style}\n{id_line}".strip()

def build_edit_prompt(desc: str, has_portrait: bool) -> str:
    id_line = "Maintain the protagonist's identity from the reference image(s) (match face & hair & beard & clothes). Change subjects body position." if has_portrait else "Maintain visual continuity with the reference image(s).Change subjects body position."
    return f"SCENE BRIEF: {desc}\n{id_line}"

# -------------------- Model calls --------------------
def run_nano_banana_t2i(prompt: str):
    return replicate.run(
        NANO_BANANA_MODEL,
        input={"prompt": prompt},
    )

def run_nano_banana_with_refs(prompt: str, refs: List[Path]):
    files = [open(p, "rb") for p in refs]
    try:
        return replicate.run(
            NANO_BANANA_MODEL,
            input={
                "prompt": prompt,
                "image_input": files,  # refs drive identity/continuity
            },
        )
    finally:
        for f in files:
            with contextlib.suppress(Exception):
                f.close()

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description="Generate scene images with google/nano-banana (t2i + ref edits).")
    parser.add_argument("--limit", type=int, default=None, help="Number of scenes to generate (default: all).")
    parser.add_argument("--portrait", type=str, default=None, help="Optional path to a user face image to cast as the main character.")
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("REPLICATE_API_TOKEN"):
        print("ERROR: Missing REPLICATE_API_TOKEN in .env", file=sys.stderr)
        sys.exit(1)

    # portrait source (CLI takes precedence over ENV)
    env_portrait = os.getenv("PORTRAIT_PATH")
    portrait_path = Path(args.portrait or env_portrait) if (args.portrait or env_portrait) else None
    has_portrait = bool(portrait_path and Path(portrait_path).exists())

    scenes = load_scenes()
    if args.limit is not None:
        scenes = scenes[:max(1, args.limit)]

    print(f"🖼️  model={NANO_BANANA_MODEL}")
    print(f"   frames: {len(scenes)} (limit={'all' if args.limit is None else args.limit}), force={DEFAULT_FORCE}")
    if has_portrait:
        print(f"   portrait: {portrait_path}")
    print("   edit refs: #2 uses [scene_001, portrait?]; #3+ uses [scene_001, previous, portrait?]")

    manifest: Dict[str, Dict[str, Any]] = {}
    prompt_log: Dict[str, Dict[str, str]] = {}

    ref_png: Optional[Path] = None   # path to scene_001.png
    prev_png: Optional[Path] = None  # path to most recently generated frame

    for i, s in enumerate(scenes, start=1):
        sid = f"{i:03d}"
        out_path = SCENES_DIR / f"scene_{sid}.png"

        if out_path.exists() and out_path.stat().st_size > 1024 and not DEFAULT_FORCE:
            manifest[sid] = {"image_path": str(out_path), "mode": "skip-existing"}
            if i == 1 and ref_png is None:
                ref_png = out_path
            prev_png = out_path
            continue

        desc = (s.get("scene_description") or s.get("narration") or "Simple scene.").strip()

        tries = 0
        while True:
            tries += 1
            try:
                if i == 1:
                    # If portrait exists: treat scene 1 as a ref-guided generation
                    if has_portrait:
                        prompt = build_t2i_prompt(desc=desc, style=DEFAULT_STYLE, has_portrait=True)
                        out = run_nano_banana_with_refs(prompt, [portrait_path])  # type: ignore[arg-type]
                        mode = "t2i+portrait-ref"
                        prompt_log[sid] = {
                            "mode": mode,
                            "model": NANO_BANANA_MODEL,
                            "prompt": prompt,
                            "references": Path(portrait_path).name if portrait_path else "",  # type: ignore[arg-type]
                        }
                    else:
                        prompt = build_t2i_prompt(desc=desc, style=DEFAULT_STYLE, has_portrait=False)
                        out = run_nano_banana_t2i(prompt)
                        mode = "t2i"
                        prompt_log[sid] = {
                            "mode": mode,
                            "model": NANO_BANANA_MODEL,
                            "prompt": prompt,
                        }
                else:
                    # EDIT
                    refs: List[Path] = []
                    if i == 2:
                        if ref_png is None or not ref_png.exists():
                            raise RuntimeError("scene_001.png not found; cannot perform edit for scene 002.")
                        refs = [ref_png]
                        if has_portrait:
                            refs.append(Path(portrait_path))  # type: ignore[arg-type]
                    else:
                        if ref_png is None or not ref_png.exists() or prev_png is None or not prev_png.exists():
                            raise RuntimeError(f"Missing references for scene {sid}.")
                        refs = [ref_png, prev_png]
                        if has_portrait:
                            refs.append(Path(portrait_path))  # type: ignore[arg-type]

                    edit_prompt = build_edit_prompt(desc=desc, has_portrait=has_portrait)
                    out = run_nano_banana_with_refs(edit_prompt, refs)
                    mode = "edit"
                    prompt_log[sid] = {
                        "mode": mode,
                        "model": NANO_BANANA_MODEL,
                        "prompt": edit_prompt,
                        "references": ", ".join([p.name for p in refs]),
                    }

                data = outputs_to_image_bytes(out)
                if not data:
                    print(f"DEBUG output type: {type(out)}; sample: {str(out)[:300]}")
                    raise RuntimeError(f"No usable image bytes in model output (scene {sid}).")

                save_png_no_resize(out_path, data)

                if i == 1 and ref_png is None:
                    ref_png = out_path
                prev_png = out_path

                manifest[sid] = {
                    "image_path": str(out_path),
                    "mode": mode,
                    **({"edit_refs": [str(p) for p in refs]} if mode == "edit" else {}),
                }
                break  # success -> exit retry loop

            except Exception as e:
                if tries < MAX_TRIES:
                    print(f"⚠️  Scene {sid} attempt {tries} failed: {e} — retrying in {RETRY_SLEEP_S}s…")
                    time.sleep(RETRY_SLEEP_S)
                else:
                    print(f"❌ Scene {sid} failed after {tries} attempts: {e}")
                    # Write a placeholder so the pipeline keeps moving
                    placeholder = Image.new("RGB", (64, 64), (220, 220, 220))
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    placeholder.save(out_path, "PNG")
                    if i == 1 and ref_png is None:
                        ref_png = out_path
                    manifest[sid] = {"image_path": str(out_path), "error": str(e)}
                    break

    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    PROMPT_JSON.write_text(json.dumps(prompt_log, indent=2), encoding="utf-8")
    print(f"✅ Done. Images in {SCENES_DIR} | Manifest: {MANIFEST}")
    print(f"📝 Prompts logged to: {PROMPT_JSON}")

if __name__ == "__main__":
    main()
