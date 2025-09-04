# generate_images_flux_schnell.py
# Minimal pipeline with prompt logging:
#   • If PORTRAIT_REF is set and exists:
#       - Scene 1: EDIT using [portrait] (to lock identity)
#       - Scene 2..N: EDIT using [portrait, previous]
#   • Else (no portrait provided):
#       - Scene 1: T2I
#       - Scene 2: EDIT from scene_001
#       - Scene 3+: EDIT from [scene_001, previous]
#
#   • Only fields used: prompt (always), image_input (for edits).
#   • Prompts logged to scenes/prompt.json

from __future__ import annotations
import os, io, sys, json, argparse, contextlib
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
def build_t2i_prompt(desc: str, style: str) -> str:
    return f"{desc}\n{style}"

def build_edit_prompt(desc: str, with_identity_hint: bool) -> str:
    if with_identity_hint:
        return (
            "SCENE BRIEF: " + desc + "\n"
            "Maintain visual continuity with the reference image(s). "
            "Preserve the main character’s identity (face, hair, outfit) from the portrait reference."
        )
    else:
        return (
            "SCENE BRIEF: " + desc + "\n"
            "Maintain visual continuity with the reference image(s)."
        )

# -------------------- Model calls --------------------
def run_nano_banana_t2i(prompt: str):
    return replicate.run(
        NANO_BANANA_MODEL,
        input={"prompt": prompt},
    )

def run_nano_banana_edit(prompt: str, refs: List[Path]):
    files = []
    for p in refs:
        f = open(p, "rb")
        files.append(f)
    try:
        return replicate.run(
            NANO_BANANA_MODEL,
            input={
                "prompt": prompt,
                "image_input": files,   # accepts one or more refs
            },
        )
    finally:
        for f in files:
            with contextlib.suppress(Exception):
                f.close()

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description="Generate scene images with google/nano-banana (text & edit).")
    parser.add_argument("--limit", type=int, default=None, help="Number of scenes to generate (default: all).")
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("REPLICATE_API_TOKEN"):
        print("ERROR: Missing REPLICATE_API_TOKEN in .env", file=sys.stderr)
        sys.exit(1)

    # Optional portrait reference from worker (absolute path)
    portrait_env = os.getenv("PORTRAIT_REF", "").strip()
    PORTRAIT: Optional[Path] = Path(portrait_env) if portrait_env else None
    if PORTRAIT and not PORTRAIT.exists():
        print(f"⚠️  PORTRAIT_REF set but file missing: {PORTRAIT}", file=sys.stderr)
        PORTRAIT = None
    if PORTRAIT:
        print(f"👤 Using portrait reference: {PORTRAIT}")

    scenes = load_scenes()
    if args.limit is not None:
        scenes = scenes[:max(1, args.limit)]

    print(f"🖼️  model={NANO_BANANA_MODEL}")
    if PORTRAIT:
        print("   strategy: scene_001 = EDIT [portrait], others = EDIT [portrait, previous]")
    else:
        print("   strategy: scene_001 = T2I, scene_002 = EDIT [scene_001], scene_003+ = EDIT [scene_001, previous]")
    print(f"   frames: {len(scenes)} (limit={'all' if args.limit is None else args.limit}), force={DEFAULT_FORCE}")

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
            # Decide mode + refs
            if PORTRAIT:
                # With portrait: always use EDIT
                if i == 1:
                    refs = [PORTRAIT]
                else:
                    refs = [PORTRAIT]
                    if prev_png and prev_png.exists():
                        refs.append(prev_png)
                edit_prompt = build_edit_prompt(desc=desc, with_identity_hint=True)
                out = run_nano_banana_edit(edit_prompt, refs)
                mode = "edit"
                prompt_log[sid] = {
                    "mode": mode,
                    "model": NANO_BANANA_MODEL,
                    "prompt": edit_prompt,
                    "references": ", ".join([p.name for p in refs]),
                }

            else:
                # Original behavior (no portrait)
                if i == 1:
                    prompt = build_t2i_prompt(desc=desc, style=DEFAULT_STYLE)
                    out = run_nano_banana_t2i(prompt)
                    mode = "t2i"
                    prompt_log[sid] = {
                        "mode": mode,
                        "model": NANO_BANANA_MODEL,
                        "prompt": prompt,
                    }
                else:
                    if i == 2:
                        if ref_png is None or not ref_png.exists():
                            raise RuntimeError("scene_001.png not found; cannot perform edit for scene 002.")
                        refs = [ref_png]
                    else:
                        if ref_png is None or not ref_png.exists() or prev_png is None or not prev_png.exists():
                            raise RuntimeError(f"Missing references for scene {sid}.")
                        refs = [ref_png, prev_png]

                    edit_prompt = build_edit_prompt(desc=desc, with_identity_hint=False)
                    out = run_nano_banana_edit(edit_prompt, refs)
                    mode = "edit"
                    prompt_log[sid] = {
                        "mode": mode,
                        "model": NANO_BANANA_MODEL,
                        "prompt": edit_prompt,
                        "references": ", ".join([p.name for p in refs]),
                    }

            # Harvest image bytes
            data = outputs_to_image_bytes(out)
            if not data:
                print(f"DEBUG output type: {type(out)}; sample: {str(out)[:200]}")
                raise RuntimeError(f"No usable image bytes in model output (scene {sid}).")

            save_png_no_resize(out_path, data)

            if i == 1 and ref_png is None:
                ref_png = out_path
            prev_png = out_path

            man_entry: Dict[str, Any] = {"image_path": str(out_path), "mode": mode}
            if mode == "edit":
                man_entry["edit_refs"] = [str(p) for p in (refs if PORTRAIT else (refs))]
            manifest[sid] = man_entry

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
    print(f"✅ Done. Images in {SCENES_DIR} | Manifest: {MANIFEST}")
    print(f"📝 Prompts logged to: {PROMPT_JSON}")

if __name__ == "__main__":
    main()
