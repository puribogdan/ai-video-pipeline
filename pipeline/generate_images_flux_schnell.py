# generate_images_flux_schnell.py
# Portrait-aware minimal pipeline with prompt logging.
# - If PORTRAIT_PATH is set and exists:
#     • Scene 1 = EDIT using [portrait]
#     • Scene 2 = EDIT using [scene_001, portrait]
#     • Scene 3+ = EDIT using [scene_001, previous, portrait]
# - Else (no portrait):
#     • Scene 1 = T2I
#     • Scene 2 = EDIT using [scene_001]
#     • Scene 3+ = EDIT using [scene_001, previous]
# - Only the *style* line is variable (chosen by user). Everything else in prompts stays the same.
# - Saves native PNGs, logs prompts to scenes/prompt.json

from __future__ import annotations
import os, io, sys, json, argparse, contextlib, time, hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
from shutil import copy2

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

# -------------------- Style Library --------------------
STYLE_LIBRARY: Dict[str, str] = {
    "anime":    "Vibrant, anime-inspired, expressive characters, detailed backgrounds, dynamic movement, emotional atmosphere",
    "3d":       "Cinematic 3D, semi-realistic textures, smooth animation, soft character designs, warm and immersive lighting",
    "kid":      "Bright, kid-friendly, simple shapes, bold outlines, soft lighting, playful and cheerful tone",
    "storybook":"Storybook style, hand-drawn textures, watercolor and crayon feel, soft edges, cozy and nostalgic mood",
    "fantasy":  "Whimsical fantasy, glowing colors, sparkles and dreamy lighting, enchanting environments, magical atmosphere",
    "japanese_kawaii": "A kawaii Japanese chibi-style animation with pastel colors, big sparkly eyes, bouncy playful motion, and simple cozy backgrounds inspired by Sanrio characters.",
    "claymation": "A whimsical modern claymation-style animation inspired by Tumble Leaf, with hand-crafted textures, soft lighting, smooth stop-motion movement, and a colorful, playful atmosphere.",
    "watercolor": "A dreamy watercolor-style animation with flowing brushstrokes, soft pastel tones, subtle texture, and gentle hand-painted transitions that feel like a moving storybook.",
    "pixel_art": "A pixel art animation inspired by Minecraft, with blocky characters and environments, chunky textures, simple lighting, and smooth, playful motion in a colorful 8-bit/voxel style.",
    "paper_cutout": "A South Park cartoon–style lo-fi animation with flat 2D paper-cutout characters, stiff jerky movements, limited arm/leg motion, simple mouth flaps for speech, and exaggerated head/eye reactions for comedic effect.",
    "van_gogh": "A Van Gogh–style animation with swirling brushstrokes, bold textures, vivid contrasting colors, and flowing painterly motion that feels alive like a moving canvas.",
    "felt_needle": "A kawaii stop-motion felt animation with soft fuzzy textures, chubby plush-like characters, pastel colors, and playful handmade charm.",
    "stop_motion_felt_clay": "A modern stop-motion animation inspired by Pokémon Concierge, with handcrafted felt and clay textures, plush-like characters, detailed miniature sets, soft natural lighting, and a cozy, soothing atmosphere.",
    "hybrid_mix": "A hybrid animation style with flat 2D cartoon characters interacting in semi-realistic 3D worlds, inspired by The Amazing World of Gumball, with playful contrast and imaginative settings.",
    "silhouette": "A silhouette animation style with characters shown in shadow or cut-out forms, mysterious yet safe, using high contrast lighting and expressive movement to tell folktale-like stories.",
    "cutout_collage": "A playful cut-out collage animation style with magazine-style paper pieces, layered textures, and quirky movement, blending humor and creativity suitable for older kids.",
    "graphic_novel": "A graphic novel–style animation with bold inking, dramatic shading, high contrast, and stylized panels, edgy and cinematic while staying approachable for younger audiences.",
    "motion_comic": "A motion comic animation style where panels move with narration, limited character motion, bold manga-style linework, and text integration, designed for teen storytelling.",
    "comic_book": "A comic book–style animation with bold outlines, halftone textures, vibrant colors, and dynamic action poses, evoking classic superhero comics.",
    "art_deco": "An Art Deco–style animation with sleek geometric forms, metallic tones, elegant symmetry, and glamorous 1920s poster-inspired design.",
    "impressionism": "An Impressionist-style animation with soft brushstrokes, dreamy lighting, pastel color palettes, and flowing transitions that feel hand-painted and emotional.",
    "cubism": "A Cubist-style animation inspired by Picasso, with fragmented geometric faces, abstracted worlds, quirky proportions, and surreal overlapping perspectives.",
    "tim_burton": "A Tim Burton–style animation with dark yet whimsical gothic charm, elongated characters, moody lighting, and playful spookiness that is eerie but not too scary.",
    "dr_seuss": "A whimsical surrealist animation blending Dr. Seuss's playful, rhyming world with bizarre surrealist landscapes, quirky characters, and imaginative, dreamlike settings.",
    "ink_parchment": "A Kung Fu Panda intro–style animation with parchment textures, bold ink-brush strokes, stylized martial arts characters, fast paper-cutout transitions, dynamic camera angles, and a red-and-gold Chinese folklore aesthetic.",
    "ukiyo_e": "A Japanese-inspired animation style with hand-printed textures, ukiyo-e woodblock aesthetics, bold flat colors, flowing ink lines, and stylized characters moving like a traditional picture scroll or illustrated storybook.",
    "sumi_e": "A sumi-e ink wash–style animation with flowing black brushstrokes, soft gradients, and expressive negative space, evoking traditional Japanese calligraphic painting.",
    "byobu": "A Japanese folding screen–style animation with golden backgrounds, stylized cranes and pines, flowing seasonal motifs, and elegant hand-painted textures.",
    "emakimono": "A Japanese emakimono picture scroll–style animation with horizontal narrative flow, stylized characters, delicate linework, and painted seasonal landscapes unfolding like a story scroll.",
    "yamato_e": "A Yamato-e–style animation inspired by classical Japanese court paintings, with soft flowing lines, seasonal landscapes, gentle colors, and elegant storytelling of nature and human life.",
}

DEFAULT_STYLE_KEY = "kid"  # fallback if env not set

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

def _sha12(path: Path) -> str:
    import hashlib
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]

# -------------------- Prompt builders (ONLY style line varies) --------------------
def build_scene1_prompt(desc: str, style_line: str, has_portrait: bool) -> str:
    if has_portrait:
        return (
            f"{desc}\n"
            f"{style_line}\n"
            "Cast the person from the reference photo as the main protagonist. "
            "Match face structure, age, skin tone, and hair shape -ignore the background. "
            "Adapt the person to match the existing visual style"
            "Keep a friendly, appealing look suitable for a kids' story. "
            
        )
    else:
        return f"{desc}\n{style_line}"

def build_edit_prompt(desc: str, style_line: str, has_portrait: bool) -> str:
    if has_portrait:
        return (
            f"SCENE BRIEF: {desc}\n"
            f"{style_line}\n"
            "Maintain the protagonist's identity from the reference image(s) (match face & hair)."
        )
    else:
        return (
            f"SCENE BRIEF: {desc}\n"
            f"{style_line}\n"
            "Maintain visual continuity with the reference image(s)."
        )

# -------------------- Model calls (with tiny retry) --------------------
def _run_with_retry(fn, *args, retries: int = 2, delay: float = 1.5, **kwargs):
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

def run_nano_banana_t2i(prompt: str):
    return _run_with_retry(replicate.run, NANO_BANANA_MODEL, input={"prompt": prompt})

def run_nano_banana_edit(prompt: str, refs: List[Path]):
    files = []
    for p in refs:
        f = open(p, "rb")
        files.append(f)
    try:
        return _run_with_retry(
            replicate.run,
            NANO_BANANA_MODEL,
            input={"prompt": prompt, "image_input": files},
        )
    finally:
        for f in files:
            with contextlib.suppress(Exception):
                f.close()

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description="Generate scene images with google/nano-banana (portrait-aware, style-selectable).")
    parser.add_argument("--limit", type=int, default=None, help="Number of scenes to generate (default: all).")
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
            portrait_path = portrait_local
            print(f"👤 Using portrait reference: {portrait_path} (size={portrait_path.stat().st_size} sha1={_sha12(portrait_path)})")
        else:
            print(f"⚠️ PORTRAIT_PATH set but file not found: {src}")

    scenes = load_scenes()
    if args.limit is not None:
        scenes = scenes[:max(1, args.limit)]

    print("🖼️  model=google/nano-banana")
    if portrait_path:
        print("   strategy: scene_001 = EDIT [portrait], scene_002 = EDIT [scene_001, portrait], scene_003+ = EDIT [scene_001, previous, portrait]")
    else:
        print("   strategy: scene_001 = T2I, scene_002 = EDIT [scene_001], scene_003+ = EDIT [scene_001, previous]")
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
            if portrait_path:
                # All scenes are EDIT with portrait in refs
                if i == 1:
                    refs = [portrait_path]
                    prompt = build_scene1_prompt(desc=desc, style_line=style_line, has_portrait=True)
                elif i == 2:
                    if ref_png is None or not ref_png.exists():
                        raise RuntimeError("scene_001.png not found; cannot perform edit for scene 002.")
                    refs = [ref_png, portrait_path]
                    prompt = build_edit_prompt(desc=desc, style_line=style_line, has_portrait=True)
                else:
                    if ref_png is None or not ref_png.exists() or prev_png is None or not prev_png.exists():
                        raise RuntimeError(f"Missing references for scene {sid}.")
                    refs = [ref_png, prev_png, portrait_path]
                    prompt = build_edit_prompt(desc=desc, style_line=style_line, has_portrait=True)

                out = run_nano_banana_edit(prompt, refs)
                mode = "edit"
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
                    out = run_nano_banana_t2i(prompt)
                    mode = "t2i"
                    prompt_log[sid] = {
                        "mode": mode,
                        "model": NANO_BANANA_MODEL,
                        "prompt": prompt,
                        "style_key": style_key,
                        "style_line": style_line,
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

                    prompt = build_edit_prompt(desc=desc, style_line=style_line, has_portrait=False)
                    out = run_nano_banana_edit(prompt, refs)
                    mode = "edit"
                    prompt_log[sid] = {
                        "mode": mode,
                        "model": NANO_BANANA_MODEL,
                        "prompt": prompt,
                        "references": ", ".join([Path(r).name for r in refs]),
                        "references_abs": ", ".join([str(r) for r in refs]),
                        "style_key": style_key,
                        "style_line": style_line,
                    }

            data = outputs_to_image_bytes(out)
            if not data:
                print(f"DEBUG output type: {type(out)}; sample: {str(out)[:200]}")
                raise RuntimeError(f"No usable image bytes in model output (scene {sid}).")

            save_png_no_resize(out_path, data)

            if i == 1 and ref_png is None:
                ref_png = out_path
            prev_png = out_path

            entry: Dict[str, Any] = {"image_path": str(out_path), "mode": mode}
            if mode == "edit" and 'refs' in locals():
                entry["edit_refs"] = [str(p) for p in refs]
            manifest[sid] = entry

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
