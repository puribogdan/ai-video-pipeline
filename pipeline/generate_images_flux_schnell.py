# generate_images_flux_schnell.py
# Portrait-aware minimal pipeline with prompt logging.
# - If PORTRAIT_PATH is set and exists:
#     • Scene 1 = EDIT using [portrait]
#     • Scene 2+ = EDIT using [scene_001, previous] (no portrait)
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
    "cubism": "A Cubist style inspired by Pablo Picasso and Georges Braque. Preserve geometric fragmentation, layered abstract faces, and overlapping perspectives. Preserve the exact style and subject from the start frame.",
    "graphic_novel": "A graphic novel style inspired by Frank Miller. Preserve heavy inking, dramatic chiaroscuro shading, and bold panel-style composition. Preserve the exact style and subject from the start frame.",
    "motion_comic": "A manga style inspired by Japanese comics. Preserve bold linework, screentone textures, expressive facial features, and iconic manga aesthetics. Preserve the exact style and subject from the start frame.",
    "comic_book": "A comic book style inspired by Jack Kirby and classic superhero comics. Preserve bold outlines, halftone dots, vibrant colors, and exaggerated action poses. Preserve the exact style and subject from the start frame.",
    "gothic": "A gothic whimsical style inspired by Tim Burton and Edward Gorey. Preserve elongated characters, moody lighting, and playful gothic charm. Preserve the exact style and subject from the start frame.",
    "silhouette": "A silhouette style inspired by Lotte Reiniger and traditional shadow puppetry. Preserve stark black figures, strong cut-out shapes, and glowing backgrounds. Preserve the exact style and subject from the start frame.",
    "fantasy_magic_glow": "A glowing fantasy style inspired by Studio Ghibli and Disney's Fantasia. Preserve luminous glows, soft gradients, spark-like particles, and magical enchanted settings. Preserve the exact style and subject from the start frame.",
    "surrealism_hybrid": "A surrealist whimsical style blending Dr. Seuss and Salvador Dalí. Preserve Seussian curves, surreal dreamscapes, and imaginative distorted forms. Preserve the exact style and subject from the start frame.",
    "ink_parchment": "A parchment style inspired by Qi Baishi and Xu Beihong. Preserve parchment textures, bold ink strokes, red-and-gold folkloric colors, and martial arts themes. Preserve the exact style and subject from the start frame.",
    "japanese_woodblock": "A woodblock style inspired by Hokusai and Hiroshige. Preserve bold outlines, flat layered colors, and dramatic wave and mountain compositions. Preserve the exact style and subject from the start frame.",
    "ink_wash": "An ink wash style inspired by Sesshū Tōyō and Zen painters. Preserve bold flowing brushstrokes, soft gradients, and expressive negative space. Preserve the exact style and subject from the start frame.",
    "japanese_gold_screen": "A gold screen style inspired by Kano Eitoku and Ogata Kōrin. Preserve shimmering gold leaf textures, stylized cranes, pines, and seasonal decorative motifs. Preserve the exact style and subject from the start frame.",
    "japanese_scroll": "A narrative scroll style inspired by the Ban Dainagon Ekotoba. Preserve continuous horizontal flow, delicate linework, and stylized story scenes. Preserve the exact style and subject from the start frame.",
    "japanese_court": "A Yamato-e style inspired by the Tale of Genji scrolls. Preserve delicate brush lines, soft colors, and courtly seasonal motifs. Preserve the exact style and subject from the start frame.",
}

DEFAULT_STYLE_KEY = "kid_friendly_cartoon"  # fallback if env not set

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
        return f"{desc}\n{style_line}"
    else:
        return f"{desc}\n{style_line}"

def build_edit_prompt(desc: str, style_line: str, has_portrait: bool) -> str:
    return f"SCENE BRIEF: {desc}\n{style_line}"

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
        print("   strategy: scene_001 = EDIT [portrait], scene_002+ = EDIT [scene_001, previous] (no portrait)")
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
                # Only scene 1 uses portrait, subsequent scenes use previous scene references only
                if i == 1:
                    refs = [portrait_path]
                    prompt = build_scene1_prompt(desc=desc, style_line=style_line, has_portrait=True)
                else:
                    if ref_png is None or not ref_png.exists():
                        raise RuntimeError("scene_001.png not found; cannot perform edit for scene 002.")
                    if i == 2:
                        refs = [ref_png]
                    else:
                        if prev_png is None or not prev_png.exists():
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
