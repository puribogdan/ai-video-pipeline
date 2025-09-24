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
    "fantasy_magic_glow": "A fantasy glowing animation style inspired by Studio Ghibli's magical scenes and Disney's Fantasia. Preserve the luminous glows, soft gradients, and spark-like effects. Animate with floating light particles, glowing auras that pulse gently, and smooth camera pans that reveal enchanted landscapes filled with magical energy.",
    "kid_friendly_cartoon": "A kid-friendly cartoon animation style inspired by Nickelodeon and Cartoon Network classics. Preserve the bright colors, rounded character designs, and exaggerated expressions. Animate with bouncy squash-and-stretch movements, cheerful scene transitions with stars or balloons, and playful timing that emphasizes humor and safety for children.",
    "realistic_3d": "A realistic 3D animation style inspired by Pixar and DreamWorks films. Preserve the high-quality rendering, expressive eyes, and cinematic lighting. Animate with smooth lifelike motion capture, dynamic camera sweeps, and emotional close-ups that bring characters and environments to life with a film-quality look.",
    "japanese_anime": "A Japanese anime-inspired animation style referencing studios like Kyoto Animation and Toei. Preserve the sharp linework, expressive eyes, and stylized hair and costumes. Animate with dramatic camera angles, speed-line transitions, glowing aura effects, and fluid fight or transformation sequences that capture the energy of anime.",
    "storybook_illustrated": "A storybook illustrated cut-out style inspired by classic children's book illustrators like Eric Carle and Ezra Jack Keats. Preserve the textured collage feel, hand-painted elements, and playful paper layering. Animate with sliding cut-out transitions, puppet-like hinged movements, and gentle page-turn effects that feel like a moving picture book.",
    "japanese_kawaii": "A kawaii Japanese chibi-style animation inspired by Sanrio characters like Hello Kitty and Pompompurin. Preserve the pastel colors, big sparkly eyes, and rounded shapes. Animate with bouncy, looping playful motions, cheerful blushing expressions, and cozy background transitions that sparkle with stars, hearts, and flowers.",
    "claymation": "A whimsical modern claymation-style animation inspired by Tumble Leaf and Aardman films. Preserve the hand-crafted clay textures, soft lighting, and slightly imperfect stop-motion charm. Animate with smooth yet tactile stop-motion movement, playful hops, and environmental interactions that highlight the handmade detail of the clay world.",
    "watercolor_storybook": "A dreamy watercolor-style animation inspired by traditional hand-painted children's books. Preserve the flowing brushstrokes, pastel tones, and textured paper feel. Animate with watercolor washes that bloom dynamically, soft dissolves between scenes, and gentle motion of characters as if painted and brought to life on the page.",
    "pixel_art": "A pixel art animation inspired by Minecraft and retro 8-bit games. Preserve the blocky characters, chunky textures, and colorful voxel-like environments. Animate with playful choppy motions, simple looping cycles (walking, waving), and camera pans across block-built landscapes that feel like a living game world.",
    "paper_cutout": "A paper cut-out-style animation inspired by South Park's early seasons and traditional collage puppetry. Preserve the flat 2D cut-out look, bold outlines, and lo-fi charm. Animate with stiff jerky movements, limited arm/leg motion, simple mouth flaps for speech, and exaggerated head tilts and eye reactions for comedic timing.",
    "van_gogh": "A Van Gogh-style animation inspired by Starry Night and his expressive brushwork. Preserve the swirling strokes, vivid contrasting colors, and textured oil-paint feel. Animate with brush strokes that flow and morph across the canvas, landscapes that ripple with energy, and cinematic camera pans that feel like entering a moving painting.",
    "needle_felted": "A kawaii felted stop-motion animation inspired by Pui Pui Molcar. Preserve the fuzzy wool textures, chubby plush-like forms, and pastel color palette. Animate with endearing stop-motion jitter, rolling or hopping motions, and simple playful expressions that highlight the handmade toy-like charm.",
    "stop_motion_felt_clay": "A modern stop-motion animation inspired by Pokémon Concierge. Preserve the handcrafted felt and clay textures, miniature set design, and soft natural lighting. Animate with smooth stop-motion pacing, cozy environmental details, and warm cinematic transitions that create a soothing, healing (iyashikei) atmosphere.",
    "hybrid_mix": "A hybrid animation style inspired by The Amazing World of Gumball, with flat 2D cartoon characters interacting in semi-realistic 3D environments. Preserve the playful contrast between flat drawings and textured worlds. Animate with exaggerated character motions against camera pans, zooms, and dynamic lighting in 3D space, creating a whimsical clash of dimensions.",
    "silhouette": "A silhouette animation style inspired by traditional shadow puppetry and Lotte Reiniger's films. Preserve the strong contrast between cut-out black figures and glowing backgrounds. Animate with elegant hand-puppet-like gestures, layered silhouettes moving against colored gradients, and smooth sliding transitions that evoke folktales told through light and shadow.",
    "cutout_collage": "A cut-out collage-style animation inspired by Terry Gilliam's Monty Python animations and children's magazine art. Preserve the layered paper textures, bold cut edges, and playful mismatched proportions. Animate with quirky stop-motion-like hops, rotating paper joints, and exaggerated cut-out transitions that give a handcrafted and humorous feel.",
    "graphic_novel": "A graphic novel-style animation inspired by Frank Miller and modern comic art. Preserve the heavy inking, dramatic chiaroscuro shading, and high-contrast panels. Animate with cinematic panel shifts, slow-motion character reveals, and bold speech bubble pop-ins, creating a moody yet accessible visual storytelling experience.",
    "motion_comic": "A motion comic-style animation inspired by Japanese manga panels. Preserve the bold linework, expressive screentones, and iconic manga speed lines. Animate with minimal character motions (eye blinks, hair sways), panel transitions that slide or zoom, and narration syncing with text bubbles, blending manga and cinematic storytelling.",
    "comic_book": "A classic comic book-style animation inspired by Jack Kirby and American superhero comics. Preserve the bold outlines, halftone dot textures, and exaggerated action poses. Animate with comic panel punches, speed-line transitions, and onomatopoeia bursts ('BAM!', 'POW!') that explode onto the screen with dynamic energy.",
    "art_deco": "An Art Deco-style animation inspired by Tamara de Lempicka and vintage 1920s poster design. Preserve the sleek geometry, metallic tones, and glamorous symmetry. Animate with elegant panning across geometric patterns, rotating poster-style frames, and shimmering gold highlights to capture the sophistication of Jazz Age visuals.",
    "impressionism": "An Impressionist-style animation inspired by Claude Monet and Pierre-Auguste Renoir. Preserve the soft brushstrokes, dappled light, and pastel color palettes. Animate with shifting light that flickers like sun on water, brush strokes appearing dynamically, and gentle crossfades between scenes to evoke a dreamy hand-painted world.",
    "cubism": "A Cubist-style animation inspired by Pablo Picasso and Georges Braque. Preserve the fragmented geometric shapes, overlapping perspectives, and surreal faces. Animate with shifting fractured layers, rotating planes, and sliding geometric compositions that constantly reassemble into new abstract figures.",
    "gothic": "A gothic whimsical animation style inspired by Tim Burton's films and Edward Gorey's illustrations. Preserve the elongated character forms, moody lighting, and playful gothic charm. Animate with shadowy flickers, stop-motion-like jerks, and sweeping camera tilts that enhance the eerie yet fantastical tone.",
    "surrealism_hybrid": "A whimsical surrealist animation blending Dr. Seuss's playful rhyming illustrations with surrealist dreamscapes. Preserve the curving, whimsical Seussian architecture and the bizarre, shifting surrealist backgrounds. Animate with bouncing, rhyming motion loops, exaggerated perspective shifts, and quirky transitions that feel both childlike and dreamlike.",
    "ink_parchment": "An ink-and-parchment-style animation inspired by traditional Chinese scroll painting and calligraphy, referencing artists like Qi Baishi and Xu Beihong. Preserve the parchment textures, bold brush strokes, and red-and-gold folkloric palette. Animate with sliding scroll reveals, fast cut-out transitions, and dynamic martial arts poses that flip, swipe, and flow like shadow puppetry across a painted scroll.",
    "ukiyo_e": "A Ukiyo-e woodblock print-style animation inspired by masters like Hokusai and Hiroshige. Preserve the bold outlines, flat vibrant colors, and dramatic compositions of the floating world. Animate with flowing waves, dynamic camera pans, sliding layers of clouds and mountains, and stylized character movements that mimic the flat yet powerful energy of a living woodblock print.",
    "sumi_e": "A sumi-e ink wash-style animation inspired by Sesshū Tōyō and Zen Buddhist painters. Preserve the bold, flowing black brushstrokes, expressive negative space, and minimalist tonal gradients. Animate with flowing ink spreading across paper, brush strokes appearing dynamically, and dissolving transitions that feel like fresh ink soaking into parchment.",
    "byobu": "A Japanese folding screen-style animation inspired by artists such as Kano Eitoku and Ogata Kōrin. Preserve the golden-leaf backgrounds, stylized cranes, pines, and seasonal landscapes with rich decorative patterns. Animate with screen-fold transitions, shimmering gold textures that catch the light, and layered parallax effects to mimic the unfolding of a grand golden screen.",
    "emakimono": "An emakimono picture scroll-style animation inspired by classical narrative scrolls like the Ban Dainagon Ekotoba. Preserve the horizontal narrative flow, delicate linework, and stylized characters. Animate with continuous side-scrolling motion, unfolding scenes that glide across the frame, with characters and landscapes revealing themselves gradually like a living story scroll.",
    "yamato_e": "A Yamato-e-style animation inspired by classical Heian-period Japanese court paintings, referencing works like the Tale of Genji scrolls. Preserve the delicate brush lines, soft flowing colors, and seasonal motifs. Animate with elegant sliding panel transitions, gentle panning across landscapes, and subtle character gestures to evoke the feeling of a painted courtly romance.",
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
