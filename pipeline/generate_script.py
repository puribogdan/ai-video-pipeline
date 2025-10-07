# generate_script.py — Split using FULL subtitles (word-level), keep exact word timings
from __future__ import annotations
import json, math, argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

from tenacity import retry, wait_exponential, stop_after_attempt
from providers.factory import get_llm_provider
from config import settings

PROJECT_ROOT = Path(__file__).parent
SUBS_PATH = PROJECT_ROOT / "subtitles" / "input_subtitles.json"
OUT_PATH   = PROJECT_ROOT / "scripts" / "input_script.json"

MIN_S = 5.0
MAX_S = 10.0

# ---------- IO ----------
def load_words() -> List[Dict[str, Any]]:
    if not SUBS_PATH.exists():
        raise FileNotFoundError(f"Missing {SUBS_PATH}. Run get_subtitles.py first.")
    data = json.loads(SUBS_PATH.read_text(encoding="utf-8"))
    words = data.get("words", [])
    if not words:
        raise ValueError("No words found in subtitles JSON.")
    for w in words:
        w["start"] = float(w["start"]); w["end"] = float(w["end"])
    return words

def write_scenes(scenes: List[Dict[str, Any]]):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(scenes, indent=2), encoding="utf-8")
    print(f"✅ Wrote {len(scenes)} scenes to: {OUT_PATH}")

# ---------- LLM Provider ----------
# Model selection is now handled by the provider

def detect_portrait_image() -> bool:
    """Check if a portrait image is available for use in prompts."""
    import os
    from pathlib import Path

    portrait_env = os.getenv("PORTRAIT_PATH", "").strip()
    if not portrait_env:
        return False

    portrait_path = Path(portrait_env)
    return portrait_path.exists() and portrait_path.is_file()

def get_system_prompt(has_portrait: bool = False) -> str:
    """Generate the system prompt based on whether portrait images are available."""
    if has_portrait:
        return (
            "You are a creative video editor with good storytelling instincts. You must output ONLY valid JSON, no other text.\n\n"
            "Input: a list WORDS = [{index, start, end, word}] with times relative to 0.\n\n"
            "Task: split into contiguous SCENES using WORD INDICES ONLY.\n"
            "Each scene should be between 5-10 seconds long based on word timings.\n\n"
            "Guidelines:\n"
            "- Do NOT invent, remove, or paraphrase words. Use exactly the provided words.\n"
            "- No overlaps, no gaps; scenes must cover ALL words in order (each word exactly once).\n"
            "- Choose scene durations between 5.0 and 10.0 seconds that make sense for the story flow.\n"
            "- The first scene must start at time 0.0.\n"
            "- Cuts must occur at word boundaries (by index).\n"
            "- Consider natural story breaks, pauses, and narrative rhythm when choosing scene lengths.\n\n"
            "Output ONLY JSON in this exact format:\n"
            "{\"scenes\":[{\"start_word_index\":int,\"end_word_index\":int,\"scene_description\":\"text-to-image prompt\"}]}\n\n"
            "Scene Duration Strategy:\n"
            "- Analyze the word timings and story content to find natural scene breaks\n"
            "- Choose durations that feel right for each scene - some moments need more time, others less\n"
            "- Consider: dramatic pauses, important descriptions, character emotions, scene changes\n"
            "- Aim for 5-10 seconds per scene, but prioritize what serves the story best\n\n"
            "Notes for scene_description image generation:\n"
            "- Use the full story context to ensure characters, setting, and continuity remain consistent.\n"
            "- At the very beginning of each scene description, clearly list all characters who appear in that scene, including any who appeared in previous scenes and remain present.\n"
            "- Always describe each character SPECIFICALLY, not vaguely. Avoid generic terms like 'a person', 'friend', or 'someone'. Instead, specify clear visible traits such as gender, age, species, or role.\n"
            "- Start each scene with this exact line:\n"
            "  The characters in the image are: ...\n"
            "  Then list all characters with short specific descriptions.\n"
            "  Example: 'Portrait Subject (person from image[0])', 'a young boy with a backpack', 'a middle-aged woman wearing a red scarf', 'a brown dog', 'an elephant'.\n"
            "- If the story mentions a new person or animal, infer a simple, concrete descriptor (for example, if 'a friend' is mentioned, decide if it’s 'a girl', 'a boy', or 'a man' based on context).\n"
            "- Describe the scene as a static image — no actions, movement, or unfolding events.\n"
            "- Write in present tense, with kid-friendly, simple, and concrete language.\n"
            "- Only include characters, objects, and details that are already established or implied in the story.\n"
            "- Focus on visual details: appearance, colors, environment, mood, and arrangement in the frame.\n"


        )
    else:
        return (
            "You are a creative video editor with good storytelling instincts. You must output ONLY valid JSON, no other text.\n\n"
            "Input: a list WORDS = [{index, start, end, word}] with times relative to 0.\n\n"
            "Task: split into contiguous SCENES using WORD INDICES ONLY.\n"
            "Each scene should be between 5-10 seconds long based on word timings.\n\n"
            "Guidelines:\n"
            "- Do NOT invent, remove, or paraphrase words. Use exactly the provided words.\n"
            "- No overlaps, no gaps; scenes must cover ALL words in order (each word exactly once).\n"
            "- Choose scene durations between 5.0 and 10.0 seconds that make sense for the story flow.\n"
            "- The first scene must start at time 0.0.\n"
            "- Cuts must occur at word boundaries (by index).\n"
            "- Consider natural story breaks, pauses, and narrative rhythm when choosing scene lengths.\n\n"
            "Output ONLY JSON in this exact format:\n"
            "{\"scenes\":[{\"start_word_index\":int,\"end_word_index\":int,\"scene_description\":\"text-to-image prompt\"}]}\n\n"
            "Scene Duration Strategy:\n"
            "- Analyze the word timings and story content to find natural scene breaks\n"
            "- Choose durations that feel right for each scene - some moments need more time, others less\n"
            "- Consider: dramatic pauses, important descriptions, character emotions, scene changes\n"
            "- Aim for 5-10 seconds per scene, but prioritize what serves the story best\n\n"
            "Notes for scene_description image generation:\n"
            "- Use the full story context to ensure characters, setting, and continuity remain consistent.\n"
            "- At the very beginning of each scene description, clearly list all characters who appear in that scene, including any who appeared in previous scenes and remain present.\n"
            "- Always describe each character SPECIFICALLY, not vaguely. Avoid generic terms like 'a person', 'friend', or 'someone'. Instead, specify clear visible traits such as gender, age, species, or role.\n"
            "- Start each scene with this exact line:\n"
            "  The characters in the image are: ...\n"
            "  Then list all characters with short specific descriptions.\n"
            "  Example: ''a young boy with a backpack', 'a middle-aged woman wearing a red scarf', 'a brown dog', 'an elephant'.\n"
            "- If the story mentions a new person or animal, infer a simple, concrete descriptor (for example, if 'a friend' is mentioned, decide if it’s 'a girl', 'a boy', or 'a man' based on context).\n"
            "- Describe the scene as a static image — no actions, movement, or unfolding events.\n"
            "- Write in present tense, with kid-friendly, simple, and concrete language.\n"
            "- Only include characters, objects, and details that are already established or implied in the story.\n"
            "- Focus on visual details: appearance, colors, environment, mood, and arrangement in the frame.\n"


        )

def chat_json(messages: list, temperature: float | None = None, **kwargs):
    """Strict-JSON chat using the configured LLM provider."""
    provider = get_llm_provider()
    if temperature is not None:
        kwargs["temperature"] = temperature
    kwargs["messages"] = messages
    return provider.chat_json(**kwargs)

SYSTEM_PROMPT = (
    "You are a creative video editor with good storytelling instincts. You must output ONLY valid JSON, no other text.\n\n"
    "Input: a list WORDS = [{index, start, end, word}] with times relative to 0.\n\n"
    "Task: split into contiguous SCENES using WORD INDICES ONLY.\n"
    "Each scene should be between 5-10 seconds long based on word timings.\n\n"
    "Guidelines:\n"
    "- Do NOT invent, remove, or paraphrase words. Use exactly the provided words.\n"
    "- No overlaps, no gaps; scenes must cover ALL words in order (each word exactly once).\n"
    "- Choose scene durations between 5.0 and 10.0 seconds that make sense for the story flow.\n"
    "- The first scene must start at time 0.0.\n"
    "- Cuts must occur at word boundaries (by index).\n"
    "- Consider natural story breaks, pauses, and narrative rhythm when choosing scene lengths.\n\n"
    "Output ONLY JSON in this exact format:\n"
    "{\"scenes\":[{\"start_word_index\":int,\"end_word_index\":int,\"scene_description\":\"text-to-image prompt\"}]}\n\n"
    "Scene Duration Strategy:\n"
    "- Analyze the word timings and story content to find natural scene breaks\n"
    "- Choose durations that feel right for each scene - some moments need more time, others less\n"
    "- Consider: dramatic pauses, important descriptions, character emotions, scene changes\n"
    "- Aim for 5-10 seconds per scene, but prioritize what serves the story best\n\n"
    "Notes for scene_description:\n"
    "- Use the full story context to ensure characters, setting, and continuity remain consistent.\n"
    "- At the very beginning of each scene description, clearly list all characters who appear in that scene, including any who appeared in previous scenes and remain present.\n"
    "  Example format: 'Characters in this scene: [Lena, two children, a small dog].'\n"
    "- If there is a portrait image added instead of the text format, use this prompt: 'Portrait Subject (person from image[0])', 'an elephant', 'a small dog'\n"
    "- Describe the scene as a static image — no actions, movement, or unfolding events.\n"
    "- Write in present tense, with kid-friendly, simple, and concrete language.\n"
    "- Only include characters, objects, and details that are already established or implied in the story.\n"
    "- Focus on visual details: appearance, colors, environment, mood, and arrangement in the frame.\n"


)

@retry(wait=wait_exponential(multiplier=1, min=2, max=12), stop=stop_after_attempt(3))
def call_llm_api(words_payload: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    print("[DEBUG] Making LLM API call...")

    # Detect if portrait image is available
    has_portrait = detect_portrait_image()
    if has_portrait:
        print(f"[DEBUG] Portrait image detected, using portrait-aware prompt format")
    else:
        print(f"[DEBUG] No portrait image detected, using standard prompt format")

    # Get appropriate system prompt based on portrait availability
    system_prompt = get_system_prompt(has_portrait)

    payload = {
        "constraints": {"min_secs": MIN_S, "max_secs": MAX_S},
        "words": words_payload,
        "instruction": (
            "Return a JSON object with 'scenes' array. Each scene must have start_word_index, end_word_index, and scene_description. "
            "Analyze word timings and create scenes that are between 5-10 seconds each based on what fits the story best. "
            "Calculate word ranges so that (end_time - start_time) is between 5.0 and 10.0 seconds for every scene. "
            "Consider natural story breaks and narrative flow when choosing scene lengths. "
            "Cover all words exactly once, in order, with cuts at word indices. First scene starts at 0.0."
        ),
    }

    response_data = None
    try:
        response_data = chat_json(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload)},
            ],
            temperature=None,
        )
        print(f"[DEBUG] LLM API call successful, response length: {len(json.dumps(response_data))} characters")

        scenes = response_data.get("scenes", [])
        if not isinstance(scenes, list) or not scenes:
            raise ValueError("Model returned no scenes.")
        print(f"[DEBUG] Parsed {len(scenes)} scenes from response")
        return scenes

    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse JSON response from LLM: {e}")
        try:
            print(f"[ERROR] Raw response: {json.dumps(response_data) if 'response_data' in locals() else 'No response data'}")
        except:
            print("[ERROR] Could not serialize response data")
        raise
    except Exception as e:
        print(f"[ERROR] LLM API call failed: {type(e).__name__}: {e}")
        raise

# ---------- Validation / rebuild ----------
def validate_and_build(
    words: List[Dict[str, Any]],
    plan: List[Dict[str, Any]],
    snap_integers: bool = False
) -> List[Dict[str, Any]]:
    """
    Build final scenes from word-index ranges; keep exact timings from words.
    Validate: coverage in order, no gaps/overlaps. Optionally snap to integers (off by default).
    """
    n = len(words)
    used: List[int] = []
    scenes: List[Dict[str, Any]] = []
    for it in plan:
        start_idx = it.get("start_word_index")
        end_idx = it.get("end_word_index")
        if start_idx is None or end_idx is None:
            raise ValueError(f"Missing word indices in scene: {it}")
        a = int(start_idx)
        b = int(end_idx)
        if a < 0 or b < a or b >= n:
            raise ValueError(f"Invalid word index range [{a},{b}]")
        used.extend(range(a, b + 1))
        st = words[a]["start"]; en = words[b]["end"]
        narration = " ".join(w["word"] for w in words[a:b+1]).strip()
        scenes.append({
            "start_time": float(st),
            "end_time":   float(en),
            "narration": narration,
            "scene_description": (it.get("scene_description") or "").strip(),
        })

    expect = list(range(n))
    if used != expect:
        raise ValueError("Coverage failure: scenes must cover all words exactly once, in order, no gaps/overlaps.")

    # Ensure scenes start from 0.0 and respect natural durations within 5-10 second range
    current_time = 0.0
    for i, s in enumerate(scenes):
        s["start_time"] = current_time
        dur = s["end_time"] - s["start_time"]
        dur_i = int(round(dur))
        # Allow natural scene durations between 5-10 seconds
        if dur_i < 5: dur_i = 5
        if dur_i > 10: dur_i = 10
        s["end_time"] = current_time + float(dur_i)
        current_time = s["end_time"]

    return scenes

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Auto-split scenes using FULL subtitles (word-level).")
    ap.add_argument("--snap-integers", action="store_true",
                      help="OPTIONAL: snap scene durations to integer 4..5s (will move off exact word times).")
    args = ap.parse_args()

    if not settings.CLAUDE_API_KEY and not settings.OPENAI_API_KEY:
        raise RuntimeError("Neither CLAUDE_API_KEY nor OPENAI_API_KEY is set. Add at least one to your .env.")

    print("[DEBUG] Loading words from subtitles...")
    words = load_words()
    print(f"[DEBUG] Loaded {len(words)} words from subtitles")

    start0 = words[0]["start"]
    words_payload = [
        {"index": i, "start": round(w["start"] - start0, 3), "end": round(w["end"] - start0, 3), "word": w["word"]}
        for i, w in enumerate(words)
    ]

    print(f"[DEBUG] Sending {len(words_payload)} word-level subtitles to LLM provider to choose scene splits (5–10s preferred)…")
    try:
        plan = call_llm_api(words_payload)
        print(f"[DEBUG] LLM returned {len(plan)} scene plans")
    except Exception as e:
        print(f"[ERROR] LLM API call failed: {e}")
        import traceback
        print(f"[ERROR] Full traceback: {traceback.format_exc()}")
        raise

    print("[DEBUG] Validating and building scenes...")
    scenes = validate_and_build(words, plan, snap_integers=args.snap_integers)
    print(f"[DEBUG] Built {len(scenes)} scenes")

    write_scenes(scenes)

if __name__ == "__main__":
    main()
