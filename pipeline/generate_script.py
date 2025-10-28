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
        return """You are a creative video editor and storyteller. You must output ONLY valid JSON — no explanations, markdown, or extra text.

---

Input:
A list WORDS = [{word, start, end}] where start and end are timestamps in seconds relative to 0.

---

Task:
Split the narration into contiguous SCENES using word indices only.

Each scene must:
- Use words in order with no overlaps or gaps.
- Start at index 0 (the first word) and end at the last word.
- Be between 5.0 and 10.0 seconds long, based on actual word timings.
- Cuts must occur only at word boundaries.
- Scene durations must align exactly with word timestamps — do not invent timing.
- Choose natural narrative breaks (pauses, content changes, or rhythm shifts).

---

Output format (JSON only):
{
  "scenes": [
    {
      "start_time": int,
      "end_time": int,
      "narration": exact words from subtitles,
      "scene_description": "text-to-image prompt"
    }
  ]
}

---

Scene Duration Strategy:
- Calculate each scene's duration as: end_time(end_word_index) - start_time(start_word_index)
- Ensure each scene is between 5.0–10.0 seconds long.
- Adjust the word index boundaries to meet the duration range while keeping all words included once.
- Total duration must exactly match the narration timing (from first word start to last word end).

---

Scene Description Guidelines (for text-to-image generation):

1. Start with this exact line:
   "The characters in the image are: ..."
   - List all characters who appear in the scene.
   - Include recurring characters from previous scenes with identical appearance.
   - The main subject from image[0] must always appear consistently (same species, same look).

2. Then describe:
   - Each character's appearance (species, clothing, colors, or key traits).
   - The environment, atmosphere, and lighting in specific, visual detail.
   - The moment as a static, cinematic frame — not continuous action.
   - Avoid generic or abstract terms like "someone" or "a person."

3. Maintain continuity across all scenes (consistent settings, outfits, colors, etc.).
4. Use clear, concrete, kid-friendly language focused only on what can be visually seen.

---

Example:

Narration:
"They jumped into it. They had fireproof jackets."

Scene description:
"The characters in the image are: a white duck with a yellow beak and orange feet wearing a shiny silver fireproof jacket, and a brown chicken with a red comb wearing a shiny silver fireproof jacket. Both birds are mid-air inside the glowing orange crater of a volcano, surrounded by rising sparks and shimmering heat waves. Their metallic jackets gleam against the fiery background as they fall bravely together. The volcano interior glows bright orange and red, with molten rock swirling below and smoke curling upward."

---

Summary of Key Rules:
- Output only JSON.
- No missing or extra words.
- Scene timing must align exactly with provided word timestamps.
- Each scene 5–10 seconds long, covering all words in sequence.
- Maintain visual and character continuity across all scenes.
"""
    else:
        return """You are a creative video editor and storyteller. You must output ONLY valid JSON — no explanations, no markdown, no text before or after.

---

Input:
A list WORDS = [{word, start, end}] where start and end are timestamps in seconds (relative to 0). 

---

Task:
Split the narration into contiguous SCENES using WORD INDICES ONLY.

Each scene must:
- Use words in order with no overlaps or gaps (each word appears exactly once).
- Start at index 0 (the first word) and end at the last word.
- Be between 5.0 and 10.0 seconds long, based on actual word timings.
- Cuts must occur only at word boundaries (by index).
- The first scene must start at time 0.0.
- Scene durations must align exactly with word timings — no fabricated times.
- Choose natural story breaks (pauses, subject changes, or emotional shifts).

---

Output ONLY JSON in this exact format:
{
  "scenes": [
    {
      "start_time": int,
      "end_time": int,
      "narration": "exact words from subtitles",
      "scene_description": "text-to-image prompt"
    }
  ]
}

---

Scene Duration Strategy:
- Use actual start/end timestamps to determine each scene's duration:
  duration = end_time(end_word_index) - start_time(start_word_index)
- Ensure total coverage from start=0 to the final word's end.
- Each scene should last 5–10 seconds when possible, while maintaining narrative rhythm.
- Prioritize story flow and natural pacing — some scenes can be slightly shorter or longer if needed.

---

Notes for scene_description (image generation):
- Always base your image on the narration context and story continuity.
- Each scene should start with:
  "The characters in the image are: ..."
  Then list all characters visible in that moment (e.g., "a young boy with a backpack", "a brown dog", "a middle-aged woman wearing a red scarf").
- If a character appears in multiple scenes, maintain consistent appearance (clothing, color, size, etc.).
- Describe visible traits specifically — no vague words like "someone" or "a person".
- Describe the image as a **frozen cinematic moment**, not continuous action.
  - Acceptable: "mid-air above the lava," "looking up at the sky," "leaning toward each other."
  - Not acceptable: "running across the field," "jumping continuously," or "walking slowly."
- Include environmental and sensory details (lighting, mood, background elements like mist, sparks, water, fog, or reflections).
- Keep descriptions concrete, vivid, and child-friendly.
- Only include characters, objects, or settings established earlier in the narration. If new ones appear, infer simple, logical visuals that fit the story.

---

Example of tone and structure:

Narration:
"They jumped into it. They had fireproof jackets."

Scene description:
"The characters in the image are: a white duck with a yellow beak and orange feet wearing a shiny silver fireproof jacket, and a brown chicken with a red comb wearing a shiny silver fireproof jacket. Both birds are mid-air inside the glowing orange crater of a volcano, surrounded by rising sparks and shimmering heat waves. Their metallic jackets gleam against the fiery background as they fall bravely together. The volcano glows bright orange and red, with molten rock swirling below and smoke curling upward through the air."
"""


def chat_json(messages: list, temperature: float | None = None, **kwargs):
    """Strict-JSON chat using the configured LLM provider."""
    provider = get_llm_provider()
    if temperature is not None:
        kwargs["temperature"] = temperature
    kwargs["messages"] = messages
    return provider.chat_json(**kwargs)


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
            "Return a JSON object with 'scenes' array. Each scene must have start_time, end_time, narration, and scene_description. "
            "Analyze word timings and create scenes that are between 5-10 seconds each based on what fits the story best. "
            "Set start_time and end_time to create contiguous scenes from 0.0, each 5-10 seconds long. "
            "Use exact words from the input for narration, covering all words exactly once in order. "
            "Consider natural story breaks and narrative flow when choosing scene lengths."
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


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Auto-split scenes using FULL subtitles (word-level).")
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

    print("[DEBUG] Using scenes directly from LLM response...")
    scenes = plan
    print(f"[DEBUG] Using {len(scenes)} scenes from LLM")

    write_scenes(scenes)

if __name__ == "__main__":
    main()
