# generate_script.py — Split using FULL subtitles (word-level), keep exact word timings
from __future__ import annotations
import json, math, argparse, asyncio
from pathlib import Path
from typing import List, Dict, Any, Tuple
from PIL import Image
import io
import base64

from tenacity import retry, wait_exponential, stop_after_attempt
from providers.factory import get_llm_provider
from config import settings

PROJECT_ROOT = Path(__file__).parent
SUBS_PATH = PROJECT_ROOT / "subtitles" / "input_subtitles.json"
OUT_PATH   = PROJECT_ROOT / "scripts" / "input_script.json"

MIN_S = 5.0
MAX_S = 10.0

PORTRAIT_DESCRIPTION_PROMPT = "Describe the main subject in this image in one short paragraph. Include: - What it is (person, animal, character, etc.) - Key visual features (age/size, colors, distinctive characteristics) - What they're wearing or how they look. Use this format: A [subject] with [key features] wearing/looking [appearance details]. Keep it under 30 words and focus only on visual details needed to recognize them in an illustration."

# State variables
portraitDescription = ""

# ---------- Image Compression ----------
def compress_image_under_5mb(image_path):
    """Compress image to under 5MB for Claude API"""
    max_size_bytes = 5 * 1024 * 1024  # 5MB in bytes
    
    # Open image
    img = Image.open(image_path)
    
    # Convert to RGB if needed (for PNG with transparency)
    if img.mode in ('RGBA', 'LA', 'P'):
        img = img.convert('RGB')
    
    # Resize to reasonable dimensions first
    img.thumbnail((1024, 1024))  # Max 1024px on longest side, keeps aspect ratio
    
    # Save with optimization
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=85, optimize=True)
    
    # If still too large, reduce quality iteratively
    quality = 85
    while buffer.tell() > max_size_bytes and quality > 20:
        buffer = io.BytesIO()
        quality -= 5
        img.save(buffer, format='JPEG', quality=quality, optimize=True)
    
    # Get the compressed bytes and convert to base64
    buffer.seek(0)
    compressed_bytes = buffer.getvalue()
    base64_image = base64.b64encode(compressed_bytes).decode('utf-8')
    
    print(f"[DEBUG] Compressed image to {len(compressed_bytes) / 1024 / 1024:.2f}MB")
    return base64_image

# ---------- Portrait Description ----------
def getPortraitDescription(image_path):
    """
    Get portrait description from Claude API using the portrait image and prompt.
    
    Args:
        image_path: Path to the image file to analyze
        
    Returns:
        Text description from Claude
    """
    print("[DEBUG] getPortraitDescription called")
    
    try:
        # Import Anthropic client and compress image
        from anthropic import Anthropic
        
        # Compress the image first
        print(f"[DEBUG] Compressing image: {image_path}")
        compressed_image_base64 = compress_image_under_5mb(image_path)
        
        # Initialize Anthropic client
        if not settings.CLAUDE_API_KEY:
            raise ValueError("CLAUDE_API_KEY not found in settings")
        
        client = Anthropic(api_key=settings.CLAUDE_API_KEY)
        
        print("[DEBUG] Calling Anthropic API for portrait description...")
        
        # Create the message with image and prompt
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": compressed_image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": PORTRAIT_DESCRIPTION_PROMPT
                        }
                    ]
                }
            ]
        )
        
        # Extract the text from the response
        description = message.content[0].text
        
        print(f"[DEBUG] Final description: {description}")
        return description
        
    except Exception as e:
        print(f"[ERROR] getPortraitDescription failed: {type(e).__name__}: {e}")
        import traceback
        print(f"[ERROR] Full traceback: {traceback.format_exc()}")
        raise

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
- Calculate each scene's duration as: end_time - start_time
- Ensure each scene is between 5.0–10.0 seconds long, not less than 5.0s or more than 10.0s.
- Adjust the word index boundaries to meet the duration range while keeping all words included once.
- Total duration must exactly match the narration timing (from first word start to last word end).

---

Scene Description Guidelines (for text-to-image generation):

1. **CRITICAL - Character Descriptions Must Be Explicit:**
   - Start with: "The characters in the image are: ..."
   - List EVERY character with SPECIFIC visual details
   - ❌ NEVER use generic terms like: " "a group of travelers", "someone", "a person"
   - ✅ ALWAYS describe with concrete details: "a purple striped cat wearing a yellow bandana", "a 7-year-old boy with a red baseball cap and blue overalls"
   - Always start the scenes with the Portrait Subject (person from image[0]) as one of the characters dressed the same as in the referance image.
2. **Character Detail Requirements:**
   - Species/type (human, animal, creature)
   - Age or size (if human/humanoid: child, teen, adult, elderly)
   - Distinctive physical features (fur color, hair color/style, build)
   - Clothing colors, patterns, and style
   - Key accessories (hats, glasses, jewelry, bags)
   

3. **Continuity Rules:**
   - Once a character is introduced with specific traits, use THE EXACT SAME description in every subsequent scene
   - Maintain consistent character appearances, clothing, and colors throughout all scenes
   - Keep environmental details and color palettes consistent

4. Then describe:
   - The environment, atmosphere, and lighting in specific, visual detail
   - The moment as a static, cinematic frame — not continuous action
   - Use clear, concrete, kid-friendly language focused only on what can be visually seen

---

Example:

Narration:
"They jumped into it. They had fireproof jackets."

❌ BAD Scene description:
"The characters in the image are: 'Portrait Subject (person from image[0])', 'a group of travelers'..."

✅ GOOD Scene description:
"The characters in the image are: Portrait Subject (person from image[0]), a purple striped cat with orange eyes wearing a yellow bandana, a 7-year-old boy with a red baseball cap and blue overalls, a brown dog with floppy ears wearing a red collar, a gray elephant with white tusks. They are all wearing shiny silver fireproof jackets. The scene shows them mid-jump entering a glowing orange portal surrounded by swirling flames..."

---

Summary of Key Rules:
- Output only JSON.
- No missing or extra words.
- Scene timing must align exactly with provided word timestamps.
- Each scene 5–10 seconds long, covering all words in sequence.
- **Every character must have explicit visual details — NO generic references.**
- Maintain exact character descriptions across all scenes for continuity.
"""
    else:
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
- Calculate each scene's duration as: end_time - start_time
- Ensure each scene is between 5.0–10.0 seconds long, not less than 5.0s or more than 10.0s.
- Adjust the word index boundaries to meet the duration range while keeping all words included once.
- Total duration must exactly match the narration timing (from first word start to last word end).

---

Scene Description Guidelines (for text-to-image generation):

1. **CRITICAL - Character Descriptions Must Be Explicit:**
   - Start with: "The characters in the image are: ..."
   - List EVERY character with SPECIFIC visual details
   - ❌ NEVER use generic terms like: "the person from image[0]", "a group of travelers", "someone", "a person"
   - ✅ ALWAYS describe with concrete details: "a purple striped cat wearing a yellow bandana", "a 7-year-old boy with a red baseball cap and blue overalls"
   
2. **Character Detail Requirements:**
   - Species/type (human, animal, creature)
   - Age or size (if human/humanoid: child, teen, adult, elderly)
   - Distinctive physical features (fur color, hair color/style, build)
   - Clothing colors, patterns, and style
   - Key accessories (hats, glasses, jewelry, bags)
   

3. **Continuity Rules:**
   - Once a character is introduced with specific traits, use THE EXACT SAME description in every subsequent scene
   - Maintain consistent character appearances, clothing, and colors throughout all scenes
   - Keep environmental details and color palettes consistent

4. Then describe:
   - The environment, atmosphere, and lighting in specific, visual detail
   - The moment as a static, cinematic frame — not continuous action
   - Use clear, concrete, kid-friendly language focused only on what can be visually seen

---

Example:

Narration:
"They jumped into it. They had fireproof jackets."

❌ BAD Scene description:
"The characters in the image are: ' 'a group of travelers'..."

✅ GOOD Scene description:
"The characters in the image are: a young girl with long blonde hair wearing a green dress and white sneakers, a purple striped cat with orange eyes wearing a yellow bandana, a 7-year-old boy with a red baseball cap and blue overalls, a brown dog with floppy ears wearing a red collar, a gray elephant with white tusks. They are all wearing shiny silver fireproof jackets. The scene shows them mid-jump entering a glowing orange portal surrounded by swirling flames..."

---

Summary of Key Rules:
- Output only JSON.
- No missing or extra words.
- Scene timing must align exactly with provided word timestamps.
- Each scene 5–10 seconds long, covering all words in sequence.
- **Every character must have explicit visual details — NO generic references.**
- Maintain exact character descriptions across all scenes for continuity.

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

    # Check for portrait image and generate description if available
    global portraitDescription
    if detect_portrait_image():
        print("[DEBUG] Portrait image detected, generating description...")
        import os
        portrait_path = os.getenv("PORTRAIT_PATH", "").strip()
        
        try:
            # Call the function to get portrait description
            portraitDescription = getPortraitDescription(portrait_path)
            print(f"[DEBUG] Portrait description generated and saved: {portraitDescription}")
        except Exception as e:
            print(f"[WARNING] Failed to generate portrait description: {e}")
            portraitDescription = ""  # Reset to empty string on failure
    else:
        print("[DEBUG] No portrait image detected, skipping portrait description")
        portraitDescription = ""

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
