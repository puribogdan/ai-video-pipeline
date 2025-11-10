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

PORTRAIT_DESCRIPTION_PROMPT = "Describe the main subject in this image in one short paragraph. Include: - What it is (person, animal, character, etc.) - Key visual features (age/size, colors, distinctive characteristics) - What they're wearing or how they look. No gesture or body position details. Use this format: A [subject] with [key features] wearing/looking [appearance details]. Keep it under 30 words and focus only on visual details needed to recognize them in an illustration."

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

# ---------- Scene Timing Split ----------
def get_scene_timing_prompt() -> str:
    """Get prompt for splitting scenes with timing only, no descriptions."""
    return """You are a video editor and storyteller. Output ONLY valid JSON.

CRITICAL TIMING RULES (MUST FOLLOW):
1. Each scene MUST be between 5 and 10 seconds long
2. Scene duration = end_time - start_time
3. Use word timestamps exactly as provided
4. All words must be included exactly once, in order
5. Split only at word boundaries

NARRATIVE SPLIT RULES (EQUALLY IMPORTANT):
1. Look for natural story breaks:
   - Topic/location changes ("Then they went...", "Meanwhile...", "Suddenly...")
   - Character actions completing ("She opened the door.", "He smiled.")
   - Dialogue shifts (new speaker or response)
   - Emotional beats (pauses, reactions, revelations)
   - Time transitions ("The next day...", "Hours later...")

2. Prioritize story coherence within timing constraints:
   - Keep related thoughts together in one scene
   - Don't cut mid-action or mid-sentence if possible
   - Scene should feel like a complete visual moment

3. Decision process:
   - Find natural break points in the narration
   - Check if the resulting scene duration is 5-10 seconds
   - If yes: split there
   - If no: look for the nearest natural break within the 5-10 second window
   - Balance story logic with timing constraints

---

Input: WORDS = [{"word": "text", "start": 0, "end": 2}, ...]

Output:
{
  "scenes": [
    {
      "start_time": <int>,
      "end_time": <int>,
      "duration": <int>,  // MUST be between 5-10
      "narration": "<exact words from timestamps>",
      "scene_description": ""  // leave empty for now
    }
  ]
}

VALIDATION CHECKLIST:
- [ ] 5 <= duration <= 10 for ALL scenes
- [ ] end_time of scene[n] = start_time of scene[n+1]
- [ ] first scene starts at first word's start
- [ ] last scene ends at last word's end
- [ ] Each scene break occurs at a natural narrative point

---

EXAMPLES:

❌ BAD (mechanical 8-second splits):
Narration: "The cat walked to the door. She paused and listened carefully. Then she opened it and gasped."
- Scene 1 (0-8s): "The cat walked to the door. She paused and listened"
- Scene 2 (8-15s): "carefully. Then she opened it and gasped."
(Cuts mid-thought and splits "listened carefully")

✅ GOOD (story-aware splits):
- Scene 1 (0-6s): "The cat walked to the door."
- Scene 2 (6-14s): "She paused and listened carefully. Then she opened it and gasped."
(Natural break after action completes, keeps related moments together)

---

Example Process:

Narration: "They jumped into the portal. Flames surrounded them but their fireproof jackets kept them safe. On the other side was a crystal cave."

Timing: 0s to 18s total

Step 1: Identify natural breaks:
- After "portal." (topic complete)
- After "safe." (action complete) 
- After "cave." (new location revealed)

Step 2: Check durations:
- 0s to 4s ("They jumped into the portal.") = 4s ❌ too short
- 0s to 11s ("...kept them safe.") = 11s ❌ too long
- 0s to 7s ("...surrounded them") = 7s ✓ within range
- 7s to 18s ("...crystal cave.") = 11s ❌ too long
- 7s to 13s ("...kept them safe.") = 6s ✓ within range
- 13s to 18s ("...crystal cave.") = 5s ✓ within range

Step 3: Final split:
{
  "scenes": [
    {"start_time": 0, "end_time": 7, "duration": 7, "narration": "They jumped into the portal. Flames surrounded them", "scene_description": ""},
    {"start_time": 7, "end_time": 13, "duration": 6, "narration": "but their fireproof jackets kept them safe.", "scene_description": ""},
    {"start_time": 13, "end_time": 18, "duration": 5, "narration": "On the other side was a crystal cave.", "scene_description": ""}
  ]
}

---

Remember: Find the best story break WITHIN the 5-10 second timing constraint. Don't just split at exactly 8 seconds every time.
"""

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
        # Get the global portrait description
        global portraitDescription
        portrait_subject = portraitDescription if portraitDescription else "Portrait Subject"
        
        return f"""You are a creative director for children's storybook illustrations. Output ONLY valid JSON.

Input: A JSON with scenes containing narration. You will fill in the scene_description field.

Task: Generate detailed visual descriptions for image generation based on the narration.

Output: Return the SAME JSON structure with scene_description filled in. Keep all other fields (start_time, end_time, duration, narration) exactly as provided.

---

CRITICAL CHARACTER DESCRIPTION RULES:

1. **ALWAYS Start scene_description With Character List:**
   "The characters in the image are: {portrait_subject}, [character 2 with details], ..."

2. **Character Detail Requirements:**
   - ❌ NEVER: "a person", "someone", "travelers", "a group"
   - ✅ ALWAYS: Explicit details
     * Species/type (human, animal, creature)
     * Age/size (child, teen, adult, elderly)
     * Physical features (fur color, hair style, build)
     * Clothing (colors, patterns, style)
     * Accessories (hats, glasses, jewelry)

3. **Continuity (CRITICAL):**
   - First scene: Introduce all characters with full details
   - All subsequent scenes: Use EXACT SAME character descriptions
   - Only change: environment, lighting, composition
   - Maintain: character appearance, clothing colors, accessories across ALL scenes

4. **Portrait Subject Rule:**
   - ALWAYS list first: {portrait_subject}
   - Keep this description identical across ALL scenes

5. **After Character List, Describe:**
   - Environment and setting
   - Atmosphere and lighting
   - Composition as a single cinematic frame
   - Use concrete, visual, kid-friendly language

---

EXAMPLE:

Input JSON:
{{
  "scenes": [
    {{"start_time": 0, "end_time": 7, "duration": 7, "narration": "They jumped into the portal.", "scene_description": ""}},
    {{"start_time": 7, "end_time": 14, "duration": 7, "narration": "They had fireproof jackets on.", "scene_description": ""}}
  ]
}}

Output JSON:
{{
  "scenes": [
    {{
      "start_time": 0,
      "end_time": 7,
      "duration": 7,
      "narration": "They jumped into the portal.",
      "scene_description": "The characters in the image are: {portrait_subject}, a purple striped cat with orange eyes wearing a yellow bandana, a 7-year-old boy with a red baseball cap and blue overalls. They are mid-jump entering a glowing orange portal with swirling flames around the edges. The portal emits bright white light. Dark stormy sky in background."
    }},
    {{
      "start_time": 7,
      "end_time": 14,
      "duration": 7,
      "narration": "They had fireproof jackets on.",
      "scene_description": "The characters in the image are: {portrait_subject}, a purple striped cat with orange eyes wearing a yellow bandana, a 7-year-old boy with a red baseball cap and blue overalls. They are now wearing shiny silver fireproof jackets over their regular clothing. They stand confidently inside a cavern with orange glowing lava in the background."
    }}
  ]
}}

---

Remember: 
- Keep start_time, end_time, duration, narration EXACTLY as provided
- ONLY fill in scene_description field
- Use IDENTICAL character descriptions across all scenes
"""
    else:
        return """You are a creative director for children's storybook illustrations. Output ONLY valid JSON.

Input: A JSON with scenes containing narration. You will fill in the scene_description field.

Task: Generate detailed visual descriptions for image generation based on the narration.

Output: Return the SAME JSON structure with scene_description filled in. Keep all other fields (start_time, end_time, duration, narration) exactly as provided.

---

CRITICAL CHARACTER DESCRIPTION RULES:

1. **ALWAYS Start scene_description With Character List:**
   "The characters in the image are: [character 1 with details], [character 2 with details], ..."

2. **Character Detail Requirements:**
   - ❌ NEVER: "a person", "someone", "travelers", "a group"
   - ✅ ALWAYS: Explicit details
     * Species/type (human, animal, creature)
     * Age/size (child, teen, adult, elderly)
     * Physical features (fur color, hair style, build)
     * Clothing (colors, patterns, style)
     * Accessories (hats, glasses, jewelry)

3. **Continuity (CRITICAL):**
   - First scene: Introduce all characters with full details
   - All subsequent scenes: Use EXACT SAME character descriptions
   - Only change: environment, lighting, composition
   - Maintain: character appearance, clothing colors, accessories across ALL scenes

4. **After Character List, Describe:**
   - Environment and setting
   - Atmosphere and lighting
   - Composition as a single cinematic frame
   - Use concrete, visual, kid-friendly language

---

EXAMPLE:

Input JSON:
{
  "scenes": [
    {"start_time": 0, "end_time": 7, "duration": 7, "narration": "They jumped into the portal.", "scene_description": ""},
    {"start_time": 7, "end_time": 14, "duration": 7, "narration": "They had fireproof jackets on.", "scene_description": ""}
  ]
}

Output JSON:
{
  "scenes": [
    {
      "start_time": 0,
      "end_time": 7,
      "duration": 7,
      "narration": "They jumped into the portal.",
      "scene_description": "The characters in the image are: a young blonde girl wearing a green dress and white sneakers, a purple striped cat with orange eyes wearing a yellow bandana, a 7-year-old boy with a red baseball cap and blue overalls. They are mid-jump entering a glowing orange portal with swirling flames around the edges. The portal emits bright white light. Dark stormy sky in background."
    },
    {
      "start_time": 7,
      "end_time": 14,
      "duration": 7,
      "narration": "They had fireproof jackets on.",
      "scene_description": "The characters in the image are: a young blonde girl wearing a green dress and white sneakers, a purple striped cat with orange eyes wearing a yellow bandana, a 7-year-old boy with a red baseball cap and blue overalls. They are now wearing shiny silver fireproof jackets over their regular clothing. They stand confidently inside a cavern with orange glowing lava in the background."
    }
  ]
}

---

Remember: 
- Keep start_time, end_time, duration, narration EXACTLY as provided
- ONLY fill in scene_description field
- Use IDENTICAL character descriptions across all scenes

"""


def chat_json(messages: list, temperature: float | None = None, **kwargs):
    """Strict-JSON chat using the configured LLM provider."""
    provider = get_llm_provider()
    if temperature is not None:
        kwargs["temperature"] = temperature
    kwargs["messages"] = messages
    return provider.chat_json(**kwargs)


@retry(wait=wait_exponential(multiplier=1, min=2, max=12), stop=stop_after_attempt(3))
def call_scene_timing_api(words_payload: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Call LLM API for scene timing split only, no descriptions."""
    print("[DEBUG] Making scene timing LLM API call...")
    
    system_prompt = get_scene_timing_prompt()
    
    payload = {
        "constraints": {"min_secs": MIN_S, "max_secs": MAX_S},
        "words": words_payload,
        "instruction": (
            "Return a JSON object with 'scenes' array. Each scene must have start_time, end_time, duration, and narration. "
            "Analyze word timings and create scenes that are between 5-10 seconds each. "
            "Set start_time and end_time to create contiguous scenes from 0.0, each 5-10 seconds long. "
            "Use exact words from the input for narration, covering all words exactly once in order. "
            "DO NOT create scene_description - only timing and narration."
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
        print(f"[DEBUG] Scene timing LLM API call successful, response length: {len(json.dumps(response_data))} characters")
        
        scenes = response_data.get("scenes", [])
        if not isinstance(scenes, list) or not scenes:
            raise ValueError("Model returned no scenes.")
        print(f"[DEBUG] Parsed {len(scenes)} scenes from timing response")
        return scenes
        
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse JSON response from timing LLM: {e}")
        try:
            print(f"[ERROR] Raw response: {json.dumps(response_data) if 'response_data' in locals() else 'No response data'}")
        except:
            print("[ERROR] Could not serialize response data")
        raise
    except Exception as e:
        print(f"[ERROR] Scene timing LLM API call failed: {type(e).__name__}: {e}")
        raise


@retry(wait=wait_exponential(multiplier=1, min=2, max=12), stop=stop_after_attempt(3))
def call_llm_api_with_timing(words_payload: List[Dict[str, Any]], timing_scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Call LLM API to add scene descriptions to pre-split timing scenes."""
    print("[DEBUG] Making scene description LLM API call with pre-split timing...")
    
    # Detect if portrait image is available
    has_portrait = detect_portrait_image()
    if has_portrait:
        print(f"[DEBUG] Portrait image detected, using portrait-aware prompt format")
    else:
        print(f"[DEBUG] No portrait image detected, using standard prompt format")
    
    # Get appropriate system prompt based on portrait availability
    system_prompt = get_system_prompt(has_portrait)
    
    payload = {
        "timing_scenes": timing_scenes,
        "words": words_payload,
        "instruction": (
            "Use the EXACT timing splits from timing_scenes. DO NOT change start_time, end_time, or narration. "
            "ONLY add the 'scene_description' field to each scene. "
            "Create detailed scene descriptions for each scene based on the narration content. "
            "Maintain character consistency and follow all scene description guidelines from the system prompt."
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
        print(f"[DEBUG] Scene description LLM API call successful, response length: {len(json.dumps(response_data))} characters")
        
        scenes = response_data.get("scenes") or response_data.get("timing_scenes", [])
        if not isinstance(scenes, list) or not scenes:
            print(f"[ERROR] No scenes found in response. Available keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'Not a dict'}")

            raise ValueError("Model returned no scenes.")
        scenes = response_data.get("scenes", [])
        if not isinstance(scenes, list) or not scenes:
            raise ValueError("Model returned no scenes.")
        print(f"[DEBUG] Parsed {len(scenes)} scenes with descriptions from response")
        return scenes
        
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse JSON response from scene description LLM: {e}")
        try:
            print(f"[ERROR] Raw response: {json.dumps(response_data) if 'response_data' in locals() else 'No response data'}")
        except:
            print("[ERROR] Could not serialize response data")
        raise
    except Exception as e:
        print(f"[ERROR] Scene description LLM API call failed: {type(e).__name__}: {e}")
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

    print(f"[DEBUG] Step 1: Getting scene timing splits...")
    try:
        timing_scenes = call_scene_timing_api(words_payload)
        print(f"[DEBUG] Scene timing API returned {len(timing_scenes)} scenes")
    except Exception as e:
        print(f"[ERROR] Scene timing LLM API call failed: {e}")
        import traceback
        print(f"[ERROR] Full traceback: {traceback.format_exc()}")
        raise

    print(f"[DEBUG] Step 2: Getting scene descriptions for timing splits...")
    try:
        scenes_with_descriptions = call_llm_api_with_timing(words_payload, timing_scenes)
        print(f"[DEBUG] Scene description API returned {len(scenes_with_descriptions)} scenes with descriptions")
    except Exception as e:
        print(f"[ERROR] Scene description LLM API call failed: {e}")
        import traceback
        print(f"[ERROR] Full traceback: {traceback.format_exc()}")
        raise

    print(f"[DEBUG] Using {len(scenes_with_descriptions)} final scenes with descriptions")
    write_scenes(scenes_with_descriptions)

if __name__ == "__main__":
    main()
