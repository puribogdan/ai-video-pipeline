# generate_script.py — Split using FULL subtitles (word-level), keep exact word timings
from __future__ import annotations
import json, math, argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

from tenacity import retry, wait_exponential, stop_after_attempt
from openai import OpenAI
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

# ---------- OpenAI ----------
client = OpenAI(api_key=settings.OPENAI_API_KEY)

# Use the Instant model (fast responses)
MODEL_NAME = "gpt-5"

def chat_json(model: str, messages: list, temperature: float | None = None):
    """Strict-JSON chat; Thinking models may ignore temperature."""
    kwargs = {"model": model, "messages": messages, "response_format": {"type": "json_object"}}
    if temperature is not None and not model.startswith("gpt-5"):
        kwargs["temperature"] = temperature
    return client.chat.completions.create(**kwargs)

SYSTEM_PROMPT = (
    "You are a careful video editor.\n"
    "Input: a list WORDS = [{index, start, end, word}] with times relative to 0.\n\n"
    "Task: split into contiguous SCENES using WORD INDICES ONLY.\n"
    "Hard constraints:\n"
    "- Do NOT invent, remove, or paraphrase words. Use exactly the provided words.\n"
    "- No overlaps, no gaps; scenes must cover ALL words in order (each word exactly once).\n"
    "- Prefer scene durations between 5 and 10 seconds based on the words’ original times.\n"
    "- Cuts must occur at word boundaries (by index).\n\n"
    "Output ONLY JSON:\n"
    "{\"scenes\":[{"
    "  \"start_word_index\": int,  \"end_word_index\": int,  # inclusive range in WORDS\n"
    "  \"scene_description\": \"<text-to-image prompt reflecting a detailed description of the scene>\"\n"
    "} ...]}\n"
    "Notes for scene_description:\n"
    "- Use the entire story context, not only the words in this scene, to understand characters, setting, and continuity. Each scene must feel consistent with the others.\n"
    "- Describe what is visually happening in this specific scene (characters, environment, actions, emotions, atmosphere) while ensuring it aligns with the story so far and what follows.\n"
    "- Output a detailed scene description\n"
    "- No new objects, characters, or events beyond what is implied in the story.\n"
    "- Present tense, kid-friendly, concrete. Focus on what the image should show.\n"
)

@retry(wait=wait_exponential(multiplier=1, min=2, max=12), stop=stop_after_attempt(3))
def call_openai_full_words(words_payload: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    print(f"[DEBUG] Making OpenAI API call to {MODEL_NAME}...")
    payload = {
        "constraints": {"min_secs": MIN_S, "max_secs": MAX_S},
        "words": words_payload,
        "instruction": (
            "Return only 'scenes' with {start_word_index, end_word_index, scene_description}. "
            "Cover all WORDS exactly once, in order, with cuts at word indices. "
            "Prefer 5–10 second scenes."
        ),
    }

    try:
        resp = chat_json(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload)},
            ],
            temperature=None,
        )
        print(f"[DEBUG] OpenAI API call successful, response length: {len(resp.choices[0].message.content)} characters")

        data = json.loads(resp.choices[0].message.content)
        scenes = data.get("scenes", [])
        if not isinstance(scenes, list) or not scenes:
            raise ValueError("Model returned no scenes.")
        print(f"[DEBUG] Parsed {len(scenes)} scenes from response")
        return scenes

    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse JSON response from OpenAI: {e}")
        if 'resp' in locals():
            print(f"[ERROR] Raw response: {resp.choices[0].message.content}")
        raise
    except Exception as e:
        print(f"[ERROR] OpenAI API call failed: {type(e).__name__}: {e}")
        # Check if this is an OpenAI API error with response details
        if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
            print(f"[ERROR] Response status: {e.response.status_code}")
            print(f"[ERROR] Response text: {e.response.text}")
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
        a = int(it.get("start_word_index"))
        b = int(it.get("end_word_index"))
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

    if snap_integers:
        t0 = scenes[0]["start_time"]
        for i, s in enumerate(scenes):
            dur = s["end_time"] - s["start_time"]
            dur_i = int(round(dur))
            if dur_i < 5: dur_i = 5
            if dur_i > 10: dur_i = 10
            s["start_time"] = float(int(round(s["start_time"] - t0)) + (0 if i == 0 else scenes[i-1]["end_time"]))
            s["end_time"]   = float(s["start_time"] + dur_i)

    return scenes

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Auto-split scenes using FULL subtitles (word-level).")
    ap.add_argument("--snap-integers", action="store_true",
                     help="OPTIONAL: snap scene durations to integer 5..10s (will move off exact word times).")
    args = ap.parse_args()

    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set. Add it to your .env.")

    print("[DEBUG] Loading words from subtitles...")
    words = load_words()
    print(f"[DEBUG] Loaded {len(words)} words from subtitles")

    start0 = words[0]["start"]
    words_payload = [
        {"index": i, "start": round(w["start"] - start0, 3), "end": round(w["end"] - start0, 3), "word": w["word"]}
        for i, w in enumerate(words)
    ]

    print(f"[DEBUG] Sending {len(words_payload)} word-level subtitles to {MODEL_NAME} to choose scene splits (5–10s preferred)…")
    try:
        plan = call_openai_full_words(words_payload)
        print(f"[DEBUG] OpenAI returned {len(plan)} scene plans")
    except Exception as e:
        print(f"[ERROR] OpenAI API call failed: {e}")
        import traceback
        print(f"[ERROR] Full traceback: {traceback.format_exc()}")
        raise

    print("[DEBUG] Validating and building scenes...")
    scenes = validate_and_build(words, plan, snap_integers=args.snap_integers)
    print(f"[DEBUG] Built {len(scenes)} scenes")

    write_scenes(scenes)

if __name__ == "__main__":
    main()
