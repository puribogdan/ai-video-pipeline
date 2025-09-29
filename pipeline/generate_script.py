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

MIN_S = 4.0
MAX_S = 5.0

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
MODEL_NAME = "claude-opus-4-1-20250805"

def chat_json(model: str, messages: list, temperature: float | None = None):
    """Strict-JSON chat using the configured LLM provider."""
    provider = get_llm_provider()
    kwargs = {"model": model, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature
    return provider.chat_json(**kwargs)

SYSTEM_PROMPT = (
    "You are a careful video editor. You must output ONLY valid JSON, no other text.\n\n"
    "Input: a list WORDS = [{index, start, end, word}] with times relative to 0.\n\n"
    "Task: split into contiguous SCENES using WORD INDICES ONLY.\n"
    "CRITICAL: Each scene MUST be exactly 4-5 seconds long based on word timings.\n\n"
    "Hard constraints:\n"
    "- Do NOT invent, remove, or paraphrase words. Use exactly the provided words.\n"
    "- No overlaps, no gaps; scenes must cover ALL words in order (each word exactly once).\n"
    "- FORCE each scene duration to be between 4.0 and 5.0 seconds by choosing appropriate word ranges.\n"
    "- The first scene must start at time 0.0.\n"
    "- Cuts must occur at word boundaries (by index).\n"
    "- Calculate word ranges so that (end_time - start_time) is between 4.0 and 5.0 seconds.\n\n"
    "Output ONLY JSON in this exact format:\n"
    "{\"scenes\":[{\"start_word_index\":int,\"end_word_index\":int,\"scene_description\":\"text-to-image prompt\"}]}\n\n"
    "CRITICAL TIMING RULES:\n"
    "- Analyze the word timings first to find ranges that give 4-5 second durations\n"
    "- Prioritize ranges closest to 4.5 seconds\n"
    "- If no perfect 4-5s range exists, choose the range closest to 4-5s\n"
    "- DO NOT exceed 5.0 seconds or go below 4.0 seconds\n\n"
    "Notes for scene_description:\n"
    "- Use the entire story context to understand characters, setting, and continuity.\n"
    "- Describe what is visually happening in this specific scene.\n"
    "- Present tense, kid-friendly, concrete. Describe how the image should look like, no actions.\n"
    "- No new objects, characters, or events beyond what is implied in the story.\n"
)

@retry(wait=wait_exponential(multiplier=1, min=2, max=12), stop=stop_after_attempt(3))
def call_llm_api(words_payload: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    print(f"[DEBUG] Making LLM API call to {MODEL_NAME}...")
    payload = {
        "constraints": {"min_secs": MIN_S, "max_secs": MAX_S},
        "words": words_payload,
        "instruction": (
            "Return a JSON object with 'scenes' array. Each scene must have start_word_index, end_word_index, and scene_description. "
            "CRITICAL: Analyze word timings and create scenes that are EXACTLY 4-5 seconds each. "
            "Calculate word ranges so that (end_time - start_time) is between 4.0 and 5.0 seconds for every scene. "
            "Cover all words exactly once, in order, with cuts at word indices. First scene starts at 0.0."
        ),
    }

    response_data = None
    try:
        response_data = chat_json(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
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

    # Force scenes to start from 0.0 and be exactly 4-5 seconds each
    current_time = 0.0
    for i, s in enumerate(scenes):
        s["start_time"] = current_time
        dur = s["end_time"] - s["start_time"]
        dur_i = int(round(dur))
        if dur_i < 4: dur_i = 4
        if dur_i > 5: dur_i = 5
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

    print(f"[DEBUG] Sending {len(words_payload)} word-level subtitles to {MODEL_NAME} to choose scene splits (4–5s preferred)…")
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
