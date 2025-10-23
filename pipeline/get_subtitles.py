# get_subtitles.py
from __future__ import annotations
import json
import argparse
from pathlib import Path

import requests
from tenacity import retry, wait_exponential, stop_after_attempt
from config import settings  # loads DEEPGRAM_API_KEY from .env

"""
Usage:
  python get_subtitles.py
  python get_subtitles.py --audio /path/to/file.mp3

Default behavior:
  - If audio_input/input_trimmed.mp3 exists and is non-empty -> use that.
  - Else fallback to audio_input/input.mp3.

Output:
  subtitles/input_subtitles.json (overwrites)
"""

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"


def simplify_words(dg_json: dict) -> tuple[str, list[dict], str]:
    """Extract {word, start, end} list + full transcript + detected language from Deepgram response."""
    words = []
    transcript = ""
    detected_language = ""
    try:
        alt = dg_json["results"]["channels"][0]["alternatives"][0]
        transcript = alt.get("transcript", "")
        for w in alt.get("words", []):
            words.append({
                "word": w.get("word", ""),
                "start": float(w.get("start", 0.0)),
                "end": float(w.get("end", 0.0)),
            })
        # Extract detected language from metadata
        detected_language = dg_json.get("metadata", {}).get("detected_language", "")
    except (KeyError, IndexError, TypeError):
        pass
    return transcript, words, detected_language


@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3))
def deepgram_transcribe(mp3_bytes: bytes) -> dict:
    if not settings.DEEPGRAM_API_KEY:
        raise RuntimeError("Missing DEEPGRAM_API_KEY in environment/.env")
    headers = {
        "Authorization": f"Token {settings.DEEPGRAM_API_KEY}",
        "Content-Type": "audio/mpeg",
    }
    params = {
        "model": "nova-2",
        "smart_format": "true",
        "punctuate": "true",
        "diarize": "false",
        "utterances": "false",
        "detect_language": "true",  # ✅ correct way to auto-detect language
        
    }
    
    resp = requests.post(DEEPGRAM_URL, headers=headers, params=params, data=mp3_bytes, timeout=120)
    if resp.status_code >= 400:
        raise RuntimeError(f"Deepgram error {resp.status_code}: {resp.text[:500]}")
    return resp.json()


def _pick_audio(root: Path, cli_path: str | None) -> Path:
    if cli_path:
        p = Path(cli_path)
        return p if p.is_absolute() else (root / p)
    audio_dir = root / "audio_input"
    trimmed = audio_dir / "input_trimmed.mp3"
    original = audio_dir / "input.mp3"
    if trimmed.exists() and trimmed.stat().st_size > 0:
        return trimmed
    return original


def main() -> None:
    project_root = Path(__file__).parent
    out_path = project_root / "subtitles" / "input_subtitles.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ap = argparse.ArgumentParser(description="Transcribe audio to word-level subtitles (Deepgram).")
    ap.add_argument("--audio", default=None, help="Optional path to audio MP3 (otherwise prefers input_trimmed.mp3).")
    args = ap.parse_args()

    input_path = _pick_audio(project_root, args.audio)

    if not input_path.exists():
        print(f"❌ Could not find audio file at: {input_path}")
        print("   Place your file under ./audio_input/ or pass --audio /path/to/file.mp3")
        raise SystemExit(1)

    print(f"🎧 Reading audio: {input_path}", flush=True)
    mp3_bytes = input_path.read_bytes()

    print("🛰️ Sending to Deepgram…", flush=True)
    dg_json = deepgram_transcribe(mp3_bytes)

    print("🧹 Simplifying to word-level timestamps…", flush=True)
    transcript, words, detected_language = simplify_words(dg_json)

    result = {
        "audio_file": str(input_path),
        "transcript": transcript,
        "words": words,
        "detected_language": detected_language,
    }
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"✅ Wrote {len(words)} words to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
