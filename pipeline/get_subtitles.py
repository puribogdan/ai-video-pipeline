# get_subtitles.py
import json
import sys
from pathlib import Path
import requests
from tenacity import retry, wait_exponential, stop_after_attempt
from config import settings  # loads DEEPGRAM_API_KEY from .env

"""
Usage:
  python get_subtitles.py [optional_input_mp3]
Defaults to audio_input/input.mp3
Outputs to subtitles/input_subtitles.json (overwrites)
"""

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"

def simplify_words(dg_json: dict):
    """
    Extract a simple list of {word, start, end} from Deepgram response.
    """
    words = []
    try:
        alts = dg_json["results"]["channels"][0]["alternatives"][0]
        for w in alts.get("words", []):
            words.append({
                "word": w.get("word", ""),
                "start": float(w.get("start", 0.0)),
                "end": float(w.get("end", 0.0)),
            })
        transcript = alts.get("transcript", "")
    except (KeyError, IndexError, TypeError):
        transcript = ""
    return transcript, words

@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3))
def deepgram_transcribe(mp3_bytes: bytes):
    headers = {
        "Authorization": f"Token {settings.DEEPGRAM_API_KEY}",
        "Content-Type": "audio/mpeg",
    }
    # Request params: ask for punctuated transcript + word-level timestamps
    params = {
        "punctuate": "true",
        "model": "nova-2",      # Good general English model; adjust if needed
        "smart_format": "true",
        "diarize": "false",
        "utterances": "false"
    }
    resp = requests.post(DEEPGRAM_URL, headers=headers, params=params, data=mp3_bytes, timeout=120)
    if resp.status_code >= 400:
        raise RuntimeError(f"Deepgram error {resp.status_code}: {resp.text[:500]}")
    return resp.json()

def main():
    project_root = Path(__file__).parent
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else project_root / "audio_input" / "input.mp3"
    output_path = project_root / "subtitles" / "input_subtitles.json"

    if not input_path.exists():
        print(f"❌ Could not find audio file at: {input_path}")
        print("   Make sure your MP3 is placed there (or pass a custom path):")
        print("   python get_subtitles.py path\\to\\your.mp3")
        return

    print(f"🎧 Reading audio: {input_path}")
    mp3_bytes = input_path.read_bytes()

    print("🛰️ Sending to Deepgram…")
    dg_json = deepgram_transcribe(mp3_bytes)

    print("🧹 Simplifying to word-level timestamps…")
    transcript, words = simplify_words(dg_json)

    # Save a neat JSON we can use later
    output = {
        "audio_file": str(input_path),
        "transcript": transcript,
        "words": words,
    }
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"✅ Wrote {len(words)} words to: {output_path}")

if __name__ == "__main__":
    main()
