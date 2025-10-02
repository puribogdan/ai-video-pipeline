#!/usr/bin/env python3
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHUNKS_DIR = ROOT / "video_chunks"
AUDIO_DIR = ROOT / "audio_input"
# Prefer trimmed audio if it exists, else fall back to original
AUDIO_TRIMMED = AUDIO_DIR / "input_trimmed.mp3"
AUDIO_ORIG = AUDIO_DIR / "input.mp3"
FINAL_VIDEO = ROOT / "final_video.mp4"
LIST_FILE = CHUNKS_DIR / "chunks.txt"

def main():
    # Collect chunk files (only final output files, not raw files)
    chunks = sorted(CHUNKS_DIR.glob("chunk_*.mp4"))
    # Filter out any raw files that might still exist (defensive programming)
    chunks = [c for c in chunks if not c.name.endswith('.raw.mp4')]
    if not chunks:
        print(f"ERROR: No chunks found in {CHUNKS_DIR}", file=sys.stderr)
        sys.exit(1)

    # Pick audio
    print(f"[LOG] merge_and_add_audio checking: {AUDIO_TRIMMED} (exists: {AUDIO_TRIMMED.exists()}, size: {AUDIO_TRIMMED.stat().st_size if AUDIO_TRIMMED.exists() else 0}), {AUDIO_ORIG} (exists: {AUDIO_ORIG.exists()}, size: {AUDIO_ORIG.stat().st_size if AUDIO_ORIG.exists() else 0})")

    if AUDIO_TRIMMED.exists() and AUDIO_TRIMMED.stat().st_size > 0:
        audio_path = AUDIO_TRIMMED
    elif AUDIO_ORIG.exists() and AUDIO_ORIG.stat().st_size > 0:
        audio_path = AUDIO_ORIG
    else:
        print(f"ERROR: No audio found. Expected {AUDIO_TRIMMED} or {AUDIO_ORIG}", file=sys.stderr)
        sys.exit(1)

    print(f"[LOG] Selected audio: {audio_path} (ext: {audio_path.suffix})")

    # Write concat list file for ffmpeg
    LIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LIST_FILE.open("w", encoding="utf-8") as f:
        for c in chunks:
            # concat demuxer requires this exact format
            f.write(f"file '{c.as_posix()}'\n")

    # ffmpeg command:
    # - concat demuxer reads chunks sequentially (low RAM)
    # - copy video stream (no re-encode); encode audio to AAC
    # - apad pads audio with silence so video length is preserved
    # - -shortest stops at the end of video (audio is now >= video thanks to apad)
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(LIST_FILE),
        "-i", str(audio_path),
        "-filter_complex", "[1:a]apad[a]",
        "-map", "0:v:0", "-map", "[a]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(FINAL_VIDEO),
    ]

    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(f"✅ Wrote: {FINAL_VIDEO}")

if __name__ == "__main__":
    main()
 