#!/usr/bin/env python3
from __future__ import annotations
import os
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
    print(f"[DEBUG] Starting merge_and_add_audio.py")
    print(f"[DEBUG] Working directory: {os.getcwd()}")
    print(f"[DEBUG] Looking for chunks in: {CHUNKS_DIR}")

    # Collect chunk files
    chunks = sorted(CHUNKS_DIR.glob("chunk_*.mp4"))
    print(f"[DEBUG] Found {len(chunks)} chunks: {[c.name for c in chunks]}")

    if not chunks:
        print(f"ERROR: No chunks found in {CHUNKS_DIR}", file=sys.stderr)
        sys.exit(1)

    # Pick audio
    print(f"[DEBUG] Checking audio files:")
    print(f"[DEBUG]   Trimmed: {AUDIO_TRIMMED} (exists: {AUDIO_TRIMMED.exists()}, size: {AUDIO_TRIMMED.stat().st_size if AUDIO_TRIMMED.exists() else 0})")
    print(f"[DEBUG]   Original: {AUDIO_ORIG} (exists: {AUDIO_ORIG.exists()}, size: {AUDIO_ORIG.stat().st_size if AUDIO_ORIG.exists() else 0})")

    if AUDIO_TRIMMED.exists() and AUDIO_TRIMMED.stat().st_size > 0:
        audio_path = AUDIO_TRIMMED
        print(f"[DEBUG] Using trimmed audio")
    elif AUDIO_ORIG.exists() and AUDIO_ORIG.stat().st_size > 0:
        audio_path = AUDIO_ORIG
        print(f"[DEBUG] Using original audio")
    else:
        print(f"ERROR: No audio found. Expected {AUDIO_TRIMMED} or {AUDIO_ORIG}", file=sys.stderr)
        sys.exit(1)

    print(f"[DEBUG] Selected audio: {audio_path} (size: {audio_path.stat().st_size} bytes)")

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
    # - Video plays to full length (no -shortest cutting)
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(LIST_FILE),
        "-i", str(audio_path),
        "-filter_complex", "[1:a]apad[a]",
        "-map", "0:v:0", "-map", "[a]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        # Removed "-shortest" to prevent cutting video endings
        "-movflags", "+faststart",
        str(FINAL_VIDEO),
    ]

    print("RUN:", " ".join(cmd), flush=True)
    try:
        # Add timeout to prevent hanging
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        print(f"✅ Wrote: {FINAL_VIDEO}")

        # Verify the output file was created and has content
        if FINAL_VIDEO.exists():
            size = FINAL_VIDEO.stat().st_size
            print(f"[DEBUG] Output file size: {size} bytes")
            if size < 1024:
                print(f"⚠️  WARNING: Output file is very small ({size} bytes)")
        else:
            print(f"❌ ERROR: Output file was not created!")
            raise RuntimeError("Output file not created")

    except subprocess.TimeoutExpired:
        print(f"❌ ffmpeg timed out after 5 minutes")
        raise
    except subprocess.CalledProcessError as e:
        print(f"❌ ffmpeg failed with return code {e.returncode}")
        print(f"❌ stderr: {e.stderr}")
        print(f"❌ stdout: {e.stdout}")
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise

if __name__ == "__main__":
    main()
 