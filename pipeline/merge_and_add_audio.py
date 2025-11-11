#!/usr/bin/env python3
"""
merge_and_add_audio.py - Merge video chunks with audio.

Simple implementation that uses ffmpeg to merge video chunks with audio.
Audio processing is now handled by external APIs.
"""

from __future__ import annotations
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
CHUNKS_DIR = ROOT / "video_chunks"
# Audio will be provided via environment variable or use the input file directly
AUDIO_INPUT = Path(os.environ.get("AUDIO_INPUT_FILE", ROOT / "input.mp3"))
FINAL_VIDEO = ROOT / "final_video.mp4"
LIST_FILE = CHUNKS_DIR / "chunks.txt"

# Logo animation configuration (using settings if available)
try:
    from config import settings
    LOGO_ANIMATION = ROOT / settings.LOGO_ANIMATION_PATH
    LOGO_ENABLED = settings.LOGO_ENABLED
    LOGO_POSITION = getattr(settings, 'LOGO_POSITION', 'end')
except ImportError:
    # Fallback to hardcoded values if settings not available
    LOGO_ANIMATION = ROOT / "logo_animation.mp4"
    LOGO_ENABLED = True
    LOGO_POSITION = 'end'


def prepare_video_with_logo(chunks: List[Path], output_path: Path) -> Path:
    """
    Prepare the final video by concatenating chunks and adding logo animation if enabled.
    
    Args:
        chunks: List of video chunk files
        output_path: Path where the intermediate video (with logo) will be saved
        
    Returns:
        Path to the video with logo added
    """
    if not LOGO_ENABLED or not LOGO_ANIMATION.exists():
        print(f"[LOG] Logo animation disabled or not found, using chunks only")
        return concatenate_chunks_only(chunks, output_path)
    
    print(f"[LOG] Adding logo animation at {LOGO_POSITION}: {LOGO_ANIMATION}")
    
    # Create a temporary file for chunks-only concatenation
    temp_video = output_path.parent / "temp_chunks_only.mp4"
    concatenate_chunks_only(chunks, temp_video)
    
    # Create a list file for ffmpeg concat based on position
    concat_list = output_path.parent / "logo_concat_list.txt"
    with concat_list.open("w", encoding="utf-8") as f:
        if LOGO_POSITION == 'start':
            # Logo at start
            f.write(f"file '{LOGO_ANIMATION.as_posix()}'\n")
            f.write(f"file '{temp_video.as_posix()}'\n")
        else:
            # Logo at end (default)
            f.write(f"file '{temp_video.as_posix()}'\n")
            f.write(f"file '{LOGO_ANIMATION.as_posix()}'\n")
    
    # Use ffmpeg to concatenate the chunks with logo
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-c", "copy",
        "-movflags", "+faststart",
        str(output_path),
    ]
    
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    
    # Clean up temporary files
    try:
        temp_video.unlink()
        concat_list.unlink()
    except Exception as e:
        print(f"Warning: Could not clean up temporary files: {e}")
    
    return output_path


def concatenate_chunks_only(chunks: List[Path], output_path: Path) -> Path:
    """
    Concatenate video chunks without adding logo.
    
    Args:
        chunks: List of video chunk files
        output_path: Path where the concatenated video will be saved
        
    Returns:
        Path to the concatenated video
    """
    # Write concat list file for ffmpeg
    LIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LIST_FILE.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(f"file '{chunk.as_posix()}'\n")
    
    # ffmpeg command for concatenation only
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(LIST_FILE),
        "-c", "copy",
        "-movflags", "+faststart",
        str(output_path),
    ]
    
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    
    return output_path


def main():
    # Collect chunk files (only final output files, not raw files)
    chunks = sorted(CHUNKS_DIR.glob("chunk_*.mp4"))
    # Filter out any raw files that might still exist (defensive programming)
    chunks = [c for c in chunks if not c.name.endswith('.raw.mp4')]
    if not chunks:
        print(f"ERROR: No chunks found in {CHUNKS_DIR}", file=sys.stderr)
        sys.exit(1)

    # Step 1: Prepare video with chunks and logo animation
    video_with_logo = FINAL_VIDEO.parent / "video_with_logo.mp4"
    print(f"[LOG] Preparing video with logo animation...")
    prepare_video_with_logo(chunks, video_with_logo)

    # Step 2: Add audio to the video with logo
    # Use audio from environment variable or default location
    if not AUDIO_INPUT.exists():
        print(f"ERROR: Audio file not found: {AUDIO_INPUT}", file=sys.stderr)
        sys.exit(1)

    print(f"[LOG] Using audio: {AUDIO_INPUT} (ext: {AUDIO_INPUT.suffix})")

    # Use ffmpeg to add audio to the video with logo
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_with_logo),
        "-i", str(AUDIO_INPUT),
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
    
    # Clean up intermediate file
    try:
        video_with_logo.unlink()
        print(f"[LOG] Cleaned up intermediate file: {video_with_logo}")
    except Exception as e:
        print(f"Warning: Could not clean up intermediate file: {e}")

    # Get video duration after successful merge
    def get_video_duration(file_path: Path) -> int:
        """Get video duration in full seconds (integer)."""
        try:
            result = subprocess.run([
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "csv=p=0", str(file_path)
            ], capture_output=True, text=True)
            if result.returncode == 0:
                duration = float(result.stdout.strip())
                return int(round(duration))  # Round to nearest second and convert to int
            return 0
        except Exception as e:
            logger.warning(f"Could not get video duration: {e}")
            return 0

    video_duration = get_video_duration(FINAL_VIDEO)
    print(f"✅ Wrote: {FINAL_VIDEO}")
    print(f"VIDEO_DURATION: {video_duration}")
    
    if LOGO_ENABLED and LOGO_ANIMATION.exists():
        print(f"✅ Logo animation added at {LOGO_POSITION}: {LOGO_ANIMATION}")
    else:
        print(f"[LOG] Logo animation not added (disabled or file not found)")


if __name__ == "__main__":
    main()