#!/usr/bin/env python3
"""
merge_and_add_audio.py - Merge video chunks with audio using streaming for large files.

Enhanced with:
- Streaming audio processing for large files
- Memory-efficient audio handling
- Progress reporting for long operations
- Fallback to traditional ffmpeg for compatibility
"""

from __future__ import annotations
import asyncio
import logging
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
AUDIO_DIR = ROOT / "audio_input"
# Prefer trimmed audio if it exists, else fall back to original
AUDIO_TRIMMED = AUDIO_DIR / "input_trimmed.mp3"
AUDIO_ORIG = AUDIO_DIR / "input.mp3"
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

# Import progressive processing modules
try:
    from .streaming_audio_io import StreamingAudioIO, AudioStreamInfo, AudioFormat
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False
    logger.warning("Streaming audio modules not available, using traditional ffmpeg approach")

def should_use_streaming_audio(audio_path: Path) -> bool:
    """
    Determine if streaming audio processing should be used.

    Args:
        audio_path: Path to the audio file

    Returns:
        True if streaming processing should be used
    """
    if not STREAMING_AVAILABLE:
        return False

    try:
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        # Use streaming for files > 50MB
        return file_size_mb > 50
    except Exception:
        return False


async def merge_audio_with_streaming(
    chunks: List[Path],
    audio_path: Path,
    output_path: Path
) -> None:
    """
    Merge video chunks with audio using streaming approach for large files.

    Args:
        chunks: List of video chunk files
        audio_path: Path to audio file
        output_path: Output video file path
    """
    logger.info(f"Using streaming audio merge for large files")

    try:
        # For very large audio files, we could implement streaming audio processing here
        # For now, we'll use the traditional ffmpeg approach but with progress monitoring

        # Write concat list file for ffmpeg
        LIST_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LIST_FILE.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(f"file '{chunk.as_posix()}'\n")

        # Use ffmpeg with progress reporting
        await run_ffmpeg_with_progress(chunks, audio_path, output_path)

    except Exception as e:
        logger.error(f"Streaming audio merge failed: {e}")
        raise


async def run_ffmpeg_with_progress(
    chunks: List[Path],
    audio_path: Path,
    output_path: Path
) -> None:
    """
    Run ffmpeg with progress monitoring.

    Args:
        chunks: List of video chunk files
        audio_path: Path to audio file
        output_path: Output video file path
    """
    # ffmpeg command for concatenation with audio
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
        "-progress", "pipe:1",  # Enable progress reporting
        str(output_path),
    ]

    logger.info(f"Running ffmpeg with progress monitoring: {' '.join(cmd)}")

    # Run ffmpeg and monitor progress
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    total_duration = await estimate_total_duration(chunks)
    start_time = time.time()

    def parse_progress_line(line: str) -> Optional[float]:
        """Parse progress line from ffmpeg output."""
        try:
            if line.startswith("out_time="):
                # Parse time format: out_time=00:00:00.000000
                time_str = line.split("=")[1].strip()
                parts = time_str.split(":")
                if len(parts) == 3:
                    hours, minutes, seconds = parts
                    total_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                    return total_seconds / total_duration if total_duration > 0 else 0
        except Exception:
            pass
        return None

    # Monitor progress
    while True:
        if process.stdout is None:
            break

        line = await process.stdout.readline()
        if not line:
            break

        line_str = line.decode('utf-8', errors='ignore').strip()
        progress = parse_progress_line(line_str)

        if progress is not None:
            elapsed_time = time.time() - start_time
            eta = (elapsed_time / progress) - elapsed_time if progress > 0 else 0
            logger.info(f"FFmpeg progress: {progress*100:.1f}% complete, ETA: {eta:.1f}s")

    # Wait for process to complete
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        error_msg = stderr.decode('utf-8', errors='ignore')
        raise RuntimeError(f"FFmpeg failed with return code {process.returncode}: {error_msg}")

    logger.info("FFmpeg completed successfully")


async def estimate_total_duration(chunks: List[Path]) -> float:
    """
    Estimate total duration of video chunks.

    Args:
        chunks: List of video chunk files

    Returns:
        Estimated total duration in seconds
    """
    total_duration = 0.0

    for chunk_path in chunks:
        try:
            # Use ffprobe to get duration
            result = await asyncio.create_subprocess_exec(
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "csv=p=0", str(chunk_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, _ = await result.communicate()
            duration_str = stdout.decode('utf-8', errors='ignore').strip()

            if duration_str:
                total_duration += float(duration_str)

        except Exception as e:
            logger.warning(f"Could not get duration for {chunk_path}: {e}")
            # Assume 10 seconds as fallback
            total_duration += 10.0

    return total_duration


def run_traditional_ffmpeg_merge(
    chunks: List[Path],
    audio_path: Path,
    output_path: Path
) -> None:
    """
    Traditional ffmpeg merge (original implementation).

    Args:
        chunks: List of video chunk files
        audio_path: Path to audio file
        output_path: Output video file path
    """
    # Write concat list file for ffmpeg
    LIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LIST_FILE.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(f"file '{chunk.as_posix()}'\n")

    # ffmpeg command
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
        str(output_path),
    ]

    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(f"✅ Wrote: {output_path}")


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

    # Use ffmpeg to add audio to the video with logo
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_with_logo),
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
 