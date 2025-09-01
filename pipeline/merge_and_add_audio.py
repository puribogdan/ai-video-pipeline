# merge_and_add_audio.py — concatenate chunks and add narration with silent padding (no video trimming)
from __future__ import annotations
from pathlib import Path
import re
from typing import List

# MoviePy imports (hub first; fallback if needed)
try:
    from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
    from moviepy.audio.AudioClip import AudioArrayClip, CompositeAudioClip
except Exception:
    from moviepy.video.io.VideoFileClip import VideoFileClip  # type: ignore
    from moviepy.audio.io.AudioFileClip import AudioFileClip  # type: ignore
    from moviepy.video.compositing.concatenate import concatenate_videoclips  # type: ignore
    from moviepy.audio.AudioClip import AudioArrayClip, CompositeAudioClip  # type: ignore

import numpy as np

PROJECT_ROOT = Path(__file__).parent
CHUNKS_DIR   = PROJECT_ROOT / "video_chunks"
AUDIO_PATH   = PROJECT_ROOT / "audio_input" / "input_trimmed.mp3"
OUT_DIR      = PROJECT_ROOT / "output"
FINAL_PATH   = OUT_DIR / "final_video.mp4"

FPS = 30

def find_chunks() -> List[Path]:
    CHUNKS_DIR.mkdir(exist_ok=True)
    files = sorted(
        CHUNKS_DIR.glob("chunk_*.mp4"),
        key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)) if re.search(r"(\d+)", p.stem) else 0
    )
    if not files:
        raise FileNotFoundError(f"No chunks found in {CHUNKS_DIR}")
    return files

def make_silence(duration: float, fps: int, nchannels: int = 2) -> AudioArrayClip:
    """Return a silent AudioArrayClip of given duration/fps/channels."""
    samples = max(1, int(round(duration * fps)))
    arr = np.zeros((samples, nchannels), dtype=np.float32)
    return AudioArrayClip(arr, fps=fps)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = find_chunks()

    # Load & normalize video chunks
    clips = []
    for p in files:
        clip = VideoFileClip(str(p))
        if clip.fps and abs(clip.fps - FPS) > 0.1:
            clip = clip.set_fps(FPS)
        clips.append(clip)

    # Concatenate video (no trimming)
    final_v = clips[0] if len(clips) == 1 else concatenate_videoclips(clips, method="compose")
    v_dur = float(final_v.duration or 0.0)

    # Load narration
    if not AUDIO_PATH.exists():
        raise FileNotFoundError(f"Audio file not found: {AUDIO_PATH}")
    audio = AudioFileClip(str(AUDIO_PATH))
    a_dur = float(audio.duration or 0.0)
    a_fps = int(getattr(audio, "fps", 44100) or 44100)
    a_ch  = int(getattr(audio, "nchannels", 2) or 2)

    print(f"[merge] video={v_dur:.2f}s  audio={a_dur:.2f}s  (fps={FPS}, afps={a_fps}, ch={a_ch})")

    # If audio is shorter, pad with silence at the end (do NOT cut video or narration)
    if a_dur < v_dur:
        pad = v_dur - a_dur
        silence = make_silence(pad, fps=a_fps, nchannels=a_ch).set_start(a_dur)
        audio_full = CompositeAudioClip([audio, silence])
    else:
        # Audio longer is fine — writer will only encode up to video duration
        audio_full = audio

    final = final_v.set_audio(audio_full)

    try:
        final.write_videofile(
            str(FINAL_PATH),
            fps=FPS,
            codec="libx264",
            audio_codec="aac",
            preset="medium",
            threads=2,
            verbose=False,
            logger=None,
        )
    finally:
        # Close all clips to release file handles on Windows
        for obj in (final, final_v, audio_full, audio, *clips):
            try:
                obj.close()
            except Exception:
                pass

    print(f"✅ Done! Final video: {FINAL_PATH}")

if __name__ == "__main__":
    main()
