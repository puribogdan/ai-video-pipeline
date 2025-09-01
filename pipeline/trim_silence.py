#!/usr/bin/env python3
"""
trim_silence_strict_keep_head.py

Rule:
- Detect silences >= min_silence_ms.
- If a silence is longer than target_silence_ms:
    Keep the FIRST keep_head_ms (default = 1000 ms) from the original audio,
    then (if needed) fill the remaining (target - keep_head_ms) with true silence
    so the total kept silence equals target_silence_ms.
- If a silence is <= target_silence_ms, leave it untouched.

Defaults:
  input  : ./audio_input/input.mp3
  output : ./audio_input/input_trimmed.mp3
"""

from __future__ import annotations
from pathlib import Path
import argparse
from typing import Tuple

from pydub import AudioSegment
from pydub.silence import detect_silence

# ---------- Single source of truth for defaults ----------
ROOT = Path(__file__).parent
DEFAULT_INPUT  = ROOT / "audio_input" / "input.mp3"
DEFAULT_OUTPUT = ROOT / "audio_input" / "input_trimmed.mp3"

DEFAULT_TARGET_SILENCE_MS   = 1000   # total silence we want to keep when trimming (ms)
DEFAULT_KEEP_HEAD_MS        = 1000   # keep this much original audio at the START of a detected silence (ms)
DEFAULT_MIN_SILENCE_MS      = 1000   # detect silences >= this length (ms)
DEFAULT_THRESHOLD_OFFSET_DB = 20     # silence threshold = audio.dBFS - offset (if absolute not set)
DEFAULT_ABS_THRESH_DBFS     = None   # e.g., -45 (None = disabled; use relative)
DEFAULT_SEEK_STEP_MS        = 10     # detection step (ms)
DEFAULT_CROSSFADE_MS        = 10     # tiny crossfade on joins (ms)

# ---------------------------------------------------------

def trim_only_oversized_silences_keep_head(
    audio: AudioSegment,
    *,
    target_silence_ms: int,
    keep_head_ms: int,
    min_silence_ms: int,
    threshold_offset_db: int,
    absolute_thresh_dbfs: float | None,
    seek_step_ms: int,
    crossfade_ms: int,
) -> Tuple[AudioSegment, int, float]:
    """
    Return (processed_audio, num_trimmed_segments, seconds_removed)

    For each detected silence segment [start, end):
      - If (end-start) > target_silence_ms:
          Keep the FIRST keep_head_ms from the original silence,
          then fill (target_silence_ms - keep_head_ms) with true silence (if positive).
      - Else: keep the whole segment unchanged.
    """
    # Compute silence threshold
    if absolute_thresh_dbfs is not None:
        silence_thresh = float(absolute_thresh_dbfs)
    else:
        silence_thresh = max(audio.dBFS - float(threshold_offset_db), -60.0)

    # Detect candidate silences (>= min_silence_ms)
    silences = detect_silence(
        audio,
        min_silence_len=min_silence_ms,
        silence_thresh=silence_thresh,  # keep float
        seek_step=seek_step_ms,
    )

    if not silences:
        return audio, 0, 0.0

    # Clamp keep_head_ms to target (can't keep more than total target)
    keep_head_ms = max(0, min(keep_head_ms, target_silence_ms))

    # Prepared “true silence” filler for any remainder
    silent_unit = AudioSegment.silent(duration=1, frame_rate=audio.frame_rate)
    silent_unit = silent_unit.set_channels(audio.channels).set_sample_width(audio.sample_width)

    out = AudioSegment.silent(duration=0, frame_rate=audio.frame_rate)
    cursor = 0
    trimmed_count = 0
    removed_ms_total = 0

    for start, end in silences:
        # Add audio before the silence
        if start > cursor:
            out = out.append(audio[cursor:start], crossfade=crossfade_ms if len(out) and crossfade_ms > 0 else 0)

        dur = end - start
        if dur > target_silence_ms:
            # Keep the first part of the ORIGINAL silence (protects word tails)
            keep_orig = keep_head_ms
            if keep_orig > 0:
                out = out.append(audio[start:start + keep_orig], crossfade=crossfade_ms if len(out) and crossfade_ms > 0 else 0)

            # If target is larger than what we kept from the original, fill the remainder with true silence
            remainder = target_silence_ms - keep_orig
            if remainder > 0:
                out = out.append(silent_unit * remainder, crossfade=0)

            removed_ms_total += (dur - target_silence_ms)
            trimmed_count += 1
        else:
            # Keep the original silence as-is
            out = out.append(audio[start:end], crossfade=crossfade_ms if len(out) and crossfade_ms > 0 else 0)

        cursor = end

    # Tail after the last silence
    if cursor < len(audio):
        out = out.append(audio[cursor:], crossfade=crossfade_ms if len(out) and crossfade_ms > 0 else 0)

    return out, trimmed_count, removed_ms_total / 1000.0


def main():
    ap = argparse.ArgumentParser(description="Trim only long silences; keep the first part (e.g., 1s) to avoid clipping words.")
    ap.add_argument("--input", "-i", default=str(DEFAULT_INPUT), help="Input audio (default: audio_input/input.mp3)")
    ap.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT), help="Output audio (default: audio_input/input_trimmed.mp3)")
    # CLI defaults are None; we fall back to constants so editing a single place works.
    ap.add_argument("--target_silence_ms", type=int, default=None, help=f"Total silence to keep when trimming (ms) [default: {DEFAULT_TARGET_SILENCE_MS}]")
    ap.add_argument("--keep_head_ms", type=int, default=None, help=f"Keep this much from START of detected silence (ms) [default: {DEFAULT_KEEP_HEAD_MS}]")
    ap.add_argument("--min_silence_ms", type=int, default=None, help=f"Detect silences >= this length (ms) [default: {DEFAULT_MIN_SILENCE_MS}]")
    ap.add_argument("--threshold_offset_db", type=int, default=None, help=f"Silence threshold = dBFS - offset [default: {DEFAULT_THRESHOLD_OFFSET_DB}]")
    ap.add_argument("--absolute_thresh_dbfs", type=float, default=None, help=f"Absolute silence threshold (dBFS), e.g., -45 [default: {DEFAULT_ABS_THRESH_DBFS}]")
    ap.add_argument("--seek_step_ms", type=int, default=None, help=f"Silence scanning step (ms) [default: {DEFAULT_SEEK_STEP_MS}]")
    ap.add_argument("--crossfade_ms", type=int, default=None, help=f"Crossfade at joins (ms) [default: {DEFAULT_CROSSFADE_MS}]")
    args = ap.parse_args()

    # Resolve effective settings
    target_sil_ms = args.target_silence_ms if args.target_silence_ms is not None else DEFAULT_TARGET_SILENCE_MS
    keep_head_ms  = args.keep_head_ms if args.keep_head_ms is not None else DEFAULT_KEEP_HEAD_MS
    min_sil_ms    = args.min_silence_ms if args.min_silence_ms is not None else DEFAULT_MIN_SILENCE_MS
    off_db        = args.threshold_offset_db if args.threshold_offset_db is not None else DEFAULT_THRESHOLD_OFFSET_DB
    abs_dbfs      = args.absolute_thresh_dbfs if args.absolute_thresh_dbfs is not None else DEFAULT_ABS_THRESH_DBFS
    step_ms       = args.seek_step_ms if args.seek_step_ms is not None else DEFAULT_SEEK_STEP_MS
    xfade_ms      = args.crossfade_ms if args.crossfade_ms is not None else DEFAULT_CROSSFADE_MS

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    audio = AudioSegment.from_file(in_path)

    print("— Trim settings —")
    print(f"  target_silence_ms   : {target_sil_ms}")
    print(f"  keep_head_ms        : {keep_head_ms}")
    print(f"  min_silence_ms      : {min_sil_ms}")
    if abs_dbfs is None:
        print(f"  threshold           : relative (dBFS - {off_db})")
    else:
        print(f"  threshold           : absolute {abs_dbfs} dBFS")
    print(f"  seek_step_ms        : {step_ms}")
    print(f"  crossfade_ms        : {xfade_ms}")

    processed, n_trimmed, seconds_removed = trim_only_oversized_silences_keep_head(
        audio,
        target_silence_ms=target_sil_ms,
        keep_head_ms=keep_head_ms,
        min_silence_ms=min_sil_ms,
        threshold_offset_db=off_db,
        absolute_thresh_dbfs=abs_dbfs,
        seek_step_ms=step_ms,
        crossfade_ms=xfade_ms,
    )

    fmt = out_path.suffix.lstrip(".").lower() or "mp3"
    processed.export(out_path, format=fmt)

    print(f"\n✅ Wrote: {out_path}")
    print(f"   original: {len(audio)/1000:.2f}s | trimmed: {len(processed)/1000:.2f}s")
    print(f"   silences shortened: {n_trimmed} | time removed: {seconds_removed:.2f}s")


if __name__ == "__main__":
    main()
