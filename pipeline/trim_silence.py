#!/usr/bin/env python3
"""
trim_silence_and_enhance.py

1) Trim long silences while keeping the first part of each pause (to avoid clipping word tails).
2) Optionally denoise + sweeten the audio on export using FFmpeg filters.

Defaults:
  input  : ./audio_input/input.mp3
  output : ./audio_input/input_trimmed.mp3

Enhancement chain (when enabled):
  highpass -> lowpass -> afftdn (denoise) -> deesser -> compand (gentle comp) -> dynaudnorm

If FFmpeg filters are missing on your system, we fall back to a simpler
in-process chain (high/low-pass + peak normalize).

Usage examples:
  python trim_silence_and_enhance.py
  python trim_silence_and_enhance.py -i path/to/in.wav -o out.mp3 --no-enhance
  python trim_silence_and_enhance.py --denoise_db 25 --hp 60 --lp 12000
"""

from __future__ import annotations
from pathlib import Path
import argparse
from typing import Tuple, List

from pydub import AudioSegment, effects
from pydub.silence import detect_silence


# ---------- Single source of truth for defaults ----------
ROOT = Path(__file__).parent
DEFAULT_INPUT  = ROOT / "audio_input" / "input.mp3"
DEFAULT_OUTPUT = ROOT / "audio_input" / "input_trimmed.mp3"

DEFAULT_TARGET_SILENCE_MS   = 1000   # total silence to keep when trimming (ms)
DEFAULT_KEEP_HEAD_MS        = 1000   # keep this much of the original silence head (ms)
DEFAULT_MIN_SILENCE_MS      = 1000   # detect silences >= this (ms)
DEFAULT_THRESHOLD_OFFSET_DB = 20     # silence threshold = audio.dBFS - offset (if absolute not set)
DEFAULT_ABS_THRESH_DBFS     = None   # e.g., -45 (None = disabled; use relative)
DEFAULT_SEEK_STEP_MS        = 10     # detection step (ms)
DEFAULT_CROSSFADE_MS        = 10     # tiny crossfade on joins (ms)

# Enhancement defaults
DEFAULT_ENHANCE           = True
DEFAULT_DENOISE_DB        = 25     # afftdn noise floor reduction in dB (approx)
DEFAULT_HIGHPASS_HZ       = 60     # rumble cut
DEFAULT_LOWPASS_HZ        = 12000  # hiss cut
DEFAULT_DEESS_FREQ        = 6000   # center freq for de-esser
DEFAULT_DEESS_THRES       = 0.5    # de-esser threshold (0..1-ish)
DEFAULT_COMPAND_GAIN      = 4      # makeup gain after compand (dB)


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
    silences: List[List[int]] = detect_silence(
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


def build_ffmpeg_filter_chain(
    *,
    hp_hz: int,
    lp_hz: int,
    denoise_db: int,
    deess_freq: int,
    deess_thresh: float,
    compand_gain: float,
) -> str:
    """
    Build a conservative, widely-supported FFmpeg filter chain.
    Designed to work on ffmpeg ~4.2+.
    """
    parts = [
        f"highpass=f={hp_hz}",
        f"lowpass=f={lp_hz}",
        # afftdn: frequency-domain denoiser. nf is target noise floor (lower = more reduction).
        f"afftdn=nr={denoise_db}",
        # de-esser around 6k (FFmpeg deesser is available in 4.1+)
        f"deesser=f={deess_freq}:t={deess_thresh}",
        # Gentle compand curve with make-up gain
        "compand=attacks=0.3:decays=0.8:points=-80/-80|-40/-20|0/-10|20/-8:soft-knee=6"
        + f":gain={compand_gain}",
        # Dynamic loudness normalization (faster than two-pass loudnorm)
        "dynaudnorm=f=150:g=10",
    ]
    return ",".join(parts)


def export_with_enhancement(
    seg: AudioSegment,
    out_path: Path,
    fmt: str,
    *,
    enhance: bool,
    hp_hz: int,
    lp_hz: int,
    denoise_db: int,
    deess_freq: int,
    deess_thresh: float,
    compand_gain: float,
) -> bool:
    """
    Try to export using an FFmpeg -af filter chain. If that fails (filters missing),
    fall back to a light in-process chain (HP/LP + peak normalize) and export clean.
    Returns True if FFmpeg chain was used; False if we fell back.
    """
    if enhance:
        af = build_ffmpeg_filter_chain(
            hp_hz=hp_hz,
            lp_hz=lp_hz,
            denoise_db=denoise_db,
            deess_freq=deess_freq,
            deess_thresh=deess_thresh,
            compand_gain=compand_gain,
        )
        try:
            seg.export(str(out_path), format=fmt, parameters=["-af", af])
            print(f"🔊 Enhancement: FFmpeg filters applied [-af {af}]")
            return True
        except Exception as e:
            print(f"⚠️ FFmpeg enhancement failed ({e}); falling back to simple processing…")

    # Fallback: simple high/low-pass + peak normalize (no denoise/compand)
    try:
        processed = effects.high_pass_filter(seg, hp_hz)
        processed = effects.low_pass_filter(processed, lp_hz)
        processed = effects.normalize(processed)  # peak normalize, not LUFS
        processed.export(str(out_path), format=fmt)
        print("🔊 Enhancement: fallback (HP/LP + peak normalize)")
        return False
    except Exception as e:
        # Last resort: raw export
        print(f"⚠️ Fallback processing failed ({e}); exporting without enhancement.")
        seg.export(str(out_path), format=fmt)
        return False


def main():
    ap = argparse.ArgumentParser(description="Trim long silences; optionally denoise + sweeten the audio.")
    ap.add_argument("--input", "-i", default=str(DEFAULT_INPUT), help="Input audio (default: audio_input/input.mp3)")
    ap.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT), help="Output audio (default: audio_input/input_trimmed.mp3)")

    # Trim controls
    ap.add_argument("--target_silence_ms", type=int, default=None, help=f"Total silence to keep when trimming (ms) [default: {DEFAULT_TARGET_SILENCE_MS}]")
    ap.add_argument("--keep_head_ms", type=int, default=None, help=f"Keep this much from START of detected silence (ms) [default: {DEFAULT_KEEP_HEAD_MS}]")
    ap.add_argument("--min_silence_ms", type=int, default=None, help=f"Detect silences >= this length (ms) [default: {DEFAULT_MIN_SILENCE_MS}]")
    ap.add_argument("--threshold_offset_db", type=int, default=None, help=f"Silence threshold = dBFS - offset [default: {DEFAULT_THRESHOLD_OFFSET_DB}]")
    ap.add_argument("--absolute_thresh_dbfs", type=float, default=None, help=f"Absolute silence threshold (dBFS), e.g., -45 [default: {DEFAULT_ABS_THRESH_DBFS}]")
    ap.add_argument("--seek_step_ms", type=int, default=None, help=f"Silence scanning step (ms) [default: {DEFAULT_SEEK_STEP_MS}]")
    ap.add_argument("--crossfade_ms", type=int, default=None, help=f"Crossfade at joins (ms) [default: {DEFAULT_CROSSFADE_MS}]")

    # Enhancement controls
    ap.add_argument("--enhance", dest="enhance", action="store_true", help="Enable denoise + sweeten (default ON)")
    ap.add_argument("--no-enhance", dest="enhance", action="store_false", help="Disable enhancement filters")
    ap.set_defaults(enhance=DEFAULT_ENHANCE)
    ap.add_argument("--denoise_db", type=int, default=DEFAULT_DENOISE_DB, help="Denoise amount for afftdn (approx dB reduction)")
    ap.add_argument("--hp", type=int, default=DEFAULT_HIGHPASS_HZ, help="High-pass cutoff Hz (rumble)")
    ap.add_argument("--lp", type=int, default=DEFAULT_LOWPASS_HZ, help="Low-pass cutoff Hz (hiss)")
    ap.add_argument("--deess_freq", type=int, default=DEFAULT_DEESS_FREQ, help="De-esser center frequency (Hz)")
    ap.add_argument("--deess_thresh", type=float, default=DEFAULT_DEESS_THRES, help="De-esser threshold (0..1 approx)")
    ap.add_argument("--compand_gain", type=float, default=DEFAULT_COMPAND_GAIN, help="Makeup gain (dB) after compression")
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
    print(f"  min_silence_ms      : {min_silence_ms}")
    if abs_dbfs is None:
        print(f"  threshold           : relative (dBFS - {off_db})")
    else:
        print(f"  threshold           : absolute {abs_dbfs} dBFS")
    print(f"  seek_step_ms        : {step_ms}")
    print(f"  crossfade_ms        : {xfade_ms}")
    print("— Enhance —")
    print(f"  enabled             : {args.enhance}")
    print(f"  hp / lp             : {args.hp} Hz / {args.lp} Hz")
    print(f"  denoise_db          : {args.denoise_db} dB (afftdn)")
    print(f"  deesser             : f={args.deess_freq} Hz, t={args.deess_thresh}")
    print(f"  compand_makeup_gain : {args.compand_gain} dB")

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

    used_ffmpeg_filters = export_with_enhancement(
        processed, out_path, fmt,
        enhance=args.enhance,
        hp_hz=args.hp, lp_hz=args.lp,
        denoise_db=args.denoise_db,
        deess_freq=args.deess_freq, deess_thresh=args.deess_thresh,
        compand_gain=args.compand_gain,
    )

    print(f"\n✅ Wrote: {out_path}")
    print(f"   original: {len(audio)/1000:.2f}s | trimmed: {len(processed)/1000:.2f}s")
    print(f"   silences shortened: {n_trimmed} | time removed: {seconds_removed:.2f}s")
    if args.enhance:
        print(f"   enhancement mode   : {'ffmpeg filters' if used_ffmpeg_filters else 'fallback (HP/LP + normalize)'}")


if __name__ == "__main__":
    main()
