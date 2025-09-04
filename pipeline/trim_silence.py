#!/usr/bin/env python3
"""
trim_silence.py  —  Trim long silences + strong automatic speech enhancement.

Pipeline (no knobs needed):
  1) Trim only long silences (keep first part to avoid clipping words).
  2) Enhance speech automatically:
       • WebRTC VAD: detect speech; non-speech attenuated ~ -80 dB
       • Transient clamp in non-speech (footsteps/door thuds)
       • Noise reduction (noisereduce) with strength picked from SNR
       • Tight speech band (120–7000 Hz) to kill rumble & extreme hiss
       • Gentle de-esser if sibilance detected
       • Loudness normalize to ~ -16 LUFS
       • Soft limiter for safety peaks

Default IO:
  input  : ./audio_input/input.mp3
  output : ./audio_input/input_trimmed.mp3
"""

from __future__ import annotations
from pathlib import Path
import argparse
from typing import Tuple

import io
import numpy as np
from scipy.signal import butter, sosfiltfilt

import librosa
import soundfile as sf
import noisereduce as nr
import pyloudnorm as pyln
import webrtcvad

from pydub import AudioSegment, effects
from pydub.silence import detect_silence

# ---------- Paths & defaults ----------
ROOT = Path(__file__).parent
DEFAULT_INPUT  = ROOT / "audio_input" / "input.mp3"
DEFAULT_OUTPUT = ROOT / "audio_input" / "input_trimmed.mp3"

# Trimming
DEFAULT_TARGET_SILENCE_MS   = 1000
DEFAULT_KEEP_HEAD_MS        = 1000
DEFAULT_MIN_SILENCE_MS      = 1000
DEFAULT_THRESHOLD_OFFSET_DB = 20
DEFAULT_ABS_THRESH_DBFS     = None
DEFAULT_SEEK_STEP_MS        = 10
DEFAULT_CROSSFADE_MS        = 10

# Enhancement
TARGET_LUFS          = -16.0
LIMITER_MARGIN_DB    = 0.8
MIN_SR               = 16000         # resample up to at least 16 kHz
SPEECH_BAND          = (120.0, 7000.0)   # very effective for wide-band junk
SIBILANCE_BAND       = (5000.0, 9000.0)
MAX_DEESS_DB         = 6.0

# =============================================================================
# pydub <-> numpy
# =============================================================================
def audiosegment_to_float_np(seg: AudioSegment) -> tuple[np.ndarray, int, int]:
    """AudioSegment -> float32 np array in [-1,1], shape (n, ch)."""
    samples = np.array(seg.get_array_of_samples())
    if seg.channels > 1:
        samples = samples.reshape((-1, seg.channels))
    else:
        samples = samples.reshape((-1, 1))

    # Scale based on sample width
    if seg.sample_width == 2:
        y = samples.astype(np.float32) / 32768.0
    elif seg.sample_width == 4:
        y = samples.astype(np.float32) / 2147483648.0
    else:
        y = samples.astype(np.float32) / 32768.0
    return y, int(seg.frame_rate), int(seg.channels)


def float_np_to_audiosegment(y: np.ndarray, sr: int, channels: int) -> AudioSegment:
    """float32 np [-1,1] (n or (n,ch)) -> 16-bit WAV -> AudioSegment."""
    y = np.atleast_2d(y)
    y = np.clip(y, -1.0, 1.0).astype(np.float32)
    if y.shape[1] != channels:
        if channels == 1:
            y = y.mean(axis=1, keepdims=True)
        else:
            y = np.tile(y.mean(axis=1, keepdims=True), (1, channels))

    buf = io.BytesIO()
    pcm = (y * 32767.0).astype(np.int16)  # (frames, channels)
    # IMPORTANT: specify format="WAV" when writing to BytesIO
    sf.write(buf, pcm, sr, subtype="PCM_16", format="WAV")
    buf.seek(0)
    return AudioSegment.from_file(buf, format="wav")

# =============================================================================
# DSP helpers
# =============================================================================
def butter_sos(kind: str, cutoff_hz, sr: int, order: int = 2):
    return butter(order, cutoff_hz, btype=kind, fs=sr, output="sos")

def highpass(y: np.ndarray, sr: int, hz: float) -> np.ndarray:
    sos = butter_sos("highpass", hz, sr)
    return sosfiltfilt(sos, y, axis=0).astype(np.float32)

def lowpass(y: np.ndarray, sr: int, hz: float) -> np.ndarray:
    sos = butter_sos("lowpass", hz, sr)
    return sosfiltfilt(sos, y, axis=0).astype(np.float32)

def bandpass(y: np.ndarray, sr: int, f1: float, f2: float) -> np.ndarray:
    sos = butter(2, [f1, f2], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, y, axis=0).astype(np.float32)

def soft_limiter(y: np.ndarray, margin_db: float = 0.8) -> np.ndarray:
    target_peak = 10 ** (-margin_db / 20.0)
    peak = float(np.max(np.abs(y)) + 1e-9)
    if peak <= target_peak:
        return y
    return (y * (target_peak / peak)).astype(np.float32)

def apply_gain(y: np.ndarray, gain_db: float) -> np.ndarray:
    g = 10 ** (gain_db / 20.0)
    return (y * g).astype(np.float32)

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)) + 1e-12)

# =============================================================================
# Heuristics
# =============================================================================
def estimate_snr_db(y_mono: np.ndarray) -> float:
    """Very rough SNR estimate using percentile energies."""
    E = np.square(y_mono).astype(np.float64)
    noise = np.sqrt(np.percentile(E, 10))
    signal = np.sqrt(np.percentile(E, 90))
    return float(20 * np.log10((signal + 1e-9) / (noise + 1e-9)))

def detect_sibilance(sr: int, y_mono: np.ndarray) -> float:
    """Return de-ess attenuation (0..MAX_DEESS_DB) based on 5–9k band energy."""
    y_bp = bandpass(y_mono[:, None], sr, *SIBILANCE_BAND).squeeze()
    ratio = rms(y_bp) / (rms(y_mono) + 1e-9)
    if ratio < 0.10:
        return 0.0
    if ratio > 0.25:
        return MAX_DEESS_DB
    frac = (ratio - 0.10) / (0.25 - 0.10)
    return float(MAX_DEESS_DB * np.clip(frac, 0.0, 1.0))

def apply_deesser(y: np.ndarray, sr: int, attn_db: float) -> np.ndarray:
    """Static de-esser: subtract a scaled sibilance band (5–9k)."""
    if attn_db <= 0.25:
        return y
    sib = bandpass(y, sr, *SIBILANCE_BAND)
    scale = 1.0 - (10 ** (-attn_db / 20.0))
    return (y - scale * sib).astype(np.float32)

# =============================================================================
# VAD & transient clamp
# =============================================================================
def apply_vad_mask(y: np.ndarray, sr: int, aggressiveness: int = 3,
                   pad_ms: int = 140, atten_db: float = 80.0):
    """
    WebRTC VAD to find speech; non-speech is attenuated ~ -80 dB.
    Returns (processed_audio, speech_mask).
    """
    vad = webrtcvad.Vad(aggressiveness)
    target_sr = 16000
    hop_t = 0.02  # 20 ms frames
    hop = int(target_sr * hop_t)

    mono = y.mean(axis=1).astype(np.float32)
    mono_vad = librosa.resample(mono, orig_sr=sr, target_sr=target_sr) if sr != target_sr else mono

    mask = np.zeros_like(mono_vad, dtype=bool)
    for i in range(0, len(mono_vad), hop):
        chunk = mono_vad[i:i+hop]
        if len(chunk) < hop:
            tmp = np.zeros(hop, dtype=np.float32)
            tmp[:len(chunk)] = chunk
            chunk = tmp
        pcm16 = (np.clip(chunk, -1, 1) * 32767).astype(np.int16).tobytes()
        try:
            is_speech = vad.is_speech(pcm16, sample_rate=target_sr)
        except Exception:
            is_speech = True
        mask[i:i+hop] = is_speech

    pad = int((pad_ms/1000.0) * target_sr)
    if pad > 0 and mask.any():
        speak_idx = np.where(mask)[0]
        lo = max(speak_idx[0] - pad, 0)
        hi = min(speak_idx[-1] + pad, len(mask)-1)
        mask[max(0, lo-pad):min(len(mask), hi+pad)] = True

    if sr != target_sr:
        idx = np.linspace(0, len(mask)-1, num=y.shape[0]).astype(np.int64)
        mask_full = mask[idx]
    else:
        mask_full = mask[:y.shape[0]]

    out = y.copy()
    out[~mask_full, :] *= 10 ** (-atten_db/20.0)
    return out, mask_full


def clamp_transients(y: np.ndarray, sr: int, speech_mask: np.ndarray,
                     win_ms: int = 10, jump_db: float = 12.0, reduce_db: float = 28.0):
    """
    Suppress sudden spikes (thuds/doors) in *non-speech* regions.
    """
    win = max(1, int(sr * win_ms / 1000.0))
    mono = y.mean(axis=1)
    kernel = np.ones(win, dtype=np.float32) / win
    rms_fast = np.sqrt(np.convolve(mono**2, kernel, mode='same') + 1e-12)
    med = np.maximum(1e-6, np.median(rms_fast))
    inst_db = 20*np.log10(rms_fast + 1e-12)
    med_db  = 20*np.log10(med)
    spikes = (inst_db - med_db) > jump_db

    clamp_idx = spikes & (~speech_mask[:len(spikes)])
    if not np.any(clamp_idx):
        return y

    gain = np.ones_like(mono, dtype=np.float32)
    gain[clamp_idx] *= 10 ** (-reduce_db/20.0)

    smooth = max(1, int(sr * 0.015))
    kernel = np.ones(smooth, dtype=np.float32) / smooth
    gain = np.convolve(gain, kernel, mode='same').astype(np.float32)

    return (y * gain[:, None]).astype(np.float32)

# =============================================================================
# Enhancement (AUTO)
# =============================================================================
def enhance_auto(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Auto denoise -> VAD mute non-speech -> transient clamp -> NR -> band-pass ->
    (optional) de-ess -> loudness normalize -> limiter.
    """
    y = np.atleast_2d(y)
    if y.shape[1] == 0:
        y = y.reshape(-1, 1)
    _, ch = y.shape

    target_sr = max(sr, MIN_SR)
    if sr != target_sr:
        y = np.stack([librosa.resample(y[:, c], orig_sr=sr, target_sr=target_sr) for c in range(ch)], axis=1)
        sr = target_sr

    # 1) VAD mute non-speech
    y, speech_mask = apply_vad_mask(y, sr, aggressiveness=3, pad_ms=140, atten_db=80.0)

    # 2) Clamp non-speech transients
    y = clamp_transients(y, sr, speech_mask, win_ms=10, jump_db=12.0, reduce_db=28.0)

    # 3) Noise reduction (strength from SNR)
    mono_ref = y.mean(axis=1).astype(np.float32)
    snr_db = estimate_snr_db(mono_ref)
    if   snr_db <= 3:   prop = 0.96
    elif snr_db <= 8:   prop = 0.92
    elif snr_db <= 15:  prop = 0.88
    elif snr_db <= 25:  prop = 0.82
    else:               prop = 0.78

    y_dn = np.zeros_like(y, dtype=np.float32)
    for c in range(ch):
        y_dn[:, c] = nr.reduce_noise(
            y=y[:, c],
            sr=sr,
            stationary=False,
            prop_decrease=prop,
            time_mask_smooth_ms=60,
            freq_mask_smooth_hz=250,
            clip_noise_stationary=False,
        ).astype(np.float32)

    # 4) Tight speech band (kills rumble & high hiss)
    hp, lp = SPEECH_BAND
    y_bp = bandpass(y_dn, sr, hp, lp)

    # 5) Gentle de-ess if needed
    attn_db = detect_sibilance(sr, y_bp.mean(axis=1))
    y_ds = apply_deesser(y_bp, sr, attn_db)

    # 6) Loudness normalize to ~ -16 LUFS
    meter = pyln.Meter(sr)
    try:
        lufs = meter.integrated_loudness(y_ds.mean(axis=1))
    except Exception:
        lufs = -24.0
    gain_db = float(np.clip(TARGET_LUFS - lufs, -12.0, 12.0))
    y_ln = apply_gain(y_ds, gain_db)

    # 7) Limiter (safety)
    y_out = soft_limiter(y_ln, margin_db=LIMITER_MARGIN_DB)
    return y_out

# =============================================================================
# Silence trimming
# =============================================================================
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
    if absolute_thresh_dbfs is not None:
        silence_thresh = float(absolute_thresh_dbfs)
    else:
        silence_thresh = max(audio.dBFS - float(threshold_offset_db), -60.0)

    silences = detect_silence(
        audio,
        min_silence_len=min_silence_ms,
        silence_thresh=silence_thresh,
        seek_step=seek_step_ms,
    )
    if not silences:
        return audio, 0, 0.0

    keep_head_ms = max(0, min(keep_head_ms, target_silence_ms))

    silent_unit = AudioSegment.silent(duration=1, frame_rate=audio.frame_rate)
    silent_unit = silent_unit.set_channels(audio.channels).set_sample_width(audio.sample_width)

    out = AudioSegment.silent(duration=0, frame_rate=audio.frame_rate)
    cursor = 0
    trimmed_count = 0
    removed_ms_total = 0

    for start, end in silences:
        if start > cursor:
            out = out.append(audio[cursor:start], crossfade=crossfade_ms if len(out) and crossfade_ms > 0 else 0)

        dur = end - start
        if dur > target_silence_ms:
            if keep_head_ms > 0:
                out = out.append(audio[start:start + keep_head_ms], crossfade=crossfade_ms if len(out) and crossfade_ms > 0 else 0)
            remainder = target_silence_ms - keep_head_ms
            if remainder > 0:
                out = out.append(silent_unit * remainder, crossfade=0)

            removed_ms_total += (dur - target_silence_ms)
            trimmed_count += 1
        else:
            out = out.append(audio[start:end], crossfade=crossfade_ms if len(out) and crossfade_ms > 0 else 0)

        cursor = end

    if cursor < len(audio):
        out = out.append(audio[cursor:], crossfade=crossfade_ms if len(out) and crossfade_ms > 0 else 0)

    return out, trimmed_count, removed_ms_total / 1000.0

# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Trim long silences, then auto-enhance speech (denoise, band-pass, de-ess, LUFS, limiter).")
    ap.add_argument("--input", "-i", default=str(DEFAULT_INPUT), help="Input audio (default: audio_input/input.mp3)")
    ap.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT), help="Output audio (default: audio_input/input_trimmed.mp3)")

    # Trimming (sane defaults)
    ap.add_argument("--target_silence_ms", type=int, default=DEFAULT_TARGET_SILENCE_MS)
    ap.add_argument("--keep_head_ms", type=int, default=DEFAULT_KEEP_HEAD_MS)
    ap.add_argument("--min_silence_ms", type=int, default=DEFAULT_MIN_SILENCE_MS)
    ap.add_argument("--threshold_offset_db", type=int, default=DEFAULT_THRESHOLD_OFFSET_DB)
    ap.add_argument("--absolute_thresh_dbfs", type=float, default=DEFAULT_ABS_THRESH_DBFS)
    ap.add_argument("--seek_step_ms", type=int, default=DEFAULT_SEEK_STEP_MS)
    ap.add_argument("--crossfade_ms", type=int, default=DEFAULT_CROSSFADE_MS)

    ap.add_argument("--no-enhance", action="store_true", help="Disable enhancement stage (just trim + peak normalize).")

    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    audio = AudioSegment.from_file(in_path)

    # Logs
    print("— Trim settings —")
    print(f"  target_silence_ms   : {args.target_silence_ms}")
    print(f"  keep_head_ms        : {args.keep_head_ms}")
    print(f"  min_silence_ms      : {args.min_silence_ms}")
    if args.absolute_thresh_dbfs is None:
        print(f"  threshold           : relative (dBFS - {args.threshold_offset_db})")
    else:
        print(f"  threshold           : absolute {args.absolute_thresh_dbfs} dBFS")
    print(f"  seek_step_ms        : {args.seek_step_ms}")
    print(f"  crossfade_ms        : {args.crossfade_ms}")

    # Trim
    processed, n_trimmed, seconds_removed = trim_only_oversized_silences_keep_head(
        audio,
        target_silence_ms=args.target_silence_ms,
        keep_head_ms=args.keep_head_ms,
        min_silence_ms=args.min_silence_ms,
        threshold_offset_db=args.threshold_offset_db,
        absolute_thresh_dbfs=args.absolute_thresh_dbfs,
        seek_step_ms=args.seek_step_ms,
        crossfade_ms=args.crossfade_ms,
    )

    # Enhance
    if args.no_enhance:
        out_audio = effects.normalize(processed)  # quick peak normalize
        enh_note = "disabled (peak normalize only)"
    else:
        y, sr, ch = audiosegment_to_float_np(processed)
        if y.ndim == 1:
            y = y[:, None]
        y_enh = enhance_auto(y, sr)
        out_audio = float_np_to_audiosegment(y_enh, max(sr, MIN_SR), channels=ch)
        enh_note = "VAD mute + transient clamp + NR + band-pass + de-ess + LUFS + limiter"

    # Export
    fmt = out_path.suffix.lstrip(".").lower() or "mp3"
    out_audio.export(out_path, format=fmt)

    print(f"\n✅ Wrote: {out_path}")
    print(f"   original: {len(audio)/1000:.2f}s | trimmed: {len(processed)/1000:.2f}s")
    print(f"   silences shortened: {n_trimmed} | time removed: {seconds_removed:.2f}s")
    print(f"   enhancement        : {enh_note}")


if __name__ == "__main__":
    main()
