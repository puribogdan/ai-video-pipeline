#!/usr/bin/env python3
"""
trim_silence.py (AUTO)

Pipeline:
  1) Trim only long silences (keep the first part to avoid clipping words).
  2) Auto-enhance (no manual tuning required):
       • Noise reduction strength picked from estimated SNR
       • Auto high-pass (rumble/hum cleanup; 50/60 Hz aware)
       • Auto low-pass if hissy content
       • Light de-esser (only if sibilance detected)
       • Loudness normalize to ~ -16 LUFS
       • Soft limiter

All enhancement is pure Python DSP (librosa/noisereduce/pyloudnorm/scipy), so it is
consistent across platforms. pydub is used for file I/O and silence detection.
"""

from __future__ import annotations
from pathlib import Path
import argparse
from typing import Tuple

import numpy as np
from scipy.signal import sosfiltfilt, butter
import librosa
import noisereduce as nr
import pyloudnorm as pyln

from pydub import AudioSegment, effects
from pydub.silence import detect_silence
import webrtcvad

# ---------- Defaults ----------
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

# Enhancement (auto)
TARGET_LUFS          = -16.0
LIMITER_MARGIN_DB    = 0.8
MIN_SR               = 16000   # resample up to at least this for DSP stability
HISS_BAND_HZ         = 10000   # energy above this counted as "hiss band"
SIBILANCE_BAND       = (5000, 9000)   # de-ess check/attenuation band
MAX_DEESS_DB         = 8.0            # max attenuation if sibilant
# ---------------------------------------------------------


# ====== Utility: pydub <-> numpy ======
def audiosegment_to_float_np(seg: AudioSegment) -> tuple[np.ndarray, int, int]:
    """Convert pydub AudioSegment -> float32 np array in [-1,1], shape (n, ch)."""
    samples = np.array(seg.get_array_of_samples())
    ch = seg.channels
    if ch > 1:
        samples = samples.reshape((-1, ch))
    else:
        samples = samples.reshape((-1, 1))

    sw = seg.sample_width
    if sw == 2:
        y = samples.astype(np.float32) / 32768.0
    elif sw == 4:
        y = samples.astype(np.float32) / 2147483648.0
    elif sw == 1:
        # 8-bit PCM is often unsigned; pydub provides signed 'b' array
        y = samples.astype(np.float32) / 128.0
    else:
        y = samples.astype(np.float32) / 32768.0
    return y, int(seg.frame_rate), int(ch)


def float_np_to_audiosegment(y: np.ndarray, sr: int, channels: int) -> AudioSegment:
    """
    Convert float32 np array in [-1,1] (n or (n,ch)) to 16-bit PCM AudioSegment
    without going through ffmpeg (avoids platform codec differences).
    """
    y = np.atleast_2d(y).astype(np.float32)
    if y.shape[1] != channels:
        if channels == 1:
            y = y.mean(axis=1, keepdims=True)
        else:
            y = np.tile(y.mean(axis=1, keepdims=True), (1, channels))
    y = np.clip(y, -1.0, 1.0)
    pcm_i16 = (y * 32767.0).astype(np.int16)
    raw = pcm_i16.tobytes()
    return AudioSegment(
        data=raw,
        sample_width=2,
        frame_rate=sr,
        channels=channels,
    )


# ====== DSP helpers ======
def butter_sos(kind: str, cutoff_hz: float, sr: int, order: int = 2):
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
    g = 10.0 ** (gain_db / 20.0)
    return (y * g).astype(np.float32)


def apply_filter(y: np.ndarray, sr: int, hp_hz: float | None, lp_hz: float | None) -> np.ndarray:
    out = y
    if hp_hz and hp_hz > 0:
        out = highpass(out, sr, hp_hz)
    if lp_hz and lp_hz > 0 and lp_hz < (sr * 0.45):
        out = lowpass(out, sr, lp_hz)
    return out.astype(np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)) + 1e-12)

def frame_gen(y_mono: np.ndarray, sr: int, ms: int = 20):
    hop = int(sr * ms / 1000)
    for i in range(0, len(y_mono), hop):
        yield i, y_mono[i:i+hop]

def apply_vad_mask(y: np.ndarray, sr: int, aggressiveness: int = 2) -> np.ndarray:
    """
    y: (n, ch) float32 [-1,1]
    Returns y with non-speech regions attenuated ~ -40 dB.
    """
    vad = webrtcvad.Vad(aggressiveness)  # 0-3 (3 = most strict)
    # VAD expects 8/16/32/48 kHz mono 16-bit PCM frames of 10/20/30 ms.
    target_sr = 16000
    if sr != target_sr:
        mono = librosa.resample(y.mean(axis=1), orig_sr=sr, target_sr=target_sr)
    else:
        mono = y.mean(axis=1)
    # build boolean mask at target_sr
    hop = int(target_sr * 0.02)  # 20 ms
    mask = np.zeros_like(mono, dtype=bool)
    # convert per-frame to 16-bit PCM bytes
    for i in range(0, len(mono), hop):
        chunk = mono[i:i+hop]
        if len(chunk) < hop:
            pad = np.zeros(hop, dtype=np.float32)
            pad[:len(chunk)] = chunk
            chunk = pad
        pcm16 = (np.clip(chunk, -1, 1) * 32767).astype(np.int16).tobytes()
        try:
            is_speech = vad.is_speech(pcm16, sample_rate=target_sr)
        except Exception:
            is_speech = True
        mask[i:i+hop] = is_speech
    # up/down sample mask to original sr length
    if sr != target_sr:
        # resample mask by simple stretch
        idx = np.linspace(0, len(mask) - 1, num=y.shape[0]).astype(np.int64)
        mask_full = mask[idx]
    else:
        mask_full = mask[:y.shape[0]]
    # edge padding (avoid clipping fricatives): expand speech by 40 ms on both ends
    pad = int(sr * 0.04)
    speech_idx = np.where(mask_full)[0]
    if speech_idx.size:
        start = max(0, speech_idx[0] - pad)
        end   = min(len(mask_full), speech_idx[-1] + pad)
        mask_full[max(0, start-pad):min(len(mask_full), end+pad)] = True
    # apply attenuation to non-speech frames
    out = y.copy()
    attn = 10 ** (-40/20)  # -40 dB
    out[~mask_full, :] *= attn
    return out

def downward_expander(y: np.ndarray, threshold_db=-45.0, ratio=2.0) -> np.ndarray:
    # simple sample-wise expander in dB domain (approx)
    eps = 1e-8
    mag = np.maximum(np.abs(y), eps)
    db  = 20*np.log10(mag)
    over = db - threshold_db
    gain_db = np.where(over < 0, over*(1-1/ratio), 0.0)
    gain = 10**(gain_db/20.0)
    return (y * gain).astype(np.float32)


# ====== Auto heuristics ======
def estimate_snr_db(y_mono: np.ndarray) -> float:
    """Very rough SNR estimate using percentiles."""
    E = np.square(y_mono).astype(np.float64)
    noise_floor = np.sqrt(np.percentile(E, 10))
    signal_lvl  = np.sqrt(np.percentile(E, 90))
    snr = 20 * np.log10((signal_lvl + 1e-9) / (noise_floor + 1e-9))
    return float(snr)


def detect_hum_or_rumble_hp(sr: int, y_mono: np.ndarray) -> float:
    """
    If strong energy around 50/60 Hz, bump HP to ~70 Hz; else ~60 Hz default.
    Uses same FFT length for spectrum and frequency bins to avoid mismatches.
    """
    y = np.asarray(y_mono, dtype=np.float32).ravel()
    if y.size < 1024 or sr <= 0:
        return 60.0

    y = y - float(np.mean(y))
    n = min(y.size, 1 << 18)  # cap for speed/memory
    spec = np.abs(np.fft.rfft(y[:n], n=n))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    def band_power(f_center: float, bw: float = 5.0) -> float:
        lo, hi = f_center - bw, f_center + bw
        mask = (freqs >= lo) & (freqs <= hi)
        if not np.any(mask):
            return 0.0
        return float(spec[mask].mean())

    p50 = band_power(50.0)
    p60 = band_power(60.0)
    low_mask = freqs <= 120.0
    baseline = float(spec[low_mask].mean()) if np.any(low_mask) else 0.0

    if max(p50, p60) > 3.0 * (baseline + 1e-9):
        return 70.0
    return 60.0


def detect_hiss_lp(sr: int, y_mono: np.ndarray) -> float | None:
    """If high-band (≥10 kHz) energy is large vs mid-band, apply low-pass ~10–12 kHz."""
    if sr < 22050:
        return None
    S = np.abs(librosa.stft(y_mono, n_fft=1024, hop_length=256))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    hi = S[(freqs >= HISS_BAND_HZ), :]
    mid = S[(freqs >= 1000) & (freqs < 4000), :]

    hi_mean  = float(np.mean(hi)) if hi.size else 0.0
    mid_mean = float(np.mean(mid)) if mid.size else 1e-9
    ratio = hi_mean / mid_mean
    if ratio > 0.55:
        return 10000.0
    if ratio > 0.40:
        return 12000.0
    return None


def detect_sibilance(sr: int, y_mono: np.ndarray) -> float:
    """Return de-ess dB (0..MAX_DEESS_DB) from band/broadband ratio in 5–9 kHz."""
    bp = bandpass(y_mono[:, None], sr, *SIBILANCE_BAND).squeeze()
    r = rms(bp) / (rms(y_mono) + 1e-9)
    if r < 0.10:
        return 0.0
    if r > 0.25:
        return MAX_DEESS_DB
    frac = (r - 0.10) / (0.25 - 0.10)
    return float(MAX_DEESS_DB * np.clip(frac, 0.0, 1.0))


def apply_deesser(y: np.ndarray, sr: int, attn_db: float) -> np.ndarray:
    """Static, gentle de-esser: subtract a scaled sibilance band (5–9 kHz)."""
    if attn_db <= 0.2:
        return y
    sibilant = bandpass(y, sr, *SIBILANCE_BAND)
    scale = 1.0 - (10 ** (-attn_db / 20.0))  # amount to subtract
    out = (y - scale * sibilant).astype(np.float32)
    return out


# ====== Enhancement (auto) ======
def enhance_auto(y: np.ndarray, sr: int) -> np.ndarray:
    """Auto denoise → de-ess (if needed) → HP/LP (auto) → LUFS → limiter."""
    # Ensure shape (n, ch)
    y = np.atleast_2d(y).astype(np.float32)
    if y.shape[1] == 0:
        y = y.reshape(-1, 1)
    _, ch = y.shape

    # Resample up if too low for stable filters
    target_sr = max(sr, MIN_SR)
    if sr != target_sr:
        y = np.stack([librosa.resample(y[:, c], orig_sr=sr, target_sr=target_sr) for c in range(ch)], axis=1)
        sr = target_sr

    # --- Noise reduction (strength from SNR) ---
    mono = y.mean(axis=1)
    snr_db = estimate_snr_db(mono)
    if   snr_db <= 3:   prop = 0.97
    elif snr_db <= 8:   prop = 0.93
    elif snr_db <= 15:  prop = 0.88
    elif snr_db <= 25:  prop = 0.83
    else:               prop = 0.78


    y = apply_vad_mask(y, sr, aggressiveness=3)  # 0..3; 3 is strictest

    y_dn = np.zeros_like(y, dtype=np.float32)
    for c in range(ch):
        y_dn[:, c] = nr.reduce_noise(
            y=y[:, c],
            sr=sr,
            stationary=False,
            prop_decrease=prop,
            time_mask_smooth_ms=50,
            freq_mask_smooth_hz=200,
            clip_noise_stationary=False,
        ).astype(np.float32)

    # --- De-ess if needed ---
    attn_db = detect_sibilance(sr, y_dn.mean(axis=1))
    y_ds = apply_deesser(y_dn, sr, attn_db)

    # --- Auto HP/LP ---
    hp_hz = detect_hum_or_rumble_hp(sr, y_ds.mean(axis=1))
    lp_hz = detect_hiss_lp(sr, y_ds.mean(axis=1))
    y_f = apply_filter(y_ds, sr, hp_hz, lp_hz)

    y_post = downward_expander(y_ds, threshold_db=-45.0, ratio=2.0)


    # --- Loudness normalize (~ -16 LUFS) ---
    meter = pyln.Meter(sr)
    try:
        lufs = meter.integrated_loudness(y_f.mean(axis=1))
    except Exception:
        lufs = -24.0  # silence fallback
    gain_db = min(12.0, max(-12.0, TARGET_LUFS - lufs))
    y_ln = apply_gain(y_f, gain_db)

    # --- Limiter ---
    y_out = soft_limiter(y_ln, margin_db=LIMITER_MARGIN_DB)
    return y_out


# ====== Silence trimming ======
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


# ====== Main ======
def main():
    ap = argparse.ArgumentParser(description="Trim long silences, then auto-enhance speech audio.")
    ap.add_argument("--input", "-i", default=str(DEFAULT_INPUT), help="Input audio (default: audio_input/input.mp3)")
    ap.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT), help="Output audio (default: audio_input/input_trimmed.mp3)")

    # Trimming controls
    ap.add_argument("--target_silence_ms", type=int, default=DEFAULT_TARGET_SILENCE_MS)
    ap.add_argument("--keep_head_ms", type=int, default=DEFAULT_KEEP_HEAD_MS)
    ap.add_argument("--min_silence_ms", type=int, default=DEFAULT_MIN_SILENCE_MS)
    ap.add_argument("--threshold_offset_db", type=int, default=DEFAULT_THRESHOLD_OFFSET_DB)
    ap.add_argument("--absolute_thresh_dbfs", type=float, default=DEFAULT_ABS_THRESH_DBFS)
    ap.add_argument("--seek_step_ms", type=int, default=DEFAULT_SEEK_STEP_MS)
    ap.add_argument("--crossfade_ms", type=int, default=DEFAULT_CROSSFADE_MS)

    # Toggle enhancement (on by default)
    ap.add_argument("--no-enhance", action="store_true", help="Disable enhancement stage.")

    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    audio = AudioSegment.from_file(in_path)

    # ---- Logs ----
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

    # ---- Trim silences ----
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

    # ---- Enhance (auto) ----
    if args.no_enhance:
        out_audio = effects.normalize(processed)
        enh_note = "disabled (peak normalize only)"
    else:
        y, sr, ch = audiosegment_to_float_np(processed)
        if y.ndim == 1:
            y = y[:, None]
        y_enh = enhance_auto(y, sr)
        out_audio = float_np_to_audiosegment(y_enh, max(sr, MIN_SR), channels=ch)
        enh_note = "auto NR + de-ess(if needed) + HP/LP + LUFS + limiter"

    # ---- Export ----
    fmt = out_path.suffix.lstrip(".").lower() or "mp3"
    out_audio.export(out_path, format=fmt)

    print(f"\n✅ Wrote: {out_path}")
    print(f"   original: {len(audio)/1000:.2f}s | trimmed: {len(processed)/1000:.2f}s")
    print(f"   silences shortened: {n_trimmed} | time removed: {seconds_removed:.2f}s")
    print(f"   enhancement        : {enh_note}")


if __name__ == "__main__":
    main()
