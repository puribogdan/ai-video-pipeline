#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert Deepgram-style JSON (word-level timestamps) into WebVTT.

New usage for your pipeline (synchronous, no webhook):
- Right after you receive the Deepgram transcript JSON in-memory:
    from get_vtt_subtitles import convert_and_write_from_response
    vtt_path = convert_and_write_from_response(transcript_json, job_id, tmp_dir=os.getenv("TMP_DIR", "/tmp"))
- Then, at your final artifact upload step, push `vtt_path` to Backblaze:
    key = f"vtts/{job_id}.vtt"  (Content-Type: text/vtt; charset=utf-8)

JSON input shape (example):
{
  "audio_file": ".../input_trimmed.mp3",
  "transcript": "Once upon a time ...",
  "words": [
    {"word": "once", "start": 0.40, "end": 0.88},
    {"word": "upon", "start": 0.88, "end": 1.38},
    ...
  ],
  "detected_language": "en"
}
"""

from __future__ import annotations
import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Dict, Any, Optional

# -------- Defaults (CLI/back-compat) ----------
DEFAULT_INPUT  = os.path.join("subtitles", "input_subtitles.json")
DEFAULT_OUTPUT = os.path.join("subtitles", "final.en.vtt")

# Segmentation heuristics
MAX_DURATION_S       = 5.0   # max seconds per cue
MAX_CHARS_TOTAL      = 84    # rough guard for total chars in a cue
MAX_CHARS_PER_LINE   = 42    # wrap to ~42 chars per line
MAX_LINES            = 2
MAX_CHAR_PER_SECOND  = 17.0  # reading speed
GAP_THRESHOLD_S      = 0.60  # new cue if long pause between words
NUDGE_MS             = 33    # avoid overlaps by nudging ~1 frame
MIN_CUE_MS           = 300   # ensure a visible minimum

EoS_TOKENS = {".", "?", "!", "…"}  # prefer to break after these


# -------- Data structures ----------
@dataclass
class Word:
    text: str
    start: float
    end: float

@dataclass
class Cue:
    start: float
    end: float
    text: str


# -------- Helpers ----------
def _load_words(data: Dict[str, Any]) -> List[Word]:
    if "words" not in data or not isinstance(data["words"], list):
        raise ValueError("Input JSON missing 'words' list.")
    words: List[Word] = []
    for w in data["words"]:
        text = w.get("word") or w.get("text")
        if not text:
            continue
        start = float(w["start"])
        end = float(w["end"])
        words.append(Word(text=text, start=start, end=end))
    words.sort(key=lambda w: (w.start, w.end))
    if not words:
        raise ValueError("No usable word entries found.")
    return words

def _tidy_punctuation_spacing(s: str) -> str:
    s = re.sub(r"\s+([.,?!…])", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _wrap_lines(text: str, max_chars_per_line: int, max_lines: int) -> str:
    words = text.split()
    lines: List[str] = []
    cur: List[str] = []
    for i, w in enumerate(words):
        cand = (" ".join(cur + [w])).strip()
        if len(cand) <= max_chars_per_line:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
            if len(lines) == max_lines - 1:
                # dump the rest into the last line (may exceed a bit)
                cur = words[i:]
                break
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines[:max_lines])

def _format_timestamp_vtt(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hh = total_ms // 3_600_000
    mm = (total_ms // 60_000) % 60
    ss = (total_ms // 1_000) % 60
    mmm = total_ms % 1_000
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{mmm:03d}"

def _segment_words(
    words: List[Word],
    max_duration: float,
    max_chars_total: int,
    max_cps: float,
    gap_threshold: float,
    max_chars_per_line: int,
    max_lines: int
) -> List[Cue]:
    if not words:
        return []
    cues: List[Cue] = []
    seg: List[Word] = [words[0]]
    seg_start = words[0].start

    for i in range(1, len(words)):
        w_prev = words[i-1]
        w = words[i]
        candidate_text = " ".join([*map(lambda x: x.text, seg), w.text])
        candidate_text = _tidy_punctuation_spacing(candidate_text)
        dur = max(w.end - seg_start, 1e-9)
        cps = len(candidate_text) / dur
        gap = w.start - w_prev.end
        end_sentence = any(w_prev.text.endswith(tok) for tok in EoS_TOKENS)

        should_break = (
            gap > gap_threshold or
            (w.end - seg_start) > max_duration or
            len(candidate_text) > max_chars_total or
            cps > max_cps or
            (end_sentence and (w_prev.end - seg_start) >= 1.6)
        )

        if should_break:
            text = _tidy_punctuation_spacing(" ".join(x.text for x in seg))
            wrapped = _wrap_lines(text, max_chars_per_line, max_lines)
            cues.append(Cue(start=seg_start, end=seg[-1].end, text=wrapped))
            seg = [w]
            seg_start = w.start
        else:
            seg.append(w)

    if seg:
        text = _tidy_punctuation_spacing(" ".join(x.text for x in seg))
        wrapped = _wrap_lines(text, max_chars_per_line, max_lines)
        cues.append(Cue(start=seg_start, end=seg[-1].end, text=wrapped))

    # enforce monotonic times / minimal visibility
    for i in range(1, len(cues)):
        prev = cues[i-1]
        cur = cues[i]
        if cur.start <= prev.end:
            cur.start = prev.end + NUDGE_MS / 1000.0
            if cur.end < cur.start:
                cur.end = cur.start + MIN_CUE_MS / 1000.0
    for c in cues:
        if (c.end - c.start) * 1000 < MIN_CUE_MS:
            c.end = c.start + MIN_CUE_MS / 1000.0

    return cues


# -------- Public API you’ll call from your pipeline ----------
def convert_json_to_vtt(
    data: Dict[str, Any],
    *,
    max_duration: float = MAX_DURATION_S,
    max_chars_total: int = MAX_CHARS_TOTAL,
    max_cps: float = MAX_CHAR_PER_SECOND,
    gap_threshold: float = GAP_THRESHOLD_S,
    max_chars_per_line: int = MAX_CHARS_PER_LINE,
    max_lines: int = MAX_LINES
) -> str:
    """Deepgram-like JSON (with 'words') -> WebVTT string."""
    words = _load_words(data)
    cues = _segment_words(
        words,
        max_duration=max_duration,
        max_chars_total=max_chars_total,
        max_cps=max_cps,
        gap_threshold=gap_threshold,
        max_chars_per_line=max_chars_per_line,
        max_lines=max_lines,
    )
    out_lines = ["WEBVTT", ""]
    if data.get("detected_language"):
        out_lines += [f"NOTE language={data['detected_language']}", ""]
    for i, cue in enumerate(cues, 1):
        start = _format_timestamp_vtt(cue.start)
        end = _format_timestamp_vtt(cue.end)
        out_lines.append(str(i))
        out_lines.append(f"{start} --> {end}")
        out_lines.append(cue.text)
        out_lines.append("")
    return "\n".join(out_lines).strip() + "\n"


def write_vtt_from_deepgram(
    transcript_json: Optional[Dict[str, Any]] = None,
    json_path: Optional[str] = None,
    vtt_out_path: str = "",
    *,
    ensure_crlf: bool = False
) -> str:
    """
    Accepts either a parsed Deepgram JSON dict or a path to a JSON file,
    converts to WebVTT, and writes UTF-8 to vtt_out_path. Returns vtt_out_path.
    """
    if not vtt_out_path:
        raise ValueError("vtt_out_path is required.")

    if transcript_json is None and not json_path:
        raise ValueError("Provide either transcript_json or json_path.")

    if transcript_json is None:
        with open(json_path, "r", encoding="utf-8") as f:
            transcript_json = json.load(f)

    vtt_text = convert_json_to_vtt(transcript_json)

    # Some players prefer CRLF; make it optional
    if ensure_crlf:
        vtt_text = vtt_text.replace("\n", "\r\n")

    os.makedirs(os.path.dirname(vtt_out_path) or ".", exist_ok=True)
    with open(vtt_out_path, "w", encoding="utf-8", newline="") as f:
        f.write(vtt_text)

    return vtt_out_path


def convert_and_write_from_response(
    transcript_json: Dict[str, Any],
    job_id: str,
    *,
    tmp_dir: str = "/tmp",
    ensure_crlf: bool = False
) -> str:
    """
    PRIMARY ENTRY for your pipeline (synchronous, no webhook).
    - Takes the Deepgram transcript JSON you already have in memory,
    - writes a VTT to {tmp_dir}/{job_id}.vtt,
    - returns that absolute path.
    """
    vtt_out_path = os.path.join(tmp_dir, f"{job_id}.vtt")
    return write_vtt_from_deepgram(
        transcript_json=transcript_json,
        json_path=None,
        vtt_out_path=vtt_out_path,
        ensure_crlf=ensure_crlf,
    )


# -------- CLI (kept for local testing/back-compat) ----------
def main():
    ap = argparse.ArgumentParser(description="Convert Deepgram JSON to WebVTT.")
    ap.add_argument("--in", dest="inp", default=DEFAULT_INPUT,
                    help=f"Path to input JSON (default: {DEFAULT_INPUT})")
    ap.add_argument("--out", dest="out", default=DEFAULT_OUTPUT,
                    help=f"Path to output VTT (default: {DEFAULT_OUTPUT})")
    ap.add_argument("--max-duration", type=float, default=MAX_DURATION_S)
    ap.add_argument("--max-cps", type=float, default=MAX_CHAR_PER_SECOND)
    ap.add_argument("--gap", type=float, default=GAP_THRESHOLD_S)
    ap.add_argument("--max-chars-total", type=int, default=MAX_CHARS_TOTAL)
    ap.add_argument("--max-chars-per-line", type=int, default=MAX_CHARS_PER_LINE)
    ap.add_argument("--max-lines", type=int, default=MAX_LINES)
    ap.add_argument("--crlf", action="store_true", help="Write CRLF line endings")
    args = ap.parse_args()

    with open(args.inp, "r", encoding="utf-8") as f:
        data = json.load(f)

    vtt = convert_json_to_vtt(
        data,
        max_duration=args.max_duration,
        max_chars_total=args.max_chars_total,
        max_cps=args.max_cps,
        gap_threshold=args.gap,
        max_chars_per_line=args.max_chars_per_line,
        max_lines=args.max_lines,
    )

    if args.crlf:
        vtt = vtt.replace("\n", "\r\n")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        f.write(vtt)

    print(f"[OK] Wrote WebVTT: {args.out}")


if __name__ == "__main__":
    main()
