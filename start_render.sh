#!/usr/bin/env bash
set -euo pipefail
python -m app.prepare_ffmpeg
export PATH="/tmp/bin:$PATH"
export IMAGEIO_FFMPEG_EXE=/tmp/bin/ffmpeg
python rq_worker.py &
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
