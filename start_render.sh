#!/usr/bin/env bash
set -euo pipefail
python -m app.prepare_ffmpeg
export PATH="/tmp/bin:$PATH"
rq worker video-jobs &
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
