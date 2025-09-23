#!/usr/bin/env bash
set -euo pipefail
python -m app.prepare_ffmpeg
export PATH="/tmp/bin:$PATH"
python worker.py &
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"


#!/usr/bin/env bash
set -euo pipefail

python -m app.prepare_ffmpeg
export PATH="/tmp/bin:$PATH"
export IMAGEIO_FFMPEG_EXE=/tmp/bin/ffmpeg   # <- add this line

rq worker video-jobs &
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
