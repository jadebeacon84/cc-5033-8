#!/usr/bin/env bash
# Start the underwater temporal-denoise UI.
#
#   ./tools/run_ui.sh              # port 8765
#   ./tools/run_ui.sh 9000         # custom port
#   PYTHON=/path/to/python ./tools/run_ui.sh
#
# Requires: python3 with numpy + opencv-python, and ffmpeg (or imageio-ffmpeg).
# If those are missing the script runs tools/fetch_deps.py once, which vendors
# matching manylinux wheels into .tools/pylib (stdlib only, no pip needed).
# Open http://127.0.0.1:<port>/ once "listening" is printed.
set -euo pipefail

cd "$(dirname "$0")/.."

PORT="${1:-8765}"
PY="${PYTHON:-python3}"
VENDOR=".tools/pylib"

check() {
  "$PY" - <<'PY' >/dev/null 2>&1
import sys
sys.path.insert(0, ".tools/pylib")
import numpy, cv2
PY
}

if ! check; then
  if [ -d "$VENDOR/numpy" ]; then
    echo "已有本地依赖但仍无法导入：请检查 $VENDOR 是否匹配 $($PY -V 2>&1)。" >&2
    exit 1
  fi
  echo "未检测到 numpy/opencv，正在获取本地依赖（.tools/pylib，约 260 MB）…" >&2
  "$PY" -I tools/fetch_deps.py || {
    echo "获取失败。可执行 pip install -r requirements.txt 或手动放置依赖。" >&2
    exit 1
  }
fi

if [ -d "$VENDOR" ]; then
  export PYTHONPATH="$PWD/$VENDOR${PYTHONPATH:+:$PYTHONPATH}"
fi

if ! "$PY" - <<'PY' >/dev/null 2>&1
from denoise.media import ffmpeg_exe
ffmpeg_exe()
PY
then
  echo "未找到 ffmpeg。请安装 ffmpeg，或 pip install imageio-ffmpeg。" >&2
  exit 1
fi

echo "水下降噪界面： http://127.0.0.1:${PORT}/"
echo "按 Ctrl+C 停止。"
exec "$PY" -m denoise.server "$PORT"
