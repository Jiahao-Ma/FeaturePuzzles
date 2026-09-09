#!/usr/bin/env bash
# Launch the promptable-segmentation demo server.
#
#   ./run_demo.sh [PORT]
#
# The web UI is served on PORT (default 7860); each result's interactive viser
# viewer gets its own port starting at PORT+1.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${1:-7860}"
# The demo needs the lingbot-map environment (compiled render_cuda_ext, Open3D,
# viser). Falling back to whatever python is on PATH keeps the script usable on
# a machine that installed that environment somewhere else.
DEFAULT_PYTHON=/home/ps/miniconda3/envs/lingbot-map/bin/python
PYTHON="${PYTHON:-$DEFAULT_PYTHON}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3 || command -v python || true)"
fi

if [[ -z "$PYTHON" || ! -x "$PYTHON" ]]; then
  echo "no usable python found -- set PYTHON=/path/to/lingbot-map/bin/python" >&2
  exit 1
fi

# ffmpeg ships inside the conda env; make sure the compositor can find it.
export PATH="$(dirname "$PYTHON"):$PATH"

cd "$HERE"
echo "UI      http://localhost:${PORT}"
echo "viser   http://localhost:$((PORT + 1))  (allocated per result)"
exec "$PYTHON" -m app.server --port "$PORT" --viser_base_port "$((PORT + 1))"
