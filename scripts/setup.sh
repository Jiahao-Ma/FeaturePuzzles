#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Pi3 Segmentation Project Setup ==="
echo "Project root: $PROJECT_ROOT"

# ---- 1. Install Python dependencies ----
echo ""
echo "[1/3] Installing Python dependencies ..."
pip install -r "$PROJECT_ROOT/requirements_seg.txt"

# ---- 2. Clone Pi3 repository ----
REPO_DIR="$PROJECT_ROOT/repo"
PI3_DIR="$REPO_DIR/Pi3"
mkdir -p "$REPO_DIR"
if [ -d "$PI3_DIR" ]; then
    echo ""
    echo "[2/3] Pi3 repository already exists at $PI3_DIR — skipping clone."
else
    echo ""
    echo "[2/3] Cloning Pi3 repository into $REPO_DIR ..."
    git clone https://github.com/yyfz/Pi3.git "$PI3_DIR"
fi

# ---- 3. Download Pi3X pretrained weights ----
CKPT_FILE="$PROJECT_ROOT/checkpoints/model.safetensors"
if [ -f "$CKPT_FILE" ]; then
    echo ""
    echo "[3/3] Pi3X weights already present at $CKPT_FILE — skipping download."
else
    echo ""
    echo "[3/3] Downloading Pi3X pretrained weights ..."
    python "$SCRIPT_DIR/download_weights.py" --local-dir "$PROJECT_ROOT/checkpoints"
fi

echo ""
echo "=== Setup complete ==="
