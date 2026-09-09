#!/usr/bin/env bash
# Download nuScenes (v1.0-mini) from the server and extract locally.
#
# WHY THIS IS FAST:
#   The server holds the data both as an extracted tree (~9.1 GB, ~31k small
#   files: 4848 sample files + 26358 sweep files) AND as a single compressed
#   `v1.0-mini.tgz` (3.9 GB). rsync'ing 31k tiny files over ssh is ~10x slower
#   per byte than one big sequential file, so we download the .tgz and extract
#   it locally. That also halves the bytes transferred. nuScenes has no public
#   depth modality (only RGB / lidar / radar + ego_pose & calibrated_sensor
#   tables), so this archive is the complete dataset.
#
# USAGE (run on your LOCAL machine):
#   1. Edit SERVER and REMOTE_ROOT below.
#   2. bash download_nuscenes_local.sh
#
# Resumable: re-run anytime; rsync --partial --append-verify skips the completed
# portion of the .tgz, and extraction is guarded so it won't redo work.
set -euo pipefail

# ============================================================================
# CONFIG — edit these lines
# ============================================================================
# SSH alias/host for the server, e.g. "user@10.0.0.5" or a ~/.ssh alias "gpu".
SERVER="root@10.108.2.160"
# Extra SSH options (e.g. non-default port). Empty string if not needed.
SSH_PORT="3069"
# Dir on the server that holds the nuScenes mini subset (contains
# v1.0-mini.tgz and the extracted maps/ samples/ sweeps/ v1.0-mini/).
REMOTE_ROOT="/mnt/dataset/marko/puzzles/puzzles/ESAMBench/data/nuscenes"
# ============================================================================

# Where to save the archive + extracted data on your LOCAL machine.
LOCAL_DIR="${LOCAL_DIR:-$/home/ps/code/puzzles_release/data/nuscenes}"
ARCHIVE="$LOCAL_DIR/v1.0-mini.tgz"
EXTRACT_DIR="$LOCAL_DIR/data"        # becomes data/{maps,samples,sweeps,v1.0-mini}
mkdir -p "$LOCAL_DIR" "$EXTRACT_DIR"

SSH_OPTS=(-o ServerAliveInterval=15 -o ServerAliveCountMax=3)
if [[ -n "${SSH_PORT}" ]]; then
    SSH_OPTS+=(-p "${SSH_PORT}")
fi

# ----------------------------------------------------------------------------
# 1. Download the single compressed archive (resumable).
# ----------------------------------------------------------------------------
echo "[1/2] rsync v1.0-mini.tgz  (~3.9 GB, resumable)"
rsync -P --partial --append-verify -S \
    -e "ssh ${SSH_OPTS[*]}" \
    "${SERVER}:${REMOTE_ROOT}/v1.0-mini.tgz" \
    "${ARCHIVE}"

# ----------------------------------------------------------------------------
# 2. Extract locally (guarded: skip if already extracted).
# ----------------------------------------------------------------------------
# The official archive extracts to: maps/ samples/ sweeps/ v1.0-mini/
SENTINEL="$EXTRACT_DIR/.extracted_ok"
if [[ -f "$SENTINEL" ]]; then
    echo "[2/2] already extracted -> $EXTRACT_DIR (remove $SENTINEL to force re-extract)"
else
    echo "[2/2] extracting -> $EXTRACT_DIR"
    tar -xzf "$ARCHIVE" -C "$EXTRACT_DIR"
    touch "$SENTINEL"
fi

echo
echo "done."
echo "  archive : $ARCHIVE"
echo "  data    : $EXTRACT_DIR/{maps,samples,sweeps,v1.0-mini}"
echo
echo "Contents (v1.0-mini, 10 scenes):"
echo "  RGB        : samples/CAM_{FRONT,BACK,FRONT_LEFT,FRONT_RIGHT,BACK_LEFT,BACK_RIGHT}/*.jpg"
echo "               (+ sweeps/CAM_* intermediate frames)"
echo "  Lidar PC   : samples/LIDAR_TOP/*.pcd.bin  (+ sweeps/LIDAR_TOP)"
echo "  Radar      : samples/RADAR_*/*.pcd  (+ sweeps/RADAR_*)   [5 radars, optional]"
echo "  Camera pose: v1.0-mini/ego_pose.json (world<->ego)  x"
echo "               v1.0-mini/calibrated_sensor.json (ego<->sensor + camera_intrinsic)"
echo "  Maps       : maps/*.png  (4 HD maps)"
echo "  NO depth  : nuScenes has no depth modality (derive via lidar->cam projection"
echo "               or a monocular depth model if needed)."
