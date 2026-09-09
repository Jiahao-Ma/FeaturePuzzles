#!/usr/bin/env bash
# Download EmbodiedSplat (ScanNet + ScanNet++) from the server and extract locally.
#
# WHY THIS IS FAST:
#   The server already has the data as 15 compressed tar.zst shards (87 GB total,
#   large sequential files) instead of the extracted dataset (156 GB / 151k small
#   files). rsync'ing 15 big files saturates bandwidth; rsync'ing 151k tiny files
#   over ssh is ~10x slower per byte. Downloading the compressed shards also halves
#   the byte count. Downloading from HF directly is slow on this network (the
#   server needed a proxy to reach HF), so pulling the shards off the server wins.
#
# USAGE (run on your LOCAL machine):
#   1. Edit SERVER and REMOTE_ROOT below.
#   2. bash download_embodiedsplat.sh
#
# Resumable: re-run anytime; rsync --partial --append-verify skips completed files.
set -euo pipefail

# ============================================================================
# CONFIG — edit these two lines
# ============================================================================
# SSH alias/host for the server, e.g. "user@10.0.0.5" or a ~/.ssh alias "gpu".
SERVER="root@10.108.2.160"
# Extra SSH options (e.g. non-default port). Empty string if not needed.
SSH_PORT="3069"
# Root dir on the server that holds dataset-shard-XXXX.tar.zst
REMOTE_ROOT="/mnt/dataset/marko/puzzles/embodiedsplat_dl"
# ============================================================================

# Where to save shards + extracted data on your LOCAL machine.
LOCAL_DIR="${LOCAL_DIR:-$HOME/embodiedsplat}"
SHARD_DIR="$LOCAL_DIR/shards"
EXTRACT_DIR="$LOCAL_DIR/dataset"      # becomes dataset/scannet + dataset/scannetpp
mkdir -p "$SHARD_DIR" "$EXTRACT_DIR"

# Relevant shards only (skip nuscenes/visdrone/iggt/etc. that you don't need).
#   ScanNet   : 0001 0004 0005 0006 0007 0008 0009 0011 0013
#   ScanNet++ : 0014 0016 0019 0020 0023 0024
SCANNET_SHARDS=(0001 0004 0005 0006 0007 0008 0009 0011 0013)
SCANNETPP_SHARDS=(0014 0016 0019 0020 0023 0024)
ALL_SHARDS=("${SCANNET_SHARDS[@]}" "${SCANNETPP_SHARDS[@]}")

# Set DOWNLOAD=scannet|scannetpp|all to control what to fetch (default: all).
DOWNLOAD="${DOWNLOAD:-all}"
case "$DOWNLOAD" in
  scannet)   SHARDS=("${SCANNET_SHARDS[@]}") ;;
  scannetpp) SHARDS=("${SCANNETPP_SHARDS[@]}") ;;
  all)       SHARDS=("${ALL_SHARDS[@]}") ;;
  *) echo "DOWNLOAD must be scannet|scannetpp|all" >&2; exit 1 ;;
esac

echo "==> Server: $SERVER (port ${SSH_PORT})"
echo "==> Remote: $REMOTE_ROOT"
echo "==> Local:  $LOCAL_DIR"
echo "==> Fetching ${#SHARDS[@]} shards ($DOWNLOAD): ${SHARDS[*]}"
echo

# ----------------------------------------------------------------------------
# 1. rsync the shards (resumable, with progress).
#    --partial --append-verify  : resume interrupted downloads safely
#    -P                         : progress + --partial
#    -S                         : sparse handling (minor)
# Tip: if a single rsync stream can't saturate your link, install GNU parallel
#      and use the PARALLEL=1 env var below to fetch shards concurrently.
# ----------------------------------------------------------------------------
rsync_one () {
  local idx="$1"
  rsync -P --partial --append-verify -S -e "ssh -p ${SSH_PORT}" \
    "$SERVER:$REMOTE_ROOT/dataset-shard-${idx}.tar.zst" \
    "$SHARD_DIR/dataset-shard-${idx}.tar.zst"
}

if [[ "${PARALLEL:-0}" == "1" ]] && command -v parallel >/dev/null 2>&1; then
  printf '%s\n' "${SHARDS[@]}" | parallel -j 4 --line-buffer rsync_one
else
  for idx in "${SHARDS[@]}"; do
    rsync_one "$idx"
  done
fi

echo "==> All shards downloaded. Verifying sizes..."
for idx in "${SHARDS[@]}"; do
  f="$SHARD_DIR/dataset-shard-${idx}.tar.zst"
  sz=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f")
  printf "  %s  %s bytes\n" "$(basename "$f")" "$sz"
done

# ----------------------------------------------------------------------------
# 2. Extract locally. tar.zst -> scannet/ + scannetpp/ under $EXTRACT_DIR.
#    Requires `zstd` (brew install zstd / apt install zstd).
# ----------------------------------------------------------------------------
if ! command -v zstd >/dev/null 2>&1; then
  echo "ERROR: zstd not installed. Install it: brew install zstd  (macOS)  /  apt install zstd  (Linux)" >&2
  exit 1
fi

echo "==> Extracting into $EXTRACT_DIR ..."
for idx in "${SHARDS[@]}"; do
  f="$SHARD_DIR/dataset-shard-${idx}.tar.zst"
  echo "  extracting $f ..."
  tar --use-compress-program=unzstd -xf "$f" -C "$EXTRACT_DIR"
done

echo
echo "==> DONE. Extracted data:"
echo "    $EXTRACT_DIR/scannet   (ScanNet)"
echo "    $EXTRACT_DIR/scannetpp (ScanNet++)"
echo
echo "You can now delete the shards to reclaim space:"
echo "    rm -rf $SHARD_DIR"
