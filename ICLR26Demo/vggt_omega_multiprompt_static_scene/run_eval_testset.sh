#!/usr/bin/env bash
# Run the iterative + geo_cc + exclusive pipeline over the whole ScanNet test
# split. Results land in <out_dir>/<scene>/; nothing outside it is touched.
# Already-finished scenes are skipped, so the script can be re-run to resume.
#
# Usage:
#   ./run_eval_testset.sh                                  # SAM prompts
#   ./run_eval_testset.sh eval_iterative_excl_gt_mask --gt_prompt
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
TEST_DIR="$ROOT/data/dataset/scannet/test"

OUT_NAME="${1:-eval_iterative_excl}"
shift || true
EXTRA_FLAGS=("$@")

OUT_ROOT="$HERE/$OUT_NAME"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$LOG_DIR"

echo "[cfg ] out=$OUT_ROOT extra=${EXTRA_FLAGS[*]:-<none>}"
cd "$HERE"
for scene_dir in "$TEST_DIR"/scene*; do
    scene="$(basename "$scene_dir")"
    out="$OUT_ROOT/$scene"
    if [ -f "$out/${scene}_masks_overlay.mp4" ]; then
        echo "[skip] $scene already done"
        continue
    fi
    echo "[run ] $scene ..."
    start=$(date +%s)
    python -u render_multiprompt_demo.py \
        --scene_dir "$scene_dir" \
        --out_dir "$OUT_ROOT" \
        --iterative --geo_cc --exclusive \
        --modes seg,hl \
        "${EXTRA_FLAGS[@]}" \
        --masks_cache "$out/${scene}_pred_masks.npz" \
        > "$LOG_DIR/${scene}.log" 2>&1
    rc=$?
    dur=$(( $(date +%s) - start ))
    if [ $rc -eq 0 ]; then
        echo "[ ok ] $scene in ${dur}s"
    else
        echo "[FAIL] $scene rc=$rc after ${dur}s -- see $LOG_DIR/${scene}.log"
        tail -5 "$LOG_DIR/${scene}.log"
    fi
done
echo "[done] all scenes -> $OUT_ROOT"
