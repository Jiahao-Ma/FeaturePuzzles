#!/usr/bin/env python3
"""Sweep window size for dynamic propagation from a single point prompt.

Checks whether feeding the whole clip in one pass keeps the mask on the prompted
object, and compares it against shorter windows closer to the head's 6-view
training distribution.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import paths, sam_prompt                # noqa: E402
from app.frames import load_video                # noqa: E402
from app.segment import segment_dynamic          # noqa: E402
from app.static_ref import ref                   # noqa: E402


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--n_frames", type=int, default=24)
    ap.add_argument("--px", type=float, required=True, help="prompt x, 0..1")
    ap.add_argument("--py", type=float, required=True, help="prompt y, 0..1")
    ap.add_argument("--windows", default="24,12,8,6")
    ap.add_argument("--out", default="runs/debug_dynamic")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    R = ref()

    src = load_video(Path(args.video), args.n_frames)
    S, (H, W) = src.size, src.hw
    print(f"clip: {S} frames @ {W}x{H}")

    pt = {"x": args.px, "y": args.py, "label": 1}
    seed = sam_prompt.masks_from_points(src.images[0], [{"points": [pt]}], image_id="dbg")
    print(f"seed area = {int(seed.sum())} px ({seed.sum() / (H * W) * 100:.1f}% of frame)")
    cv2.imwrite(str(out / "seed.png"),
                sam_prompt.encode_mask_preview(src.images[0], seed, R.object_colors_bgr(1)))
    sam_prompt.release()

    _, bboxes = R.vggt_preprocess(src.images, 256)
    seed_256 = R.map_masks_to_256(seed, bboxes[0], 256)

    for wsz in [int(x) for x in args.windows.split(",")]:
        wsz = min(wsz, S)
        if wsz >= S:
            pred = segment_dynamic(src.images, seed_256, 0, 256, device, exclusive=False)
            tag = f"w{wsz}_oneshot"
        else:
            # Sliding windows that always carry the reference frame, as in the
            # static pipeline, but without any geometric post-processing.
            pred = R.vggt_multiprompt_segment(
                src.images, paths.OMEGA_CKPT, paths.SEG_CKPT,
                seed_256, 0, 256, device, thr=0.5, m_chunk=1, window_size=wsz)
            tag = f"w{wsz}_sliding"

        ref_iou = iou(pred[0, 0] > 0, seed[0] > 0)
        areas = [int(pred[v, 0].sum()) for v in range(S)]
        print(f"{tag:16s} ref-frame IoU vs seed = {ref_iou:.3f}; "
              f"area first/mid/last = {areas[0]}/{areas[S//2]}/{areas[-1]}")
        R.encode_mask_overlay_video(src.images, pred, R.object_colors_bgr(1),
                                    out / f"{tag}.mp4", fps=6, log=lambda *_: None)


if __name__ == "__main__":
    main()
