"""Diagnostic: dump per-object predicted masks for a few frames to verify
that the multiprompt head keeps the K SAM-prompted objects *separate* across
views (no over-merging) and to confirm what the propagation looks like.

Saves, for a small window of frames:
  diag_frame{f}_obj{i}.png      per-object predicted mask (white on black)
  diag_frame{f}_overlay.png     all-K coloured overlay on the frame
  diag_summary.txt              per-(frame,object) pixel counts
"""
import sys
from pathlib import Path
import numpy as np
import cv2
import torch

RELEASE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RELEASE_ROOT / 'ICLR26Demo' / 'vggt_omega_multiprompt'))
sys.path.insert(0, str(RELEASE_ROOT))

from render_multiprompt_demo import (
    load_scannet_scene, sam_top_k_masks, object_colors_bgr,
    vggt_preprocess, map_masks_to_256, vggt_multiprompt_segment,
)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--sam_ckpt',
                    default=str(RELEASE_ROOT / 'checkpoints' / 'sam' / 'sam_vit_h_4b8939.pth'))
    ap.add_argument('--omega_ckpt',
                    default=str(RELEASE_ROOT / 'checkpoints' / 'vggt_omega_1b_256_text.pt'))
    ap.add_argument('--seg_ckpt',
                    default=str(RELEASE_ROOT / 'weights' / 'seg_omega_lora_multiprompt' /
                                'checkpoint-best.pth'))
    ap.add_argument('--n_frames', type=int, default=12)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--dump_frames', type=int, nargs='+', default=[0, 5, 11])
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    out = Path(args.out_dir) / (Path(args.scene_dir).name + '_diag')
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    def _log(m): print(m, flush=True)
    data = load_scannet_scene(Path(args.scene_dir), args.n_frames, 640, 480, log=_log)
    images = data['images']
    S = len(images)
    _log(f'[diag] {S} frames loaded')

    sam_masks = sam_top_k_masks(images[0], Path(args.sam_ckpt), args.top_k, dev, log=_log)
    K = sam_masks.shape[0]
    obj_bgr = object_colors_bgr(K)

    _, bboxes = vggt_preprocess(images, 256)
    prompt_masks_256 = map_masks_to_256(sam_masks, bboxes[0], 256)

    _log('[diag] running multiprompt seg ...')
    pred = vggt_multiprompt_segment(
        images, Path(args.omega_ckpt), Path(args.seg_ckpt),
        prompt_masks_256, 0, 256, dev, thr=0.5, m_chunk=2, window_size=6, log=_log)
    # pred: (S, K, H, W) uint8

    # Geometric gating
    from render_multiprompt_demo import geometric_anchor_and_gate
    _log('[diag] geometric anchor + gating ...')
    gated = geometric_anchor_and_gate(
        pred, data['depth'], data['c2w'], data['K'], sam_masks, 0,
        occ_rel=0.05, min_overlap_px=10, log=_log)

    lines = ['frame,obj,px_before,px_after,area_prompt_px,iou_vs_prompt\n']
    H, W = images.shape[1:3]
    for f in args.dump_frames:
        if f >= S:
            continue
        bgr = cv2.cvtColor(images[f], cv2.COLOR_RGB2BGR)
        overlay = bgr.copy()
        for i in range(K):
            m = gated[f, i]
            px = int(m.sum())
            if f == 0:
                pm = sam_masks[i]
                inter = int((m & pm).sum())
                uni = int((m | pm).sum())
                iou = inter / uni if uni else 0.0
            else:
                iou = float('nan')
            lines.append(f'{f},{i},{int(pred[f,i].sum())},{px},{int(sam_masks[i].sum())},{iou:.3f}\n')
            cv2.imwrite(str(out / f'frame{f:02d}_obj{i:02d}.png'), m * 255)
            if px > 0:
                overlay[m > 0] = (overlay[m > 0].astype(np.float32) * 0.4 +
                                  obj_bgr[i].astype(np.float32) * 0.6).astype(np.uint8)
        cv2.imwrite(str(out / f'frame{f:02d}_overlay_gated.png'), overlay)
    (out / 'diag_summary.csv').write_text(''.join(lines))
    _log(f'[diag] wrote per-object masks + overlays to {out}')
    _log('[diag] summary:\n' + ''.join(lines))


if __name__ == '__main__':
    main()
