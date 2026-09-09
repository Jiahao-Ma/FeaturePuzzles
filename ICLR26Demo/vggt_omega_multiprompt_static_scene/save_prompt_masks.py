"""Save the first-frame SAM prompt masks (top-K) for a scene into its output folder.

Produces, per scene:
  - prompt_masks_firstframe_overlay.png  : all K masks coloured + outlined on the first frame
  - prompt_masks_firstframe_grid.png      : K binary masks in a grid
  - prompt_mask_256_{i:02d}.png            : each mask mapped into the 256x256 padded square
                                             (exactly what is fed to vggt_omega_multiprompt)
  - prompt_masks_firstframe.npz           : raw arrays (work masks, 256 masks, colors, bbox)
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
    vggt_preprocess, map_masks_to_256,
)


def save_scene_prompt_masks(scene_dir: Path, out_dir: Path, sam_ckpt: Path,
                            n_frames: int = 100, top_k: int = 10,
                            work_w: int = 640, work_h: int = 480,
                            resolution: int = 256, ref_view: int = 0,
                            device: str = 'cuda'):
    scene_name = scene_dir.name
    out = out_dir / scene_name
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')

    def _log(m): print(m, flush=True)
    _log(f'[prompt] {scene_name}: loading scene ...')
    data = load_scannet_scene(scene_dir, n_frames, work_w, work_h, log=_log)
    images = data['images']
    ref_rgb = images[ref_view]  # (H,W,3) uint8 RGB at working res

    _log(f'[prompt] {scene_name}: SAM segment-everything -> top-{top_k} ...')
    sam_masks = sam_top_k_masks(ref_rgb, sam_ckpt, top_k, dev, log=_log)  # (K,H,W) uint8
    K = sam_masks.shape[0]
    obj_bgr = object_colors_bgr(K)  # (K,3) uint8 BGR

    # 256x256 padded-square prompt masks (what is fed to the seg head)
    _, bboxes = vggt_preprocess(images, resolution)
    ref_bbox = bboxes[ref_view]
    prompt_masks_256 = map_masks_to_256(sam_masks, ref_bbox, resolution)  # (K,256,256)

    ref_bgr = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2BGR)

    # 1) Coloured overlay on the first frame.
    overlay = ref_bgr.copy()
    blend = ref_bgr.copy()
    for i in range(K):
        m = sam_masks[i].astype(bool)
        if not m.any():
            continue
        blend[m] = obj_bgr[i]
    overlay = cv2.addWeighted(blend, 0.5, overlay, 0.5, 0)
    # draw outlines
    for i in range(K):
        m = sam_masks[i].astype(np.uint8)
        if not m.any():
            continue
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1,
                         (int(obj_bgr[i][0]), int(obj_bgr[i][1]), int(obj_bgr[i][2])), 2)
    ov_path = out / f'{scene_name}_prompt_masks_firstframe_overlay.png'
    cv2.imwrite(str(ov_path), overlay)
    _log(f'[prompt] wrote {ov_path}')

    # 2) Grid of binary masks.
    cols = int(np.ceil(np.sqrt(K)))
    rows = int(np.ceil(K / cols))
    cell_h, cell_w = work_h // 2, work_w // 2
    grid = np.full((rows * cell_h, cols * cell_w, 3), 255, np.uint8)
    for i in range(K):
        r, c = i // cols, i % cols
        m = sam_masks[i]
        cell = np.full((cell_h, cell_w, 3), 255, np.uint8)
        mm = cv2.resize(m, (cell_w, cell_h), interpolation=cv2.INTER_NEAREST)
        col = obj_bgr[i]
        cell[mm > 0] = col
        cv2.putText(cell, f'#{i} a={int(m.sum())}', (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        grid[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = cell
    grid_path = out / f'{scene_name}_prompt_masks_firstframe_grid.png'
    cv2.imwrite(str(grid_path), grid)
    _log(f'[prompt] wrote {grid_path}')

    # 3) Individual 256x256 prompt masks (exactly fed to the model).
    for i in range(K):
        p = out / f'{scene_name}_prompt_mask_256_{i:02d}.png'
        cv2.imwrite(str(p), prompt_masks_256[i] * 255)

    # 4) Raw npz.
    npz_path = out / f'{scene_name}_prompt_masks_firstframe.npz'
    np.savez(str(npz_path),
             sam_masks_work=sam_masks,
             prompt_masks_256=prompt_masks_256,
             colors_bgr=obj_bgr,
             ref_bbox=np.array(ref_bbox),
             ref_rgb=ref_rgb)
    _log(f'[prompt] wrote {npz_path}  (K={K}, bbox={ref_bbox})')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene_dir', required=True)
    ap.add_argument('--out_dir',
                    default=str(RELEASE_ROOT / 'ICLR26Demo' / 'vggt_omega_multiprompt'))
    ap.add_argument('--sam_ckpt',
                    default=str(RELEASE_ROOT / 'checkpoints' / 'sam' / 'sam_vit_h_4b8939.pth'))
    ap.add_argument('--n_frames', type=int, default=100)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    save_scene_prompt_masks(
        Path(args.scene_dir), Path(args.out_dir), Path(args.sam_ckpt),
        n_frames=args.n_frames, top_k=args.top_k, device=args.device)
    print('[prompt] done')


if __name__ == '__main__':
    main()
