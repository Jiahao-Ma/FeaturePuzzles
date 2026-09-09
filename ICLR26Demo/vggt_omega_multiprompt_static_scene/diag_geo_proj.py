"""Debug: project ref-view obj0's 3D points to all frames, report
visible / in-bounds / occluded counts per view to diagnose whether the
geometric gating empty results are a visibility truth or a projection bug."""
import sys
from pathlib import Path
import numpy as np
import cv2
import torch

RELEASE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RELEASE_ROOT / 'ICLR26Demo' / 'vggt_omega_multiprompt'))
sys.path.insert(0, str(RELEASE_ROOT))
from render_multiprompt_demo import (
    load_scannet_scene, sam_top_k_masks, _backproject_view, _project_to_view,
)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene_dir', required=True)
    ap.add_argument('--sam_ckpt',
                    default=str(RELEASE_ROOT / 'checkpoints' / 'sam' / 'sam_vit_h_4b8939.pth'))
    ap.add_argument('--n_frames', type=int, default=12)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--obj', type=int, default=0)
    args = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _log(m): print(m, flush=True)
    data = load_scannet_scene(Path(args.scene_dir), args.n_frames, 640, 480, log=_log)
    depths, c2w, Ks = data['depth'], data['c2w'], data['K']
    S = len(depths)
    sam_masks = sam_top_k_masks(data['images'][0], Path(args.sam_ckpt), args.top_k, dev, log=_log)
    world_ref = _backproject_view(depths[0], c2w[0], Ks[0])
    pts = world_ref[sam_masks[args.obj] > 0]
    print(f'[dbg] obj{args.obj}: {pts.shape[0]} seed pts')
    H, W = depths.shape[1:]
    for v in range(S):
        ui, vi, zc, inb = _project_to_view(pts, Ks[v], c2w[v], H, W)
        neg = int((zc < 1e-3).sum())
        pos = int((zc >= 1e-3).sum())
        zmap = np.full(pts.shape[0], np.inf, np.float32)
        zmap[inb] = depths[v][vi[inb], ui[inb]]
        occ = inb & (zc > zmap * (1.05))
        vis = inb & (zc <= zmap * (1.05))
        inb_n = int(inb.sum())
        zc_inb = zc[inb] if inb_n else np.array([np.nan])
        print(f'[dbg] view{v}: inb={inb_n} vis={int(vis.sum())} '
              f'behind_cam={neg} in_front={pos} '
              f'z_all=[{zc.min():.2f},{zc.max():.2f}] '
              f'z_inb=[{zc_inb.min():.2f},{zc_inb.max():.2f}]')


if __name__ == '__main__':
    main()
