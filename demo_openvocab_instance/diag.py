#!/usr/bin/env python3
"""Diagnose reconstruction coverage vs ScanNet++ GT extent."""
import json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import load_scene_frames, load_depth, unproject, VoxelGrid, R_Z_FLIP

SC = sys.argv[1] if len(sys.argv) > 1 else "9071e139d9"
DATA = Path("/mnt/dataset/marko/puzzles/embodiedsplat_dl/dataset/scannetpp/data") / SC
frames, (fx, fy, cx, cy, W, H) = load_scene_frames(DATA, 8, 160)
print(f"{SC}: {len(frames)} frames, full {W}x{H}, fx={fx:.1f} cx={cx:.1f} cy={cy:.1f}")

g = VoxelGrid(0.05)
p = torch.load(DATA / f"{SC}.pth", map_location="cpu", weights_only=False)
gxyz = np.asarray(p["sampled_coords"], np.float64) @ R_Z_FLIP
print(f"GT: {len(gxyz)} pts  min={gxyz.min(0).round(2)}  max={gxyz.max(0).round(2)}")
gcode = np.unique(g.encode(g.keys(gxyz)))
print(f"GT occupies {len(gcode)} voxels @5cm")

for ws in (0.25, 0.5, 1.0):
    Wh, Ww = int(round(H * ws)), int(round(W * ws))
    Kw = (fx * ws, fy * ws, cx * ws, cy * ws)
    codes = []
    vf = []
    for fr in frames:
        d = load_depth(fr["depth"])
        dw = np.array(Image.fromarray(d).resize((Ww, Wh), Image.NEAREST))
        vf.append((dw > 0).mean())
        pts, _, _ = unproject(dw, fr["c2w"], Kw, gl=True)
        codes.append(np.unique(g.encode(g.keys(pts))))
    uc = np.unique(np.concatenate(codes))
    hit = np.isin(gcode, uc)
    inter = np.isin(g.encode(g.keys(gxyz)), uc)
    xyz = g.centers(g.decode(uc))
    print(f"work_scale={ws}: valid_depth={np.mean(vf)*100:.1f}%  ours={len(uc)} vox  "
          f"GT-vox recall={hit.mean()*100:.1f}%  GT-pt coverage={inter.mean()*100:.1f}%  "
          f"min={xyz.min(0).round(2)} max={xyz.max(0).round(2)}")

# depth range sanity
d = load_depth(frames[0]["depth"])
print(f"depth[0]: dtype ok, valid={np.mean(d>0)*100:.1f}%, "
      f"range=[{d[d>0].min():.2f}, {d[d>0].max():.2f}] m, median={np.median(d[d>0]):.2f}")
