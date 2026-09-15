#!/usr/bin/env python3
"""Search the 24 axis-aligned proper rotations for the ScanNet++ GT -> nerfstudio
alignment, scoring each by voxel coverage of our reconstruction."""
import itertools, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import load_scene_frames, load_depth, unproject, VoxelGrid

DATA = Path("/mnt/dataset/marko/puzzles/embodiedsplat_dl/dataset/scannetpp/data")


def proper_rotations():
    out = []
    for perm in itertools.permutations(range(3)):
        for sx, sy, sz in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, (sx, sy, sz))):
                R[p, i] = s                      # (x,y,z) @ R  picks axis perm[i]*s
            if abs(np.linalg.det(R) - 1) < 1e-6:
                out.append(R)
    return out


def ours_voxels(scene, stride=16, maxf=60, voxel=0.05):
    d = DATA / scene
    frames, (fx, fy, cx, cy, W, H) = load_scene_frames(d, stride, maxf)
    g = VoxelGrid(voxel)
    s = 0.25
    Wh, Ww = int(round(H * s)), int(round(W * s))
    K = (fx * s, fy * s, cx * s, cy * s)
    codes = []
    for fr in frames:
        dm = np.array(Image.fromarray(load_depth(fr["depth"])).resize((Ww, Wh), Image.NEAREST))
        pts, _, _ = unproject(dm, fr["c2w"], K, gl=True)
        codes.append(np.unique(g.encode(g.keys(pts))))
    return np.unique(np.concatenate(codes)), g, len(frames)


for scene in sys.argv[1:] or ["9071e139d9"]:
    uc, g, nf = ours_voxels(scene)
    p = torch.load(DATA / scene / f"{scene}.pth", map_location="cpu", weights_only=False)
    raw = np.asarray(p["sampled_coords"], np.float64)
    sub = raw[np.random.RandomState(0).choice(len(raw), min(300_000, len(raw)), replace=False)]
    print(f"\n=== {scene}  ({nf} frames, {len(uc)} ours voxels) ===")
    scores = []
    for R in proper_rotations():
        cov = np.isin(g.encode(g.keys(sub @ R)), uc).mean()
        scores.append((cov, R))
    scores.sort(key=lambda t: -t[0])
    for cov, R in scores[:4]:
        print(f"  coverage={cov*100:5.1f}%   R rows={R.astype(int).tolist()}")
    cur = np.diag([1.0, 1.0, -1.0])
    cov_cur = np.isin(g.encode(g.keys(sub @ cur)), uc).mean()
    print(f"  [current repo R_Z_90 = diag(1,1,-1)] coverage={cov_cur*100:.1f}%")
