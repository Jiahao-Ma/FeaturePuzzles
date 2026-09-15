#!/usr/bin/env python3
"""Compare predicted instance boxes against ScanNet++ GT instance boxes."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render import load_bundle

b = load_bundle(sys.argv[1] if len(sys.argv) > 1 else "/tmp/demo_test/9071e139d9_bundle.npz")
gi = b["gt_instances"]
voc = b["vocab"]
centers = b["vox_centers"]
print(f"scene {b['scene']}  voxels={len(centers)}  "
      f"cloud min={centers.min(0).round(2)} max={centers.max(0).round(2)}")

print("\n--- raw per-view detections (before association) ---")
raw = b["raw_dets"]
ext = np.array([(lambda p: (p.max(0) - p.min(0)))(
    centers[np.searchsorted(np.sort(np.unique(np.concatenate([d['codes'] for d in raw[:1]]))), [0])] )
    for d in raw[:0]]) if False else None
from pipeline import VoxelGrid
g = VoxelGrid(float(b["voxel"]))
exts, nv = [], []
for d in raw:
    p = g.centers(g.decode(d["codes"]))
    exts.append(p.max(0) - p.min(0)); nv.append(len(d["codes"]))
exts = np.array(exts); nv = np.array(nv)
diag = np.linalg.norm(exts, axis=1)
print(f"  n={len(raw)}  voxels/det: med={np.median(nv):.0f} p90={np.percentile(nv,90):.0f} max={nv.max()}")
print(f"  det box diagonal (m): med={np.median(diag):.2f} p75={np.percentile(diag,75):.2f} "
      f"p90={np.percentile(diag,90):.2f} max={diag.max():.2f}")
for c in range(int(b["n_targets"])):
    m = np.array([d["cls"] == c for d in raw])
    if m.any():
        print(f"    {voc[c]:8s} n={m.sum():4d} med_diag={np.median(diag[m]):.2f} "
              f"p90_diag={np.percentile(diag[m],90):.2f} med_vox={np.median(nv[m]):.0f}")

print("\n--- predicted instances ---")
for f in b["instances"]:
    print(f"  #{f['id']:2d} {voc[f['cls']]:8s} nvox={len(f['vox_idx']):6d} "
          f"dets={f['n_det']:3d} box={f['bmin'].round(2)}..{f['bmax'].round(2)} "
          f"ext={(f['bmax']-f['bmin']).round(2)}")

print("\n--- GT instances ---")
for o in gi["objs"]:
    print(f"  gt{o['gt_id']:2d} {o['label']:8s} npts={o['n_pts']:6d} ncodes={len(o['codes']):6d} "
          f"box={o['bmin'].round(2)}..{o['bmax'].round(2)} ext={(o['bmax']-o['bmin']).round(2)}")

print("\n--- pairwise IoU / containment (same class only) ---")
for f in b["instances"]:
    row = []
    for o in gi["objs"]:
        if o["cls"] != f["cls"]:
            continue
        inter = np.intersect1d(f["codes"], o["codes"], assume_unique=True).size
        iou = inter / (f["codes"].size + o["codes"].size - inter) if inter else 0.0
        cov = inter / f["codes"].size
        row.append(f"gt{o['gt_id']}:IoU={iou:.2f},cov={cov:.2f}")
    print(f"  #{f['id']:2d} {voc[f['cls']]:8s} " + "  ".join(row))

# do the two clouds even occupy the same voxels?
allgt = np.unique(np.concatenate([o["codes"] for o in gi["objs"]]))
ours = set(b["instances"][0]["codes"].tolist()) if b["instances"] else set()
print(f"\nGT target-class voxels: {len(allgt)}")
