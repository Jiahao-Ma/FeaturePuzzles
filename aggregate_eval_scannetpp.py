#!/usr/bin/env python3
"""Aggregate per-scene SN++ _pred.npy files into a combined mIoU/mACC
(same logic as eval_scannetpp_3d.py multi-scene aggregation)."""
import sys
from pathlib import Path
import numpy as np
from eval_scannet_3d import _metrics
from utils.scannetpp_labels import SCANNETPP_EVAL

out = Path(sys.argv[1])
scenes = sys.argv[2:] if len(sys.argv) > 2 else ["09c1414f1b", "9071e139d9", "a24f64f7fb", "c49a8c6cff"]

all_preds = []; all_gts = []
per_scene = []
for sc in scenes:
    p = out / f"{sc}_pred.npy"
    if not p.exists():
        print(f"[{sc}] no _pred.npy, skip"); continue
    d = np.load(p)
    pred = d[:, 3].astype(np.int64)
    gt_cls = d[:, 4].astype(np.int64)
    valid = pred >= 0
    m = (gt_cls >= 0) & valid
    all_preds.append(pred[m]); all_gts.append(gt_cls[m])
    # per-scene
    s_miou, s_macc, s_nc = _metrics(pred[m], gt_cls[m], len(SCANNETPP_EVAL))
    per_scene.append((sc, int(m.sum()), s_miou * 100, s_macc * 100))
    print(f"[{sc}] pts={int(m.sum())} mIoU={s_miou*100:.2f} mACC={s_macc*100:.2f}")

print("\n=== per-scene ===")
for sc, n, mi, ma in per_scene:
    print(f"  {sc}  pts={n}  mIoU={mi:.2f}  mACC={ma:.2f}")

if all_preds:
    preds = np.concatenate(all_preds); gts = np.concatenate(all_gts)
    miou, macc, nc = _metrics(preds, gts, len(SCANNETPP_EVAL))
    print(f"\n================ SN++ aggregated ({len(scenes)} scenes) ================")
    print(f"eval_points={len(gts)} classes_evaluated={nc}")
    print(f"mIoU = {miou*100:.2f}   mACC = {macc*100:.2f}")
    (out / "result_scannetpp_aggregated.txt").write_text(
        f"mIoU={miou*100:.2f} mACC={macc*100:.2f} classes={nc} points={len(gts)} scenes={len(scenes)}\n")
