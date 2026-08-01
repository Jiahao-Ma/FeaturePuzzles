"""Aggregate per-scene _pred.npy files into global mIoU / mACC.

Each _pred.npy is (N,5): x,y,z,pred_cls,gt_cls  (gt_cls/pred_cls -1 = void).
Usage:  python aggregate_eval_scannet3d.py --dir <eval_out> --dataset scannet20
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from utils.scannet_labels import SCANNET20  # noqa: E402
from utils.scannet200_labels import SCANNET200_EVAL  # noqa: E402

# mirror eval_scannet_3d.py class sets exactly (EmbodiedSplat src/evaluation/database)
SCANNET15 = ["wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
             "window", "bookshelf", "counter", "desk", "curtain", "toilet", "sink"]
SCANNET10 = ["wall", "floor", "bed", "chair", "sofa", "table", "door",
             "window", "bookshelf", "toilet"]
SETS = {"scannet20": SCANNET20, "scannet15": SCANNET15, "scannet10": SCANNET10,
        "scannet200": SCANNET200_EVAL}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="eval output dir with *_pred.npy")
    p.add_argument("--dataset", default="scannet20", choices=list(SETS))
    args = p.parse_args()

    cls_labels = SETS[args.dataset]
    K = len(cls_labels)
    d = Path(args.dir)
    files = sorted(d.glob("*_pred.npy"))
    if not files:
        print(f"no _pred.npy in {d}")
        return
    print(f"aggregating {len(files)} scenes from {d}")
    all_preds, all_gts = [], []
    per_scene = []
    for f in files:
        arr = np.load(f)                       # (N,5)
        pred = arr[:, 3].astype(np.int64)
        gt = arr[:, 4].astype(np.int64)
        # match eval_scannet_3d.py eval_mask: GT-valid AND covered (pred>=0).
        # uncovered points have pred=-1 and must be excluded.
        m = (gt >= 0) & (pred >= 0)
        per_scene.append((f.stem.replace("_pred", ""), m.sum(),
                          (pred[m] == gt[m]).mean() if m.sum() else 0.0))
        all_preds.append(pred[m])
        all_gts.append(gt[m])
    preds = np.concatenate(all_preds)
    gts = np.concatenate(all_gts)

    ious, accs = [], []
    for c in range(K):
        m = gts == c
        if m.sum() == 0:
            continue
        p = (preds == c) & m
        u = (preds == c) | m
        ious.append(p.sum() / max(u.sum(), 1))
        accs.append((preds[m] == c).mean())
    miou = float(np.mean(ious)) if ious else 0.0
    macc = float(np.mean(accs)) if accs else 0.0

    print("\n=== per-scene accuracy ===")
    for name, n, a in per_scene:
        print(f"  {name:14s} pts={n:7d} acc={a*100:5.2f}")
    print(f"\n================ {args.dataset} 3D open-vocab eval (aggregated) ================")
    print(f"scenes={len(files)} eval_points={len(gts)} classes_evaluated={len(ious)}")
    print(f"mIoU = {miou*100:.2f}   mACC = {macc*100:.2f}")
    (d / f"result_{args.dataset}.txt").write_text(
        f"{args.dataset}: mIoU={miou*100:.2f} mACC={macc*100:.2f}\n")


if __name__ == "__main__":
    main()
