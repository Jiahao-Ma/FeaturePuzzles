#!/usr/bin/env python3
"""Sweep 3D association hyper-parameters against ScanNet++ GT instances.

Reads the raw per-view detections cached in a bundle, so no GPU / no re-running
the network. Prints a ranked table and the best setting.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import associate                                             # noqa: E402
from pipeline import VoxelGrid, TARGET_PROMPTS, R_GT_TO_NERF  # noqa: E402
from render import load_bundle                               # noqa: E402

DATA = Path("/mnt/dataset/marko/puzzles/embodiedsplat_dl/dataset/scannetpp/data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", nargs="+", required=True)
    ap.add_argument("--data_root", default=str(DATA))
    ap.add_argument("--metric", default="sum",
                    choices=["f1_25", "f1_50", "mean_best_iou", "sum"])
    ap.add_argument("--stage", default="link", choices=["link", "filter", "both"])
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()

    scenes = []
    for bp in a.bundle:
        b = load_bundle(bp)
        # GT is re-derived here (not taken from the bundle) so the matching
        # protocol can be changed without re-running the network
        g = VoxelGrid(float(b["voxel"]))
        gi = associate.load_gt_instances(Path(a.data_root) / b["scene"], TARGET_PROMPTS,
                                         R_GT_TO_NERF, g.encode, g.keys)
        if gi is None:
            print(f"skip {bp}: no GT instances")
            continue
        scenes.append((b["scene"], b["raw_dets"], gi["objs"], float(b["voxel"]),
                       int(b["n_targets"])))
        cnt = {t: sum(1 for o in gi["objs"] if o["cls"] == i)
               for i, t in enumerate(TARGET_PROMPTS)}
        print(f"{b['scene']}: {len(b['raw_dets'])} raw detections, "
              f"{len(gi['objs'])} GT instances {cnt}")

    if a.stage == "link":
        gridspec = dict(
            merge_iou=[0.12, 0.20, 0.30, 0.42, 0.55],
            merge_containment=[0.60, 0.75, 0.90],
            box_iou_gate=[0.05, 0.15, 0.30],
            min_dets=[2], min_instance_vox=[200], cross_class_absorb=[0.80])
    elif a.stage == "filter":
        gridspec = dict(
            merge_iou=[0.42], merge_containment=[0.75], box_iou_gate=[0.05],
            min_dets=[8, 11, 14, 18, 24], min_instance_vox=[150, 200, 300],
            cross_class_absorb=[0.60, 0.80, 0.95])
    else:
        gridspec = dict(
            merge_iou=[0.20, 0.30], merge_containment=[0.75, 0.90],
            box_iou_gate=[0.15], min_dets=[2, 4, 8],
            min_instance_vox=[200, 500], cross_class_absorb=[0.80])
    keys = list(gridspec)
    rows = []
    combos = list(itertools.product(*[gridspec[k] for k in keys]))
    print(f"\nevaluating {len(combos)} settings on {len(scenes)} scene(s) ...")
    for ci, combo in enumerate(combos):
        p = dict(zip(keys, combo))
        agg = dict(f1_25=[], f1_50=[], mean_best_iou=[], n_pred=0, n_gt=0)
        for name, dets, objs, voxel, n_tgt in scenes:
            keep = associate.fuse(dets, decode=VoxelGrid.decode, voxel=voxel,
                                  verbose=False, **p)
            m = associate.match_instances(keep, objs, n_tgt)
            agg["f1_25"].append(m["iou25"]["f1"])
            agg["f1_50"].append(m["iou50"]["f1"])
            agg["mean_best_iou"].append(m["mean_best_iou"])
            agg["n_pred"] += m["n_pred"]; agg["n_gt"] += m["n_gt"]
        r = {k: float(np.mean(agg[k])) for k in ("f1_25", "f1_50", "mean_best_iou")}
        r["sum"] = r["f1_25"] + r["f1_50"] + r["mean_best_iou"]
        r.update(p, n_pred=agg["n_pred"], n_gt=agg["n_gt"])
        rows.append(r)
        if (ci + 1) % 50 == 0:
            print(f"  {ci+1}/{len(combos)}", flush=True)

    rows.sort(key=lambda r: -r[a.metric])
    hdr = f"{'F1@25':>7} {'F1@50':>7} {'mBIoU':>7} {'pred':>5} {'gt':>4}  " + \
          "  ".join(f"{k:>17}" for k in keys)
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows[:a.top]:
        print(f"{r['f1_25']*100:7.1f} {r['f1_50']*100:7.1f} {r['mean_best_iou']:7.3f} "
              f"{r['n_pred']:5d} {r['n_gt']:4d}  " +
              "  ".join(f"{r[k]:>17}" for k in keys))
    best = rows[0]
    print("\nbest:", " ".join(f"--{k.replace('_','_')} {best[k]}" for k in keys))


if __name__ == "__main__":
    main()
