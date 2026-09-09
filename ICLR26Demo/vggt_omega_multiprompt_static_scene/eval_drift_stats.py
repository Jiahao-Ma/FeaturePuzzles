#!/usr/bin/env python3
"""Unsupervised drift metrics for the multiprompt tracking results.

No GT is used. The metrics target the two failure modes seen in this pipeline:

  * object collapse -- two prompts end up covering the same physical surface.
    Measured as containment  inter(a,b) / min(area_a, area_b)  over all views.
    A pair above ``--tau`` is counted as collapsed. For a healthy tracker the
    K prompts stay disjoint, so the count should be ~0.

  * area inflation -- a mask spreads far beyond the object it was prompted on.
    Measured per object as max-over-views area / ref-view area, plus the number
    of views whose area exceeds 3x the ref area.

Also reports coverage (fraction of views where the object is present at all),
which guards against "fixing" drift by simply predicting nothing.

Usage:
    python eval_drift_stats.py --root eval_iterative_excl
    python eval_drift_stats.py --root eval_iterative_excl --csv out.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def scene_stats(npz_path: Path, tau: float = 0.3, inflate: float = 3.0) -> dict:
    masks = np.load(npz_path)["pred_masks"]          # (S, K, H, W) uint8
    S, K, H, W = masks.shape
    area = masks.reshape(S, K, -1).sum(-1).astype(np.float64)   # (S, K)
    ref = area[0]
    total = area.sum(0)

    ratios = []
    n_collapsed = 0
    worst = (None, 0.0)
    for a in range(K):
        for b in range(a + 1, K):
            denom = min(total[a], total[b])
            if denom <= 0:
                continue
            r = float(np.logical_and(masks[:, a], masks[:, b]).sum()) / denom
            ratios.append(r)
            if r > tau:
                n_collapsed += 1
            if r > worst[1]:
                worst = ((a, b), r)

    with np.errstate(divide="ignore", invalid="ignore"):
        infl = np.where(ref > 0, area.max(0) / np.maximum(ref, 1), 0.0)
        n_inflated = int(sum((area[:, k] > inflate * ref[k]).sum()
                             for k in range(K) if ref[k] > 0))
    present = (area > 0).mean(0)                      # per object, frac of views

    return {
        "scene": npz_path.parent.name,
        "views": S,
        "objects": K,
        "collapsed_pairs": n_collapsed,
        "total_pairs": len(ratios),
        "mean_containment": float(np.mean(ratios)) if ratios else 0.0,
        "max_containment": float(worst[1]),
        "worst_pair": f"{worst[0][0]}&{worst[0][1]}" if worst[0] else "-",
        "mean_inflation": float(np.mean(infl)),
        "max_inflation": float(np.max(infl)),
        "inflated_view_obj": n_inflated,
        "mean_coverage": float(np.mean(present)),
        "total_px": int(masks.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="dir containing <scene>/<scene>_pred_masks.npz")
    ap.add_argument("--tau", type=float, default=0.3,
                    help="containment above this counts as a collapsed pair")
    ap.add_argument("--csv", default=None, help="optional CSV output path")
    args = ap.parse_args()

    root = Path(args.root)
    npzs = sorted(root.glob("*/*_pred_masks.npz"))
    if not npzs:
        raise SystemExit(f"no *_pred_masks.npz under {root}")

    rows = [scene_stats(p, tau=args.tau) for p in npzs]

    hdr = (f"{'scene':<16}{'views':>6}{'obj':>5}{'collapsed':>11}"
           f"{'meanCont':>10}{'maxCont':>9}{'meanInfl':>10}{'maxInfl':>9}"
           f"{'inflVO':>8}{'cover':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['scene']:<16}{r['views']:>6}{r['objects']:>5}"
              f"{r['collapsed_pairs']:>5}/{r['total_pairs']:<5}"
              f"{r['mean_containment']:>10.3f}{r['max_containment']:>9.2f}"
              f"{r['mean_inflation']:>10.2f}{r['max_inflation']:>9.1f}"
              f"{r['inflated_view_obj']:>8}{r['mean_coverage']:>7.2f}")

    print("-" * len(hdr))
    n = len(rows)
    tot_col = sum(r["collapsed_pairs"] for r in rows)
    tot_pair = sum(r["total_pairs"] for r in rows)
    print(f"{'MEAN (' + str(n) + ' scenes)':<16}{'':>6}{'':>5}"
          f"{tot_col:>5}/{tot_pair:<5}"
          f"{np.mean([r['mean_containment'] for r in rows]):>10.3f}"
          f"{np.mean([r['max_containment'] for r in rows]):>9.2f}"
          f"{np.mean([r['mean_inflation'] for r in rows]):>10.2f}"
          f"{np.mean([r['max_inflation'] for r in rows]):>9.1f}"
          f"{int(np.mean([r['inflated_view_obj'] for r in rows])):>8}"
          f"{np.mean([r['mean_coverage'] for r in rows]):>7.2f}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[csv] {args.csv}")


if __name__ == "__main__":
    main()
