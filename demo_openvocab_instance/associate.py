#!/usr/bin/env python3
"""Cross-view association of per-view promptable-segmentation masks into 3D
instances, plus ScanNet++ ground-truth instance loading and matching.

Kept separate from pipeline.py so the association hyper-parameters can be swept
offline against GT without re-running the network (see sweep.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# association
# ---------------------------------------------------------------------------

DEFAULTS = dict(merge_iou=0.30, merge_containment=0.80, min_dets=2,
                min_instance_vox=200, cross_class_absorb=0.80,
                box_iou_gate=0.10)


def _box_iou(a, b):
    lo = np.maximum(a[0], b[0]); hi = np.minimum(a[1], b[1])
    d = np.clip(hi - lo, 0, None)
    inter = float(np.prod(d))
    if inter <= 0:
        return 0.0
    va = float(np.prod(a[1] - a[0])); vb = float(np.prod(b[1] - b[0]))
    return inter / max(va + vb - inter, 1e-9)


def _occupancy(dets, key="ccodes"):
    """Dense (n_dets, n_voxels) boolean occupancy over the union of coarse voxels.

    Pairwise intersection sizes are then a single matmul, which makes the whole
    association fast enough to sweep offline.
    """
    allc = np.unique(np.concatenate([d[key] for d in dets]))
    D = np.zeros((len(dets), len(allc)), bool)
    for i, d in enumerate(dets):
        D[i, np.searchsorted(allc, d[key])] = True
    return D, allc


def _inter_matrix(D):
    return (D.astype(np.float32) @ D.astype(np.float32).T).astype(np.int64)


def fuse(dets, *, decode, voxel, merge_iou, merge_containment, min_dets,
         min_instance_vox, cross_class_absorb, box_iou_gate, verbose=True):
    """dets: list of dicts with `cls`, `codes` (fine), `ccodes` (coarse), `score`.

    Two candidates are linked when they are the same class AND their coarse
    voxel sets agree either by IoU (symmetric, safe) or by containment (one is a
    partial view of the other). Containment alone chains greedily across
    neighbouring objects, so it is additionally gated on 3D box IoU, which is
    bounded by physical object extent. Repeated until no more merges.
    """
    if not dets:
        return []
    groups = [[i] for i in range(len(dets))]
    cls = np.array([d["cls"] for d in dets])
    D, allc = _occupancy(dets)
    cvox = voxel * 2
    cxyz = decode(allc).astype(np.float64) * cvox

    def group_boxes(G):
        out = []
        for g in G:
            occ = D[g].any(0)
            p = cxyz[occ]
            out.append((p.min(0), p.max(0) + cvox))
        return out

    n_pass = 0
    for n_pass in range(1, 41):
        M = np.stack([D[g].any(0) for g in groups])            # (K, Nc)
        sizes = M.sum(1)
        order = np.argsort(-sizes)
        I = _inter_matrix(M)
        gcls = np.array([cls[g[0]] for g in groups])
        boxes = group_boxes(groups)
        used = np.zeros(len(groups), bool)
        out = []
        for i in order:
            if used[i]:
                continue
            used[i] = True
            merged = list(groups[i])
            # candidates are always compared against the *seed* group, never
            # against the growing union; that keeps one pass from chaining
            # across the room, while the outer passes still allow real growth
            for j in order:
                if used[j] or gcls[j] != gcls[i]:
                    continue
                inter = I[i, j]
                if inter == 0:
                    continue
                link = inter / (sizes[i] + sizes[j] - inter) >= merge_iou
                if not link and inter / min(sizes[i], sizes[j]) >= merge_containment:
                    link = _box_iou(boxes[i], boxes[j]) >= box_iou_gate
                if link:
                    used[j] = True
                    merged += groups[j]
            out.append(merged)
        done = len(out) == len(groups)
        groups = out
        if done:
            break

    fused = []
    for g in groups:
        fused.append(dict(
            cls=int(cls[g[0]]),
            codes=np.unique(np.concatenate([dets[i]["codes"] for i in g])),
            ccodes=np.unique(np.concatenate([dets[i]["ccodes"] for i in g])),
            score=float(max(dets[i]["score"] for i in g)),
            n_det=len(g),
            views=[v for i in g for v in dets[i].get("views", [])]))

    # a fragment almost entirely inside a bigger object of another class is a
    # duplicate of it (e.g. a table blob that swallowed the chair beside it)
    fused.sort(key=lambda f: -f["ccodes"].size)
    Df, _ = _occupancy(fused)
    If = _inter_matrix(Df)
    sf = Df.sum(1)
    drop = set()
    for i in range(len(fused)):
        if i in drop:
            continue
        for j in range(i + 1, len(fused)):
            if j in drop or fused[j]["cls"] == fused[i]["cls"]:
                continue
            if If[i, j] / sf[j] >= cross_class_absorb:
                drop.add(j)
    fused = [f for i, f in enumerate(fused) if i not in drop]

    keep = [f for f in fused
            if f["codes"].size >= min_instance_vox and f["n_det"] >= min_dets]
    keep = iou_filter(keep, same_iou=0.40, cross_iou=0.65, verbose=verbose)
    keep.sort(key=lambda f: (-f["score"], -f["codes"].size))
    for i, f in enumerate(keep):
        f["id"] = i
    if verbose:
        print(f"[assoc] {len(dets)} detections -> {len(fused)} fused "
              f"({n_pass} passes, {len(drop)} cross-class dropped) -> {len(keep)} kept")
    return keep


def iou_filter(inst, same_iou=0.40, cross_iou=0.65, verbose=True):
    """Drop overlapping 3D instances after association.

    Same-class pairs at ``same_iou`` are treated as duplicates (keep the
    higher-score / larger one). Cross-class pairs at ``cross_iou`` are the
    typical sofa-eats-chair / table-eats-cabinet collisions.
    """
    if len(inst) < 2:
        return inst
    D, _ = _occupancy(inst, key="codes")
    I = _inter_matrix(D)
    sz = D.sum(1).astype(np.float64)
    score = np.array([float(f.get("score", 0)) + 1e-3 * sz[k] / max(sz.max(), 1)
                      for k, f in enumerate(inst)])
    order = np.argsort(-score)
    keep_idx, drop = [], set()
    for i in order:
        if i in drop:
            continue
        conflict = False
        for k in keep_idx:
            inter = I[i, k]
            if inter == 0:
                continue
            iou = inter / max(sz[i] + sz[k] - inter, 1)
            same = inst[i]["cls"] == inst[k]["cls"]
            if iou >= (same_iou if same else cross_iou):
                conflict = True
                break
        if conflict:
            drop.add(i)
        else:
            keep_idx.append(i)
    out = [inst[i] for i in keep_idx]
    if verbose:
        print(f"[iou-filter] {len(inst)} -> {len(out)}  "
              f"(same>={same_iou} cross>={cross_iou})")
    return out


# ---------------------------------------------------------------------------
# ScanNet++ GT instances
# ---------------------------------------------------------------------------


# ScanNet++ annotates fine-grained labels, so the strict 20-class protocol treats
# 'dining chair' as a different class from 'chair'. For an OPEN-VOCABULARY
# evaluation that is the wrong call: the prompt "chair" should match a dining
# chair. These are the groups accepted for each prompt; everything else is a
# negative. Listed explicitly so the protocol is auditable.
SYNONYMS = {
    "sofa":    ["sofa", "folding sofa", "sofa chair"],
    "table":   ["table", "coffee table", "dining table"],
    "cabinet": ["cabinet", "kitchen cabinet", "bedside cabinet"],
    "chair":   ["chair", "dining chair", "office chair", "ottoman chair"],
}


def label_to_target(targets, synonyms=None):
    """{ScanNet++ label -> prompt index} for the given prompt list."""
    syn = synonyms if synonyms is not None else SYNONYMS
    out = {}
    for i, t in enumerate(targets):
        for name in syn.get(t, [t]):
            out[name] = i
    return out


def build_raw_to_target(targets, synonyms=None):
    """(4096,) int64: official ScanNet++ raw label id -> prompt index, else -1."""
    import os
    from utils.scannetpp_labels import SCANNETPP_CLASS_LABELS  # noqa: F401

    here = os.path.dirname(os.path.abspath(
        __import__("utils.scannetpp_labels", fromlist=["x"]).__file__))
    names = open(os.path.join(here, "scannetpp_semantic_classes_official.txt")).read().splitlines()
    l2t = label_to_target(targets, synonyms)
    m = -np.ones(4096, dtype=np.int64)
    for raw, nm in enumerate(names):
        if nm in l2t:
            m[raw] = l2t[nm]
    return m


def load_gt_instances(scene_dir: Path, targets, R_align, encode, keys, synonyms=None):
    """GT instances for `targets`, as voxel code sets in the aligned frame.

    ScanNet++ stores a per-vertex segment index (segments.json) and a
    segment -> object grouping with a label (segments_anno.json). The segment
    indices are ordered like the vertices of scans/mesh_aligned_0.05.ply --
    NOT like `sampled_coords` in <scene>.pth, which is the same point set in a
    shuffled order (verified: reconstructing an object from the .pth ordering
    scatters it across the whole scene, while the mesh ordering reproduces the
    annotated `obb` exactly).

    Labels are matched exactly against the prompt strings, the same convention
    the 20-class semantic eval uses (so 'dining chair' is NOT a 'chair').
    """
    import trimesh

    sj, aj = scene_dir / "scans/segments.json", scene_dir / "scans/segments_anno.json"
    mesh = scene_dir / "scans/mesh_aligned_0.05.ply"
    if not (sj.is_file() and aj.is_file() and mesh.is_file()):
        return None
    seg = np.asarray(json.load(open(sj))["segIndices"], dtype=np.int64)
    verts = np.asarray(trimesh.load(str(mesh), process=False).vertices, dtype=np.float64)
    if len(verts) != len(seg):
        raise RuntimeError(f"mesh has {len(verts)} verts but segIndices has {len(seg)}")
    xyz = verts @ R_align

    tset = label_to_target(targets, synonyms)
    objs = []
    for g in json.load(open(aj))["segGroups"]:
        if g["label"] not in tset:
            continue
        m = np.isin(seg, np.asarray(g["segments"], dtype=np.int64))
        if m.sum() < 50:
            continue
        p = xyz[m]
        objs.append(dict(gt_id=len(objs), cls=tset[g["label"]], label=g["label"],
                         codes=np.unique(encode(keys(p))), n_pts=int(m.sum()),
                         bmin=p.min(0), bmax=p.max(0), centroid=p.mean(0)))
    if not objs:
        return None
    return dict(objs=objs, n_verts=len(verts))


def match_instances(pred, gt_objs, n_cls, iou_thresholds=(0.25, 0.5)):
    """Greedy score-ordered matching per class (standard 3D instance protocol).

    Returns per-threshold precision/recall/F1 plus mean best-IoU per GT object.
    """
    res = {}
    ious = np.zeros((len(pred), len(gt_objs)))
    for i, p in enumerate(pred):
        for j, g in enumerate(gt_objs):
            if p["cls"] != g["cls"]:
                continue
            inter = np.intersect1d(p["codes"], g["codes"], assume_unique=True).size
            if inter:
                ious[i, j] = inter / (p["codes"].size + g["codes"].size - inter)
    order = np.argsort([-p["score"] for p in pred]) if pred else np.array([], int)
    for th in iou_thresholds:
        taken = np.zeros(len(gt_objs), bool)
        tp = 0
        for i in order:
            cand = [j for j in range(len(gt_objs))
                    if not taken[j] and ious[i, j] >= th]
            if cand:
                j = max(cand, key=lambda j: ious[i, j])
                taken[j] = True
                tp += 1
        fp = len(pred) - tp
        fn = len(gt_objs) - tp
        prec = tp / max(tp + fp, 1e-9); rec = tp / max(tp + fn, 1e-9)
        res[f"iou{int(th*100)}"] = dict(
            tp=int(tp), fp=int(fp), fn=int(fn), precision=float(prec),
            recall=float(rec), f1=float(2 * prec * rec / max(prec + rec, 1e-9)))
    res["mean_best_iou"] = float(ious.max(0).mean()) if len(pred) and len(gt_objs) else 0.0
    res["n_pred"] = len(pred); res["n_gt"] = len(gt_objs)
    res["per_class"] = {}
    for c in range(n_cls):
        gi = [j for j, g in enumerate(gt_objs) if g["cls"] == c]
        pi = [i for i, p in enumerate(pred) if p["cls"] == c]
        res["per_class"][c] = dict(
            n_gt=len(gi), n_pred=len(pi),
            mean_best_iou=float(ious[np.ix_(pi, gi)].max(0).mean()) if pi and gi else 0.0)
    return res
