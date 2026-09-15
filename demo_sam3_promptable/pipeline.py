#!/usr/bin/env python3
"""SAM3-style open-vocab instances + promptable multi-view linking + 3D IoU filter.

Open-vocabulary masks come from SAM 3 when the gated checkpoint is present,
otherwise Grounding DINO + SAM2 (same text -> all-instances contract).
Those instances become mask+point prompts for the jointly trained promptable
head, which propagates each object across the 6-view batch. Lifted 3D
detections are associated and then NMS'd by voxel IoU so overlapping objects
are dropped.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "demo_openvocab_instance"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo_openvocab_instance.pipeline import (  # noqa: E402
    VoxelGrid, attach_voxels, build_model, _sample, _seg_sum, largest_blob_3d)
import associate  # noqa: E402
from loaders import index_scene, pick_nearby_window, read_frame, unproject  # noqa: E402
from ov import OVSegmenter  # noqa: E402

TARGET_PROMPTS = ["sofa", "table", "cabinet", "chair"]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(ROOT / "weights/joint_feat_seg_lingbot/checkpoint-best.pth"))
    p.add_argument("--dataset", required=True, choices=["scannet", "scannetpp", "farm"])
    p.add_argument("--scene", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--camera", default=None)
    p.add_argument("--n_views", type=int, default=6)
    p.add_argument("--frame_stride", type=int, default=4)
    p.add_argument("--max_frames", type=int, default=48)
    p.add_argument("--start_idx", type=int, default=-1,
                   help="first indexed frame; -1 = auto-pick a nearby n_views window")
    p.add_argument("--auto_nearby", action="store_true",
                   help="ignore max_frames; pick n_views consecutive nearby cameras")
    p.add_argument("--min_travel", type=float, default=0.08,
                   help="min camera-path length (m) when auto-picking a window")
    p.add_argument("--max_angle", type=float, default=30.0,
                   help="prefer consecutive views whose camera forward differs by <= this (deg)")
    p.add_argument("--max_step", type=float, default=0.0,
                   help="cap consecutive camera translation (m); 0 = no cap")
    p.add_argument("--match_iou", type=float, default=0.30,
                   help="replace promptable mask with Grounded-SAM if IoU >= this")
    p.add_argument("--match_contain", type=float, default=0.50,
                   help="or if this fraction of the promptable mask sits inside a GSAM instance")
    p.add_argument("--depth_prop", type=int, default=1,
                   help="also warp masks across views through posed depth")
    p.add_argument("--depth_tol", type=float, default=0.10,
                   help="range disagreement (m) tolerated by the depth warp")
    p.add_argument("--agree_iou", type=float, default=0.10,
                   help="promptable vs depth warp IoU counted as agreement")
    p.add_argument("--agree_contain", type=float, default=0.30,
                   help="or containment counted as agreement")
    p.add_argument("--sam_refine", type=int, default=1,
                   help="re-run SAM2 on each propagated mask")
    p.add_argument("--voxel", type=float, default=0.05)
    p.add_argument("--work_scale", type=float, default=0.35)
    p.add_argument("--ov_long", type=int, default=1024, help="OV image long side")
    p.add_argument("--max_prompts", type=int, default=16)
    p.add_argument("--mask_thresh", type=float, default=0.5)
    p.add_argument("--nms_iou", type=float, default=0.55)
    p.add_argument("--max_mask_frac", type=float, default=0.45)
    p.add_argument("--min_view_area", type=float, default=0.002)
    p.add_argument("--min_iou_pred", type=float, default=0.35)
    p.add_argument("--max_det_diag", type=float, default=4.5)
    p.add_argument("--assoc_voxel_mult", type=float, default=2.0)
    p.add_argument("--merge_iou", type=float, default=0.30)
    p.add_argument("--merge_containment", type=float, default=0.80)
    p.add_argument("--cross_class_absorb", type=float, default=0.80)
    p.add_argument("--box_iou_gate", type=float, default=0.10)
    p.add_argument("--min_dets", type=int, default=2)
    p.add_argument("--min_instance_vox", type=int, default=80)
    p.add_argument("--same_iou", type=float, default=0.40)
    p.add_argument("--cross_iou", type=float, default=0.65)
    p.add_argument("--box_thr", type=float, default=0.22)
    p.add_argument("--text_thr", type=float, default=0.18)
    # wall / floor / ceiling / door / window are detected only to veto targets
    p.add_argument("--neg_prompt", type=int, default=1)
    p.add_argument("--neg_drop", type=float, default=0.55)
    p.add_argument("--neg_strong", type=float, default=0.40)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _resize_rgb(rgb, long):
    h, w = rgb.shape[:2]
    if max(h, w) <= long:
        return rgb, 1.0
    s = long / max(h, w)
    nw, nh = int(round(w * s)), int(round(h * s))
    return np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR)), s


def _up_mask(m, H, W):
    return np.array(Image.fromarray(m.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127


def _centroid(m):
    ys, xs = np.nonzero(m)
    return float(xs.mean()), float(ys.mean())


def _mask_box(m):
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return None
    return np.array([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])


def _box_iou(a, b):
    lo = np.maximum(a[:2], b[:2]); hi = np.minimum(a[2:], b[2:])
    d = np.clip(hi - lo, 0, None)
    inter = float(d[0] * d[1])
    if inter <= 0:
        return 0.0
    ua = float((a[2] - a[0]) * (a[3] - a[1]))
    ub = float((b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(ua + ub - inter, 1e-6)


def cluster_gsam(gsam, iou_thr=0.12, contain_thr=0.35, box_iou_thr=0.28):
    """Union same-class Grounded-SAM fragments that sit on one object."""
    n = len(gsam)
    if n == 0:
        return []
    masks = [g["mask"].astype(bool) for g in gsam]
    area = np.array([int(m.sum()) for m in masks], np.int32)
    boxes = [_mask_box(m) for m in masks]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if int(gsam[i]["cls"]) != int(gsam[j]["cls"]):
                continue
            inter = int(np.logical_and(masks[i], masks[j]).sum())
            iou = inter / max(int(area[i]) + int(area[j]) - inter, 1) if inter else 0.0
            cont = inter / max(min(int(area[i]), int(area[j])), 1) if inter else 0.0
            biou = (_box_iou(boxes[i], boxes[j])
                    if boxes[i] is not None and boxes[j] is not None else 0.0)
            if iou >= iou_thr or cont >= contain_thr or biou >= box_iou_thr:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = []
    for ids in groups.values():
        mk = np.zeros_like(masks[0], dtype=bool)
        for i in ids:
            mk |= masks[i]
        out.append(dict(
            cls=int(gsam[ids[0]]["cls"]), mask=mk,
            score=float(max(gsam[i]["score"] for i in ids)),
            n_parts=len(ids)))
    return out


def build_view_geometry(batch, R, gl):
    """Per-view 256² point map + depth in camera z, matching the mask raster."""
    import cv2
    geo = []
    for fr in batch:
        H, W = fr["H"], fr["W"]
        d = cv2.resize(fr["depth"].astype(np.float32), (R, R),
                       interpolation=cv2.INTER_NEAREST)
        fx, fy, cx, cy = fr["K"]
        Kr = (fx * R / W, fy * R / H, cx * R / W, cy * R / H)
        pts, ys, xs = unproject(d, fr["c2w"], Kr, gl=fr["gl"])
        xyz = np.full((R, R, 3), np.nan, np.float32)
        xyz[ys, xs] = pts
        geo.append(dict(xyz=xyz, depth=d, K=Kr, c2w=fr["c2w"],
                        w2c=np.linalg.inv(fr["c2w"]), gl=fr["gl"]))
    return geo


def project_mask(mask, g_src, g_dst, R, depth_tol=0.10, dilate=2):
    """Warp a mask from one view into another through the metric point cloud.

    Points whose reprojected range disagrees with the destination depth are
    occluded and dropped, so the warp cannot paint through walls.
    """
    import cv2
    P = g_src["xyz"][mask]
    P = P[np.isfinite(P).all(1)]
    if len(P) < 10:
        return np.zeros((R, R), bool)
    w2c, (fx, fy, cx, cy) = g_dst["w2c"], g_dst["K"]
    cam = P @ w2c[:3, :3].T + w2c[:3, 3]
    if g_dst["gl"]:
        z = -cam[:, 2]
        v = -cam[:, 1]
    else:
        z = cam[:, 2]
        v = cam[:, 1]
    ok = z > 1e-3
    if not ok.any():
        return np.zeros((R, R), bool)
    z = z[ok]
    u = (cam[ok, 0] / z * fx + cx)
    v = (v[ok] / z * fy + cy)
    ui, vi = np.round(u).astype(np.int32), np.round(v).astype(np.int32)
    inb = (ui >= 0) & (ui < R) & (vi >= 0) & (vi < R)
    if not inb.any():
        return np.zeros((R, R), bool)
    ui, vi, z = ui[inb], vi[inb], z[inb]
    dz = g_dst["depth"][vi, ui]
    tol = np.maximum(depth_tol, 0.05 * z)
    vis = (dz > 1e-3) & (np.abs(dz - z) <= tol)
    out = np.zeros((R, R), bool)
    out[vi[vis], ui[vis]] = True
    if dilate > 0 and out.any():
        k = np.ones((2 * dilate + 1,) * 2, np.uint8)
        out = cv2.dilate(out.astype(np.uint8), k, 1).astype(bool)
        out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, k, iterations=2).astype(bool)
    return out


def depth_propagate(cand, geo, N, M, R, depth_tol=0.10):
    """Geometric counterpart of the promptable propagation."""
    out = np.zeros((N, M, R, R), bool)
    for m, c in enumerate(cand):
        v_s = int(c["view"])
        src = c["mask"].astype(bool)
        out[v_s, m] = src
        for v_t in range(N):
            if v_t == v_s:
                continue
            out[v_t, m] = project_mask(src, geo[v_s], geo[v_t], R,
                                       depth_tol=depth_tol)
    return out


def combine_prop_depth(prop, dep, cand, agree_iou=0.10, agree_contain=0.30):
    """Cross-check the learned and geometric warps before refinement.

    Agreement -> union (each fills the other's holes). Disagreement -> trust the
    depth warp, which cannot hallucinate a new object. Depth-only -> keep it.
    """
    N, M = prop.shape[:2]
    seed = np.zeros_like(prop)
    stats = dict(both=0, prop_only=0, depth_only=0, disagree=0, empty=0)
    for v in range(N):
        for m in range(M):
            if int(cand[m]["view"]) == v:
                seed[v, m] = cand[m]["mask"].astype(bool)
                continue
            P, D = prop[v, m].astype(bool), dep[v, m].astype(bool)
            ps, ds = int(P.sum()), int(D.sum())
            if ps < 15 and ds < 15:
                stats["empty"] += 1
                continue
            if ps < 15:
                seed[v, m] = D
                stats["depth_only"] += 1
                continue
            if ds < 15:
                seed[v, m] = P
                stats["prop_only"] += 1
                continue
            inter = int(np.logical_and(P, D).sum())
            iou = inter / max(ps + ds - inter, 1)
            cont = inter / max(min(ps, ds), 1)
            if iou >= agree_iou or cont >= agree_contain:
                seed[v, m] = np.logical_or(P, D)
                stats["both"] += 1
            else:
                seed[v, m] = D
                stats["disagree"] += 1
    return seed, stats


def sam_refine_masks(seed, iou_pred, cand, ov, rgb_small, min_area=20, neg=None):
    """Send every propagated seed back through SAM2 to complete its silhouette."""
    N, M = seed.shape[:2]
    out = seed.copy()
    n_ref = 0
    for v in range(N):
        ids = [m for m in range(M)
               if int(cand[m]["view"]) != v and int(seed[v, m].sum()) >= min_area]
        if not ids:
            continue
        res = ov.refine(np.ascontiguousarray(rgb_small[v]),
                        [seed[v, m] for m in ids], min_area=min_area,
                        neg=None if neg is None else neg[v])
        for m, r in zip(ids, res):
            if r is None:
                continue
            mk, sc = r
            out[v, m] = mk
            iou_pred[v, m] = max(float(iou_pred[v, m]), 0.5 * (float(sc) + float(iou_pred[v, m])))
            n_ref += 1
    return out, n_ref


def veto_neg_masks(masks, cand, neg, drop_frac=0.55, min_area=20):
    """Final structure veto: nothing that lives on wall / floor / ceiling survives."""
    if neg is None or all(n is None for n in neg):
        return masks, 0, 0
    N, M = masks.shape[:2]
    out = masks.copy()
    n_drop = n_cut = 0
    for v in range(N):
        nv = neg[v]
        if nv is None or not nv.any():
            continue
        for m in range(M):
            mk = out[v, m].astype(bool)
            a = int(mk.sum())
            if a < min_area:
                continue
            hit = int(np.logical_and(mk, nv).sum())
            if hit == 0:
                continue
            if hit / a >= drop_frac:
                out[v, m] = False
                n_drop += 1
            else:
                out[v, m] = np.logical_and(mk, ~nv)
                n_cut += 1
    return out, n_drop, n_cut


def match_gsam_masks(prop, iou_pred, cand, ov_per_view, match_iou=0.30,
                     match_contain=0.50):
    """Promptable ID × merged Grounded-SAM silhouette.

    1. Cluster same-class GSAM fragments that overlap into one object mask.
    2. Each promptable prediction (and each source seed) is replaced by the
       best overlapping cluster — many prompts may share one complete mask.
    Unmatched neighbour views keep the raw promptable output.
    """
    N, M = prop.shape[:2]
    out = prop.copy()
    n_hit = n_merge = 0
    for v in range(N):
        clusters = cluster_gsam(ov_per_view[v])
        n_merge += sum(max(c["n_parts"] - 1, 0) for c in clusters)
        if not clusters:
            continue
        for m in range(M):
            if int(cand[m]["view"]) == v:
                P = cand[m]["mask"].astype(bool)
            else:
                P = prop[v, m].astype(bool)
            psum = int(P.sum())
            if psum < 15:
                continue
            cls = int(cand[m]["cls"])
            pb = _mask_box(P)
            best = None
            for i, G in enumerate(clusters):
                if G["cls"] != cls:
                    continue
                gm = G["mask"]
                inter = int(np.logical_and(P, gm).sum())
                gsum = int(gm.sum())
                iou = inter / max(psum + gsum - inter, 1) if inter else 0.0
                contain = inter / max(psum, 1) if inter else 0.0
                biou = (_box_iou(pb, _mask_box(gm)) if pb is not None else 0.0)
                if iou >= match_iou or contain >= match_contain or biou >= 0.35:
                    sc = iou + 0.35 * contain + 0.15 * biou
                    if best is None or sc > best[0]:
                        best = (sc, i)
            if best is None:
                continue
            G = clusters[best[1]]
            out[v, m] = G["mask"]
            iou_pred[v, m] = max(float(iou_pred[v, m]), G["score"])
            n_hit += 1
    return out, n_hit, n_merge


@torch.no_grad()
def main():
    args = get_args()
    device = torch.device("cuda")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.auto_nearby:
        metas = index_scene(args.dataset, args.scene, stride=max(args.frame_stride, 1),
                            max_frames=0, camera=args.camera)
        if args.start_idx >= 0:
            i0 = args.start_idx
            info = dict(reason="start_idx", start_stem=metas[i0]["stem"])
        else:
            i0, info = pick_nearby_window(metas, n=args.n_views,
                                         min_travel=args.min_travel,
                                         max_angle_deg=args.max_angle,
                                         max_step=(args.max_step or None))
        metas = metas[i0:i0 + args.n_views]
        print(f"[data] nearby window i0={i0} n={len(metas)} {info}")
        frames = [read_frame(m) for m in metas]
    else:
        from loaders import load_scene
        frames = load_scene(args.dataset, args.scene, stride=args.frame_stride,
                            max_frames=args.max_frames, camera=args.camera)
    print(f"[data] {args.dataset}/{args.scene}: {len(frames)} frames "
          f"stems={[f['stem'] for f in frames]}")
    if not frames:
        raise SystemExit("no frames")

    backbone, feat_head, seg_head, teacher, meta = build_model(args.ckpt, device, load_text=False)
    A = meta["args"]
    R = A["resolution"]
    print(f"[model] epoch={meta['epoch']} val_iou={meta['val_iou']:.3f} res={R}")
    ov = OVSegmenter(TARGET_PROMPTS, device=str(device),
                     box_thr=args.box_thr, text_thr=args.text_thr,
                     use_neg=bool(args.neg_prompt), neg_drop=args.neg_drop,
                     neg_strong=args.neg_strong)

    grid = VoxelGrid(args.voxel)
    coarse = VoxelGrid(args.voxel * args.assoc_voxel_mult)
    acc_code, acc_rgb, acc_cnt, acc_sim = [], [], [], []
    instances = []
    showcase = None
    cam_pos = np.stack([f["c2w"][:3, 3] for f in frames])
    n_batches = (len(frames) + args.n_views - 1) // args.n_views
    t_ov = t_seg = 0.0
    n_prompt_total = 0
    n_tgt = len(TARGET_PROMPTS)

    for bi in range(n_batches):
        batch = frames[bi * args.n_views:(bi + 1) * args.n_views]
        N = len(batch)
        imgs = torch.zeros(1, N, 3, R, R, device=device)
        rgb_small, ov_per_view, neg_per_view = [], [], []
        for j, fr in enumerate(batch):
            rgb = fr["rgb"]
            imr = np.asarray(Image.fromarray(rgb).resize((R, R), Image.LANCZOS))
            rgb_small.append(imr)
            imgs[0, j] = torch.from_numpy(imr).permute(2, 0, 1).float() / 255.0
            small, s = _resize_rgb(rgb, args.ov_long)
            t0 = time.time()
            det = ov.segment(small)
            t_ov += time.time() - t0
            neg = ov.last_neg
            neg_per_view.append(_up_mask(neg, R, R) if neg is not None else None)
            # map masks back to 256² work res used by the promptable head
            mapped = []
            for d in det:
                m256 = _up_mask(d["mask"], R, R)
                if m256.sum() < 20:
                    continue
                mapped.append(dict(cls=d["cls"], mask=m256, score=d["score"],
                                   xy=_centroid(m256)))
            ov_per_view.append(mapped)

        cand = []
        for v, dets in enumerate(ov_per_view):
            for d in dets:
                cand.append(dict(view=v, cls=d["cls"], xy=d["xy"],
                                 mask=d["mask"], score=d["score"], area=int(d["mask"].sum())))
        M = len(cand)
        n_prompt_total += M

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            tokens, geom, _ = backbone.forward_split(imgs, return_geometry=True,
                                                     external_geometry=None)

        # open-vocab 2D label map (for visualisation / 3D class cloud)
        lab_np = np.full((N, R, R), -1, np.int16)
        fg_np = np.zeros((N, R, R), bool)
        sim = np.zeros((N, n_tgt + 2, R, R), np.float32)  # targets + dummy bg
        for v, dets in enumerate(ov_per_view):
            # paint higher-score last
            for d in sorted(dets, key=lambda x: x["score"]):
                lab_np[v][d["mask"]] = d["cls"]
                fg_np[v][d["mask"]] = True
                sim[v, d["cls"]][d["mask"]] = max(d["score"], 0.05)

        seg_masks = iou_pred = None
        keep_mask = np.zeros((N, max(M, 1)), bool)
        if M:
            chunks = [cand[i:i + args.max_prompts] for i in range(0, M, args.max_prompts)]
            probs, ious = [], []
            for ch in chunks:
                Mc = len(ch)
                pm = torch.zeros(1, N, Mc, R, R, device=device)
                pv = torch.zeros(1, Mc, dtype=torch.long, device=device)
                pp = torch.zeros(1, N, Mc, 1, 2, device=device)
                pl = -torch.ones(1, N, Mc, 1, device=device)
                for m, s in enumerate(ch):
                    v = s["view"]
                    pv[0, m] = v
                    pm[0, v, m] = torch.from_numpy(s["mask"].astype(np.float32)).to(device)
                    pp[0, v, m, 0] = torch.tensor(s["xy"], device=device)
                    pl[0, v, m, 0] = 1.0
                t0 = time.time()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    so = seg_head(tokens, pm, pv, R, R, prompt_points=pp,
                                  prompt_point_labels=pl, geometry=None)
                t_seg += time.time() - t0
                probs.append(so["segmentation_logits"].float()[0].sigmoid() > args.mask_thresh)
                ious.append(so["iou_pred"].float()[0])
            seg_masks = torch.cat(probs, 1).cpu().numpy()
            iou_pred = torch.cat(ious, 1).cpu().numpy()
            prop_masks = seg_masks.copy()
            prop_iou = iou_pred.copy()

            # geometric propagation through the metric point cloud
            geo = build_view_geometry(batch, R, gl=batch[0]["gl"])
            depth_masks = (depth_propagate(cand, geo, N, M, R, depth_tol=args.depth_tol)
                           if args.depth_prop else np.zeros_like(prop_masks))
            seed_masks, cstats = combine_prop_depth(
                prop_masks, depth_masks, cand,
                agree_iou=args.agree_iou, agree_contain=args.agree_contain)
            print(f"        depth-prop {cstats}", flush=True)

            # SAM2 refinement of every propagated seed
            n_ref = 0
            if args.sam_refine:
                t0 = time.time()
                seed_masks, n_ref = sam_refine_masks(
                    seed_masks, iou_pred, cand, ov, rgb_small, neg=neg_per_view)
                t_ov += time.time() - t0

            seg_masks, n_gsam, n_merge = match_gsam_masks(
                seed_masks, iou_pred, cand, ov_per_view,
                match_iou=args.match_iou, match_contain=args.match_contain)
            print(f"        sam-refine {n_ref} | gsam-match {n_gsam} replaced, "
                  f"{n_merge} fragments merged", flush=True)
            if args.neg_prompt:
                seg_masks, n_nd, n_nc = veto_neg_masks(seg_masks, cand, neg_per_view)
                nf = np.mean([n.mean() for n in neg_per_view if n is not None]) \
                    if any(n is not None for n in neg_per_view) else 0.0
                print(f"        structure veto: {n_nd} masks dropped, {n_nc} trimmed "
                      f"(neg covers {nf * 100:.0f}% of frame)", flush=True)
            from demo_openvocab_instance.pipeline import mask_nms
            keep_mask = np.zeros((N, M), bool)
            for v in range(N):
                for i in mask_nms(seg_masks[v], iou_pred[v], args.nms_iou,
                                  0.85, args.min_view_area):
                    keep_mask[v, i] = True
            keep_mask &= iou_pred >= args.min_iou_pred

        # lift
        for j, fr in enumerate(batch):
            d_full = fr["depth"]
            s = args.work_scale
            Wh = max(int(round(fr["H"] * s)), 8)
            Ww = max(int(round(fr["W"] * s)), 8)
            import cv2
            d = cv2.resize(d_full.astype(np.float32), (Ww, Wh), interpolation=cv2.INTER_NEAREST)
            fx, fy, cx, cy = fr["K"]
            Kw = (fx * s, fy * s, cx * s, cy * s)
            pts, ys, xs = unproject(d, fr["c2w"], Kw, gl=fr["gl"])
            if len(pts) == 0:
                continue
            vk = grid.keys(pts)
            code = grid.encode(vk)
            col = np.asarray(Image.fromarray(rgb_small[j]).resize((Ww, Wh), Image.BILINEAR))
            gcol = col[ys, xs].astype(np.float32)
            # sample 2D OV scores at work res
            u = torch.from_numpy(np.stack([(xs + 0.5) / Ww, (ys + 0.5) / Wh], 1)).float()
            gsim = _sample(torch.from_numpy(sim[j]).to(device), u.to(device)).cpu().numpy()
            uc, inv = np.unique(code, return_inverse=True)
            cnt = np.bincount(inv, minlength=len(uc)).astype(np.float32)
            acc_code.append(uc)
            acc_cnt.append(cnt)
            acc_sim.append(_seg_sum(gsim, inv, len(uc)))
            acc_rgb.append(_seg_sum(gcol, inv, len(uc)))

            if seg_masks is None:
                continue
            for m in range(M):
                if not keep_mask[j, m]:
                    continue
                mk_w = _up_mask(seg_masks[j, m], Wh, Ww)
                sel = np.nonzero(mk_w[ys, xs])[0]
                if len(sel) < 25:
                    continue
                p = pts[sel]
                p = p[largest_blob_3d(p, args.voxel * args.assoc_voxel_mult)]
                if len(p) < 25:
                    continue
                ext = p.max(0) - p.min(0)
                if np.linalg.norm(ext) > args.max_det_diag:
                    continue
                instances.append(dict(cls=cand[m]["cls"],
                                      codes=np.unique(grid.encode(grid.keys(p))),
                                      ccodes=np.unique(coarse.encode(coarse.keys(p))),
                                      score=float(iou_pred[j, m]),
                                      views=[(bi, j)]))

        if showcase is None or M > showcase.get("M", 0):
            showcase = dict(batch=bi, M=M, rgb=np.stack(rgb_small),
                            lab=lab_np, fg=fg_np,
                            sim=sim.astype(np.float16),
                            masks=(seg_masks if seg_masks is not None
                                   else np.zeros((N, 0, R, R), bool)),
                            prop_masks=(prop_masks if M else np.zeros((N, 0, R, R), bool)),
                            depth_masks=(depth_masks if M else np.zeros((N, 0, R, R), bool)),
                            seed_masks=(seed_masks if M else np.zeros((N, 0, R, R), bool)),
                            refined_masks=(seg_masks if M else np.zeros((N, 0, R, R), bool)),
                            neg=np.stack([n if n is not None else
                                          np.zeros((R, R), bool)
                                          for n in neg_per_view]),
                            keep=keep_mask if M else np.zeros((N, 0), bool),
                            iou_pred=(iou_pred if M else np.zeros((N, 0), np.float32)),
                            prop_iou=(prop_iou if M else np.zeros((N, 0), np.float32)),
                            cand=[dict(view=c["view"], cls=c["cls"], xy=c["xy"],
                                       area=c["area"], score=c["score"]) for c in cand],
                            depth=geom["depth"].float()[0, :, :, :, 0].cpu().numpy(),
                            depth_conf=geom["depth_conf"].float()[0].cpu().numpy(),
                            c2w=np.stack([f["c2w"] for f in batch]),
                            stems=[f["stem"] for f in batch])
        print(f"  batch {bi+1}/{n_batches} views={N} ov_prompts={M} "
              f"raw_inst={len(instances)}", flush=True)

    print("[merge] voxels + instances")
    code = np.concatenate(acc_code)
    uc, inv = np.unique(code, return_inverse=True)
    G = len(uc)
    cnt = np.bincount(inv, weights=np.concatenate(acc_cnt), minlength=G).astype(np.float32)
    vsim = _seg_sum(np.concatenate(acc_sim), inv, G) / np.maximum(cnt[:, None], 1)
    vrgb = _seg_sum(np.concatenate(acc_rgb), inv, G) / np.maximum(cnt[:, None], 1)
    vk = grid.decode(uc)
    centers = grid.centers(vk)
    vlab = vsim[:, :n_tgt].argmax(1).astype(np.int16)
    vfg = vsim[:, :n_tgt].max(1) > 0.05
    print(f"[merge] {G} voxels, target frac={vfg.mean()*100:.1f}%")
    # renderer orients the scene from floor vs ceiling scores
    lo, hi = np.percentile(centers, [2, 98], 0)
    ax = int(np.argmin(hi - lo))
    h = centers[:, ax]
    extra = np.zeros((G, 8 - vsim.shape[1]), np.float32) if vsim.shape[1] < 8 else None
    if extra is not None:
        vsim = np.concatenate([vsim, extra], 1)
    vsim[:, 6] = (hi[ax] - h) / max(hi[ax] - lo[ax], 1e-6)   # floor
    vsim[:, 7] = (h - lo[ax]) / max(hi[ax] - lo[ax], 1e-6)   # ceiling

    assoc_params = dict(merge_iou=args.merge_iou, merge_containment=args.merge_containment,
                        min_dets=args.min_dets, min_instance_vox=args.min_instance_vox,
                        cross_class_absorb=args.cross_class_absorb,
                        box_iou_gate=args.box_iou_gate)
    keep = associate.fuse(instances, decode=VoxelGrid.decode, voxel=args.voxel,
                          **assoc_params)
    # extra IoU pass with the CLI thresholds (fuse already does a default pass)
    keep = associate.iou_filter(keep, same_iou=args.same_iou,
                                cross_iou=args.cross_iou, verbose=True)
    keep = attach_voxels(keep, uc, centers)
    for i, f in enumerate(keep):
        f["id"] = i
    for c, name in enumerate(TARGET_PROMPTS):
        print(f"        {name:8s}: {sum(1 for f in keep if f['cls'] == c)}")

    sim_cols = TARGET_PROMPTS + ["_bgmax20", "_bgmax141", "floor", "ceiling"]
    scene_name = args.scene.replace("/", "_")
    np.savez_compressed(
        out_dir / f"{scene_name}_bundle.npz",
        scene=scene_name, voxel=args.voxel,
        vox_keys=vk.astype(np.int32), vox_centers=centers.astype(np.float32),
        vox_rgb=vrgb.astype(np.uint8), vox_count=cnt.astype(np.int32),
        vox_sim=vsim.astype(np.float16), vox_label=vlab, vox_fg=vfg, vox_fg141=vfg,
        vox_sim_eval=vsim.astype(np.float16), vox_label_eval=vlab,
        vocab=np.array(TARGET_PROMPTS + ["_bg"]), n_targets=n_tgt,
        sim_cols=np.array(sim_cols),
        instances=np.array(keep, dtype=object),
        raw_dets=np.array(instances, dtype=object),
        assoc_params=np.array([assoc_params], dtype=object),
        gt_instances=np.array([None], dtype=object),
        inst_metrics=np.array([{}], dtype=object),
        cam_pos=cam_pos.astype(np.float32),
        frame_stems=np.array([f["stem"] for f in frames]),
        showcase=np.array([showcase], dtype=object),
        timing=np.array([dict(t_feat=t_ov, t_seg=t_seg, n_frames=len(frames),
                              n_batches=n_batches, n_prompts=n_prompt_total,
                              ov_backend=ov.backend)], dtype=object),
        model=np.array([meta], dtype=object),
        gt=np.array([{}], dtype=object),
        dataset=args.dataset,
    )
    print(f"[done] {out_dir / (scene_name + '_bundle.npz')}  "
          f"ov {t_ov:.1f}s  seg {t_seg:.1f}s  inst {len(keep)}")


if __name__ == "__main__":
    main()
