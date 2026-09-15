#!/usr/bin/env python3
"""Diagnostic figures for a nearby 6-view SAM3/promptable bundle.

  fig1  reconstruction (RGB point cloud + trajectory)
  fig2  Grounded-SAM / SAM3 mask on each of the 6 frames
  fig3  promptable head propagating those prompts across the 6 views
  fig4  open-vocab classes lifted to 3D voxels
  fig5  fused cross-view 3D instances
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo_openvocab_instance"))
import render as R  # noqa: E402
import pcrender as pr  # noqa: E402

CLS = R.CLS_COLORS  # sofa / table / cabinet / chair  BGR


def _upscale(rgb):
    """256 -> ~384 so 6-row grids stay readable."""
    h, w = rgb.shape[:2]
    return cv2.resize(rgb, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)


def fig1_recon(b, cam, out):
    xyz, rgb = b["vox_centers"], b["vox_rgb"][:, ::-1]
    img, _ = pr.render_points(xyz[R.VIS], rgb[R.VIS], cam["w2c"], cam["K"],
                              cam["W"], cam["H"], radius=cam["rad"])
    img = pr.draw_trajectory(img, b["cam_pos"], cam["w2c"], cam["K"], thickness=2,
                             marker_every=1)
    stems = [str(s) for s in b.get("frame_stems", [])]
    grid = pr.titled(img, f"{b['scene']} fig1  BEV reconstruction  |  {len(stems)} nearby "
                          f"frames  (orange = camera poses)", h=40)
    cv2.imwrite(str(out / "fig1_reconstruction.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 93])


def fig2_ov(b, out):
    sc = b["showcase"]
    rgb, lab, fg, cand = sc["rgb"], sc["lab"], sc["fg"], sc["cand"]
    N, Rsz = rgb.shape[0], rgb.shape[1]
    nt = b["n_targets"]
    pal = pr.instance_palette(max(len(cand), 1), seed=3)
    rows = []
    for v in range(N):
        base = np.ascontiguousarray(rgb[v][:, :, ::-1])
        ov = base.copy()
        for c in range(nt):
            ov = pr.overlay_mask(ov, (lab[v] == c) & fg[v], CLS[c], alpha=0.5)
        inst = base.copy()
        n_here = 0
        src_masks = sc["masks"]
        for m, cd in enumerate(cand):
            if cd["view"] != v or m >= src_masks.shape[1]:
                continue
            n_here += 1
            inst = pr.overlay_mask(inst, src_masks[v, m], pal[m], alpha=0.45)
            px, py = int(cd["xy"][0] + 0.5), int(cd["xy"][1] + 0.5)
            cv2.drawMarker(inst, (px, py), (255, 255, 255), cv2.MARKER_STAR, 11, 3)
            cv2.drawMarker(inst, (px, py), pal[m], cv2.MARKER_STAR, 9, 1)
        pr.put_label(ov, f"view {v}  {sc['stems'][v]}", (6, 18), (255, 255, 255), 0.42)
        pr.put_label(inst, f"{n_here} Grounded-SAM instances", (6, 18), (255, 255, 255), 0.42)
        rows.append(pr.hstack_pad([_upscale(base), _upscale(ov), _upscale(inst)], gap=4))
    hdr = pr.hstack_pad([pr.label_strip(t, Rsz * 2) for t in
                         ["RGB", "Grounded-SAM class mask", "instance seeds (star = centroid)"]],
                        gap=4)
    grid = pr.vstack_pad([hdr] + rows, gap=4)
    grid = pr.titled(grid, f"{b['scene']} fig2  Grounded-SAM / SAM3 per-frame masks  "
                           f"(sofa=red table=green cabinet=blue chair=purple)", h=40)
    cv2.imwrite(str(out / "fig2_grounded_sam.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 93])


def _union_groups(ids, masks, iou_thr=0.40, contain_thr=0.60):
    """Group mask ids that cover the same object so overlaps draw as one blob."""
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in ids:
        A = masks[a].astype(bool)
        asz = int(A.sum())
        if asz < 20:
            continue
        for b in ids:
            if b <= a:
                continue
            B = masks[b].astype(bool)
            bsz = int(B.sum())
            if bsz < 20:
                continue
            inter = int(np.logical_and(A, B).sum())
            if inter == 0:
                continue
            if (inter / max(asz + bsz - inter, 1) >= iou_thr
                    or inter / max(min(asz, bsz), 1) >= contain_thr):
                parent[find(b)] = find(a)
    g = {}
    for i in ids:
        g.setdefault(find(i), []).append(i)
    return list(g.values())


def _paint_stage(base, ids, masks, pal, cand, v, min_frac=0.002):
    """Overlay one propagation stage, unioning ids that share an object."""
    img = base.copy()
    live = [m for m in ids if m < masks.shape[0] and masks[m].mean() >= min_frac]
    n = 0
    for grp in _union_groups(live, masks):
        mk = np.zeros(masks[grp[0]].shape, bool)
        for m in grp:
            mk |= masks[m].astype(bool)
        n += 1
        img = pr.overlay_mask(img, mk, pal[grp[0]], alpha=0.50)
        if any(int(cand[m]["view"]) == v for m in grp):
            px, py = int(cand[grp[0]]["xy"][0] + 0.5), int(cand[grp[0]]["xy"][1] + 0.5)
            cv2.drawMarker(img, (px, py), (255, 255, 255), cv2.MARKER_STAR, 11, 3)
    return img, n


def fig3_prop(b, out):
    """Four stages of one source view's objects travelling to the other views.

    promptable prediction | depth warp | agreed seed | SAM2-refined result.
    """
    sc = b["showcase"]
    rgb, cand = sc["rgb"], sc["cand"]
    zero = np.zeros_like(sc["masks"])
    prop = sc.get("prop_masks", sc["masks"])
    dep = sc.get("depth_masks", zero)
    refined = sc.get("refined_masks", prop)
    N, Rsz = rgb.shape[0], rgb.shape[1]
    pal = pr.instance_palette(max(len(cand), 1), seed=3)
    counts = np.bincount([c["view"] for c in cand], minlength=N)
    src = int(np.argmax(counts)) if len(cand) else 0
    src_ids = sorted([m for m, c in enumerate(cand) if c["view"] == src],
                     key=lambda m: -cand[m]["area"])[:12]
    rows = []
    for v in range(N):
        base = np.ascontiguousarray(rgb[v][:, :, ::-1])
        p_img, n_p = _paint_stage(base, src_ids, prop[v], pal, cand, v)
        d_img, n_d = _paint_stage(base, src_ids, dep[v], pal, cand, v)
        r_img, n_r = _paint_stage(base, src_ids, refined[v], pal, cand, v)
        tag = "SOURCE" if v == src else f"from view {src}"
        pr.put_label(p_img, f"view {v}  promptable  {n_p}", (6, 18), (255, 255, 255), 0.42)
        pr.put_label(d_img, f"depth warp  {n_d}", (6, 18), (255, 255, 255), 0.42)
        pr.put_label(r_img, f"{tag}  refined  {n_r}", (6, 18), (255, 255, 255), 0.42)
        rows.append(pr.hstack_pad([_upscale(base), _upscale(p_img),
                                   _upscale(d_img), _upscale(r_img)], gap=4))
    hdr = pr.hstack_pad([pr.label_strip(t, Rsz * 2) for t in
                         ["RGB", "ours: promptable propagation",
                          "depth reprojection", "agreed + SAM2 refined"]], gap=4)
    grid = pr.vstack_pad([hdr] + rows, gap=4)
    grid = pr.titled(grid, f"{b['scene']} fig3  view {src} objects propagated 3 ways  |  "
                           f"same colour = same object", h=40)
    cv2.imwrite(str(out / "fig3_promptable_prop.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 93])


def fig4_cls3d(b, cam, out):
    xyz, rgb = b["vox_centers"], b["vox_rgb"][:, ::-1]
    lab, fg = b["vox_label"], b.get("vox_fg141", b["vox_fg"])
    tiles = []
    for c in range(b["n_targets"]):
        name = b["vocab"][c]
        m = (lab == c) & fg
        col = R.tint_rgb(rgb, m, CLS[c], dim=0.75, alpha=0.72)
        img, _ = pr.render_points(xyz[R.VIS], col[R.VIS], cam["w2c"], cam["K"],
                                  cam["W"], cam["H"], radius=cam["rad"])
        pr.put_label(img, f'{name}   {int(m.sum())} voxels ({m.mean()*100:.1f}%)',
                     (12, 28), CLS[c], scale=0.58)
        tiles.append(pr.titled(img, f'BEV open-vocab  |  "{name}"'))
    grid = pr.vstack_pad([pr.hstack_pad(tiles[:2], gap=8),
                          pr.hstack_pad(tiles[2:], gap=8)], gap=8)
    grid = pr.titled(grid, f"{b['scene']} fig4  classes lifted to 3D "
                           f"(OV 2D masks unprojected with posed RGB-D)", h=40)
    cv2.imwrite(str(out / "fig4_classes_3d.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 93])


def _place_label(img, text, xy, color, placed, scale=0.38):
    """Label near ``xy`` but nudged off any label already drawn."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = int(np.clip(xy[0] - tw / 2, 0, max(img.shape[1] - tw - 4, 0))), int(xy[1])
    for dy in [0, -16, 16, -32, 32, -48, 48, -64, 64, -80, 80]:
        box = (x - 4, y + dy - th - 6, x + tw + 6, y + dy + base + 3)
        if all(box[2] < p[0] or p[2] < box[0] or box[3] < p[1] or p[3] < box[1]
               for p in placed):
            placed.append(box)
            pr.put_label(img, text, (x, y + dy), color, scale=scale)
            return
    pr.put_label(img, text, (x, y), color, scale=scale)


def fig5_inst(b, cam, out):
    xyz, rgb = b["vox_centers"], b["vox_rgb"][:, ::-1]
    inst = b["instances"]
    pal = pr.instance_palette(max(len(inst), 1), seed=3)
    mask = np.zeros(len(xyz), bool)
    paint = np.zeros((len(xyz), 3), np.uint8)
    for f in inst:
        mask[f["vox_idx"]] = True
        paint[f["vox_idx"]] = pal[f["id"]]
    col = R.tint_rgb(rgb, mask, (0, 0, 0), dim=0.85, alpha=0.0)
    col[mask] = paint[mask]
    img, dep = pr.render_points(xyz[R.VIS], col[R.VIS], cam["w2c"], cam["K"],
                                cam["W"], cam["H"], radius=cam["rad"])
    labels = []
    for f in inst:
        corners, rect = R.oriented_box(xyz[f["vox_idx"]], cam["Rb"])
        pr.draw_box3d_corners(img, corners, cam["w2c"], cam["K"], pal[f["id"]],
                              thickness=2, depth_buf=dep)
        px, _ = pr.project(corners, cam["w2c"], cam["K"])
        w, h = rect[1]
        labels.append((f"#{f['id']} {b['vocab'][f['cls']]} {w:.1f}x{h:.1f}m",
                       (px[:, 0].mean(), px[:, 1].mean()), pal[f["id"]]))
    # labels after every box, each nudged off the previous ones: BEV instances sit
    # shoulder to shoulder and text would otherwise pile up
    placed = []
    for text, at, colr in labels:
        _place_label(img, text, at, colr, placed)
    counts = {b["vocab"][c]: int(sum(1 for f in inst if f["cls"] == c))
              for c in range(b["n_targets"])}
    legend = "  |  ".join(f"{k} x{v}" for k, v in counts.items())
    grid = pr.titled(img, f"{b['scene']} fig5  BEV fused instances, minimum-area boxes  |  "
                          f"{legend}", h=40)
    cv2.imwrite(str(out / "fig5_instances.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 93])
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument("--height", type=int, default=0, help="0 = floor-plan aspect ratio")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    b = R.load_bundle(a.bundle)
    centers = b["vox_centers"]
    up_ax, up_sg = R.detect_up(centers, b["vox_sim"].astype("float32"),
                               [str(c) for c in b["sim_cols"]])
    print(f"[{b['scene']}] {len(centers)} voxels, {len(b['instances'])} inst, "
          f"up=axis{up_ax}*{up_sg:+.0f}")
    R.UP = (up_ax, up_sg)
    R.VIS = R.below_ceiling(centers, up_ax, up_sg)
    fg = b.get("vox_fg141", b["vox_fg"])
    inst_m = np.zeros(len(centers), bool)
    for f in b["instances"]:
        inst_m[f["vox_idx"]] = True
    vis_xyz = centers[R.VIS]
    w2c, K, Wb, Hb, Rb, (ex, ey) = R.bev_camera(vis_xyz, up_ax, up_sg,
                                                a.width, a.height or None,
                                                cam_pos=b["cam_pos"],
                                                fg=(inst_m if inst_m.any() else fg)[R.VIS])
    cam = dict(w2c=w2c, K=K, W=Wb, H=Hb, Rb=Rb,
               # BEV zooms in far more than the obliques did, so allow fat splats
               rad=pr.auto_radius(vis_xyz, w2c, K, float(b["voxel"]), max_r=18))
    diag = float(np.linalg.norm(vis_xyz.max(0) - vis_xyz.min(0)))
    print(f"  BEV {Wb}x{Hb}px  floor plan {ex:.1f} x {ey:.1f} m  radius={cam['rad']}")
    fig1_recon(b, cam, out)
    fig2_ov(b, out)
    fig3_prop(b, out)
    fig4_cls3d(b, cam, out)
    counts = fig5_inst(b, cam, out)
    summary = dict(
        scene=b["scene"], voxel=float(b["voxel"]), n_voxels=int(len(centers)),
        scene_diag_m=float(diag), timing=b.get("timing"),
        vocab=b["vocab"], n_targets=b["n_targets"],
        frames=[str(s) for s in b.get("frame_stems", [])],
        ov_backend=(b.get("timing") or {}).get("ov_backend"),
        instance_counts=counts,
        instances=[dict(id=int(f["id"]), cls=b["vocab"][f["cls"]],
                        n_vox=int(len(f["vox_idx"])), n_det=int(f.get("n_det", 0)),
                        score=float(f["score"]))
                   for f in b["instances"]])
    (out / f"{b['scene']}_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[done] {out}  inst={counts}")


if __name__ == "__main__":
    main()
