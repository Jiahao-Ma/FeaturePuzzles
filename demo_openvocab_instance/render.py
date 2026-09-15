#!/usr/bin/env python3
"""Render FARM-style demo figures from a scene bundle produced by pipeline.py.

Figures (numbering mirrors the reference paper's demo layout):
  fig1_teaser              scene reconstruction + instance boxes + trajectory + crops
  fig2_openvocab_prompts   one panel per text prompt: highlighted point cloud + heatmap
  fig3_multiview           per-view RGB / feed-forward depth / open-vocab / instances
  fig4_memory              FARM Fig.10-style snapshot: detections | 3D map | object entries
  fig5_reconstruction      RGB point cloud from several viewpoints (+ GT reference)
  fig6_semantic_vs_gt      open-vocab prediction vs ScanNet++ GT, with IoU table
  fig7_instances           instance-coloured cloud with per-instance boxes
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pcrender as pr                                        # noqa: E402

# OpenCV's Hershey fonts cannot render CJK and no CJK font is installed, so
# figures are labelled in English; the Chinese prompt names live in the report.
# BGR, one per target prompt
CLS_COLORS = [(60, 76, 231), (96, 174, 39), (219, 152, 52), (182, 89, 155)]
STUFF_GREY = (196, 196, 196)

# splat radius per camera, sized from the voxel pitch in main()
RAD = {"oblique": 1, "oblique2": 1, "top": 1}
# ceiling-removal mask over the voxel cloud, and the (axis, sign) of world up;
# both are set in main() and only ever affect rendering, never metrics
VIS = None
UP = (2, 1.0)


# ---------------------------------------------------------------------------
# bundle + geometry helpers
# ---------------------------------------------------------------------------


def load_bundle(path):
    z = np.load(path, allow_pickle=True)
    b = {k: z[k] for k in z.files}
    for k in ("instances", "raw_dets", "showcase", "timing", "model", "gt",
              "gt_instances", "inst_metrics", "assoc_params"):
        if k in b and b[k].dtype == object:
            b[k] = b[k].tolist() if k in ("instances", "raw_dets") else b[k][0]
    b["scene"] = str(b["scene"])
    b["vocab"] = [str(v) for v in b["vocab"]]
    b["n_targets"] = int(b["n_targets"])
    return b


def detect_up(centers, sim, sim_cols):
    """Return (up_axis, up_sign). For room-scale scans the vertical axis is the
    shortest one (ceiling height < floor plan); the sign is read off the
    model's own 'floor' vs 'ceiling' open-vocabulary response along that axis."""
    lo = np.percentile(centers, 1, axis=0)
    hi = np.percentile(centers, 99, axis=0)
    ax = int(np.argmin(hi - lo))
    fi, ci = sim_cols.index("floor"), sim_cols.index("ceiling")
    n = max(len(centers) // 50, 200)
    h_floor = np.median(centers[np.argsort(-sim[:, fi])[:n], ax])
    h_ceil = np.median(centers[np.argsort(-sim[:, ci])[:n], ax])
    return ax, (1.0 if h_floor < h_ceil else -1.0)


def gravity_frame(centers, up_axis, up_sign):
    """Rotation taking world -> a frame whose +z is up (for consistent cameras)."""
    z = np.zeros(3); z[up_axis] = up_sign
    a = np.array([1.0, 0.0, 0.0]) if up_axis != 0 else np.array([0.0, 1.0, 0.0])
    x = a - z * (a @ z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], 0)                 # rows


def _make_cam(Rg, center_g, diag, az, el, rad):
    a, e = np.radians(az), np.radians(el)
    off_g = np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)]) * diag * rad
    eye = (center_g + off_g) @ Rg
    tgt = center_g @ Rg
    return pr.look_at_camera(eye, tgt, up=Rg[2]), az


def _view_score(xyz, fg, w2c, K, W=240, H=170):
    """Prefer views where furniture (fg) is visible and spread across the frame."""
    px, z = pr.project(xyz, w2c, K)
    in_front = z > 1e-3
    in_frame = in_front & (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
    if in_frame.mean() < 0.15:
        return -1.0
    vis_fg = in_frame & fg
    if vis_fg.sum() < 30:
        return -0.5
    # 2D spread of furniture — walls collapse to a thin strip
    u, v = px[vis_fg, 0] / W, px[vis_fg, 1] / H
    spread = float(np.clip(u.std() * 2.8, 0, 1) + np.clip(v.std() * 2.8, 0, 1))
    depth_span = float(np.percentile(z[vis_fg], 90) - np.percentile(z[vis_fg], 10))
    depth_span = np.clip(depth_span / 4.0, 0, 1)
    return 1.2 * vis_fg.mean() + 0.7 * spread + 0.5 * depth_span + 0.15 * in_frame.mean()


def scene_cameras(centers, rgb, fg, up_axis, up_sign, W, H, fov=48.0):
    """Pick 2 room-looking obliques + a top-down by scoring candidate viewpoints."""
    Rg = gravity_frame(centers, up_axis, up_sign)
    g = centers @ Rg.T
    lo, hi = np.percentile(g, 2, axis=0), np.percentile(g, 98, axis=0)
    center_g = (lo + hi) / 2
    # look across furniture sitting on the floor, not down at a wall
    center_g = center_g.copy()
    center_g[2] = lo[2] + 0.28 * (hi[2] - lo[2])
    diag = float(np.linalg.norm(hi - lo))
    Ks = pr.intrinsics(240, 170, fov)
    fg = np.asarray(fg, bool)
    scored = []
    for az in range(0, 360, 20):
        for el, rad in ((18, 1.15), (24, 1.05), (32, 0.95)):
            w2c, azv = _make_cam(Rg, center_g, diag, az, el, rad)
            s = _view_score(centers, fg, w2c, Ks)
            scored.append((s, azv, el, rad, w2c))
    scored.sort(reverse=True, key=lambda t: t[0])
    best = scored[0]
    second = next((t for t in scored if min(abs(t[1] - best[1]) % 360,
                                            360 - abs(t[1] - best[1]) % 360) > 110),
                  scored[min(6, len(scored) - 1)])
    top, _ = _make_cam(Rg, center_g, diag, best[1], 82, 0.85)
    cams = {"oblique": best[4], "oblique2": second[4], "top": top}
    print(f"  cameras: oblique az={best[1]} el={best[2]} score={best[0]:.3f} | "
          f"oblique2 az={second[1]} el={second[2]} score={second[0]:.3f}")
    return cams, pr.intrinsics(W, H, fov), diag


def _near_cameras(centers, cam_pos, fg=None, slack=1.6, floor=3.0):
    """Voxels no further from the trajectory than the furniture itself.

    Depth through glass or down a corridor sprays voxels tens of metres away,
    which would otherwise decide the size of the BEV panel.
    """
    d = np.linalg.norm(np.asarray(centers)[:, None, :] -
                       np.asarray(cam_pos)[None, :, :], axis=2).min(1)
    ref = d[fg] if fg is not None and np.count_nonzero(fg) > 50 else d
    return d <= max(floor, slack * float(np.percentile(ref, 95)))


def bev_footprint(centers, up_axis, up_sign, cell=0.20, min_count=None,
                  keep_frac=0.97, cam_pos=None, fg=None):
    """Minimum-area floor plan of the scan: (yawed frame, centre, ex, ey).

    Depth noise streaming down a corridor would inflate a plain bounding box by
    several metres, so the footprint is measured on an occupancy grid that is
    opened and reduced to the components holding ``keep_frac`` of the voxels,
    then fitted with cv2.minAreaRect. Yawing the frame onto that rectangle is
    what lets the BEV panel fill edge to edge instead of leaving white wedges.
    """
    Rg = gravity_frame(centers, up_axis, up_sign)
    sel = _near_cameras(centers, cam_pos, fg) if cam_pos is not None else None
    p = np.asarray(centers)[sel] @ Rg.T if sel is not None else np.asarray(centers) @ Rg.T
    ij = np.floor(p[:, :2] / cell).astype(np.int64)
    org = ij.min(0)
    ij -= org
    cnt = np.zeros(ij.max(0) + 1, np.int32)
    np.add.at(cnt, (ij[:, 0], ij[:, 1]), 1)
    if min_count is None:
        # grazing-incidence rays leave cells far thinner than surfaces do, and the
        # count per cell scales with the voxel size, so calibrate on the median
        min_count = max(3, int(0.35 * np.median(cnt[cnt > 0])))
    occ = (cnt >= min_count).astype(np.uint8)
    occ = cv2.morphologyEx(occ, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, _, _ = cv2.connectedComponentsWithStats(occ, 8)
    mass = [(int(cnt[lab == k].sum()), k) for k in range(1, n)]
    mass.sort(reverse=True)
    total = sum(m for m, _ in mass)
    keep, acc = np.zeros_like(occ, bool), 0
    for m, k in mass:
        keep |= lab == k
        acc += m
        if total and acc >= keep_frac * total:
            break
    cx, cy = np.nonzero(keep)
    xy = (np.stack([cx, cy], 1) + org + 0.5) * cell
    if len(xy) < 4:
        xy = p[:, :2]
    rect = cv2.minAreaRect(xy.astype(np.float32))
    # landscape panels read better, so put the long side of the plan along x
    ang = np.radians(rect[2]) + (np.pi / 2 if rect[1][1] > rect[1][0] else 0.0)
    c, s = np.cos(-ang), np.sin(-ang)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    Rb = Rz @ Rg
    r = np.column_stack([xy, np.zeros(len(xy))]) @ Rg @ Rb.T
    lo, hi = r[:, :2].min(0), r[:, :2].max(0)          # axis aligned after the yaw
    ctr = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2,
                    float((np.asarray(centers) @ Rb.T)[:, 2].max())])
    return Rb, ctr, max(hi[0] - lo[0], 1e-3), max(hi[1] - lo[1], 1e-3)


def bev_camera(centers, up_axis, up_sign, width, height=None, pad=0.01, far=12.0,
               cam_pos=None, fg=None):
    """Near-orthographic top-down camera; the floor plan fills the whole frame.

    ``height`` defaults to the floor plan's own aspect ratio, so neither axis is
    padded or cropped. A far eye with a matching long focal length keeps the
    projection orthographic enough that BEV distances stay readable.
    """
    Rb, ctr, ex, ey = bev_footprint(centers, up_axis, up_sign,
                                    cam_pos=cam_pos, fg=fg)
    ex, ey = ex * (1 + 2 * pad), ey * (1 + 2 * pad)
    W = int(width)
    H = int(height) if height else int(round(W * ey / ex))
    H = max(H, 32)
    D = far * max(ex, ey)
    f = D * min(W / ex, H / ey)
    eye = (ctr + np.array([0.0, 0.0, D])) @ Rb
    Rc = np.stack([Rb[0], -Rb[1], -Rb[2]], 0)          # x right, y down, z fwd
    w2c = np.eye(4)
    w2c[:3, :3] = Rc
    w2c[:3, 3] = -Rc @ eye
    K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])
    return w2c, K, W, H, Rb, (ex, ey)


def oriented_box(points, Rb):
    """Tightest yaw-rotated box around ``points`` (8 corners, ``box_corners`` order).

    Axis-aligned boxes leave a lot of air around anything diagonal to the world
    axes, so the footprint comes from cv2.minAreaRect in the gravity plane.
    """
    p = np.asarray(points, float) @ Rb.T
    rect = cv2.minAreaRect(p[:, :2].astype(np.float32))
    quad = cv2.boxPoints(rect).astype(np.float64)        # 4x2, ordered around
    z0, z1 = float(p[:, 2].min()), float(p[:, 2].max())
    order = [0, 0, 1, 1, 3, 3, 2, 2]
    zs = [z0, z1, z0, z1, z0, z1, z0, z1]
    local = np.array([[quad[i, 0], quad[i, 1], z] for i, z in zip(order, zs)])
    return local @ Rb, rect


def tint_rgb(rgb, mask, color, dim=0.55, alpha=0.70):
    """FARM-style query highlight: keep the room readable, paint the selected points."""
    out = rgb.astype(np.float32) * dim
    if mask is not None and np.any(mask):
        c = np.asarray(color, np.float32)
        out[mask] = rgb[mask].astype(np.float32) * (1.0 - alpha) + c * alpha
        out[mask] = np.clip(out[mask] * 1.12, 0, 255)
    return np.clip(out, 0, 255).astype(np.uint8)


def below_ceiling(xyz, up_axis, up_sign, frac=0.90):
    """Mask dropping the ceiling. Any camera placed above an indoor scene sees
    only the ceiling slab, so every 3D figure here renders the scene with the
    top of the height range removed (standard for indoor scan figures)."""
    h = np.asarray(xyz)[:, up_axis] * up_sign
    lo, hi = np.percentile(h, 1), np.percentile(h, 99.5)
    return h <= lo + frac * (hi - lo)


def colorize_sim(sim, lo=None, hi=None, cmap=cv2.COLORMAP_INFERNO):
    lo = np.percentile(sim, 5) if lo is None else lo
    hi = np.percentile(sim, 99.5) if hi is None else hi
    t = np.clip((sim - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.applyColorMap((t * 255).astype(np.uint8), cmap).reshape(-1, 3)


# ---------------------------------------------------------------------------
# instance crops (re-reads source JPGs and projects the instance into a frame)
# ---------------------------------------------------------------------------


def instance_crops(b, scene_dir, max_inst=8, size=132):
    tj = json.load(open(scene_dir / "dslr/nerfstudio/transforms_undistorted.json"))
    fx, fy, cx, cy = tj["fl_x"], tj["fl_y"], tj["cx"], tj["cy"]
    Wf, Hf = tj["w"], tj["h"]
    by_stem = {Path(f["file_path"]).stem: np.array(f["transform_matrix"], float)
               for f in tj["frames"]}
    stems = [str(s) for s in b["frame_stems"]]
    c2ws = np.stack([by_stem[s] for s in stems])
    w2cs = np.linalg.inv(c2ws)
    centers = b["vox_centers"].astype(np.float64)

    crops = []
    for f in b["instances"][:max_inst]:
        P = centers[f["vox_idx"]]
        if len(P) > 4000:
            P = P[np.random.RandomState(0).choice(len(P), 4000, replace=False)]
        best = None
        for k, w2c in enumerate(w2cs):
            cam = P @ w2c[:3, :3].T + w2c[:3, 3]
            zc = -cam[:, 2]                      # OpenGL: in front of camera => z<0
            vis = zc > 0.15
            if vis.mean() < 0.5:
                continue
            u = cam[vis, 0] / zc[vis] * fx + cx
            v = -cam[vis, 1] / zc[vis] * fy + cy
            inb = (u > 0) & (u < Wf) & (v > 0) & (v < Hf)
            if inb.mean() < 0.7:
                continue
            # prefer close + centred + large
            span = max(u[inb].max() - u[inb].min(), v[inb].max() - v[inb].min())
            ctr = abs(u[inb].mean() / Wf - 0.5) + abs(v[inb].mean() / Hf - 0.5)
            score = span / max(Wf, Hf) - 1.2 * ctr
            if best is None or score > best[0]:
                best = (score, k, u[inb], v[inb])
        if best is None:
            crops.append(None); continue
        _, k, u, v = best
        pad = 0.16 * max(u.max() - u.min(), v.max() - v.min(), 40)
        x0, x1 = int(max(u.min() - pad, 0)), int(min(u.max() + pad, Wf))
        y0, y1 = int(max(v.min() - pad, 0)), int(min(v.max() + pad, Hf))
        img = Image.open(scene_dir / "dslr/undistorted_images" / f"{stems[k]}.JPG").convert("RGB")
        c = np.asarray(img.crop((x0, y0, x1, y1)).resize((size, size), Image.LANCZOS))[:, :, ::-1]
        crops.append(np.ascontiguousarray(c))
    return crops


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def fig_reconstruction(b, cams, K, W, H, out):
    xyz, rgb = b["vox_centers"], b["vox_rgb"][:, ::-1]        # RGB -> BGR
    panels = []
    for name, title in [("oblique", "reconstructed scene point cloud"),
                        ("oblique2", "opposite view"), ("top", "top-down")]:
        img, _ = pr.render_points(xyz[VIS], rgb[VIS], cams[name], K, W, H, radius=RAD[name])
        img = pr.draw_trajectory(img, b["cam_pos"], cams[name], K, thickness=2,
                                 marker_every=10)
        panels.append(pr.titled(img, f"{title}  (camera trajectory in orange)"))
    gtd = b.get("gt") or {}
    if len(gtd):
        gvis = np.nonzero(below_ceiling(gtd["xyz"], *UP))[0]
        sel = np.random.RandomState(0).choice(gvis, min(1_500_000, len(gvis)), replace=False)
        gimg, _ = pr.render_points(gtd["xyz"][sel], np.full((len(sel), 3), 150, np.uint8),
                                   cams["oblique"], K, W, H, radius=1)
        panels.append(pr.titled(gimg, "ScanNet++ GT point cloud (reference)"))
    grid = pr.vstack_pad([pr.hstack_pad(panels[:2]), pr.hstack_pad(panels[2:])])
    cv2.imwrite(str(out / "fig5_reconstruction.jpg"), grid,
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    return grid


def fig_openvocab_prompts(b, cams, K, W, H, out):
    xyz, rgb = b["vox_centers"], b["vox_rgb"][:, ::-1]
    sim = b["vox_sim"].astype(np.float32)
    lab, fg = b["vox_label"], b.get("vox_fg141", b["vox_fg"])
    nt = b["n_targets"]
    panels = []
    for c in range(nt):
        name = b["vocab"][c]
        m = (lab == c) & fg
        col = tint_rgb(rgb, m, CLS_COLORS[c], dim=0.52, alpha=0.72)
        img, _ = pr.render_points(xyz[VIS], col[VIS], cams["oblique"], K, W, H, radius=RAD["oblique"])
        pr.put_label(img, f'query: "{name}"   {int(m.sum())} voxels ({m.mean()*100:.1f}%)',
                     (12, 26), CLS_COLORS[c], scale=0.58)
        hcol = colorize_sim(sim[:, c])
        himg, _ = pr.render_points(xyz[VIS], hcol[VIS], cams["oblique"], K, W, H,
                                   radius=RAD["oblique"], bg=(18, 18, 18), shade=False)
        pr.put_label(himg, f'cosine("{name}", feature)', (12, 26), (240, 240, 240), scale=0.55)
        panels.append(pr.hstack_pad([pr.titled(img, f'text prompt: "{name}"  -> open-vocabulary 3D'),
                                     pr.titled(himg, "dense open-vocabulary response")], gap=6))
    grid = pr.vstack_pad(panels, gap=10)
    cv2.imwrite(str(out / "fig2_openvocab_prompts.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return grid


def fig_farm_queries(b, cams, K, W, H, out, crops):
    """FARM Fig.1-style: one RGB reconstruction per text prompt, target highlighted,
    with the matching instance crops along the bottom of each panel."""
    xyz, rgb = b["vox_centers"], b["vox_rgb"][:, ::-1]
    lab, fg = b["vox_label"], b.get("vox_fg141", b["vox_fg"])
    inst = b["instances"]
    nt = b["n_targets"]
    tiles = []
    for c in range(nt):
        name = b["vocab"][c]
        m = (lab == c) & fg
        col = tint_rgb(rgb, m, CLS_COLORS[c], dim=0.55, alpha=0.72)
        img, dep = pr.render_points(xyz[VIS], col[VIS], cams["oblique"], K, W, H,
                                    radius=RAD["oblique"])
        hits = [f for f in inst if f["cls"] == c]
        for f in hits[:3]:
            pr.draw_box3d(img, f["bmin"], f["bmax"], cams["oblique"], K, CLS_COLORS[c],
                          thickness=2, depth_buf=dep, label=f"#{f['id']}")
        pr.put_label(img, f'Find "{name}"  ·  {len(hits)} instance(s)',
                     (12, 28), CLS_COLORS[c], scale=0.62)
        panel = pr.titled(img, f'open-vocab + promptable  |  "{name}"')
        if crops:
            strip = []
            for f, cr in zip(inst, crops):
                if cr is None or f["cls"] != c:
                    continue
                cr = cr.copy()
                cv2.rectangle(cr, (0, 0), (cr.shape[1] - 1, cr.shape[0] - 1), CLS_COLORS[c], 4)
                pr.put_label(cr, f"#{f['id']}", (5, 18), CLS_COLORS[c], scale=0.45)
                strip.append(cr)
            if strip:
                panel = pr.vstack_pad([panel, pr.hstack_pad(strip[:4], gap=5)], gap=4)
        tiles.append(panel)
    grid = pr.vstack_pad([pr.hstack_pad(tiles[:2], gap=8),
                          pr.hstack_pad(tiles[2:], gap=8)], gap=8)
    grid = pr.titled(grid, f"{b['scene']}: FARM-style retrieval  —  sofa / table / cabinet / chair",
                     h=40)
    cv2.imwrite(str(out / "fig0_farm_queries.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 93])
    return grid


def fig_instances(b, cams, K, W, H, out, crops=None):
    xyz, rgb = b["vox_centers"], b["vox_rgb"][:, ::-1]
    inst = b["instances"]
    pal = pr.instance_palette(max(len(inst), 1), seed=3)
    mask = np.zeros(len(xyz), bool)
    col_paint = np.zeros((len(xyz), 3), np.uint8)
    for f in inst:
        mask[f["vox_idx"]] = True
        col_paint[f["vox_idx"]] = pal[f["id"]]
    col = tint_rgb(rgb, mask, (0, 0, 0), dim=0.50, alpha=0.0)
    col[mask] = col_paint[mask]
    views = []
    for name in ("oblique", "top"):
        img, dep = pr.render_points(xyz[VIS], col[VIS], cams[name], K, W, H, radius=RAD[name])
        for f in inst:
            pr.draw_box3d(img, f["bmin"], f["bmax"], cams[name], K, pal[f["id"]],
                          thickness=2, depth_buf=dep, label=f"#{f['id']}")
        views.append(pr.titled(img, f"instance segmentation via promptable head "
                                    f"({len(inst)} instances) - {name}"))
    grid = pr.hstack_pad(views)
    if crops:
        strip = []
        for f, c in zip(inst, crops):
            if c is None:
                continue
            c = c.copy()
            cv2.rectangle(c, (0, 0), (c.shape[1] - 1, c.shape[0] - 1), pal[f["id"]], 4)
            pr.put_label(c, f"#{f['id']} {b['vocab'][f['cls']]}", (5, 18), pal[f["id"]], scale=0.45)
            strip.append(c)
        if strip:
            grid = pr.vstack_pad([grid, pr.titled(pr.hstack_pad(strip, gap=6),
                                                  "per-instance evidence crops")])
    cv2.imwrite(str(out / "fig7_instances.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return grid


def fig_teaser(b, cams, K, W, H, out, crops):
    xyz, rgb = b["vox_centers"], b["vox_rgb"][:, ::-1]
    inst = b["instances"]
    lab, fg = b["vox_label"], b.get("vox_fg141", b["vox_fg"])
    paint = rgb.copy()
    for c in range(b["n_targets"]):
        m = (lab == c) & fg
        paint[m] = tint_rgb(rgb, m, CLS_COLORS[c], dim=1.0, alpha=0.55)[m]
    img, dep = pr.render_points(xyz[VIS], paint[VIS], cams["oblique"], K, W, H,
                                radius=RAD["oblique"])
    img = pr.draw_trajectory(img, b["cam_pos"], cams["oblique"], K, thickness=2,
                             color=(0, 140, 255), marker_every=8)
    labelled = {f["id"] for f in inst[:5]}
    for f in inst:
        if f["id"] not in labelled:
            continue
        c = CLS_COLORS[f["cls"]]
        pr.draw_box3d(img, f["bmin"], f["bmax"], cams["oblique"], K, c, thickness=2,
                      depth_buf=dep, label=b["vocab"][f["cls"]])
    legend = []
    for c, nm in enumerate(b["vocab"][:b["n_targets"]]):
        n = sum(1 for f in inst if f["cls"] == c)
        legend.append(f"{nm} x{n}")
    pr.put_label(img, "  |  ".join(legend), (12, H - 14), (40, 40, 40), scale=0.55)
    top = pr.titled(img, f"{b['scene']}: scene reconstruction + open-vocab classes "
                         f"+ {len(inst)} promptable instances")
    if crops:
        strip = [c for c in crops if c is not None]
        if strip:
            top = pr.vstack_pad([top, pr.hstack_pad(strip, gap=6)])
    cv2.imwrite(str(out / "fig1_teaser.jpg"), top, [cv2.IMWRITE_JPEG_QUALITY, 93])
    return top


def fig_multiview(b, out):
    sc = b["showcase"]
    rgb, lab, fg = sc["rgb"], sc["lab"], sc["fg"]
    masks, cand, depth = sc["masks"], sc["cand"], sc["depth"]
    N, R = rgb.shape[0], rgb.shape[1]
    nt = b["n_targets"]
    pal = pr.instance_palette(max(len(cand), 1), seed=3)
    rows = []
    for v in range(N):
        base = np.ascontiguousarray(rgb[v][:, :, ::-1])
        # open-vocabulary class overlay
        ov = base.copy()
        lu = cv2.resize(lab[v].astype(np.int32), (R, R), interpolation=cv2.INTER_NEAREST)
        fu = cv2.resize(fg[v].astype(np.uint8), (R, R), interpolation=cv2.INTER_NEAREST) > 0
        for c in range(nt):
            ov = pr.overlay_mask(ov, (lu == c) & fu, CLS_COLORS[c], alpha=0.5)
        # feed-forward depth; if the head collapsed to a blob, fall back to a
        # percentile stretch of the same map so the panel is at least readable
        d = depth[v].astype(np.float32)
        lo, hi = np.percentile(d, 5), np.percentile(d, 95)
        if hi - lo < 1e-3:
            lo, hi = float(d.min()), float(d.max())
        dn = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
        dimg = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        # promptable instance masks that survived per-view NMS
        iv = base.copy()
        keep = sc["keep"]
        up = R / lab.shape[-1]
        for m, cd in enumerate(cand):
            if m >= masks.shape[1] or not keep[v, m]:
                continue
            iv = pr.overlay_mask(iv, masks[v, m], pal[m], alpha=0.45)
        for m, cd in enumerate(cand):            # seeds live on their own view
            if cd["view"] != v or m >= masks.shape[1] or not keep[v, m]:
                continue
            px, py = [int((t + 0.5) * up) for t in cd["xy"]]
            cv2.drawMarker(iv, (px, py), (255, 255, 255), cv2.MARKER_STAR, 11, 3)
            cv2.drawMarker(iv, (px, py), pal[m], cv2.MARKER_STAR, 9, 1)
        n_here = int(keep[v].sum())
        n_seed = sum(1 for cd in cand if cd["view"] == v)
        pr.put_label(iv, f"{n_seed} point prompts here, {n_here} masks",
                     (6, 18), (255, 255, 255), scale=0.42)
        rows.append(pr.hstack_pad([base, dimg, ov, iv], gap=4))
    hdr = pr.hstack_pad([pr.label_strip(t, R) for t in
                         ["RGB input", "feed-forward depth", "open-vocab (text prompts)",
                          "promptable seg (point prompts)"]], gap=4)
    grid = pr.vstack_pad([hdr] + rows, gap=4)
    grid = pr.titled(grid, f"{b['scene']} batch {sc['batch']}: one shared backbone pass -> "
                           f"both heads; prompts propagate across all {N} views", h=40)
    cv2.imwrite(str(out / "fig3_multiview.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 93])
    return grid


def fig_memory(b, cams, K, W, H, out, crops):
    """FARM Fig.10-style snapshot: detections + depth | 3D map w/ boxes | object entries."""
    sc = b["showcase"]
    R = sc["rgb"].shape[1]
    nt = b["n_targets"]
    v = int(np.bincount([c["view"] for c in sc["cand"]], minlength=sc["rgb"].shape[0]).argmax()) \
        if len(sc["cand"]) else 0
    det = np.ascontiguousarray(sc["rgb"][v][:, :, ::-1])
    lu = cv2.resize(sc["lab"][v].astype(np.int32), (R, R), interpolation=cv2.INTER_NEAREST)
    fu = cv2.resize(sc["fg"][v].astype(np.uint8), (R, R), interpolation=cv2.INTER_NEAREST) > 0
    for c in range(nt):
        det = pr.overlay_mask(det, (lu == c) & fu, CLS_COLORS[c], alpha=0.45)
    up = R / sc["lab"].shape[-1]
    for m, cd in enumerate(sc["cand"]):
        if cd["view"] != v or not sc["keep"][v, m]:
            continue
        px, py = [int((t + 0.5) * up) for t in cd["xy"]]
        cv2.drawMarker(det, (px, py), (255, 255, 255), cv2.MARKER_STAR, 13, 3)
        cv2.drawMarker(det, (px, py), CLS_COLORS[cd["cls"]], cv2.MARKER_STAR, 11, 2)
    rgb_big = cv2.resize(np.ascontiguousarray(sc["rgb"][v][:, :, ::-1]), (2 * R, 2 * R))
    left = pr.vstack_pad([pr.titled(cv2.resize(det, (2 * R, 2 * R)), "open-vocab detections"),
                          pr.titled(rgb_big, "RGB observation")], gap=6)

    inst = b["instances"]
    pal = pr.instance_palette(max(len(inst), 1), seed=3)
    xyz, rgb = b["vox_centers"], b["vox_rgb"][:, ::-1]
    mask = np.zeros(len(xyz), bool)
    paint = np.zeros((len(xyz), 3), np.uint8)
    for f in inst:
        mask[f["vox_idx"]] = True
        paint[f["vox_idx"]] = pal[f["id"]]
    col = tint_rgb(rgb, mask, (0, 0, 0), dim=0.52, alpha=0.0)
    col[mask] = paint[mask]
    mid, dep = pr.render_points(xyz[VIS], col[VIS], cams["oblique"], K, W, H, radius=RAD["oblique"])
    for f in inst:
        pr.draw_box3d(mid, f["bmin"], f["bmax"], cams["oblique"], K, pal[f["id"]],
                      thickness=2, depth_buf=dep, label=f"#{f['id']}")
    mid = pr.titled(mid, "object-level 3D memory (instances + boxes)")

    entries = []
    for f, c in list(zip(inst, crops))[:6]:
        if c is None:
            continue
        card = np.full((c.shape[0], 300, 3), 250, np.uint8)
        card[:, :c.shape[1]] = c
        cv2.rectangle(card, (0, 0), (c.shape[1] - 1, c.shape[0] - 1), pal[f["id"]], 3)
        ext = f["bmax"] - f["bmin"]
        lines = [f"#{f['id']}  {b['vocab'][f['cls']]}",
                 f"score {f['score']:.2f}   dets {f['n_det']}",
                 f"{len(f['vox_idx'])} voxels",
                 f"extent {ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f} m"]
        for i, t in enumerate(lines):
            cv2.putText(card, t, (c.shape[1] + 8, 24 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.46, (30, 30, 30), 1, cv2.LINE_AA)
        entries.append(card)
    right = pr.titled(pr.vstack_pad(entries, gap=6), "object memory entries") if entries else None
    parts = [left, mid] + ([right] if right is not None else [])
    grid = pr.hstack_pad(parts, gap=10)
    grid = pr.titled(grid, f"{b['scene']}: detect -> lift -> associate -> object-level memory", h=40)
    cv2.imwrite(str(out / "fig4_memory.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return grid


def fig_semantic_vs_gt(b, cams, K, W, H, out):
    gtd = b.get("gt") or {}
    if not len(gtd):
        return None, {}
    names = [str(n) for n in gtd["eval_names"]]
    tgt = b["vocab"][:b["n_targets"]]
    xyz, gcls, vidx = gtd["xyz"], gtd["cls"].astype(np.int64), gtd["vox_idx"]
    ok = vidx >= 0

    # (a) strict 20-class ScanNet++ protocol, comparable with eval_scannetpp_3d.py
    pred20 = -np.ones(len(xyz), np.int64)
    pred20[ok] = b["vox_label_eval"][vidx[ok]].astype(np.int64)
    m20 = ok & (gcls >= 0)
    per20, ious, accs = {}, [], []
    for i, nm in enumerate(names):
        p, g = pred20[m20] == i, gcls[m20] == i
        if not g.any():
            continue
        inter = (p & g).sum()
        iou = inter / max((p | g).sum(), 1); acc = inter / max(g.sum(), 1)
        per20[nm] = dict(iou=float(iou), acc=float(acc), n=int(g.sum()))
        ious.append(iou); accs.append(acc)

    # (b) the 4 text prompts actually asked for, under both background settings
    gtgt = gtd["tgt"].astype(np.int64) if "tgt" in gtd else None
    preds, prompt_metrics = {}, {}
    for setting, fgkey in (("bg20", "vox_fg"), ("bg141", "vox_fg141")):
        predt = -np.ones(len(xyz), np.int64)
        if gtgt is None or fgkey not in b:
            continue
        pl = b["vox_label"][vidx[ok]].astype(np.int64)
        pl[~b[fgkey][vidx[ok]]] = -1
        predt[ok] = pl
        preds[setting] = predt
        per, ti, ta = {}, [], []
        for c, nm in enumerate(tgt):
            p, g = predt[ok] == c, gtgt[ok] == c
            if not g.any():
                continue
            inter = (p & g).sum()
            iou = inter / max((p | g).sum(), 1); acc = inter / max(g.sum(), 1)
            per[nm] = dict(iou=float(iou), acc=float(acc), n=int(g.sum()))
            ti.append(iou); ta.append(acc)
        prompt_metrics[setting] = dict(mIoU=float(np.mean(ti)) if ti else 0.0,
                                       mAcc=float(np.mean(ta)) if ta else 0.0,
                                       per_class=per)
    predt = preds.get("bg141", preds.get("bg20", -np.ones(len(xyz), np.int64)))

    metrics = dict(
        n_gt=int(len(xyz)), coverage=float(ok.mean()), n_eval=int(m20.sum()),
        scannetpp20=dict(mIoU=float(np.mean(ious)) if ious else 0.0,
                         mAcc=float(np.mean(accs)) if accs else 0.0,
                         n_classes=len(per20), per_class=per20),
        prompts=prompt_metrics.get("bg20", {}),
        prompts_bg141=prompt_metrics.get("bg141", {}),
        synonyms={k: list(v) for k, v in (gtd.get("synonyms") or {}).items()})

    # --- visual: our 4 prompt classes vs GT, everything else grey ----------
    gvis = np.nonzero(below_ceiling(xyz, *UP))[0]
    sel = np.random.RandomState(0).choice(gvis, min(2_000_000, len(gvis)), replace=False)

    def color_by(lbls):
        c = np.full((len(sel), 3), STUFF_GREY, np.uint8)
        for k in range(len(tgt)):
            c[lbls[sel] == k] = CLS_COLORS[k]
        return c

    imgs = []
    for lbls, ttl in [(predt, "ours: open-vocabulary prediction (4 text prompts)"),
                      (gtgt if gtgt is not None else predt,
                       "ScanNet++ ground truth (synonyms accepted)")]:
        im, _ = pr.render_points(xyz[sel], color_by(lbls), cams["oblique"], K, W, H, radius=1)
        imgs.append(pr.titled(im, ttl))

    p20 = (metrics.get("prompts") or {}).get("per_class") or {}
    p141 = (metrics.get("prompts_bg141") or {}).get("per_class") or {}
    tbl = np.full((H, 430, 3), 252, np.uint8)
    y = 28
    cv2.putText(tbl, "text prompt   IoU/Acc  bg20     bg141", (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (20, 20, 20), 1, cv2.LINE_AA); y += 28
    for k, t in enumerate(tgt):
        cv2.rectangle(tbl, (14, y - 11), (28, y + 3), CLS_COLORS[k], -1)
        def cell(d):
            return f"{d['iou']*100:4.1f}/{d['acc']*100:4.1f}" if d else "   n/a   "
        txt = f"{t:<9s} {cell(p20.get(t))} {cell(p141.get(t))}"
        cv2.putText(tbl, txt, (36, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    (20, 20, 20), 1, cv2.LINE_AA)
        y += 26
    y += 10
    for t in [f"prompt mIoU     {metrics['prompts']['mIoU']*100:.1f}"
              f"  /  {(metrics.get('prompts_bg141') or {}).get('mIoU', 0)*100:.1f}",
              f"prompt mAcc     {metrics['prompts']['mAcc']*100:.1f}"
              f"  /  {(metrics.get('prompts_bg141') or {}).get('mAcc', 0)*100:.1f}",
              f"SN++20 mIoU     {metrics['scannetpp20']['mIoU']*100:.1f}"
              f" ({metrics['scannetpp20']['n_classes']} cls)",
              f"SN++20 mAcc     {metrics['scannetpp20']['mAcc']*100:.1f}",
              f"GT coverage     {metrics['coverage']*100:.1f}%"]:
        cv2.putText(tbl, t, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
        y += 26
    y += 12
    cv2.putText(tbl, "strict 20-class protocol, per class:", (14, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (90, 90, 90), 1, cv2.LINE_AA); y += 22
    for nm, d in sorted(per20.items(), key=lambda kv: -kv[1]["iou"]):
        if y > H - 14:
            break
        cv2.putText(tbl, f"{nm:<14s} {d['iou']*100:5.1f}  {d['acc']*100:5.1f}", (14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1, cv2.LINE_AA)
        y += 20
    grid = pr.hstack_pad(imgs + [pr.titled(tbl, "3D open-vocab accuracy")])
    cv2.imwrite(str(out / "fig6_semantic_vs_gt.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return grid, metrics


def fig_instances_vs_gt(b, cams, K, W, H, out):
    gi = b.get("gt_instances")
    if not gi:
        return None
    xyz, rgb = b["vox_centers"], b["vox_rgb"][:, ::-1]
    im = b["inst_metrics"]
    panels = []
    for which in ("pred", "gt"):
        mask = np.zeros(len(xyz), bool)
        paint = np.zeros((len(xyz), 3), np.uint8)
        items = b["instances"] if which == "pred" else gi["objs"]
        for f in items:
            idx = f["vox_idx"] if "vox_idx" in f else None
            if idx is None:
                continue
            mask[idx] = True
            paint[idx] = CLS_COLORS[f["cls"]]
        col = tint_rgb(rgb, mask, (0, 0, 0), dim=0.52, alpha=0.0)
        if mask.any():
            col[mask] = paint[mask]
        img, dep = pr.render_points(xyz[VIS], col[VIS], cams["oblique"], K, W, H, radius=RAD["oblique"])
        items = b["instances"] if which == "pred" else gi["objs"]
        for f in items:
            pr.draw_box3d(img, f["bmin"], f["bmax"], cams["oblique"], K,
                          CLS_COLORS[f["cls"]], thickness=2, depth_buf=dep,
                          label=b["vocab"][f["cls"]])
        panels.append(pr.titled(img, f"{'ours: promptable instances' if which=='pred' else 'ScanNet++ GT instances'}"
                                     f"  (n={len(items)})"))
    tbl = np.full((H, 400, 3), 252, np.uint8)
    y = 30
    rows = [("instances", f"pred {im['n_pred']}   GT {im['n_gt']}"),
            ("mean best IoU", f"{im['mean_best_iou']:.3f}"),
            ("F1 @ IoU 0.25", f"{im['iou25']['f1']*100:.1f}"
                              f"  (P {im['iou25']['precision']*100:.0f} /"
                              f" R {im['iou25']['recall']*100:.0f})"),
            ("F1 @ IoU 0.50", f"{im['iou50']['f1']*100:.1f}"
                              f"  (P {im['iou50']['precision']*100:.0f} /"
                              f" R {im['iou50']['recall']*100:.0f})")]
    for k, val in rows:
        cv2.putText(tbl, f"{k:<15s} {val}", (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (20, 20, 20), 1, cv2.LINE_AA)
        y += 28
    y += 14
    cv2.putText(tbl, "per class    GT  pred  bestIoU", (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.46, (20, 20, 20), 1, cv2.LINE_AA)
    y += 26
    for c, nm in enumerate(b["vocab"][:b["n_targets"]]):
        d = im["per_class"].get(c) or im["per_class"].get(str(c)) or {}
        cv2.rectangle(tbl, (14, y - 11), (28, y + 3), CLS_COLORS[c], -1)
        cv2.putText(tbl, f"{nm:<10s} {d.get('n_gt',0):4d} {d.get('n_pred',0):5d}"
                         f"   {d.get('mean_best_iou',0):.3f}",
                    (36, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 20, 20), 1, cv2.LINE_AA)
        y += 27
    grid = pr.hstack_pad(panels + [pr.titled(tbl, "3D instance accuracy")])
    cv2.imwrite(str(out / "fig8_instances_vs_gt.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return grid


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data_root", default="/mnt/dataset/marko/puzzles/embodiedsplat_dl/dataset/scannetpp/data")
    ap.add_argument("--width", type=int, default=860)
    ap.add_argument("--height", type=int, default=620)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    b = load_bundle(a.bundle)
    centers = b["vox_centers"]
    up_ax, up_sg = detect_up(centers, b["vox_sim"].astype(np.float32),
                             [str(c) for c in b["sim_cols"]])
    print(f"[{b['scene']}] {len(centers)} voxels, up=axis{up_ax}*{up_sg:+.0f}, "
          f"{len(b['instances'])} instances")
    global VIS, UP
    UP = (up_ax, up_sg)
    VIS = below_ceiling(centers, up_ax, up_sg)
    fg_cam = np.asarray(b.get("vox_fg141", b["vox_fg"]), bool)
    if b.get("instances"):
        inst_m = np.zeros(len(centers), bool)
        for f in b["instances"]:
            inst_m[f["vox_idx"]] = True
        fg_cam = fg_cam | inst_m
    cams, K, diag = scene_cameras(centers[VIS], b["vox_rgb"][VIS][:, ::-1],
                                 fg_cam[VIS], up_ax, up_sg, a.width, a.height)
    for name, c in cams.items():
        RAD[name] = pr.auto_radius(centers[VIS], c, K, float(b["voxel"]))
    print(f"  scene diagonal {diag:.1f} m, splat radius {RAD}, "
          f"ceiling removal keeps {VIS.mean()*100:.0f}% of voxels")

    scene_dir = Path(a.data_root) / b["scene"]
    crops = instance_crops(b, scene_dir, max_inst=8)
    print(f"  crops: {sum(c is not None for c in crops)}/{len(crops)}")

    fig_farm_queries(b, cams, K, a.width, a.height, out, crops);      print("  fig0 farm queries")
    fig_teaser(b, cams, K, a.width, a.height, out, crops);            print("  fig1 teaser")
    fig_openvocab_prompts(b, cams, K, a.width, a.height, out);        print("  fig2 open-vocab")
    fig_multiview(b, out);                                            print("  fig3 multiview")
    fig_memory(b, cams, K, a.width, a.height, out, crops);            print("  fig4 memory")
    fig_reconstruction(b, cams, K, a.width, a.height, out);           print("  fig5 reconstruction")
    _, metrics = fig_semantic_vs_gt(b, cams, K, a.width, a.height, out); print("  fig6 vs GT")
    fig_instances(b, cams, K, a.width, a.height, out, crops);         print("  fig7 instances")
    fig_instances_vs_gt(b, cams, K, a.width, a.height, out);          print("  fig8 inst vs GT")

    summary = dict(
        scene=b["scene"], voxel=float(b["voxel"]), n_voxels=int(len(centers)),
        scene_diag_m=float(diag), model=b["model"], timing=b["timing"],
        vocab=b["vocab"], n_targets=b["n_targets"],
        instances=[dict(id=int(f["id"]), cls=b["vocab"][f["cls"]], n_vox=int(len(f["vox_idx"])),
                        n_det=int(f["n_det"]), score=float(f["score"]),
                        extent=[float(x) for x in (f["bmax"] - f["bmin"])],
                        centroid=[float(x) for x in f["centroid"]])
                   for f in b["instances"]],
        instance_counts={b["vocab"][c]: int(sum(1 for f in b["instances"] if f["cls"] == c))
                         for c in range(b["n_targets"])},
        gt_instance_counts=({b["vocab"][c]: int(sum(1 for o in b["gt_instances"]["objs"]
                                                    if o["cls"] == c))
                             for c in range(b["n_targets"])} if b.get("gt_instances") else {}),
        assoc_params=b.get("assoc_params"),
        instance_metrics=b.get("inst_metrics"),
        metrics=metrics)
    (out / f"{b['scene']}_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[done] figures + summary in {out}")
    if metrics:
        print(f"  prompt mIoU: bg141={(metrics.get('prompts_bg141') or {}).get('mIoU', 0)*100:.1f}"
              f" bg20={metrics['prompts']['mIoU']*100:.1f}  "
              f"SN++20 mIoU={metrics['scannetpp20']['mIoU']*100:.1f}  "
              f"coverage={metrics['coverage']*100:.1f}%")


if __name__ == "__main__":
    main()
