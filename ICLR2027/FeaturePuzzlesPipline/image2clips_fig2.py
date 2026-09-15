"""Figure-2 material generator, following the official Puzzles Image2Clips.

Faithfully ports the crop-generation / calibration / rotation-augmentation logic
of https://github.com/Jiahao-Ma/puzzles-code (Image2Clips.py + Puzzles/
puzzles_utils.py + Puzzles/visualization.py), and reproduces its Open3D
point-cloud rendering with a NumPy z-buffer point-splat rasterizer so it runs
headless: non-occluded regions render dense (no holes, splat = point_size 3),
occluded regions show the white background, exactly like the paper's Figure 2.

Materials reused from a previous pipeline run (--bk_dir):
  02_moge2/geometry.npz      MoGe-2 points/depth/mask/intrinsics (threshold-style
                             depth-edge removal already applied to `mask`)
  01_fastsam/instance_ids.png  SAM mask labels (colored instance overlay)
  04_dinov3/parent_pca_hr.png  DINOv3 high-res PCA feature visualization

Outputs (--out_dir), one folder per Figure-2 panel:
  A_input/       input RGB, depth vis, hole-free point-cloud render
  B_patches/     numbered bbox overlay, stacked translucent region overlays,
                 per-puzzle RGB / depth / mask-label / dinov3-label crops
  C_calibration/ point cloud + PnP camera frustums (pre-rotation)
  D_augment/     rotated cameras in the cloud, hole-free renders of the rotated
                 views (RGB / depth / mask label / dinov3 label), post-rotation
                 visibility regions on the original image, sequence strips
  clips_meta.npz boxes, template box, shared K, per-puzzle w2c (before/after
                 rotation), rotation flags, seed

Usage (mirrors the official lamp.jpg example):
  python image2clips_fig2.py --image room2.png --bk_dir outputs/room2_bk \
      --out_dir outputs/room2_fig2 --NPuzzles 6 \
      --min_box_size_scale 0.2 --max_box_size_scale 0.5 \
      --min_overlap 0.1 --max_overlap 0.3 --Augment \
      --rotate_ratio 0.2 --rotate_min_coverage 0.2 --rotate_max_coverage 0.8 \
      --max_rotation_angle 60.0 --front_back_ratio_thresh 0.6
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# Official puzzles_utils.py ports
# --------------------------------------------------------------------------

def calculate_iou(box1, box2) -> float:
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    x_left, y_top = max(x1_1, x1_2), max(y1_1, y1_2)
    x_right, y_bottom = min(x2_1, x2_2), min(y2_1, y2_2)
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    inter = (x_right - x_left) * (y_bottom - y_top)
    a1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    a2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    return inter / (a1 + a2 - inter)


def image_plane_uv(width: int, height: int, dtype=None) -> np.ndarray:
    u = np.linspace(0.5, width - 0.5, width, dtype=dtype)
    v = np.linspace(0.5, height - 0.5, height, dtype=dtype)
    u, v = np.meshgrid(u, v, indexing="xy")
    return np.stack([u, v], axis=-1)


def generate_overlapping_boxes(image, mask, n_boxes, min_box_size_scale,
                               max_box_size_scale, min_overlap, max_overlap,
                               rng, max_attempts=1000, minTrue=0.6):
    """Official ordered-overlapping-box sampler (square boxes, IoU-chained)."""
    h, w = image.shape[:2]
    cover_region = np.zeros((h, w), dtype=np.bool_)
    boxes: List[Tuple[int, int, int, int]] = []

    def _default_box():
        min_edge = min(w, h)
        left, top = (w - min_edge) // 2, (h - min_edge) // 2
        return (left, top, left + min_edge, top + min_edge)

    def is_valid_box(new_box) -> bool:
        if not boxes:
            return True
        ious = [calculate_iou(new_box, b) for b in boxes]
        has_valid_overlap = any(min_overlap <= i <= max_overlap for i in ious)
        no_excessive_overlap = all(i <= max_overlap for i in ious)
        temp = cover_region.copy()
        temp[new_box[1]:new_box[3], new_box[0]:new_box[2]] = True
        overlap = (temp & mask).sum() / mask.sum()
        true_overlap = overlap >= min(cover_region.mean() * 1.2, 0.7)
        return has_valid_overlap and no_excessive_overlap and true_overlap

    def generate_box_with_overlap(box_size, max_tries=50):
        if not boxes:
            best_cover, cand = 0.0, _default_box()
            while max_tries > 0:
                x1 = int(rng.integers(0, w - box_size))
                y1 = int(rng.integers(0, h - box_size))
                cover = mask[y1:y1 + box_size, x1:x1 + box_size].mean()
                if cover > best_cover:
                    best_cover, cand = cover, (x1, y1, x1 + box_size, y1 + box_size)
                if cover >= minTrue:
                    return (x1, y1, x1 + box_size, y1 + box_size)
                max_tries -= 1
            return cand
        ref = boxes[int(rng.integers(0, len(boxes)))]
        rx1, ry1 = ref[0], ref[1]
        max_shift = int(box_size * (1 - min_overlap))
        x_shift = int(rng.integers(-max_shift, max_shift))
        y_shift = int(rng.integers(-max_shift, max_shift))
        x1 = max(0, min(w - box_size, rx1 + x_shift))
        y1 = max(0, min(h - box_size, ry1 + y_shift))
        return (int(x1), int(y1), int(x1 + box_size), int(y1 + box_size))

    attempts = 0
    while len(boxes) < n_boxes and attempts < max_attempts:
        box_size = int(rng.integers(int(min_box_size_scale * min(h, w)),
                                    int(max_box_size_scale * min(h, w))))
        new_box = generate_box_with_overlap(box_size)
        if is_valid_box(new_box):
            cover_region[new_box[1]:new_box[3], new_box[0]:new_box[2]] = True
            boxes.append(new_box)
        attempts += 1

    if len(boxes) < n_boxes:  # jittered supplements, official fallback
        for _ in range(n_boxes - len(boxes)):
            sbox = boxes[int(rng.integers(0, len(boxes)))]
            x1, y1, x2, y2 = sbox
            bw, bh = x2 - x1, y2 - y1
            off = int(0.2 * min(bw, bh))
            nx1 = max(0, min(w - bw, x1 + int(rng.integers(-off, off + 1))))
            ny1 = max(0, min(h - bh, y1 + int(rng.integers(-off, off + 1))))
            boxes.append((int(nx1), int(ny1), int(nx1 + bw), int(ny1 + bh)))

    bw0, bh0 = boxes[0][2] - boxes[0][0], boxes[0][3] - boxes[0][1]
    tx1, ty1 = w // 2 - bw0 // 2, h // 2 - bh0 // 2
    template_box = (tx1, ty1, tx1 + bw0, ty1 + bh0)
    return boxes, template_box


def estimate_camera_intrinsics(pc: np.ndarray, uv: np.ndarray) -> np.ndarray:
    X, Y, Z = pc[:, 0], pc[:, 1], pc[:, 2]
    u, v = uv[:, 0], uv[:, 1]
    n = len(pc)
    A = np.zeros((2 * n, 4))
    A[0:2 * n:2, 0], A[0:2 * n:2, 2] = X, Z
    A[1:2 * n:2, 1], A[1:2 * n:2, 3] = Y, Z
    b = np.zeros(2 * n)
    b[0:2 * n:2], b[1:2 * n:2] = u * Z, v * Z
    p = np.linalg.lstsq(A, b, rcond=None)[0]
    return np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]])


def calibrate_intrinsics_by_pts(template, points, downsample_size=12):
    x1, y1, x2, y2 = template
    bw, bh = x2 - x1, y2 - y1
    box_points = points[y1:y2, x1:x2]
    box_uvs = image_plane_uv(bw, bh)
    steph, stepw = max(1, bh // downsample_size), max(1, bw // downsample_size)
    pts = box_points[::steph, ::stepw].reshape(-1, 3)
    uvs = box_uvs[::steph, ::stepw].reshape(-1, 2)
    ok = np.all(np.isfinite(pts), axis=1) & (np.abs(pts).sum(axis=1) > 0)
    return estimate_camera_intrinsics(pts[ok], uvs[ok])


def estimate_camera_pose_ransac(points_3d, points_2d, K):
    dist = np.zeros((4, 1))
    ok, rvec, tvec, _ = cv2.solvePnPRansac(
        points_3d.astype(np.float64), points_2d.astype(np.float64),
        K.astype(np.float64), dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnPRansac failed")
    R, _ = cv2.Rodrigues(rvec)
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = tvec.reshape(-1)
    return w2c.astype(np.float32)


def normal_estimation(points):
    p_up, p_down = points[:-2, 1:-1], points[2:, 1:-1]
    p_left, p_right = points[1:-1, :-2], points[1:-1, 2:]
    normals = np.cross(p_right - p_left, p_down - p_up, axis=2)
    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / np.where(norms == 0, 1, norms)
    full = np.zeros_like(points)
    full[1:-1, 1:-1] = normals
    return full


def fast_camera_coverage(K, w2c, points_3d, width, height, max_rotate=80):
    pts = points_3d.reshape(-1, 3)
    normals = normal_estimation(points_3d).reshape(-1, 3)
    c2w = np.linalg.inv(w2c)
    v = c2w[:3, 3].reshape(1, 3) - pts
    vn = np.linalg.norm(v, axis=1).reshape(-1, 1)
    v = v / np.where(vn == 0, 1, vn)
    dot = np.einsum("ij,ij->i", normals, v)
    # inward-pointing normals (official comment): front if dot < cos(180-max_rotate)
    front = np.zeros_like(dot)
    front[dot < np.cos(np.radians(180 - max_rotate))] = 1
    cam = (w2c[:3, :3] @ pts.T + w2c[:3, 3:]).T
    uvz = (K @ cam.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = uvz[:, :2] / uvz[:, 2:]
    vis = (np.isfinite(uv).all(axis=1) & (uvz[:, 2] > 0)
           & (uv[:, 0] >= 0) & (uv[:, 0] < width)
           & (uv[:, 1] >= 0) & (uv[:, 1] < height))
    canvas = np.zeros((height, width), dtype=np.float32)
    canvas[uv[vis, 1].astype(int), uv[vis, 0].astype(int)] = 1.0
    return float(canvas.mean()), float(front.mean())


def random_axis_angle_rotation(angle_deg, center, rng):
    a = np.deg2rad(angle_deg)
    axis = rng.normal(0, 1, 3)
    axis /= np.linalg.norm(axis)
    c, s, oc = np.cos(a), np.sin(a), 1 - np.cos(a)
    x, y, z = axis
    R = np.array([[c + x * x * oc, x * y * oc - z * s, x * z * oc + y * s],
                  [y * x * oc + z * s, c + y * y * oc, y * z * oc - x * s],
                  [z * x * oc - y * s, z * y * oc + x * s, c + z * z * oc]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = center - R @ center
    return T


def random_rotate_around_pc_mean(w2c, point_cloud, max_angle, jitter_std, rng):
    centroid = np.mean(point_cloud, axis=0)
    c2w = np.linalg.inv(w2c)
    angle = rng.uniform(0, max_angle) + rng.normal(0, jitter_std)
    T = random_axis_angle_rotation(angle, centroid, rng)
    return np.linalg.inv(T @ c2w)


def random_rotate(points, masks, K, w2c, width, height, max_rotation_angle,
                  min_coverage, max_coverage, front_back_ratio_thresh, rng,
                  jitter_std=30.0, max_tries=3):
    """Official: best rotated extrinsic passing FrontCov + coverage checks."""
    best, best_front = w2c, 0.0
    for _ in range(max_tries):
        cand = random_rotate_around_pc_mean(
            w2c, points[masks].reshape(-1, 3), max_rotation_angle, jitter_std, rng)
        coverage, front = fast_camera_coverage(K, cand, points, width, height)
        if front < front_back_ratio_thresh:
            continue
        if min_coverage < coverage < max_coverage and front > best_front:
            best, best_front = cand, front
    return best


# --------------------------------------------------------------------------
# NumPy z-buffer point-splat rasterizer (replaces Open3D rasterization)
# --------------------------------------------------------------------------

def splat_render(points_flat: np.ndarray, colors_list: List[np.ndarray],
                 K: np.ndarray, w2c: np.ndarray, width: int, height: int,
                 bg=255, splat: int = 1):
    """Render a point cloud with painter's algorithm + square splats.

    Sorting far-to-near and overwriting means the nearest point wins per pixel;
    the (2*splat+1)^2 splat mimics Open3D point_size=3 so non-occluded surfaces
    render densely with no holes, while genuinely occluded/revealed regions
    keep the background color (white in the paper's Figure 2).

    Returns ([rendered_color_images], depth) — every colors array in
    ``colors_list`` is rendered with the same projection (e.g. RGB, mask-label
    colors, DINOv3-PCA colors in one pass).
    """
    cam = points_flat @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    front = z > 1e-6
    cam, z = cam[front], z[front]
    uvz = cam @ K.T
    u, v = uvz[:, 0] / z, uvz[:, 1] / z
    ui = np.round(u).astype(np.int64)
    vi = np.round(v).astype(np.int64)
    inb = (ui >= -splat) & (ui < width + splat) & (vi >= -splat) & (vi < height + splat)
    ui, vi, z = ui[inb], vi[inb], z[inb]
    order = np.argsort(-z)                       # far first, near last
    ui, vi, z = ui[order], vi[order], z[order]
    cols = [c[front][inb][order] for c in colors_list]

    outs = [np.full((height * width, 3), bg, dtype=np.uint8) for _ in colors_list]
    dep = np.zeros(height * width, dtype=np.float32)
    for dy in range(-splat, splat + 1):
        for dx in range(-splat, splat + 1):
            uu, vv = ui + dx, vi + dy
            m = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
            idx = vv[m] * width + uu[m]
            for o, c in zip(outs, cols):
                o[idx] = c[m]
            dep[idx] = z[m]
    return [o.reshape(height, width, 3) for o in outs], dep.reshape(height, width)


# --------------------------------------------------------------------------
# Visualization helpers (official style)
# --------------------------------------------------------------------------

PALETTE = [(230, 57, 70), (46, 134, 222), (38, 166, 91), (241, 143, 1),
           (142, 68, 173), (0, 168, 168), (214, 93, 177), (120, 111, 166)]


def visualize_boxes(image_rgb, boxes, save_path, colors):
    vis = image_rgb.copy()
    for i, (box, color) in enumerate(zip(boxes, colors)):
        x1, y1, x2, y2 = box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 7)
        cv2.putText(vis, f"#{i}", (x1 + 8, y1 + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 4, cv2.LINE_AA)
    cv2.imwrite(str(save_path), vis[..., ::-1])


def visualize_puzzles_regions(image_rgb, boxes=None, masks=None, colors=None,
                              save_path=None, alpha_ratio=0.4,
                              save_per_frame=True):
    """Official stacked translucent overlay (PIL alpha compositing)."""
    h, w = image_rgb.shape[:2]
    img = Image.fromarray(image_rgb).convert("RGBA")
    pil_masks = []
    if boxes is not None:
        for box in boxes:
            m = np.zeros((h, w), dtype=np.uint8)
            m[box[1]:box[3], box[0]:box[2]] = 255
            pil_masks.append(Image.fromarray(m).convert("L"))
    else:
        for m in masks:
            pil_masks.append(Image.fromarray(
                m.astype(np.uint8).reshape(h, w) * 255).convert("L"))
    base_dir = Path(str(save_path)).with_suffix("")
    if save_per_frame:
        base_dir.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(pil_masks):
        layer = Image.new("RGBA", img.size)
        layer.paste(tuple(colors[i]), mask=m)
        alpha = int(255 * alpha_ratio)
        layer.putalpha(m.point(lambda p: alpha if p > 0 else 0))
        img = Image.alpha_composite(img, layer)
        if save_per_frame:
            img.save(base_dir / f"puzzle_{i}.png")
    img.save(save_path)


def puzzles_visibility_on_canvas(points, masks, intrinsics, extrinsics,
                                 height, width):
    pts = points.reshape(-1, 3).T
    out = []
    for w2c, K in zip(extrinsics, intrinsics):
        cam = w2c[:3, :3] @ pts + w2c[:3, 3:4]
        uvz = K @ cam
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = uvz[:2] / uvz[2]
        m = (masks.reshape(-1) & np.isfinite(uv).all(axis=0) & (uvz[2] > 0)
             & (uv[0] >= 0) & (uv[0] < width) & (uv[1] >= 0) & (uv[1] < height))
        out.append(m)
    return out


def depth_edge_mask(depth: np.ndarray, valid: np.ndarray, rtol: float) -> np.ndarray:
    """True where the 3x3 local depth range exceeds ``rtol * depth``.

    With a mild rtol this only flags genuine occlusion silhouettes (flying
    pixels); the thin removed strips are then covered by neighbouring splats
    during rendering, so non-occluded surfaces stay hole-free.
    """
    from scipy.ndimage import maximum_filter, minimum_filter
    d = depth.astype(np.float64)
    BIG = np.finfo(np.float64).max / 2
    dmax = maximum_filter(np.where(valid, d, -BIG), size=3, mode="nearest")
    dmin = minimum_filter(np.where(valid, d, BIG), size=3, mode="nearest")
    return ((dmax - dmin) > rtol * np.maximum(d, 1e-6)) & valid


def depth_to_vis(depth: np.ndarray, valid: np.ndarray, cmap="Spectral"):
    vis = np.full((*depth.shape, 3), 255, dtype=np.uint8)
    if valid.sum() == 0:
        return vis
    d = depth[valid]
    lo, hi = np.percentile(d, 2), np.percentile(d, 98)
    t = np.clip((depth - lo) / max(hi - lo, 1e-9), 0, 1)
    rgb = (matplotlib.colormaps[cmap](t)[..., :3] * 255).astype(np.uint8)
    vis[valid] = rgb[valid]
    return vis


def frustum_segments(K, w2c, width, height, scale):
    """3D wireframe segments (world coords) of a camera frustum + image plane."""
    c2w = np.linalg.inv(w2c)
    Kinv = np.linalg.inv(K)
    corners_px = np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]],
                          dtype=np.float64)
    corners_cam = (Kinv @ corners_px.T).T * scale
    apex_w = c2w[:3, 3]
    corners_w = (c2w[:3, :3] @ corners_cam.T).T + apex_w
    segs = [(apex_w, corners_w[i]) for i in range(4)]
    segs += [(corners_w[i], corners_w[(i + 1) % 4]) for i in range(4)]
    return segs


def fit_focal_for(w2c_view, view_wh, points_world, margin=0.06, f_max=None):
    """Largest focal length that keeps all ``points_world`` inside the frame."""
    W, H = view_wh
    cam = points_world @ w2c_view[:3, :3].T + w2c_view[:3, 3]
    cam = cam[cam[:, 2] > 1e-6]
    rx = np.abs(cam[:, 0]) / cam[:, 2]
    ry = np.abs(cam[:, 1]) / cam[:, 2]
    fx = (W / 2) * (1 - margin) / max(rx.max(), 1e-9)
    fy = (H / 2) * (1 - margin) / max(ry.max(), 1e-9)
    f = min(fx, fy)
    if f_max is not None:
        f = min(f, f_max)
    return f


def render_cloud_with_frustums(points_flat, colors_flat, K_full, w2c_view,
                               view_wh, cam_list, save_path, splat=1,
                               frustum_scale=0.35, thickness=3):
    """Hole-free cloud render from an offset viewpoint + projected frustums."""
    W, H = view_wh
    (img,), _ = splat_render(points_flat, [colors_flat], K_full, w2c_view, W, H,
                             bg=255, splat=splat)
    img = np.ascontiguousarray(img)
    for (K_c, w2c_c, wh_c, color) in cam_list:
        for a, b in frustum_segments(K_c, w2c_c, wh_c[0], wh_c[1], frustum_scale):
            pa = w2c_view[:3, :3] @ a + w2c_view[:3, 3]
            pb = w2c_view[:3, :3] @ b + w2c_view[:3, 3]
            if pa[2] <= 1e-6 or pb[2] <= 1e-6:
                continue
            ua = (K_full @ pa)[:2] / pa[2]
            ub = (K_full @ pb)[:2] / pb[2]
            cv2.line(img, tuple(np.round(ua).astype(int)),
                     tuple(np.round(ub).astype(int)), color, thickness, cv2.LINE_AA)
    cv2.imwrite(str(save_path), img[..., ::-1])
    return img


def look_at_w2c(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """w2c for a camera at ``eye`` looking at ``target`` (world y points down)."""
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, fwd); right /= np.linalg.norm(right)
    dn = np.cross(fwd, right)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([right, dn, fwd], axis=1)
    c2w[:3, 3] = eye
    return np.linalg.inv(c2w)


def offset_view(K_full, points_valid, yaw_deg=-24.0, pitch_deg=-14.0, zoom=1.35):
    """A third-person viewpoint orbiting the cloud centroid (panel A)."""
    centroid = points_valid.mean(axis=0)
    yaw, pitch = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg)
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]])
    R = Ry @ Rx
    cam_pos = centroid + R @ (np.zeros(3) - centroid) * zoom
    return look_at_w2c(cam_pos, centroid)


def strip(images: List[np.ndarray], save_path, pad=6):
    h = min(im.shape[0] for im in images)
    ims = [cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h)) for im in images]
    bar = np.full((h, pad, 3), 255, dtype=np.uint8)
    out = ims[0]
    for im in ims[1:]:
        out = np.concatenate([out, bar, im], axis=1)
    cv2.imwrite(str(save_path), out[..., ::-1])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(args):
    root = Path(__file__).resolve().parent
    bk = root / args.bk_dir
    out = root / args.out_dir
    for sub in ["A_input", "B_patches", "C_calibration", "D_augment"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- load reused materials -------------------------------------------
    image = cv2.imread(str(root / args.image))[..., ::-1].copy()      # RGB
    H, W = image.shape[:2]
    g = np.load(bk / "02_moge2" / "geometry.npz")
    points = g["points"].astype(np.float64)                            # (H,W,3)
    depth = g["depth"].astype(np.float64)
    stored_valid = g["mask"].astype(bool)
    # The cached geometry removed depth edges with an aggressive rtol (9% of
    # pixels, zeroing their points). Recover those pixels by unprojecting the
    # stored depth with the (normalized) MoGe intrinsics, then re-remove only
    # genuine occlusion silhouettes with a milder tolerance so non-occluded
    # regions render without holes (official --threshold behaviour).
    K_norm = g["intrinsics"].astype(np.float64)
    fx, fy = K_norm[0, 0] * W, K_norm[1, 1] * H
    cx, cy = K_norm[0, 2] * W, K_norm[1, 2] * H
    uv_full = image_plane_uv(W, H)
    pts_rec = np.dstack([(uv_full[..., 0] - cx) / fx * depth,
                         (uv_full[..., 1] - cy) / fy * depth, depth])
    old_edge = g["depth_edge"].astype(bool) if "depth_edge" in g else np.zeros((H, W), bool)
    points[old_edge] = pts_rec[old_edge]
    base_valid = (stored_valid | old_edge) & np.isfinite(depth) & (depth > 0)
    base_valid &= np.isfinite(points).all(axis=2)
    valid = base_valid & ~depth_edge_mask(depth, base_valid, args.edge_rtol)
    points[~valid] = 0.0

    mask_label = cv2.imread(str(bk / "01_fastsam" / "instance_ids.png"))[..., ::-1]
    mask_label = cv2.resize(mask_label, (W, H), interpolation=cv2.INTER_NEAREST)
    feat_label = cv2.imread(str(bk / "04_dinov3" / "parent_pca_hr.png"))[..., ::-1]
    feat_label = cv2.resize(feat_label, (W, H), interpolation=cv2.INTER_NEAREST)

    pts_flat = points.reshape(-1, 3)[valid.reshape(-1)]
    rgb_flat = image.reshape(-1, 3)[valid.reshape(-1)]
    mlab_flat = mask_label.reshape(-1, 3)[valid.reshape(-1)]
    flab_flat = feat_label.reshape(-1, 3)[valid.reshape(-1)]

    # ---- Panel A: input RGB-D --------------------------------------------
    cv2.imwrite(str(out / "A_input" / "image.png"), image[..., ::-1])
    cv2.imwrite(str(out / "A_input" / "depth.png"),
                depth_to_vis(depth, valid)[..., ::-1])
    cv2.imwrite(str(out / "A_input" / "mask_label.png"), mask_label[..., ::-1])
    cv2.imwrite(str(out / "A_input" / "feat_label.png"), feat_label[..., ::-1])
    view_W = 1280
    view_H = int(view_W * H / W)
    K_view = np.array([[0.9 * view_W, 0, view_W / 2],
                       [0, 0.9 * view_W, view_H / 2], [0, 0, 1]])
    # Panel A: close orbit focused on the cloud itself.
    w2c_view = offset_view(K_view, pts_flat, yaw_deg=-18, pitch_deg=-10, zoom=1.25)
    (cloud_img,), _ = splat_render(pts_flat, [rgb_flat], K_view, w2c_view,
                                   view_W, view_H, bg=255, splat=1)
    cv2.imwrite(str(out / "A_input" / "pointcloud.png"), cloud_img[..., ::-1])
    # Same orbit rendered with mask-label colours and DINOv3 PCA colours so the
    # 3D cloud carries the same supervision signals as the 2D crops.
    (mlab_cloud,), _ = splat_render(pts_flat, [mlab_flat], K_view, w2c_view,
                                    view_W, view_H, bg=255, splat=1)
    cv2.imwrite(str(out / "A_input" / "pointcloud_mask.png"), mlab_cloud[..., ::-1])
    (flab_cloud,), _ = splat_render(pts_flat, [flab_flat], K_view, w2c_view,
                                    view_W, view_H, bg=255, splat=1)
    cv2.imwrite(str(out / "A_input" / "pointcloud_feat.png"), flab_cloud[..., ::-1])
    # Panels C/D: viewpoint behind & above the crop cameras (which sit near the
    # world origin = the original MoGe camera), looking at the cloud centroid,
    # so the frustums appear in front of the point cloud like Figure 2.C.
    # Eye distance and focal length are fitted later, once the frustums exist.
    centroid = pts_flat.mean(axis=0)

    # ---- Panel B: generate ordered overlapping patches --------------------
    boxes, template_box = generate_overlapping_boxes(
        image, valid, args.NPuzzles, args.min_box_size_scale,
        args.max_box_size_scale, args.min_overlap, args.max_overlap, rng)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(boxes))]
    tw = template_box[2] - template_box[0]
    th = template_box[3] - template_box[1]

    visualize_boxes(image, boxes, out / "B_patches" / "bbox_overlay.png", colors)
    visualize_puzzles_regions(image, boxes=boxes, colors=colors,
                              save_path=out / "B_patches" / "regions_overlay.png")

    # ---- Panel C: calibration (shared K, per-patch PnP pose) --------------
    boxK = calibrate_intrinsics_by_pts(template_box, points)
    puzzles, puzzle_depths, puzzle_valids = [], [], []
    puzzle_mlabs, puzzle_flabs, extrinsics = [], [], []
    for bi, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        crop_img = cv2.resize(image[y1:y2, x1:x2], (tw, th), interpolation=cv2.INTER_AREA)
        crop_pts = cv2.resize(points[y1:y2, x1:x2].astype(np.float32), (tw, th),
                              interpolation=cv2.INTER_NEAREST).astype(np.float64)
        crop_val = cv2.resize(valid[y1:y2, x1:x2].astype(np.uint8), (tw, th),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        crop_dep = cv2.resize(depth[y1:y2, x1:x2].astype(np.float32), (tw, th),
                              interpolation=cv2.INTER_NEAREST).astype(np.float64)
        crop_mlab = cv2.resize(mask_label[y1:y2, x1:x2], (tw, th),
                               interpolation=cv2.INTER_NEAREST)
        crop_flab = cv2.resize(feat_label[y1:y2, x1:x2], (tw, th),
                               interpolation=cv2.INTER_NEAREST)
        uv = image_plane_uv(tw, th).reshape(-1, 2)
        pts3 = crop_pts.reshape(-1, 3)
        vm = crop_val.reshape(-1)
        idx = np.arange(vm.size)[vm]
        if idx.size > 1024:
            idx = rng.choice(idx, 1024, replace=False)
        w2c = estimate_camera_pose_ransac(pts3[idx], uv[idx], boxK)
        puzzles.append(crop_img); puzzle_depths.append(crop_dep)
        puzzle_valids.append(crop_val); extrinsics.append(w2c)
        puzzle_mlabs.append(crop_mlab); puzzle_flabs.append(crop_flab)
        cv2.imwrite(str(out / "B_patches" / f"puzzle_{bi}.png"), crop_img[..., ::-1])
        cv2.imwrite(str(out / "B_patches" / f"puzzle_{bi}_depth.png"),
                    depth_to_vis(crop_dep, crop_val)[..., ::-1])
        cv2.imwrite(str(out / "B_patches" / f"puzzle_{bi}_mask.png"), crop_mlab[..., ::-1])
        cv2.imwrite(str(out / "B_patches" / f"puzzle_{bi}_feat.png"), crop_flab[..., ::-1])
    strip(puzzles, out / "B_patches" / "puzzles_strip.png")

    med_depth = float(np.median(depth[valid]))
    frustum_scale = 0.22 * med_depth
    cam_list = [(boxK, extrinsics[i], (tw, th), colors[i]) for i in range(len(boxes))]

    # ---- Panel D: rotation augmentation + hole-free rendering -------------
    rot_extrinsics, rotated = [], []
    if args.Augment:
        for pi in range(len(boxes)):
            if rng.random() > args.rotate_ratio:        # official: rand > ratio => rotate
                pts_hw3 = cv2.resize(points[boxes[pi][1]:boxes[pi][3],
                                            boxes[pi][0]:boxes[pi][2]].astype(np.float32),
                                     (tw, th), interpolation=cv2.INTER_NEAREST).astype(np.float64)
                rot = random_rotate(pts_hw3, puzzle_valids[pi], boxK, extrinsics[pi],
                                    tw, th, args.max_rotation_angle,
                                    args.rotate_min_coverage, args.rotate_max_coverage,
                                    args.front_back_ratio_thresh, rng)
                rot_extrinsics.append(rot)
                rotated.append(not np.allclose(rot, extrinsics[pi]))
            else:
                rot_extrinsics.append(extrinsics[pi])
                rotated.append(False)

        rot_puzzles, rot_depth_vis, rot_mlabs, rot_flabs = [], [], [], []
        ss = 2  # supersampling: render 2x with a wider splat, then downsample
        K_ss = boxK.copy() * ss
        K_ss[2, 2] = 1.0
        for pi in range(len(boxes)):
            if rotated[pi]:
                (rimg, rml, rfl), rdep = splat_render(
                    pts_flat, [rgb_flat, mlab_flat, flab_flat],
                    K_ss, rot_extrinsics[pi], tw * ss, th * ss, bg=255, splat=3)
                rvalid = rdep > 0
                dvis = depth_to_vis(rdep, rvalid)
                rot_puzzles.append(cv2.resize(rimg, (tw, th), interpolation=cv2.INTER_AREA))
                rot_depth_vis.append(cv2.resize(dvis, (tw, th), interpolation=cv2.INTER_AREA))
                rot_mlabs.append(cv2.resize(rml, (tw, th), interpolation=cv2.INTER_NEAREST))
                rot_flabs.append(cv2.resize(rfl, (tw, th), interpolation=cv2.INTER_NEAREST))
            else:
                rot_puzzles.append(puzzles[pi])
                rot_depth_vis.append(depth_to_vis(puzzle_depths[pi], puzzle_valids[pi]))
                rot_mlabs.append(puzzle_mlabs[pi]); rot_flabs.append(puzzle_flabs[pi])
            tag = "rot" if rotated[pi] else "orig"
            cv2.imwrite(str(out / "D_augment" / f"rot_puzzle_{pi}_{tag}.png"),
                        rot_puzzles[pi][..., ::-1])
            cv2.imwrite(str(out / "D_augment" / f"rot_puzzle_{pi}_{tag}_depth.png"),
                        rot_depth_vis[pi][..., ::-1])
            cv2.imwrite(str(out / "D_augment" / f"rot_puzzle_{pi}_{tag}_mask.png"),
                        rot_mlabs[pi][..., ::-1])
            cv2.imwrite(str(out / "D_augment" / f"rot_puzzle_{pi}_{tag}_feat.png"),
                        rot_flabs[pi][..., ::-1])
        strip(rot_puzzles, out / "D_augment" / "rot_puzzles_strip.png")
        strip(rot_depth_vis, out / "D_augment" / "rot_depth_strip.png")
        strip(rot_mlabs, out / "D_augment" / "rot_mask_strip.png")
        strip(rot_flabs, out / "D_augment" / "rot_feat_strip.png")

        canvas_masks = puzzles_visibility_on_canvas(
            points, valid, [boxK] * len(boxes), rot_extrinsics, th, tw)
        visualize_puzzles_regions(image, masks=canvas_masks, colors=colors,
                                  save_path=out / "D_augment" / "rot_regions_overlay.png")
    else:
        rot_extrinsics = extrinsics
        rotated = [False] * len(boxes)

    # ---- Panels C/D frustum renders (shared fitted view for consistency) ---
    rot_cam_list = [(boxK, rot_extrinsics[i], (tw, th), colors[i])
                    for i in range(len(boxes))]
    frustum_pts = []
    for K_c, w2c_c, wh_c, _ in cam_list + rot_cam_list:
        for a, b in frustum_segments(K_c, w2c_c, wh_c[0], wh_c[1], frustum_scale):
            frustum_pts.extend([a, b])
    frustum_pts = np.array(frustum_pts)
    fit_pts = np.concatenate([pts_flat[::37], frustum_pts], axis=0)
    # Start behind/above the crop cameras, then pull the eye further back until
    # every frustum vertex has comfortable view-space depth (avoids the extreme
    # magnification of cameras that sit right next to the viewpoint).
    target = centroid * 0.95
    eye = centroid + 1.45 * (np.zeros(3) - centroid)
    eye += np.array([-0.25, -0.40, 0.0]) * np.linalg.norm(centroid)
    min_depth_needed = 0.35 * float(np.linalg.norm(centroid))
    for _ in range(24):
        w2c_cam_view = look_at_w2c(eye, target)
        z = (frustum_pts @ w2c_cam_view[:3, :3].T + w2c_cam_view[:3, 3])[:, 2]
        if z.min() >= min_depth_needed:
            break
        eye = target + (eye - target) * 1.15
    f_fit = fit_focal_for(w2c_cam_view, (view_W, view_H), fit_pts,
                          f_max=0.85 * view_W)
    K_cam_view = np.array([[f_fit, 0, view_W / 2],
                           [0, f_fit, view_H / 2], [0, 0, 1]])
    render_cloud_with_frustums(pts_flat, rgb_flat, K_cam_view, w2c_cam_view,
                               (view_W, view_H), cam_list,
                               out / "C_calibration" / "cameras_pointcloud.png",
                               frustum_scale=frustum_scale)
    if args.Augment:
        render_cloud_with_frustums(pts_flat, rgb_flat, K_cam_view, w2c_cam_view,
                                   (view_W, view_H), rot_cam_list,
                                   out / "D_augment" / "cameras_rot_pointcloud.png",
                                   frustum_scale=frustum_scale)

    # ---- metadata ----------------------------------------------------------
    np.savez(out / "clips_meta.npz",
             boxes=np.array(boxes), template_box=np.array(template_box),
             intrinsics=boxK,
             extrinsics=np.stack(extrinsics),
             rot_extrinsics=np.stack(rot_extrinsics),
             rotated=np.array(rotated), seed=args.seed)
    with open(out / "clips_meta.json", "w") as f:
        json.dump({"boxes": [list(map(int, b)) for b in boxes],
                   "template_box": list(map(int, template_box)),
                   "rotated": [bool(r) for r in rotated],
                   "n_puzzles": len(boxes), "seed": args.seed,
                   "args": vars(args)}, f, indent=2)
    print(f"done -> {out}")
    print(f"  boxes={len(boxes)} template={tw}x{th} rotated={rotated}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Figure-2 materials via official Image2Clips")
    p.add_argument("--image", type=str, default="room2.png")
    p.add_argument("--bk_dir", type=str, default="outputs/room2_bk")
    p.add_argument("--out_dir", type=str, default="outputs/room2_fig2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--edge_rtol", type=float, default=0.05,
                   help="relative 3x3 depth-range tolerance for occlusion-edge removal")
    p.add_argument("--NPuzzles", type=int, default=6)
    p.add_argument("--min_box_size_scale", type=float, default=0.2)
    p.add_argument("--max_box_size_scale", type=float, default=0.5)
    p.add_argument("--min_overlap", type=float, default=0.1)
    p.add_argument("--max_overlap", type=float, default=0.3)
    p.add_argument("--Augment", action="store_true")
    p.add_argument("--rotate_ratio", type=float, default=0.2)
    p.add_argument("--rotate_min_coverage", type=float, default=0.2)
    p.add_argument("--rotate_max_coverage", type=float, default=0.8)
    p.add_argument("--max_rotation_angle", type=float, default=60.0)
    p.add_argument("--front_back_ratio_thresh", type=float, default=0.6)
    main(p.parse_args())
