"""Project a labeled 3D point cloud into a camera frame -> per-pixel label map.

Used to obtain per-frame 2D semantic labels for ScanNet fine-tuning without
downloading ScanNet2D label pngs: take the EmbodiedSplat labeled point cloud
(`<scene>/<id>.npy`, col0-2 xyz, col10 nyu40) and project it into each camera
with z-buffer occlusion.

Geometry matches eval_scannet_3d.py: extrinsics.npy = cam->world poses,
intrinsic_depth.txt / intrinsic_color.txt = 3x3 (load 4x4, take [:3,:3]).
We assume depth maps (480x640, intrinsic_depth) and color frames (1296x968,
intrinsic_color) share the SAME camera centre / pose (EmbodiedSplat renders
depth at the colour pose with depth intrinsics), so the same extrinsic works
for both intrinsics.
"""
from __future__ import annotations

import numpy as np


def load_intrinsic(path) -> np.ndarray:
    """4x4 intrinsic txt -> (3,3) float64."""
    K = np.loadtxt(path).reshape(4, 4)[:3, :3].astype(np.float64)
    return K


def project_labels(
    gt_xyz: np.ndarray,        # (P,3) world coords
    gt_lbl: np.ndarray,        # (P,) int label (nyu40 or scannet200)
    K: np.ndarray,             # (3,3) intrinsics
    T_cw: np.ndarray,          # (4,4) cam->world
    H: int,
    W: int,
) -> np.ndarray:
    """Project labeled 3D points into a (H,W) label map with z-buffer.

    Returns int32 (H,W) label map, -1 = no label / occluded away.
    """
    R = T_cw[:3, :3]                       # world -> cam rotation (T is cam->world)
    t = T_cw[:3, 3]                        # cam origin in world
    p_c = (R.T @ (gt_xyz - t).T).T         # (P,3) in camera frame
    z = p_c[:, 2]
    in_front = z > 1e-4
    p_c = p_c[in_front]
    z = z[in_front]
    lbl = gt_lbl[in_front]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * (p_c[:, 0] / z) + cx
    v = fy * (p_c[:, 1] / z) + cy
    ui = np.round(u).astype(np.int64)
    vi = np.round(v).astype(np.int64)
    keep = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui, vi, z, lbl = ui[keep], vi[keep], z[keep], lbl[keep]

    label_map = np.full((H, W), -1, dtype=np.int32)
    depth_buf = np.full((H, W), np.inf, dtype=np.float64)
    # z-buffer: keep nearest point per pixel
    order = np.argsort(-z)                 # far -> near, near overwrites
    ui, vi, z, lbl = ui[order], vi[order], z[order], lbl[order]
    label_map[vi, ui] = lbl.astype(np.int32)
    return label_map


def dense_labels_via_depth(
    depth: np.ndarray,        # (Hd,Wd) uint16 mm
    Kd: np.ndarray,           # (3,3) depth intrinsics
    T_cw: np.ndarray,          # (4,4) cam->world (shared pose)
    gt_xyz: np.ndarray,        # (P,3) labeled world points
    gt_lbl: np.ndarray,        # (P,) int label
    radius: float = 0.05,      # nearest-neighbour radius (m)
):
    """Unproject depth -> 3D world, KDTree-match to labeled cloud -> dense
    per-depth-pixel label map + a boolean valid mask.

    Returns (label_map (Hd,Wd) int32 -1 void, valid (Hd,Wd) bool, pts_w (Q,3))."""
    from scipy.spatial import cKDTree
    Hd, Wd = depth.shape
    d = depth.astype(np.float32) / 1000.0
    valid_d = d > 0
    ys, xs = np.nonzero(valid_d)
    z = d[ys, xs]
    fx, fy = Kd[0, 0], Kd[1, 1]
    cx, cy = Kd[0, 2], Kd[1, 2]
    x_c = (xs - cx) * z / fx
    y_c = (ys - cy) * z / fy
    pts_c = np.stack([x_c, y_c, z, np.ones_like(z)], axis=1)        # (Q,4)
    pts_w = (T_cw @ pts_c.T).T[:, :3]                               # (Q,3)

    tree = cKDTree(gt_xyz)
    dist, idx = tree.query(pts_w, k=1, distance_upper_bound=radius,
                           workers=-1)
    # cKDTree returns idx == len(gt_xyz) when nothing within radius
    matched = idx < gt_xyz.shape[0]
    label_map = np.full((Hd, Wd), -1, dtype=np.int32)
    label_map[ys[matched], xs[matched]] = gt_lbl[idx[matched]].astype(np.int32)
    m_pts_w = pts_w[matched]
    m_lbl = gt_lbl[idx[matched]].astype(np.int32)
    return label_map, m_pts_w, m_lbl


def reproject_labels_to_camera(
    pts_w: np.ndarray,        # (Q,3) world points (e.g. unprojected depth)
    lbl: np.ndarray,          # (Q,) int label
    K: np.ndarray,            # (3,3) target camera intrinsics
    T_cw: np.ndarray,          # (4,4) cam->world (shared pose)
    H: int,
    W: int,
) -> np.ndarray:
    """Project labeled 3D world points into a target camera (z-buffered)."""
    R = T_cw[:3, :3]; t = T_cw[:3, 3]
    p_c = (R.T @ (pts_w - t).T).T
    z = p_c[:, 2]
    in_front = z > 1e-4
    p_c = p_c[in_front]; z = z[in_front]; lbl = lbl[in_front]
    fx, fy = K[0, 0], K[1, 1]; cx, cy = K[0, 2], K[1, 2]
    u = fx * (p_c[:, 0] / z) + cx
    v = fy * (p_c[:, 1] / z) + cy
    ui = np.round(u).astype(np.int64); vi = np.round(v).astype(np.int64)
    keep = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui, vi, z, lbl = ui[keep], vi[keep], z[keep], lbl[keep]
    label_map = np.full((H, W), -1, dtype=np.int32)
    order = np.argsort(-z)
    ui, vi, lbl = ui[order], vi[order], lbl[order]
    label_map[vi, ui] = lbl.astype(np.int32)
    return label_map


def label_map_to_rgb(label_map: np.ndarray, palette: np.ndarray, void=(40, 40, 40)) -> np.ndarray:
    """label_map (H,W) int (-1 void) -> (H,W,3) uint8 via palette (N,3)."""
    H, W = label_map.shape
    rgb = np.broadcast_to(np.array(void, dtype=np.uint8), (H, W, 3)).copy()
    valid = (label_map >= 0) & (label_map < len(palette))
    rgb[valid] = palette[label_map[valid]]
    return rgb
