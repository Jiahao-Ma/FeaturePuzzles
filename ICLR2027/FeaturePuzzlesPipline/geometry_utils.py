"""Pure-geometry helpers (numpy / OpenCV) following official Puzzles conventions.

Coordinate frames:
- MoGe outputs camera-frame point maps (camera origin at 0). We treat that
  camera frame as the *world* frame for the whole sequence.
- Each crop gets its own world-to-camera pose w2c from PnP, solved against the
  shared crop intrinsic boxK and template-box pixel coordinates.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np


def calculate_iou(box_a: Sequence[int], box_b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter) / float(union) if union > 0 else 0.0


def generate_coverage_boxes(
    image: np.ndarray,
    n_boxes: int,
    min_box_size_scale: float,
    max_box_size_scale: float,
    rng: np.random.Generator,
    overlap_pad: float = 0.1,
    coverage_target: float = 0.9,
    coverage_jitter: float = 0.15,
) -> Tuple[List[Tuple[int, int, int, int]], Tuple[int, int, int, int]]:
    """Place ``n_boxes`` square crops on a jittered grid so their union covers
    ~``coverage_target`` of the image (NOT the whole image) with some randomness.

    Boxes are laid out on a grid (rows x cols, factorised to match the image
    aspect ratio). Each box's centre is jittered within its cell by
    ``coverage_jitter`` of the cell size, and the box side is the larger cell
    dimension scaled by ``coverage_target`` (with a small random multiplier)
    plus ``overlap_pad`` for the seam, clamped to the configured scale range.
    The first box (top-left) defines the template.
    """
    h, w = image.shape[:2]
    short = min(h, w)
    min_s = int(round(min_box_size_scale * short))
    max_s = int(round(max_box_size_scale * short))
    # factorise n into rows x cols closest to the image aspect ratio
    best = None
    for cols in range(1, n_boxes + 1):
        rows = -(-n_boxes // cols)          # ceil
        if rows * cols < n_boxes:
            continue
        score = abs((w / cols) / (h / rows) - w / h)
        if best is None or score < best[0]:
            best = (score, rows, cols)
    _, rows, cols = best
    cell_w = w / cols
    cell_h = h / rows
    boxes = []
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= n_boxes:
                break
            cx = cell_w * (c + 0.5)
            cy = cell_h * (r + 0.5)
            # jitter the centre within the cell for randomness
            jx = float(rng.uniform(-coverage_jitter, coverage_jitter)) * cell_w
            jy = float(rng.uniform(-coverage_jitter, coverage_jitter)) * cell_h
            cx += jx
            cy += jy
            # size: cell dimension scaled to ~coverage_target (with a little
            # randomness) plus the seam overlap pad
            size_mul = coverage_target * float(rng.uniform(0.97, 1.03))
            s = int(round(max(cell_w, cell_h) * size_mul * (1.0 + overlap_pad)))
            s = max(min_s, min(max_s, s))
            x1 = int(max(0, round(cx - s / 2)))
            y1 = int(max(0, round(cy - s / 2)))
            x2 = int(min(w, round(cx + s / 2)))
            y2 = int(min(h, round(cy + s / 2)))
            # ensure the box keeps the requested square side where possible
            if x2 - x1 < s and x1 == 0:
                x2 = min(w, s)
            if y2 - y1 < s and y1 == 0:
                y2 = min(h, s)
            boxes.append((x1, y1, x2, y2))
            idx += 1
    bw = boxes[0][2] - boxes[0][0]
    bh = boxes[0][3] - boxes[0][1]
    tx1 = w // 2 - bw // 2
    ty1 = h // 2 - bh // 2
    template_box = (tx1, ty1, tx1 + bw, ty1 + bh)
    return boxes, template_box


def generate_overlapping_boxes(
    image: np.ndarray,
    n_boxes: int,
    min_box_size_scale: float,
    max_box_size_scale: float,
    min_overlap: float,
    max_overlap: float,
    rng: np.random.Generator,
    max_tries: int = 2000,
    consecutive: bool = True,
) -> Tuple[List[Tuple[int, int, int, int]], Tuple[int, int, int, int]]:
    """Ordered overlapping crop boxes; first box defines the template box.

    Puzzles samples a *trajectory* of patches: each new box must overlap the
    previous one by IoU in [min_overlap, max_overlap] (``consecutive=True``,
    the official Image2Clips behaviour, which emphasises view span). With
    ``consecutive=False`` every pair must satisfy the IoU range (stricter).
    """
    h, w = image.shape[:2]
    short = min(h, w)
    boxes: List[Tuple[int, int, int, int]] = []
    tries = 0
    while len(boxes) < n_boxes and tries < max_tries:
        tries += 1
        s = int(round(float(rng.uniform(min_box_size_scale, max_box_size_scale)) * short))
        if s < 8:
            continue
        x1 = int(rng.integers(0, max(1, w - s)))
        y1 = int(rng.integers(0, max(1, h - s)))
        box = (x1, y1, x1 + s, y1 + s)
        if not boxes:
            boxes.append(box)
            continue
        prev = [boxes[-1]] if consecutive else boxes
        ok = True
        for p in prev:
            iou = calculate_iou(box, p)
            if not (min_overlap <= iou <= max_overlap):
                ok = False
                break
        if ok:
            boxes.append(box)
    if len(boxes) < n_boxes:
        raise RuntimeError(
            f"Could only sample {len(boxes)}/{n_boxes} boxes satisfying "
            f"IoU in [{min_overlap},{max_overlap}] after {tries} tries."
        )
    bw = boxes[0][2] - boxes[0][0]
    bh = boxes[0][3] - boxes[0][1]
    tx1 = w // 2 - bw // 2
    ty1 = h // 2 - bh // 2
    template_box = (tx1, ty1, tx1 + bw, ty1 + bh)
    return boxes, template_box


def image_plane_uv(width: int, height: int) -> np.ndarray:
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    return np.stack([xs, ys], axis=-1)  # (H,W,2): x,y pixel centers


def estimate_camera_intrinsics(points_3d: np.ndarray, points_2d: np.ndarray) -> np.ndarray:
    """Linear pinhole calibration (fx, fy, cx, cy) from 3D-2D correspondences.

    Solves fx*X + cx*Z = u*Z and fy*Y + cy*Z = v*Z (i.e. u = fx*X/Z + cx),
    matching the official ``Puzzles.puzzles_utils.estimate_camera_intrinsics``.
    """
    n = points_3d.shape[0]
    X = points_3d[:, 0]; Y = points_3d[:, 1]; Z = points_3d[:, 2]
    u = points_2d[:, 0]; v = points_2d[:, 1]
    A = np.zeros((2 * n, 4), dtype=np.float64)
    A[0::2, 0] = X
    A[0::2, 2] = Z
    A[1::2, 1] = Y
    A[1::2, 3] = Z
    b = np.zeros(2 * n, dtype=np.float64)
    b[0::2] = u * Z
    b[1::2] = v * Z
    params, *_ = np.linalg.lstsq(A, b, rcond=None)
    fx, fy, cx, cy = params
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def calibrate_intrinsics_by_pts(template: Sequence[int], points: np.ndarray,
                                downsample: int = 12) -> np.ndarray:
    """Calibrate one shared crop intrinsic boxK from the template box region."""
    x1, y1, x2, y2 = template
    bw, bh = x2 - x1, y2 - y1
    step_h = max(1, bh // downsample)
    step_w = max(1, bw // downsample)
    box_points = points[y1:y2:step_h, x1:x2:step_w].reshape(-1, 3)
    box_uvs = image_plane_uv(bw, bh)[::step_h, ::step_w].reshape(-1, 2)
    finite = np.all(np.isfinite(box_points), axis=1)
    return estimate_camera_intrinsics(box_points[finite], box_uvs[finite])


def estimate_camera_pose_ransac(points_3d: np.ndarray, points_2d: np.ndarray,
                                K: np.ndarray) -> Tuple[np.ndarray, dict]:
    """PnP-RANSAC. Returns w2c (4x4) and diagnostics dict."""
    dist = np.zeros((4, 1))
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_3d.astype(np.float32), points_2d.astype(np.float32),
        K.astype(np.float32), dist, flags=cv2.SOLVEPNP_ITERATIVE,
    )
    diag = {"success": bool(success), "n_inliers": 0, "reproj_err_px": float("inf")}
    if not success:
        return np.eye(4, dtype=np.float32), diag
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(-1)
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    proj, _ = cv2.projectPoints(points_3d.astype(np.float32), rvec, tvec,
                                K.astype(np.float32), dist)
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - points_2d.astype(np.float32), axis=1)
    if inliers is not None and len(inliers):
        diag["n_inliers"] = int(len(inliers))
        diag["reproj_err_px"] = float(err[inliers.flatten()].mean())
    else:
        diag["reproj_err_px"] = float(err.mean())
    diag["mean_err_px"] = float(err.mean())
    return w2c.astype(np.float32), diag


def w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
    return np.linalg.inv(w2c)


def is_valid_rotation(R: np.ndarray, tol: float = 1e-3) -> bool:
    if R.shape != (3, 3):
        return False
    if not np.allclose(np.linalg.det(R), 1.0, atol=tol):
        return False
    return np.allclose(R @ R.T, np.eye(3), atol=tol)


def unproject_cv(uv: np.ndarray, depth: np.ndarray, K: np.ndarray,
                 w2c: np.ndarray) -> np.ndarray:
    """Unproject template-box pixels to world coordinates.

    uv: (..., 2) pixel coords; depth: (...) metric depth; K: 3x3; w2c: 4x4.
    Returns (..., 3) world points.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u, v = uv[..., 0], uv[..., 1]
    x = (u - cx) / fx * depth
    y = (v - cy) / fy * depth
    z = depth
    cam = np.stack([x, y, z], axis=-1)
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    world = np.einsum("ji,...j->...i", R, cam - t)
    return world


def project_world(points_w: np.ndarray, K: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    """Project world points to template-box pixel coords (..., 2)."""
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    cam = np.einsum("ij,...j->...i", R, points_w) + t
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = cam[..., 0] / cam[..., 2] * fx + cx
    v = cam[..., 1] / cam[..., 2] * fy + cy
    return np.stack([u, v], axis=-1)
