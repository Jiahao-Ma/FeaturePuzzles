"""Stage 5: 3D visualization (point clouds + camera frustums).

Exports three scene point clouds (RGB, mask-ID, DINO-PCA) each with the six
crop camera frustums derived from PnP poses, as both high-resolution PNGs
(matplotlib) and interactive GLB scenes (trimesh). The MoGe camera frame is
the world frame.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from pipeline_utils import colorize_id_map, save_png


def _frustum_verts(boxK: np.ndarray, w2c: np.ndarray, tw: int, th: int,
                   scale: float) -> np.ndarray:
    """Return 8 corners of a frustum in world coords (8,3)."""
    fx, fy = boxK[0, 0], boxK[1, 1]
    cx, cy = boxK[0, 2], boxK[1, 2]
    z = scale
    corners = np.array([
        [0, 0, 0],                                   # apex (camera center)
        [(0 - cx) / fx * z, (0 - cy) / fy * z, z],
        [(tw - cx) / fx * z, (0 - cy) / fy * z, z],
        [(tw - cx) / fx * z, (th - cy) / fy * z, z],
        [(0 - cx) / fx * z, (th - cy) / fy * z, z],
    ], dtype=np.float32)
    c2w = np.linalg.inv(w2c)
    R = c2w[:3, :3]; t = c2w[:3, 3]
    world = corners @ R.T + t
    return world


def _frustum_edges(verts: np.ndarray) -> List[Tuple[int, int]]:
    return [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]


def _scene_points(points: np.ndarray, mask: np.ndarray, colors: np.ndarray,
                  step: int) -> Tuple[np.ndarray, np.ndarray]:
    pts = points[::step, ::step].reshape(-1, 3)
    m = mask[::step, ::step].reshape(-1)
    cols = colors[::step, ::step].reshape(-1, 3)
    valid = m & np.all(np.isfinite(pts), axis=1) & (pts[:, 2] > 0)
    return pts[valid], cols[valid]


def _setup_axes(ax, pts: np.ndarray):
    import matplotlib.pyplot as plt
    if pts.shape[0] == 0:
        return
    ctr = pts.mean(axis=0)
    span = (pts.max(axis=0) - pts.min(axis=0)).max() / 2 + 1e-6
    ax.set_xlim(ctr[0] - span, ctr[0] + span)
    ax.set_ylim(ctr[1] - span, ctr[1] + span)
    ax.set_zlim(ctr[2] - span, ctr[2] + span)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])


def _render_scene_png(pts: np.ndarray, cols: np.ndarray, frustums: List[np.ndarray],
                      title: str, path: Path, dpi: int, elev: float = 20,
                      azim: float = -60) -> None:
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    if pts.shape[0] > 0:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols / 255.0, s=0.4,
                   marker=".", alpha=0.8)
    palette = _distinct_palette(len(frustums))
    for i, fv in enumerate(frustums):
        for a, b in _frustum_edges(fv):
            ax.plot([fv[a, 0], fv[b, 0]], [fv[a, 1], fv[b, 1]], [fv[a, 2], fv[b, 2]],
                    color=palette[i], linewidth=1.6)
        ax.scatter([fv[0, 0]], [fv[0, 1]], [fv[0, 2]], color=palette[i], s=40,
                   marker="^", edgecolors="k")
    _setup_axes(ax, pts if pts.shape[0] else np.array(frustums).reshape(-1, 3))
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _distinct_palette(n: int) -> List[Tuple[float, float, float]]:
    import colorsys
    return [colorsys.hsv_to_rgb((i * 0.61803398875) % 1.0, 0.9, 0.95) for i in range(n)]


def _glb_scene(pts: np.ndarray, cols: np.ndarray, frustums: List[np.ndarray],
              path: Path) -> None:
    """Export an interactive GLB via trimesh: point cloud + frustum line sets."""
    import trimesh
    geoms = []
    if pts.shape[0] > 0:
        pc = trimesh.PointCloud(pts, colors=(cols / 255.0))
        geoms.append(pc)
    palette = _distinct_palette(len(frustums))
    for i, fv in enumerate(frustums):
        edges = _frustum_edges(fv)
        segs = np.array([[fv[a], fv[b]] for a, b in edges])
        ls = trimesh.load_path(segs)
        rgba = np.array([int(c * 255) for c in palette[i]] + [255])
        # trimesh requires one color row per entity (group of connected segments)
        ls.colors = np.tile(rgba, (max(len(ls.entities), 1), 1))
        geoms.append(ls)
    scene = trimesh.Scene(geoms)
    scene.export(path)


def stage_visualize_3d(points: np.ndarray, mask: np.ndarray, image_rgb: np.ndarray,
                       id_map: np.ndarray, pca_rgb: np.ndarray, boxK: np.ndarray,
                       w2cs: np.ndarray, tw: int, th: int, cfg: Dict,
                       out_dir: Path) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vc = cfg["visualization"]
    step = int(vc["pointcloud_step"])
    scale = float(vc["frustum_scale"])
    frustums = [_frustum_verts(boxK, w2c, tw, th, scale) for w2c in w2cs]

    # RGB scene
    pts, cols = _scene_points(points, mask, image_rgb, step)
    _render_scene_png(pts, cols, frustums, "Scene (RGB) + PnP cameras",
                      out_dir / "cameras_rgb.png", vc["png_dpi"])
    _glb_scene(pts, cols, frustums, out_dir / "scene_rgb.glb")

    # Mask-ID scene
    id_pal = colorize_id_map(id_map)
    pts_m, cols_m = _scene_points(points, mask, id_pal, step)
    _render_scene_png(pts_m, cols_m, frustums, "Scene (mask ID) + PnP cameras",
                      out_dir / "cameras_mask.png", vc["png_dpi"])
    _glb_scene(pts_m, cols_m, frustums, out_dir / "scene_mask.glb")

    # DINO-PCA scene (parent PCA upsampled to image resolution)
    pca_full = _upsample_pca(pca_rgb, image_rgb.shape[:2])
    pts_f, cols_f = _scene_points(points, mask, pca_full, step)
    _render_scene_png(pts_f, cols_f, frustums, "Scene (DINOv3 PCA) + PnP cameras",
                      out_dir / "cameras_feature.png", vc["png_dpi"])
    _glb_scene(pts_f, cols_f, frustums, out_dir / "scene_feature.glb")
    return {"n_points_rgb": int(pts.shape[0]), "frustum_scale": scale,
            "n_frustums": len(frustums)}


def _upsample_pca(pca_rgb: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    import cv2
    if pca_rgb.shape[:2] == hw:
        return pca_rgb
    return cv2.resize(pca_rgb, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
