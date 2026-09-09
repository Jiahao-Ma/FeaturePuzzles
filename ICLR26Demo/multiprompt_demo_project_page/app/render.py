"""Point-cloud assembly, offline renders and the mask-overlay video.

Reuses the LingBot octree/renderer stack through the static reference module so
the demo's videos are produced by the same code as the paper figures. The only
addition is that renders are requested per camera mode and then handed to
``compose.py`` for the picture-in-picture layout.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from . import paths
from .static_ref import ref


def build_point_cloud(geo: Dict[str, np.ndarray], pred_masks: np.ndarray,
                      obj_rgb: np.ndarray, base_cfg, device: str = "cuda", log=print):
    """Unproject every view and colour points by the object mask covering them."""
    return ref().collect_points_and_colors(
        geo, pred_masks, obj_rgb, base_cfg.scene.max_depth,
        base_cfg.scene.downsample, base_cfg.scene.jitter, device=device, log=log)


def make_base_config(fps: float, width: int = 1920, height: int = 1080,
                     config_path: Optional[Path] = None,
                     median_depth: Optional[float] = None):
    """Render config, with the built-in side-by-side output disabled -- this
    demo does its own compositing.

    ``max_depth`` is rescaled to the scene when a median depth is supplied: the
    preset's fixed 10 m cull is right for a ScanNet room but throws away most of
    an outdoor scene, and keeps far outliers in a small one.
    """
    from rgbd_render.config import PipelineConfig
    cfg = PipelineConfig.from_yaml(str(config_path or paths.RENDER_CONFIG))
    cfg.fast_review = 0
    cfg.frame_stride = 1
    cfg.skip_first = 0
    cfg.render.width = width
    cfg.render.height = height
    cfg.render.combined_video = False
    cfg.overlay.camera_vis = "default"
    cfg.overlay.frame_tag = False
    cfg.fps = fps
    if median_depth and median_depth > 0:
        cfg.scene.max_depth = float(np.clip(4.0 * median_depth, 2.0, 60.0))
    return cfg


# Follow-camera placement, in multiples of the scene's median depth. Shared with
# the viser viewer so the interactive view matches the rendered one.
FOLLOW_BACK, FOLLOW_UP, FOLLOW_LOOK = 0.55, 0.15, 0.90


def follow_offsets(c2w: np.ndarray, median_depth: float) -> Dict[str, float]:
    """Convert the metric offsets above into the trajectory-relative fractions
    ``make_follow_path`` multiplies by."""
    cam = c2w[:, :3, 3]
    scene_scale = max(float(np.linalg.norm(cam.max(0) - cam.min(0))), 1e-3)
    return {
        "back_offset": FOLLOW_BACK * median_depth / scene_scale,
        "up_offset": FOLLOW_UP * median_depth / scene_scale,
        "look_offset": FOLLOW_LOOK * median_depth / scene_scale,
        "smooth_window": int(np.clip(len(c2w) // 3, 3, 30)),
        "scene_scale": scene_scale,
    }


def _median_depth(geo: Dict[str, np.ndarray]) -> float:
    d = geo["depth"]
    valid = d[d > 0]
    return float(np.median(valid)) if valid.size else 1.0


# How much wider than the capture camera's footprint the follow view should be.
FOLLOW_MARGIN = 1.25


def follow_fov(fy: float, image_h: int, median_depth: float,
               max_fov: float = 60.0) -> float:
    """Vertical FOV that keeps the captured content filling the follow view.

    The follow camera sits ``FOLLOW_BACK * median_depth`` behind the capture
    camera, so what that camera framed edge-to-edge is now further away and
    shrinks. Matching the FOV to the pulled-back distance undoes that, leaving
    ``FOLLOW_MARGIN`` of context around the original frame.
    """
    if not np.isfinite(fy) or fy <= 0 or median_depth <= 0:
        return max_fov
    half_h = median_depth * (image_h / 2.0) / fy
    dist = median_depth * (1.0 + FOLLOW_BACK)
    fov = 2.0 * np.degrees(np.arctan(FOLLOW_MARGIN * half_h / dist))
    return float(np.clip(fov, 20.0, max_fov))


def adaptive_camera_configs(geo: Dict[str, np.ndarray], fov: float):
    """Follow/bird's-eye configs sized to the scene rather than the trajectory.

    ``make_follow_path`` multiplies its offsets by the camera-trajectory
    diagonal. That is a good proxy for scene size indoors, where a scan walks a
    few metres through a few-metre-deep room, but not for a video that travels
    tens of metres through a shallow scene -- there it parks the camera far
    enough back that the content becomes a speck. The offsets below are chosen
    in metres relative to scene depth and then converted into the fractions the
    renderer expects, which reproduces the tuned indoor values for ScanNet.

    The follow view also narrows its FOV to compensate for standing further back
    than the camera that captured the points; the bird's-eye view keeps the
    config's FOV because it frames the cloud itself.
    """
    from rgbd_render.config import CameraConfig, CameraSegment

    median = _median_depth(geo)
    o = follow_offsets(geo["c2w"], median)
    f_fov = follow_fov(float(geo["K"][0][1, 1]), int(geo["images"].shape[1]), median, fov)
    follow = CameraConfig(fov=f_fov, transition=30, segments=[CameraSegment(
        mode="follow", frames=[0, -1],
        back_offset=o["back_offset"], up_offset=o["up_offset"],
        look_offset=o["look_offset"],
        # Long smoothing windows flatten the heading on short clips.
        smooth_window=o["smooth_window"],
        # Keep follow_scale == scene_scale; the offsets already encode the size.
        scale_frames=10 ** 9,
    )])
    birdeye = CameraConfig(fov=fov, transition=30, segments=[CameraSegment(
        mode="birdeye", frames=[0, -1], reveal_height_mult=2.5)])
    return {"follow": follow, "birdeye": birdeye}


def render_views(geo: Dict[str, np.ndarray], pred_masks: np.ndarray,
                 obj_rgb: np.ndarray, out_dir: Path, name: str,
                 mode: str = "hl", cam_modes: Tuple[str, ...] = ("follow", "birdeye"),
                 fps: float = 10.0, width: int = 1920, height: int = 1080,
                 device: str = "cuda", log=print) -> Dict[str, Path]:
    """Render one colour mode from each requested camera and return the paths.

    Args:
        mode: ``hl`` keeps every point and tints the masked ones (best for the
            demo), ``seg`` drops unmasked points, ``rgb`` ignores masks.
    """
    R = ref()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_base_config(fps, width, height, median_depth=_median_depth(geo))

    xyz, rgb_pts, seg_col, is_m, frames = build_point_cloud(
        geo, pred_masks, obj_rgb, cfg, device=device, log=log)

    if mode == "rgb":
        pts, cols, fr = xyz, rgb_pts, frames
    elif mode == "seg":
        if not is_m.any():
            raise RuntimeError("no masked points: nothing to render in 'seg' mode")
        pts, cols, fr = xyz[is_m], seg_col[is_m], frames[is_m]
    else:
        pts, cols, fr = xyz, seg_col, frames

    oct_ = R.build_octree(pts, cols, fr, cfg.scene.octree_level, log=log)
    c2w, Ks, images = geo["c2w"], geo["K"], geo["images"]
    scene_scale = float(np.linalg.norm(c2w[:, :3, 3].max(0) - c2w[:, :3, 3].min(0))) or 1.0
    scene = R.make_scene(oct_, c2w, Ks, images, scene_scale)

    cam_cfgs = adaptive_camera_configs(geo, cfg.camera.fov)
    outputs: Dict[str, Path] = {}
    try:
        for cam in cam_modes:
            if cam not in cam_cfgs:
                log(f"[render] unknown camera mode {cam!r}, skipping")
                continue
            dst = out_dir / f"{name}_{mode}_{cam}.mp4"
            t0 = time.time()
            R.render_combo(scene, cam_cfgs[cam], cfg, str(dst), log=log)
            log(f"[render] {cam} done in {time.time()-t0:.1f}s -> {dst.name}")
            outputs[cam] = dst
    finally:
        scene.destroy()
    return outputs


def mask_overlay_video(images: np.ndarray, pred_masks: np.ndarray,
                       obj_bgr: np.ndarray, out_path: Path, fps: float = 10.0,
                       log=print) -> Path:
    ref().encode_mask_overlay_video(images, pred_masks, obj_bgr, out_path,
                                    fps=fps, log=log)
    return out_path


def export_point_cloud(geo: Dict[str, np.ndarray], pred_masks: np.ndarray,
                       obj_rgb: np.ndarray, out_path: Path,
                       max_points_per_frame: int = 60_000, log=print) -> Path:
    """Save a per-frame point cloud for the viser viewer.

    Stored flat with a frame index per point so the viewer can either show one
    frame at a time (4D) or accumulate them (3D).
    """
    images, depth, c2w, Ks = geo["images"], geo["depth"], geo["c2w"], geo["K"]
    S, H, W = depth.shape
    K_obj = pred_masks.shape[1]

    obj_idx = np.full((S, H, W), -1, np.int32)
    for v in range(S):
        for k in range(K_obj):
            m = pred_masks[v, k] > 0
            obj_idx[v][m & (obj_idx[v] == -1)] = k

    xyz_l, rgb_l, seg_l, fr_l, oid_l = [], [], [], [], []
    rng = np.random.default_rng(0)
    for v in range(S):
        d = depth[v]
        valid = d > 0
        rows, cols = np.where(valid)
        if rows.size == 0:
            continue
        if rows.size > max_points_per_frame:
            sel = rng.choice(rows.size, max_points_per_frame, replace=False)
            rows, cols = rows[sel], cols[sel]
        z = d[rows, cols]
        K = Ks[v]
        x = (cols - K[0, 2]) * z / K[0, 0]
        y = (rows - K[1, 2]) * z / K[1, 1]
        cam = np.stack([x, y, z], -1)
        world = cam @ c2w[v][:3, :3].T + c2w[v][:3, 3]
        rgb = images[v][rows, cols].astype(np.float32) / 255.0
        oid = obj_idx[v][rows, cols]
        seg = rgb.copy()
        hit = oid >= 0
        if hit.any():
            seg[hit] = obj_rgb[oid[hit]]
        xyz_l.append(world.astype(np.float32)); rgb_l.append(rgb.astype(np.float32))
        seg_l.append(seg.astype(np.float32)); oid_l.append(oid.astype(np.int16))
        fr_l.append(np.full(rows.size, v, np.int32))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        xyz=np.concatenate(xyz_l), rgb=np.concatenate(rgb_l),
        seg=np.concatenate(seg_l), obj=np.concatenate(oid_l),
        frame=np.concatenate(fr_l), c2w=c2w.astype(np.float32),
        K=Ks.astype(np.float32), obj_rgb=obj_rgb.astype(np.float32),
        image_hw=np.array([H, W], np.int32),
    )
    log(f"[export] point cloud -> {out_path.name} "
        f"({sum(len(a) for a in xyz_l):,} points, {S} frames)")
    return out_path
