"""Posed RGB-D loaders for ScanNet, ScanNet++, FARM-Scenes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

SCANNET_25K = Path("/mnt/dataset/marko/puzzles/scannet_frames/scannet_frames_25k")
SCANNETPP = Path("/mnt/dataset/marko/puzzles/embodiedsplat_dl/dataset/scannetpp/data")
FARM = Path("/mnt/dataset/marko/puzzles/FARM-Scenes/scenes")


def load_depth(path, scale=0.001):
    d = np.array(Image.open(path))
    if np.issubdtype(d.dtype, np.integer):
        return d.astype(np.float32) * scale
    return d.astype(np.float32)


def unproject(depth, c2w, K, gl=False):
    fx, fy, cx, cy = K
    ys, xs = np.nonzero(depth > 0)
    z = depth[ys, xs]
    x = (xs - cx) * z / fx
    if gl:
        y = -(ys - cy) * z / fy
        zc = -z
    else:
        y = (ys - cy) * z / fy
        zc = z
    pts = np.stack([x, y, zc, np.ones_like(z)], 0)
    return (c2w @ pts).T[:, :3], ys, xs


def _read_K4(path):
    a = np.loadtxt(path).reshape(4, 4)
    return float(a[0, 0]), float(a[1, 1]), float(a[0, 2]), float(a[1, 2])


def index_scannet(scene, stride=1, max_frames=0, camera=None):
    root = SCANNET_25K / scene
    color = sorted((root / "color").glob("*.jpg"))
    if not color:
        color = sorted((root / "color").glob("*.png"))
    fx, fy, cx, cy = _read_K4(root / "intrinsics_color.txt")
    frames = []
    for img_p in color[::max(1, stride)]:
        stem = img_p.stem
        dep = root / "depth" / f"{stem}.png"
        pose = root / "pose" / f"{stem}.txt"
        if not (dep.is_file() and pose.is_file()):
            continue
        c2w = np.loadtxt(pose).reshape(4, 4)
        if not np.isfinite(c2w).all():
            continue
        frames.append(dict(
            stem=f"{scene}_{stem}", img=img_p, depth_path=dep, c2w=c2w,
            K=(fx, fy, cx, cy), gl=False, depth_scale=0.001,
            dataset="scannet", scene=scene))
        if max_frames and len(frames) >= max_frames:
            break
    return frames


def load_scannet(scene, stride=1, max_frames=0, camera=None):
    return [read_frame(m) for m in index_scannet(scene, stride, max_frames, camera)]


def index_scannetpp(scene, stride=12, max_frames=0, camera=None):
    scene_dir = SCANNETPP / scene
    t = json.loads((scene_dir / "dslr/nerfstudio/transforms_undistorted.json").read_text())
    fx, fy, cx, cy = t["fl_x"], t["fl_y"], t["cx"], t["cy"]
    w, h = t["w"], t["h"]
    sel = [f for f in t["frames"] if not f.get("is_bad", False)][::max(1, stride)]
    if max_frames:
        sel = sel[:max_frames]
    frames = []
    for fr in sel:
        stem = Path(fr["file_path"]).stem
        img = scene_dir / "dslr/undistorted_images" / f"{stem}.JPG"
        dep = scene_dir / "dslr/undistorted_depths" / f"{stem}.png"
        if not (img.is_file() and dep.is_file()):
            continue
        frames.append(dict(
            stem=stem, img=img, depth_path=dep, depth_scale=0.001,
            c2w=np.array(fr["transform_matrix"], np.float64),
            K=(fx, fy, cx, cy), gl=True, W=w, H=h,
            dataset="scannetpp", scene=scene, scene_dir=scene_dir))
    return frames


def load_scannetpp(scene, stride=12, max_frames=0, camera=None):
    return [read_frame(m) for m in index_scannetpp(scene, stride, max_frames, camera)]


def index_farm(scene, stride=8, max_frames=0, camera=None):
    """scene like 'odin1/SDH4and7_0413' or 'grandtour/2024-11-25_warehouse'."""
    root = FARM / scene
    meta = json.loads((root / "frames.json").read_text())
    scale = float((meta.get("depth_encoding") or {}).get("scale_to_metres", 0.001))
    cams = meta.get("cameras") or []
    cam = camera or (cams[0] if cams else None)
    frames = []
    for fr in meta["frames"]:
        if cam and fr.get("camera") != cam:
            continue
        frames.append(fr)
    frames = frames[::max(1, stride)]
    if max_frames:
        frames = frames[:max_frames]
    out = []
    for fr in frames:
        img_p = root / fr["rgb_path"]
        dep_p = root / fr["depth_path"]
        if not (img_p.is_file() and dep_p.is_file()):
            continue
        K = np.asarray(fr["K"], np.float64)
        out.append(dict(
            stem=f"{fr.get('camera','cam')}_{fr['frame_id']}",
            img=img_p, depth_path=dep_p, depth_scale=scale,
            c2w=np.asarray(fr["T_world_cam"], np.float64),
            K=(float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])),
            gl=False, dataset="farm", scene=scene, scene_dir=root))
    return out


def load_farm(scene, stride=8, max_frames=0, camera=None):
    return [read_frame(m) for m in index_farm(scene, stride, max_frames, camera)]


def read_frame(meta):
    """Load RGB + depth for an indexed frame dict (in place + return)."""
    if meta.get("rgb") is not None and meta.get("depth") is not None:
        return meta
    im = Image.open(meta["img"]).convert("RGB")
    rgb = np.asarray(im)
    H, W = rgb.shape[:2]
    d = load_depth(meta["depth_path"], meta.get("depth_scale", 0.001))
    if d.shape[:2] != (H, W):
        import cv2
        d = cv2.resize(d.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    meta["rgb"] = rgb
    meta["depth"] = d
    meta["W"] = W
    meta["H"] = H
    return meta


def index_scene(dataset, scene, **kw):
    if dataset == "scannet":
        return index_scannet(scene, **kw)
    if dataset == "scannetpp":
        return index_scannetpp(scene, **kw)
    if dataset in ("farm", "farm_scenes", "farm-scenes"):
        return index_farm(scene, **kw)
    raise ValueError(dataset)


def _cam_forward(c2w, gl=False):
    f = np.asarray(c2w, dtype=np.float64)[:3, 2]
    f = -f if gl else f
    n = np.linalg.norm(f)
    return f / max(n, 1e-9)


def pick_nearby_window(metas, n=6, min_travel=0.08, max_angle_deg=30.0, max_step=None):
    """Consecutive n frames with overlapping FOV (small translation *and* yaw/pitch).

    ScanNet 25k is already ~every 100 frames, so the densest window is still
    ~0.5 m/step — that is the closest this dump has. Rotation is used to skip
    windows that glance at the ceiling or turn a corner between frames.
    """
    if len(metas) <= n:
        return 0, dict(n=len(metas), travel=None, max_step=None, reason="short")
    P = np.stack([np.asarray(m["c2w"])[:3, 3] for m in metas], 0)
    F = np.stack([_cam_forward(m["c2w"], m.get("gl", False)) for m in metas], 0)

    def stats(i):
        dd = np.linalg.norm(np.diff(P[i:i + n], axis=0), axis=1)
        dots = np.clip((F[i:i + n - 1] * F[i + 1:i + n]).sum(1), -1.0, 1.0)
        ang = np.degrees(np.arccos(dots))
        travel, mx = float(dd.sum()), float(dd.max())
        mean_ang, max_ang = float(ang.mean()), float(ang.max())
        cost = travel * (1.0 + mean_ang / 25.0) + 0.12 * max_ang
        return dict(travel=travel, max_step=mx, mean_ang=mean_ang, max_ang=max_ang, cost=cost)

    tight, loose = None, None
    for i in range(len(P) - n + 1):
        s = stats(i)
        if s["travel"] < min_travel:
            continue
        if max_step is not None and s["max_step"] > max_step:
            continue
        rec = (s["cost"], i, s)
        if loose is None or rec[0] < loose[0]:
            loose = rec
        if s["max_ang"] <= max_angle_deg:
            if tight is None or rec[0] < tight[0]:
                tight = rec
    used = tight if tight is not None else loose
    if used is None:
        return 0, dict(n=n, reason="none")
    i0, s = used[1], used[2]
    return i0, dict(n=n, reason=("overlap" if tight is not None else "min_cost"),
                    travel=s["travel"], max_step=s["max_step"],
                    mean_ang=s["mean_ang"], max_ang=s["max_ang"],
                    start_stem=metas[i0]["stem"], end_stem=metas[i0 + n - 1]["stem"])


def load_scene(dataset, scene, **kw):
    if dataset == "scannet":
        return load_scannet(scene, **kw)
    if dataset == "scannetpp":
        return load_scannetpp(scene, **kw)
    if dataset in ("farm", "farm_scenes", "farm-scenes"):
        return load_farm(scene, **kw)
    raise ValueError(dataset)
