"""Stage 3: Puzzles crop sequence + per-crop PnP cameras.

Follows the official Image2Clips geometry:
  1. sample N overlapping crop boxes (IoU in [min,max])
  2. crop parent RGB / point map / valid mask / depth / FastSAM masks
  3. resize every crop to the template box size
  4. calibrate one shared crop intrinsic boxK from the template box
  5. per crop: sample valid 2D-3D correspondences, solvePnPRansac -> w2c
  6. resample crops whose PnP quality checks fail

The main sequence is unaugmented. An optional rotation-augmentation branch
applies the official random camera rotation + point-cloud rasterization to
illustrate geometry augmentation (exported separately, not used as labels).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from geometry_utils import (calibrate_intrinsics_by_pts, estimate_camera_pose_ransac,
                            generate_overlapping_boxes, image_plane_uv,
                            is_valid_rotation, unproject_cv, w2c_to_c2w)
from pipeline_utils import (colorize_id_map, distinct_colors, overlay_id_map,
                            save_npz, save_png)


def _resize_to_template(arr: np.ndarray, tw: int, th: int, interp: int) -> np.ndarray:
    if arr.dtype == bool:
        r = cv2.resize(arr.astype(np.uint8), (tw, th), interpolation=interp)
        return r.astype(bool)
    if arr.ndim == 2:
        return cv2.resize(arr, (tw, th), interpolation=interp)
    if arr.ndim == 3:
        return cv2.resize(arr, (tw, th), interpolation=interp)
    raise ValueError(f"unsupported ndim {arr.ndim}")


def _crop_and_resize(box, parent, tw, th, interp):
    x1, y1, x2, y2 = box
    sub = parent[y1:y2, x1:x2]
    if sub.shape[0] == th and sub.shape[1] == tw:
        return sub
    return _resize_to_template(sub, tw, th, interp)


def _solve_one_crop(box_points_tmpl: np.ndarray, box_mask_tmpl: np.ndarray,
                    boxK: np.ndarray, calib_num_pts: int,
                    rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
    uv = image_plane_uv(box_points_tmpl.shape[1], box_points_tmpl.shape[0])
    uv_ds = uv.reshape(-1, 2)
    pts_ds = box_points_tmpl.reshape(-1, 3)
    m_ds = box_mask_tmpl.reshape(-1)
    valid = m_ds & np.all(np.isfinite(pts_ds), axis=1) & (pts_ds[:, 2] > 0)
    n_valid = int(valid.sum())
    if n_valid < calib_num_pts:
        idx = np.where(valid)[0]
    else:
        idx = rng.choice(np.where(valid)[0], calib_num_pts, replace=False)
    w2c, diag = estimate_camera_pose_ransac(pts_ds[idx], uv_ds[idx], boxK)
    return w2c, diag


def _resample_box(boxes_done, image, rng, cfg_p, sampling="trajectory"):
    pc = cfg_p
    h, w = image.shape[:2]
    short = min(h, w)
    if sampling == "coverage":
        # coverage mode: jitter the failed box position while keeping its size
        # so the tiling still covers the image.
        s = int(round(float(rng.uniform(pc["min_box_size_scale"],
                                        pc["max_box_size_scale"])) * short))
        s = max(8, s)
        for _ in range(int(pc["max_resample_tries"])):
            x1 = int(rng.integers(0, max(1, w - s)))
            y1 = int(rng.integers(0, max(1, h - s)))
            return (x1, y1, x1 + s, y1 + s)
    # Consecutive-trajectory resample: only the previous accepted crop must
    # satisfy the IoU range, so a failed PnP crop can still be replaced.
    prev = [boxes_done[-1]] if boxes_done else []
    for _ in range(int(pc["max_resample_tries"])):
        s = int(round(float(rng.uniform(pc["min_box_size_scale"],
                                        pc["max_box_size_scale"])) * short))
        if s < 8:
            continue
        x1 = int(rng.integers(0, max(1, w - s)))
        y1 = int(rng.integers(0, max(1, h - s)))
        box = (x1, y1, x1 + s, y1 + s)
        if all(pc["min_overlap"] <= _iou(box, b) <= pc["max_overlap"] for b in prev):
            return box
    raise RuntimeError("resample budget exhausted")


def _iou(a, b):
    from geometry_utils import calculate_iou
    return calculate_iou(a, b)


def stage_puzzles(image_rgb: np.ndarray, points: np.ndarray, valid_mask: np.ndarray,
                  depth: np.ndarray, id_map: np.ndarray, cfg: Dict, out_dir: Path,
                  rng: np.random.Generator) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pc = cfg["puzzles"]
    n_crops = int(pc["n_crops"])
    tw_th = int(pc["template_long"])
    sampling = pc.get("sampling", "coverage")
    if sampling == "coverage":
        from geometry_utils import generate_coverage_boxes
        boxes, template_box = generate_coverage_boxes(
            image_rgb, n_crops, pc["min_box_size_scale"], pc["max_box_size_scale"],
            rng, overlap_pad=float(pc.get("overlap_pad", 0.1)),
            coverage_target=float(pc.get("coverage_target", 0.9)),
            coverage_jitter=float(pc.get("coverage_jitter", 0.15)))
    else:
        boxes, template_box = generate_overlapping_boxes(
            image_rgb, n_crops, pc["min_box_size_scale"], pc["max_box_size_scale"],
            pc["min_overlap"], pc["max_overlap"], rng)
    bw = template_box[2] - template_box[0]
    bh = template_box[3] - template_box[1]
    tw, th = bw, bh
    boxK = calibrate_intrinsics_by_pts(template_box, points).astype(np.float32)

    crops_rgb, crops_id, crops_mask, crops_depth, crops_points = [], [], [], [], []
    w2cs, c2ws, diagnostics = [], [], []
    final_boxes: List[Tuple[int, int, int, int]] = []
    pnp = pc["pnp"]
    calib_num_pts = int(pc["calib_num_pts"])
    for i in range(n_crops):
        box = boxes[i]
        for _ in range(int(pc["max_resample_tries"])):
            crgb = _crop_and_resize(box, image_rgb, tw, th, cv2.INTER_LINEAR)
            cpts = _crop_and_resize(box, points, tw, th, cv2.INTER_NEAREST)
            cmask = _crop_and_resize(box, valid_mask, tw, th, cv2.INTER_NEAREST).astype(bool)
            cdepth = _crop_and_resize(box, depth, tw, th, cv2.INTER_NEAREST)
            cid = _crop_and_resize(box, id_map, tw, th, cv2.INTER_NEAREST)
            w2c, diag = _solve_one_crop(cpts, cmask, boxK, calib_num_pts, rng)
            ok = (diag["success"] and diag["n_inliers"] >= pnp["min_inliers"]
                  and diag["reproj_err_px"] <= pnp["max_reproj_err_px"]
                  and is_valid_rotation(w2c[:3, :3]))
            if ok:
                break
            new_box = _resample_box(final_boxes, image_rgb, rng, pc, sampling=sampling)
            box = new_box
        if not ok:
            raise RuntimeError(f"crop {i}: PnP failed after resample budget "
                               f"(inliers={diag['n_inliers']}, "
                               f"err={diag['reproj_err_px']:.3f}px)")
        final_boxes.append(box)
        crops_rgb.append(crgb)
        crops_id.append(cid)
        crops_mask.append(cmask)
        crops_depth.append(cdepth)
        crops_points.append(cpts)
        w2cs.append(w2c)
        c2ws.append(w2c_to_c2w(w2c))
        diagnostics.append(diag)

    crops_rgb = np.stack(crops_rgb)        # (N,th,tw,3)
    crops_id = np.stack(crops_id)          # (N,th,tw)
    crops_mask = np.stack(crops_mask)      # (N,th,tw)
    crops_depth = np.stack(crops_depth)    # (N,th,tw)
    crops_points = np.stack(crops_points)  # (N,th,tw,3)
    w2cs = np.stack(w2cs)                  # (N,4,4)
    c2ws = np.stack(c2ws)                  # (N,4,4)

    save_npz({"boxes": np.array(final_boxes, dtype=np.int32),
              "template_box": np.array(template_box, dtype=np.int32),
              "boxK": boxK, "w2c": w2cs, "c2w": c2ws,
              "crops_rgb": crops_rgb, "crops_id": crops_id,
              "crops_mask": crops_mask, "crops_depth": crops_depth,
              "crops_points": crops_points}, out_dir / "sequence.npz")
    (out_dir / "crops").mkdir(exist_ok=True)
    (out_dir / "masks").mkdir(exist_ok=True)
    (out_dir / "poses").mkdir(exist_ok=True)
    palette = distinct_colors(int(id_map.max()) + 1 if id_map.size else 1, seed=1)
    for i in range(n_crops):
        save_png(crops_rgb[i], out_dir / "crops" / f"crop_{i:02d}.png")
        save_png(overlay_id_map(crops_rgb[i], crops_id[i], alpha=0.5, palette=palette),
                 out_dir / "masks" / f"mask_{i:02d}.png")
        np.savetxt(out_dir / "poses" / f"w2c_{i:02d}.txt", w2cs[i])
    save_png(_strip(crops_rgb), out_dir / "sequence_strip.png")
    save_png(_draw_crop_regions(image_rgb, final_boxes), out_dir / "crop_regions.png")

    result = {"boxes": [list(b) for b in final_boxes], "template_box": list(template_box),
              "boxK": boxK.tolist(), "w2c": w2cs.tolist(), "c2w": c2ws.tolist(),
              "diagnostics": diagnostics, "template_size": [th, tw]}

    if cfg["puzzles"]["augmentation"]["enabled"]:
        try:
            aug = _build_augmentation(image_rgb, points, valid_mask, image_rgb.shape,
                                      final_boxes, boxK, w2cs, tw, th, rng, cfg, out_dir)
            result["augmentation"] = aug
        except Exception as e:
            result["augmentation_error"] = str(e)
    return result


def _strip(crops: np.ndarray) -> np.ndarray:
    return np.concatenate(list(crops), axis=1)


def _draw_crop_regions(image: np.ndarray, boxes) -> np.ndarray:
    """Visualize every crop box on the parent image: semi-transparent numbered
    fills + solid outlines, mirroring the official Image2Clips region overlay."""
    import cv2
    out = image.copy()
    overlay = image.copy()
    palette = distinct_colors(len(boxes), seed=3)
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        color = tuple(int(c) for c in palette[i])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        color = tuple(int(c) for c in palette[i])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        cv2.putText(out, f"#{i}", (x1 + 4, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2, cv2.LINE_AA)
    return out


def _build_augmentation(image_rgb, points, valid_mask, shape, boxes, boxK, w2cs,
                        tw, th, rng, cfg, out_dir):
    """Optional rotation augmentation: rotate each crop camera around the parent
    point-cloud centroid and re-render RGB/depth via z-buffer rasterization."""
    import trimesh
    from visualization_helpers import rasterize_views
    ac = cfg["puzzles"]["augmentation"]
    aug_dir = out_dir.parent / "06_augmentation_comparison"
    aug_dir.mkdir(parents=True, exist_ok=True)
    pts = points[valid_mask].reshape(-1, 3)
    colors = image_rgb[valid_mask].reshape(-1, 3)
    centroid = pts.mean(axis=0)
    rot_w2cs = []
    for i, w2c in enumerate(w2cs):
        if rng.random() > ac["rotate_ratio"]:
            rot_w2cs.append(w2c)
            continue
        angle = float(rng.uniform(0, ac["max_rotation_angle"]))
        axis = rng.normal(0, 1, 3)
        axis /= np.linalg.norm(axis) + 1e-9
        ca, sa = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
        R = (ca * np.eye(3) + sa * _cross(axis) + (1 - ca) * np.outer(axis, axis))
        c2w = w2c_to_c2w(w2c)
        # rotate camera position around centroid, keep looking at centroid
        cam_pos = c2w[:3, 3]
        new_pos = centroid + R @ (cam_pos - centroid)
        z = centroid - new_pos
        z /= np.linalg.norm(z) + 1e-9
        up = np.array([0, 0, 1.0])
        x = np.cross(up, z); x /= np.linalg.norm(x) + 1e-9
        y = np.cross(z, x)
        Rc = np.stack([x, y, z], axis=0)
        tc = -Rc @ new_pos
        new_w2c = np.eye(4, dtype=np.float32)
        new_w2c[:3, :3] = Rc
        new_w2c[:3, 3] = tc
        rot_w2cs.append(new_w2c)
    renders = rasterize_views(pts, colors, boxK, rot_w2cs, tw, th)
    for i, r in enumerate(renders):
        save_png(r, aug_dir / f"rot_crop_{i:02d}.png")
    save_png(np.concatenate(renders, axis=1), aug_dir / "rot_sequence_strip.png")
    return {"n_rotated": int(sum(1 for a, b in zip(rot_w2cs, w2cs) if not np.allclose(a, b))),
            "w2c": [w.tolist() for w in rot_w2cs]}


def _cross(axis):
    return np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                     [-axis[1], axis[0], 0]])
