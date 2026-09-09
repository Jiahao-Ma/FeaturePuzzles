"""End-to-end VGGT-Omega pipeline for a single (long) video.

Pipeline (all stages share the *same* 256x256 padded-square frames, so depth,
camera and segmentation masks are pixel-aligned):

  1. Sample N frames from an mp4.
  2. Reconstruction  : full VGGTOmega (clean backbone) -> camera extrinsics /
                       intrinsics + depth -> unproject to a world point cloud.
  3. Segmentation    : VGGTOmegaBackbone (+LoRA) + VGGTOmegaSegmentationHead with
                       a color-seeded prompt mask on the reference view, then
                       propagated to every view.
  4. Visualisation   : 2D overlay grids/frames  +  3D point cloud (matplotlib
                       renders + .glb + .npz) with the segmented object highlighted.

This is the "long-video promptable segmentation + depth post-processing"
pipeline: 2D promptable masks from the seg_omega_lora_multiprompt model are
fused with the VGGT-Omega depth-derived world point cloud, so the segmented
object is lifted into 3D and highlighted across all views.

Example:
  python video_pipeline/video_omega_seg_pipeline.py \
      --video examples/your_video.mp4 \
      --object blue --out_dir out/video_seg/your_video
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401  (3d backend)
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))                                   # lib.py
sys.path.insert(0, str(PROJECT_ROOT / "src"))                   # models.*
sys.path.insert(0, str(PROJECT_ROOT / "repo" / "vggt-omega"))   # vggt_omega package

from lib import (  # noqa: E402
    build_omega_full,
    forward_omega_aggregator,
    forward_omega_seg,
    overlay_mask,
    pad_image_square,
)
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402


# Color presets -> HSV ranges (OpenCV H in [0,180], S/V in [0,255])
COLOR_PRESETS = {
    "blue": [((80, 60, 50), (140, 255, 255))],
    "red": [((0, 70, 40), (14, 255, 255)), ((160, 70, 40), (180, 255, 255))],
}


def extract_frames(video_path: Path, n_frames: int) -> Tuple[List[Image.Image], List[int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = [int(round(i * (total - 1) / (n_frames - 1))) for i in range(n_frames)]
    frames: List[Image.Image] = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, bgr = cap.read()
        if not ok:
            continue
        frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames, idxs[: len(frames)]


def preprocess(frames: List[Image.Image], target: int):
    tensors, padded_uint8, bboxes = [], [], []
    for im in frames:
        canvas, coords, _ = pad_image_square(im, target=target)
        arr = np.array(canvas, copy=True)
        padded_uint8.append(arr)
        bboxes.append(coords)
        tensors.append(torch.from_numpy(arr).float().div(255.0).permute(2, 0, 1))
    return torch.stack(tensors).unsqueeze(0), padded_uint8, bboxes  # (1,N,3,t,t)


def _largest_color_blob(rgb: np.ndarray, color: str, min_area: int = 20):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for lo, hi in COLOR_PRESETS[color]:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        raise RuntimeError(f"No '{color}' pixels found for prompt.")
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[best, cv2.CC_STAT_AREA] < min_area:
        raise RuntimeError(f"Largest '{color}' blob too small ({stats[best, cv2.CC_STAT_AREA]} px).")
    blob = (labels == best).astype(np.uint8)
    x, y, w, h, _ = stats[best]
    return blob, (x, y, x + w, y + h)


def color_prompt_mask(rgb: np.ndarray, color: str, min_area: int = 20) -> np.ndarray:
    """Color blob -> GrabCut -> full-object prompt mask (convex-hull fallback)."""
    H, W = rgb.shape[:2]
    blob, (x1, y1, x2, y2) = _largest_color_blob(rgb, color, min_area)
    bw, bh = x2 - x1, y2 - y1
    mx = max(4, int(round(0.30 * bw)))
    my_top = max(6, int(round(1.10 * bh)))
    my_bot = max(4, int(round(0.40 * bh)))
    rx1, ry1 = max(0, x1 - mx), max(0, y1 - my_top)
    rx2, ry2 = min(W, x2 + mx), min(H, y2 + my_bot)

    def _convex_fallback() -> np.ndarray:
        m = np.zeros((H, W), np.uint8)
        cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pts = np.concatenate(cnts, axis=0)
        cv2.fillConvexPoly(m, cv2.convexHull(pts), 1)
        k = max(3, (min(bw, bh) // 2) | 1)
        return cv2.dilate(m, np.ones((k, k), np.uint8), iterations=1).astype(np.float32)

    try:
        gc = np.zeros((H, W), np.uint8)
        gc[ry1:ry2, rx1:rx2] = cv2.GC_PR_BGD
        gc[blob.astype(bool)] = cv2.GC_FGD
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(rgb, gc, (rx1, ry1, rx2 - rx1, ry2 - ry1), bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
        fg = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
        keep_labels = set(np.unique(lab[blob.astype(bool)])) - {0}
        obj = np.isin(lab, list(keep_labels)).astype(np.uint8) if keep_labels else fg
        obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        area = int(obj.sum())
        if area < max(min_area, blob.sum()) or area > 0.6 * (ry2 - ry1) * (rx2 - rx1):
            return _convex_fallback()
        return obj.astype(np.float32)
    except Exception:
        return _convex_fallback()


def unproject(depth: np.ndarray, extr: np.ndarray, intr: np.ndarray) -> np.ndarray:
    """depth (N,H,W), extr (N,3,4), intr (N,3,3) -> world points (N,H,W,3)."""
    N, H, W = depth.shape
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    x = np.broadcast_to(x[None], (N, H, W))
    y = np.broadcast_to(y[None], (N, H, W))
    fx = intr[:, 0, 0][:, None, None]; fy = intr[:, 1, 1][:, None, None]
    cx = intr[:, 0, 2][:, None, None]; cy = intr[:, 1, 2][:, None, None]
    cam = np.stack([(x - cx) / fx * depth, (y - cy) / fy * depth, depth], axis=-1)
    R = extr[:, :3, :3]; t = extr[:, :3, 3]
    return np.einsum("sij,shwj->shwi", np.transpose(R, (0, 2, 1)), cam - t[:, None, None, :])


@torch.no_grad()
def reconstruct(imgs: torch.Tensor, ckpt: Path, device: torch.device):
    """Full VGGTOmega (clean) -> extrinsics, intrinsics, depth, depth_conf."""
    from vggt_omega.models import VGGTOmega
    model = VGGTOmega(enable_camera=True, enable_depth=True, enable_alignment=True).eval()
    state = torch.load(str(ckpt), map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.to(device)
    with torch.inference_mode():
        pred = model(imgs.to(device))
    H, W = pred["images"].shape[-2:]
    extr, intr = encoding_to_camera(pred["pose_enc"], (H, W))
    out = {
        "extrinsic": extr[0].float().cpu().numpy(),
        "intrinsic": intr[0].float().cpu().numpy(),
        "depth": pred["depth"][0, ..., 0].float().cpu().numpy(),
        "depth_conf": pred["depth_conf"][0].float().cpu().numpy(),
    }
    out["world_points"] = unproject(out["depth"], out["extrinsic"], out["intrinsic"])
    del model; torch.cuda.empty_cache()
    return out


@torch.no_grad()
def segment(imgs, omega_ckpt, seg_ckpt, prompt_mask, ref_view, target, device, thr=0.5):
    backbone, head = build_omega_full(Path(omega_ckpt), Path(seg_ckpt), device)
    tokens = forward_omega_aggregator(backbone, imgs.to(device))
    N = imgs.shape[1]
    pm = torch.zeros(1, N, 1, target, target, dtype=torch.float32, device=device)
    pm[0, ref_view, 0] = torch.from_numpy(prompt_mask).to(device)
    pv = torch.tensor([[ref_view]], dtype=torch.long, device=device)
    seg = forward_omega_seg(head, tokens, pm, pv, target, target)
    logits = seg["segmentation_logits"][0, :, 0].float()
    masks = (logits.sigmoid() > thr).cpu().numpy().astype(np.uint8)
    iou = (seg["iou_pred"][0, :, 0].float().cpu().numpy()
           if seg.get("iou_pred") is not None else None)
    del backbone, head, tokens; torch.cuda.empty_cache()
    return masks, iou


def save_2d(padded, prompt_mask, pred_masks, iou, ref_view, out_dir, title):
    N = len(padded)
    fig, axes = plt.subplots(2, N, figsize=(3.4 * N, 6.9))
    if N == 1:
        axes = axes[:, None]
    for v in range(N):
        bgr = cv2.cvtColor(padded[v], cv2.COLOR_RGB2BGR)
        if v == ref_view:
            top = overlay_mask(bgr, prompt_mask, (0, 0, 255), (0, 0, 139)); ttl = f"view {v} (ref) — prompt"
        else:
            top, ttl = bgr, f"view {v}"
        pred = overlay_mask(bgr, pred_masks[v], (0, 255, 0), (0, 100, 0))
        btl = f"pred iou={iou[v]:.2f}" if iou is not None else "pred"
        axes[0, v].imshow(cv2.cvtColor(top, cv2.COLOR_BGR2RGB)); axes[0, v].set_title(ttl, fontsize=9); axes[0, v].axis("off")
        axes[1, v].imshow(cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)); axes[1, v].set_title(btl, fontsize=9); axes[1, v].axis("off")
    fig.suptitle(title, fontsize=13); fig.tight_layout()
    fig.savefig(out_dir / "seg_2d_grid.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    fdir = out_dir / "frames"; fdir.mkdir(exist_ok=True)
    for v in range(N):
        bgr = cv2.cvtColor(padded[v], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(fdir / f"overlay_{v:02d}.png"), overlay_mask(bgr, pred_masks[v], (0, 255, 0), (0, 100, 0)))
        cv2.imwrite(str(fdir / f"mask_{v:02d}.png"), pred_masks[v] * 255)


def build_cloud(recon, padded, pred_masks, bboxes, conf_pct, max_points):
    wp = recon["world_points"]; cf = recon["depth_conf"]
    N, H, W, _ = wp.shape
    thr = float(np.percentile(cf, conf_pct))
    pos, col, msk = [], [], []
    for v in range(N):
        keep = cf[v] > thr
        x1, y1, x2, y2 = bboxes[v]
        bb = np.zeros_like(keep); bb[y1:y2, x1:x2] = True
        keep &= bb & np.isfinite(wp[v]).all(axis=-1)
        pos.append(wp[v][keep]); col.append(padded[v][keep]); msk.append(pred_masks[v].astype(bool)[keep])
    pos = np.concatenate(pos).astype(np.float32); col = np.concatenate(col).astype(np.uint8); msk = np.concatenate(msk)
    P = pos.shape[0]
    if P > max_points:
        n_m = int(msk.sum()); budget = max(0, max_points - n_m); bg = np.where(~msk)[0]
        if budget < bg.shape[0]:
            sel = np.random.default_rng(0).choice(bg, budget, replace=False)
            k = np.zeros(P, bool); k[msk] = True; k[sel] = True
            pos, col, msk = pos[k], col[k], msk[k]
    return {"positions": pos, "colors": col, "is_mask": msk}
def _highlight_colors(pcd, color_name):
    col = pcd["colors"].copy()
    hi = np.array([255, 30, 30], np.uint8) if color_name == "red" else np.array([30, 100, 255], np.uint8)
    col[pcd["is_mask"]] = (0.30 * pcd["colors"][pcd["is_mask"]].astype(np.float32) + 0.70 * hi).clip(0, 255).astype(np.uint8)
    return pcd["positions"].astype(np.float32), col


def render_3d(pcd, out_dir, color_name):
    pos, col, msk = pcd["positions"], pcd["colors"], pcd["is_mask"]
    if pos.shape[0] == 0:
        return
    c = np.median(pos, axis=0)
    lo, hi = np.percentile(pos, 2, axis=0), np.percentile(pos, 98, axis=0)
    rng = float(np.max(hi - lo)) * 0.6 + 1e-6
    bg_pos = pos[~msk]
    bg_col = (col[~msk].astype(np.float32) / 255.0).clip(0, 1)
    bg_col = (bg_col * 0.7 + 0.3).clip(0, 1)
    m_pos = pos[msk]
    hi_rgb = np.array([1.0, 0.05, 0.05]) if color_name == "red" else np.array([0.05, 0.35, 1.0])
    if bg_pos.shape[0] > 80000:
        s = np.random.default_rng(0).choice(bg_pos.shape[0], 80000, replace=False)
        bg_pos, bg_col = bg_pos[s], bg_col[s]
    views = [(18, -60), (35, 35), (78, -90)]
    fig = plt.figure(figsize=(6.4 * len(views), 6.2))
    for i, (el, az) in enumerate(views):
        ax = fig.add_subplot(1, len(views), i + 1, projection="3d")
        ax.scatter(bg_pos[:, 0], bg_pos[:, 1], bg_pos[:, 2], c=bg_col, s=1.4, marker=".", linewidths=0, alpha=0.55)
        if m_pos.shape[0] > 0:
            ax.scatter(m_pos[:, 0], m_pos[:, 1], m_pos[:, 2], c=[hi_rgb], s=22, marker="o", linewidths=0.3, edgecolors="k", depthshade=False)
        ax.set_xlim(c[0] - rng, c[0] + rng); ax.set_ylim(c[1] - rng, c[1] + rng); ax.set_zlim(c[2] - rng, c[2] + rng)
        ax.view_init(elev=el, azim=az); ax.set_title(f"elev={el} azim={az}", fontsize=10)
        ax.set_box_aspect((1, 1, 1)); ax.grid(False)
    fig.suptitle(f"3D point cloud — segmented object highlighted ({m_pos.shape[0]} pts)", fontsize=13)
    fig.tight_layout(); fig.savefig(out_dir / "seg_3d_pointcloud.png", dpi=120, bbox_inches="tight"); plt.close(fig)


def export_ply(pcd, out_dir, color_name):
    """Dependency-free binary PLY with per-point RGB (mask highlighted)."""
    pos, colors = _highlight_colors(pcd, color_name)
    n = pos.shape[0]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n").encode("ascii")
    rec = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = pos[:, 0], pos[:, 1], pos[:, 2]
    rec["r"], rec["g"], rec["b"] = colors[:, 0], colors[:, 1], colors[:, 2]
    with open(out_dir / "scene_highlighted.ply", "wb") as f:
        f.write(header); f.write(rec.tobytes())


def export_glb(pcd, out_dir, color_name):
    try:
        import trimesh
    except Exception:
        return
    pos, colors = _highlight_colors(pcd, color_name)
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=pos, colors=colors))
    scene.export(str(out_dir / "scene.glb"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--object", required=True, choices=list(COLOR_PRESETS))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--omega_ckpt", default=str(PROJECT_ROOT / "checkpoints" / "vggt_omega_1b_256_text.pt"))
    ap.add_argument("--seg_ckpt", default=str(PROJECT_ROOT / "weights" / "seg_omega_lora_multiprompt" / "checkpoint-best.pth"))
    ap.add_argument("--n_frames", type=int, default=8)
    ap.add_argument("--ref_view", type=int, default=0)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--conf_pct", type=float, default=40.0)
    ap.add_argument("--max_points", type=int, default=400_000)
    ap.add_argument("--mask_threshold", type=float, default=0.4)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = args.resolution

    print(f"[1/5] extracting {args.n_frames} frames from {args.video}")
    frames, idxs = extract_frames(Path(args.video), args.n_frames)
    imgs, padded, bboxes = preprocess(frames, target)
    N = imgs.shape[1]
    print(f"      got {N} frames @ {target}px (src idx {idxs})")

    print("[2/5] reconstruction (camera + depth -> point cloud)")
    recon = reconstruct(imgs, Path(args.omega_ckpt), device)

    print(f"[3/5] color prompt mask ('{args.object}') on ref view {args.ref_view}")
    prompt_mask = color_prompt_mask(padded[args.ref_view], args.object)
    cv2.imwrite(str(out_dir / "prompt_mask.png"), (prompt_mask * 255).astype(np.uint8))

    print("[4/5] VGGT-Omega prompt segmentation across all views")
    pred_masks, iou = segment(imgs, args.omega_ckpt, args.seg_ckpt,
                              prompt_mask, args.ref_view, target, device, args.mask_threshold)
    print(f"      pred mask px/view: {[int(m.sum()) for m in pred_masks]}")

    print("[5/5] visualisation (2D overlays + 3D point cloud)")
    title = f"{Path(args.video).stem} — '{args.object}' object"
    save_2d(padded, prompt_mask, pred_masks, iou, args.ref_view, out_dir, title)
    pcd = build_cloud(recon, padded, pred_masks, bboxes, args.conf_pct, args.max_points)
    render_3d(pcd, out_dir, args.object)
    export_ply(pcd, out_dir, args.object)
    export_glb(pcd, out_dir, args.object)

    np.savez_compressed(
        out_dir / "reconstruction.npz",
        extrinsic=recon["extrinsic"], intrinsic=recon["intrinsic"],
        depth=recon["depth"], depth_conf=recon["depth_conf"],
        world_points=recon["world_points"], pred_masks=pred_masks, frame_indices=np.array(idxs),
    )
    np.savez_compressed(out_dir / "pointcloud.npz",
                         positions=pcd["positions"], colors=pcd["colors"], is_mask=pcd["is_mask"])
    meta = {
        "video": str(args.video), "object": args.object, "n_views": int(N),
        "frame_indices": idxs, "ref_view": args.ref_view,
        "iou_pred": (iou.tolist() if iou is not None else None),
        "pred_mask_px": [int(m.sum()) for m in pred_masks],
        "cloud_points": int(pcd["positions"].shape[0]),
        "cloud_mask_points": int(pcd["is_mask"].sum()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
