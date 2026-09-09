"""Build an interactive-canvas PCD entry for a video using a BOX/MASK/POINT prompt.

Pipeline:
  1. sample N frames -> pad to square `target` (record bbox for coordinate mapping)
  2. reconstruction: full clean VGGTOmega -> extrinsics/intrinsics/depth -> world points
  3. segmentation: VGGTOmegaBackbone(+LoRA) + promptable head, prompt on ref view
  4. pack -> binary point cloud (float16 pos + uint8 rgb + packed mask bits),
     normalized cameras, and small JPEG thumbnails (views / preds / prompt)

Emits one JSON object (the PCD_DATA entry) to <out_json>.
Box is given in *normalized original-frame* coords (x0 y0 x1 y1).
"""
from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "repo" / "vggt-omega"))

from lib import (  # noqa: E402
    build_omega_full,
    forward_omega_seg,
    forward_omega_split,
)
from video_omega_seg_pipeline import (  # noqa: E402
    extract_frames,
    preprocess,
    reconstruct,
)
from PIL import Image  # noqa: E402


def extract_frames_window(video_path, n_frames, start_frac=0.0, end_frac=1.0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lo = int(round(start_frac * (total - 1))); hi = int(round(end_frac * (total - 1)))
    hi = max(hi, lo + n_frames)
    idxs = [int(round(lo + i * (hi - lo) / (n_frames - 1))) for i in range(n_frames)]
    frames, kept = [], []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, bgr = cap.read()
        if not ok:
            continue
        frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))); kept.append(fi)
    cap.release()
    return frames, kept


def map_box(norm_box, bbox, orig_wh, target):
    ow, oh = orig_wh; side = max(ow, oh)
    nx0, ny0, nx1, ny1 = norm_box
    ox0, oy0, ox1, oy1 = nx0 * ow, ny0 * oh, nx1 * ow, ny1 * oh
    left, top, _, _ = bbox
    s = target / side
    return [left + ox0 * s, top + oy0 * s, left + ox1 * s, top + oy1 * s]


def grabcut_fg(rgb, box_target):
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in box_target]
    x0 = max(0, x0); y0 = max(0, y0); x1 = min(W, x1); y1 = min(H, y1)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gc = np.zeros((H, W), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, gc, (x0, y0, max(1, x1 - x0), max(1, y1 - y0)), bgd, fgd, 6, cv2.GC_INIT_WITH_RECT)
    fg = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n > 1:
        fg = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    return fg.astype(np.float32)


@torch.no_grad()
def segment_prompt(imgs, omega_ckpt, seg_ckpt, box_target, ref_view, target, device,
                   mode="box", prompt_mask=None, points=None, labels=None, thr=0.4):
    backbone, head = build_omega_full(Path(omega_ckpt), Path(seg_ckpt), device)
    want_geom = getattr(head, "geom_pe", None) is not None
    tokens, geometry = forward_omega_split(backbone, imgs.to(device), return_geometry=want_geom)
    N = imgs.shape[1]
    pm = torch.zeros(1, N, 1, target, target, dtype=torch.float32, device=device)
    pv = torch.tensor([[ref_view]], dtype=torch.long, device=device)
    kw = {}
    if mode == "mask":
        pm[0, ref_view, 0] = torch.from_numpy(prompt_mask).to(device)
    elif mode == "point":
        P = points.shape[0]
        ppts = torch.zeros(1, N, 1, 6, 2, dtype=torch.float32, device=device)
        plabs = torch.full((1, N, 1, 6), -1, dtype=torch.long, device=device)
        ppts[0, ref_view, 0, :P] = torch.from_numpy(points)
        plabs[0, ref_view, 0, :P] = torch.from_numpy(labels)
        kw = dict(prompt_points=ppts, prompt_point_labels=plabs)
    else:  # box
        pboxes = torch.zeros(1, N, 1, 4, dtype=torch.float32, device=device)
        pboxes[0, ref_view, 0] = torch.tensor(box_target, dtype=torch.float32)
        pbox_valid = torch.zeros(1, N, 1, dtype=torch.bool, device=device)
        pbox_valid[0, ref_view, 0] = True
        kw = dict(prompt_boxes=pboxes, prompt_box_valid=pbox_valid)
    out = forward_omega_seg(head, tokens, pm, pv, target, target, geometry=geometry, **kw)
    logits = out["segmentation_logits"][0, :, 0].float()
    masks = (logits.sigmoid() > thr).cpu().numpy().astype(np.uint8)
    iou = (out["iou_pred"][0, :, 0].float().cpu().numpy()
           if out.get("iou_pred") is not None else None)
    del backbone, head, tokens; torch.cuda.empty_cache()
    return masks, iou


def build_cloud(recon, padded, pred_masks, bboxes, conf_pct, max_points, mask_conf_pct=10):
    """Point cloud with separate confidence thresholds for mask/background."""
    wp = recon["world_points"]; cf = recon["depth_conf"]
    N, H, W, _ = wp.shape
    bg_thr = float(np.percentile(cf, conf_pct)); mask_thr = float(np.percentile(cf, mask_conf_pct))
    pos, col, msk = [], [], []
    for v in range(N):
        finite = np.isfinite(wp[v]).all(axis=-1)
        x1, y1, x2, y2 = bboxes[v]
        bb = np.zeros_like(finite); bb[y1:y2, x1:x2] = True
        is_mask = pred_masks[v].astype(bool)
        keep = (is_mask & (cf[v] > mask_thr) & bb & finite) | ((~is_mask) & (cf[v] > bg_thr) & bb & finite)
        pos.append(wp[v][keep]); col.append(padded[v][keep]); msk.append(is_mask[keep])
    pos = np.concatenate(pos).astype(np.float32); col = np.concatenate(col).astype(np.uint8); msk = np.concatenate(msk)
    P = pos.shape[0]; rng = np.random.default_rng(0)
    if P > max_points:
        mask_idx = np.where(msk)[0]; bg_idx = np.where(~msk)[0]
        mask_budget = min(mask_idx.shape[0], int(max_points * 0.40))
        if mask_idx.shape[0] > mask_budget:
            mask_idx = rng.choice(mask_idx, mask_budget, replace=False)
        bg_sel = rng.choice(bg_idx, min(bg_idx.shape[0], max_points - mask_idx.shape[0]), replace=False)
        keep = np.concatenate([mask_idx, bg_sel]); rng.shuffle(keep)
        pos, col, msk = pos[keep], col[keep], msk[keep]
    return pos, col, msk


def camera_params(extr):
    cams = []
    for v in range(extr.shape[0]):
        R = extr[v, :3, :3]; t = extr[v, :3, 3]
        cams.append((-R.T @ t, R[2, :3] / (np.linalg.norm(R[2, :3]) + 1e-9)))
    return cams


def normalize(pos, cams):
    center = np.median(pos, axis=0)
    lo, hi = np.percentile(pos, 2, axis=0), np.percentile(pos, 98, axis=0)
    scale = float(np.max(hi - lo)) * 0.5 + 1e-6
    npos = (pos - center) / scale
    out_cams = []
    for c, f in cams:
        cc = (c - center) / scale
        out_cams.append([round(float(cc[0]), 4), round(float(cc[1]), 4), round(float(cc[2]), 4),
                         round(float(f[0]), 4), round(float(f[1]), 4), round(float(f[2]), 4)])
    return npos.astype(np.float32), out_cams


def pack_cloud(pos, col, msk):
    blob = struct.pack("<I", pos.shape[0]) + pos.astype("<f2").tobytes() + col.astype(np.uint8).tobytes() + np.packbits(msk.astype(np.uint8)).tobytes()
    return base64.b64encode(blob).decode()


def jpeg_b64(rgb, size=128, q=72):
    bgr = cv2.cvtColor(cv2.resize(rgb, (size, size)), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf.tobytes()).decode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--cls", required=True)
    ap.add_argument("--box", type=float, nargs=4, required=True,
                    help="normalized x0 y0 x1 y1 person box in original frame")
    ap.add_argument("--mode", choices=["box", "mask", "point"], default="point",
                    help="box prompt, GrabCut mask prompt, or positive points inside box")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--omega_ckpt", default=str(PROJECT_ROOT / "checkpoints" / "vggt_omega_1b_256_text.pt"))
    ap.add_argument("--seg_ckpt", default=str(PROJECT_ROOT / "weights" / "seg_omega_lora_multiprompt" / "checkpoint-best.pth"))
    ap.add_argument("--n_frames", type=int, default=6)
    ap.add_argument("--start_frac", type=float, default=0.0)
    ap.add_argument("--end_frac", type=float, default=1.0)
    ap.add_argument("--ref_view", type=int, default=0)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--conf_pct", type=float, default=45.0)
    ap.add_argument("--max_points", type=int, default=50000)
    ap.add_argument("--mask_threshold", type=float, default=0.4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = args.resolution

    print(f"[1/4] frames from {args.video} window=[{args.start_frac},{args.end_frac}]")
    if args.start_frac > 0.0 or args.end_frac < 1.0:
        frames, idxs = extract_frames_window(Path(args.video), args.n_frames, args.start_frac, args.end_frac)
    else:
        frames, idxs = extract_frames(Path(args.video), args.n_frames)
    imgs, padded, bboxes = preprocess(frames, target)
    N = imgs.shape[1]
    orig_wh = [frames[i].size for i in range(N)]

    print("[2/4] reconstruction")
    recon = reconstruct(imgs, Path(args.omega_ckpt), device)

    print(f"[3/4] promptable {args.mode.upper()} segmentation")
    box_t = map_box(args.box, bboxes[args.ref_view], orig_wh[args.ref_view], target)
    box_t = [max(0, min(target, v)) for v in box_t]
    prompt_mask = points = labels = None
    if args.mode == "mask":
        prompt_mask = grabcut_fg(padded[args.ref_view], box_t)
        print(f"      grabcut fg px={int(prompt_mask.sum())}")
    elif args.mode == "point":
        x0, y0, x1, y1 = box_t; cx = (x0 + x1) / 2
        pts = [[cx, y0 + (y1 - y0) * f] for f in (0.30, 0.55, 0.78)]
        points = np.array(pts, dtype=np.float32); labels = np.ones(len(pts), dtype=np.int64)
        print(f"      points={[[round(p[0]),round(p[1])] for p in pts]}")
    pred_masks, iou = segment_prompt(imgs, args.omega_ckpt, args.seg_ckpt, box_t,
                                     args.ref_view, target, device, mode=args.mode,
                                     prompt_mask=prompt_mask, points=points, labels=labels,
                                     thr=args.mask_threshold)
    print(f"      box={[round(b,1) for b in box_t]} mask px/view={[int(m.sum()) for m in pred_masks]}")

    print("[4/4] cloud + pack")
    pos, col, msk = build_cloud(recon, padded, pred_masks, bboxes, args.conf_pct, args.max_points)
    cams = camera_params(recon["extrinsic"])
    npos, ncams = normalize(pos, cams)
    data_b64 = pack_cloud(npos, col, msk)

    prompt_img = padded[args.ref_view].copy()
    x0, y0, x1, y1 = [int(v) for v in box_t]
    if prompt_mask is not None:
        ov = prompt_img.copy(); ov[prompt_mask > 0.5] = [230, 40, 40]
        prompt_img = cv2.addWeighted(prompt_img, 0.55, ov, 0.45, 0)
    elif points is not None:
        for (px, py) in points:
            cv2.circle(prompt_img, (int(px), int(py)), 6, (0, 230, 0), -1)
            cv2.circle(prompt_img, (int(px), int(py)), 7, (255, 255, 255), 1)
    else:
        ov = prompt_img.copy(); cv2.rectangle(ov, (x0, y0), (x1, y1), (230, 40, 40), 3)
        prompt_img = cv2.addWeighted(prompt_img, 0.7, ov, 0.3, 0)
        cv2.rectangle(prompt_img, (x0, y0), (x1, y1), (230, 40, 40), 2)
    prompt_b64 = jpeg_b64(prompt_img)
    views_b64 = [jpeg_b64(padded[v]) for v in range(N)]
    preds_b64 = []
    for v in range(N):
        vis = padded[v].copy()
        if pred_masks[v].any():
            o = vis.copy(); o[pred_masks[v] > 0] = [0, 210, 0]
            vis = cv2.addWeighted(vis, 0.5, o, 0.5, 0)
        preds_b64.append(jpeg_b64(vis))

    entry = {
        "scene": args.scene, "cls": args.cls, "type": "dynamic",
        "n_total": int(pos.shape[0]), "n_mask": int(msk.sum()),
        "n_display": int(pos.shape[0]), "n_mask_display": int(msk.sum()),
        "cameras": ncams, "data": data_b64,
        "views": views_b64, "preds": preds_b64, "prompt": prompt_b64,
        "iou": [round(float(x), 4) for x in (iou.tolist() if iou is not None else [])],
    }
    Path(args.out_json).write_text(json.dumps(entry))
    print(f"done -> {args.out_json}  ({pos.shape[0]} pts, {int(msk.sum())} mask)")


if __name__ == "__main__":
    main()
