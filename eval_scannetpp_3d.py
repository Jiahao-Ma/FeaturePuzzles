#!/usr/bin/env python3
"""ScanNet++ 3D open-vocab eval (GT depth) for our VGGT-Omega + RGBGuidedFeatureHead.

Mirrors eval_scannet_3d.py but loads ScanNet++ data:
  - images:   dslr/undistorted_images/<DSC>.JPG  (pinhole, w=1440 h=848)
  - depth:    dslr/undistorted_depths/<DSC>.png  (meters)
  - poses+K:  dslr/transforms.json (nerfstudio c2w, OpenGL convention)
  - GT:       <scene>.pth sampled_coords @ R_z_90, sampled_labels -> 20 eval classes
"""
from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
# reuse backbone/head/teacher/predict infra from the ScanNet eval script
from eval_scannet_3d import (
    VGGTOmegaBackbone, DINOv3Talk2DinoTeacher, build_rgb_guided_feature_head,
    _load_heads, _predict_batch, _metrics,
)
from utils.scannetpp_labels import SCANNETPP_EVAL, build_raw_to_scannetpp

R_Z_90 = np.diag([1.0, 1.0, -1.0]).astype(np.float32)  # ScanNet++ mesh->COLMAP/nerfstudio frame: flip Z (verified via z-buffered color match)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_root", default="/mnt/dataset/marko/puzzles/embodiedsplat_dl/dataset/scannetpp/data")
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint", default="checkpoints/vggt_omega_1b_256_text.pt")
    p.add_argument("--scenes", nargs="+", default=["09c1414f1b"])
    p.add_argument("--n_views", type=int, default=6)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--feature_res", type=int, default=128)
    p.add_argument("--teacher_long", type=int, default=2048)
    p.add_argument("--frame_stride", type=int, default=20)
    p.add_argument("--max_frames", type=int, default=200)
    p.add_argument("--voxel", type=float, default=0.05)
    p.add_argument("--work_scale", type=float, default=0.25, help="downsample depth+intrinsics for unprojection (full-res CPU copy too slow)")
    p.add_argument("--tta_flip", action="store_true")
    p.add_argument("--ensemble_ckpts", nargs="*", default=None)
    p.add_argument("--gl", action="store_true", help="OpenGL cam convention (nerfstudio default)")
    return p.parse_args()


def load_scene_frames(scene_dir: Path, stride, max_frames):
    """Return list of (img_path, depth_path, c2w 4x4, fx, fy, cx, cy, w, h). Uses undistorted pinhole."""
    t = json.load(open(scene_dir / "dslr" / "nerfstudio" / "transforms_undistorted.json"))
    fx, fy, cx, cy = t["fl_x"], t["fl_y"], t["cx"], t["cy"]
    w, h = t["w"], t["h"]
    frames = t["frames"]
    # skip is_bad frames (if present)
    frames = [fr for fr in frames if not fr.get("is_bad", False)]
    sel = frames[::max(1, stride)]
    if max_frames > 0:
        sel = sel[:max_frames]
    out = []
    for fr in sel:
        stem = Path(fr["file_path"]).stem
        img = scene_dir / "dslr" / "undistorted_images" / f"{stem}.JPG"
        dep = scene_dir / "dslr" / "undistorted_depths" / f"{stem}.png"
        if not img.is_file() or not dep.is_file():
            continue
        c2w = np.array(fr["transform_matrix"], dtype=np.float64)   # cam->world (nerfstudio OpenGL)
        out.append((img, dep, c2w, fx, fy, cx, cy, w, h))
    return out, fx, fy, cx, cy, w, h


def load_depth(path: Path):
    d = np.array(Image.open(path))
    if d.dtype == np.uint16:
        return d.astype(np.float32) / 1000.0      # mm -> m
    return d.astype(np.float32)                    # already meters (float)


@torch.no_grad()
def run_scene(scene_dir, frames, K, args, device, backbone, heads):
    fx, fy, cx, cy, w, h = K
    r = args.resolution
    Dh, Dw = h, w
    # voxel accumulator
    vox_keys = None; vox_sum = None; vox_cnt = None
    inf_time = 0.0; n_inf_f = 0; n_inf_b = 0
    all_keys = []; all_sums = []; all_cnts = []   # per-frame reduced voxels; merged once at end

    def lin_keys(keys, vmin, span):
        return ((keys[:, 0] - vmin[0]) * span[1] * span[2]
                + (keys[:, 1] - vmin[1]) * span[2]
                + (keys[:, 2] - vmin[2])).astype(np.int64)

    for i in range(0, len(frames), args.n_views):
        batch = frames[i:i + args.n_views]
        N = len(batch)
        imgs = torch.zeros(N, 3, r, r, device=device)
        dmaps = []; poses = []
        for j, (ip, dp, c2w, fxs, fys, cxs, cys, ww, hh) in enumerate(batch):
            im = Image.open(ip).convert("RGB").resize((r, r), Image.LANCZOS)
            imgs[j] = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255.0
            dmaps.append(load_depth(dp))
            poses.append(c2w)
        poses = np.stack(poses)
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        pred = _predict_batch(backbone, heads, imgs, tta_flip=args.tta_flip, device=device)
        if device.type == "cuda": torch.cuda.synchronize()
        inf_time += time.time() - t0; n_inf_b += 1; n_inf_f += N
        # work-resolution (downsampled) for unprojection — avoids full-res CPU copy
        s = args.work_scale
        Wh, Ww = int(round(Dh * s)), int(round(Dw * s))
        pred_work = F.interpolate(pred, (Wh, Ww), mode="bilinear", align_corners=False)  # (N,C,Wh,Ww) GPU
        for j in range(N):
            ip, dp, c2w, fxs, fys, cxs, cys, ww, hh = batch[j]
            d = dmaps[j]
            # downsample depth to work res (nearest) + scale intrinsics
            d_w = np.array(Image.fromarray(d).resize((Ww, Wh), Image.NEAREST))
            fxs_w, fys_w, cxs_w, cys_w = fxs * s, fys * s, cxs * s, cys * s
            valid = d_w > 0
            T = poses[j]                            # c2w
            ys, xs = np.nonzero(valid); z = d_w[ys, xs]
            x_c = (xs - cxs_w) * z / fxs_w
            y_c = -(ys - cys_w) * z / fys_w          # image v-down -> cam y-up
            if args.gl:
                zc = -z                            # OpenGL: forward = -z
            else:
                zc = z
            pts_c = np.stack([x_c, y_c, zc, np.ones_like(z)], axis=0)  # (4,P)
            pts_w = (T @ pts_c).T[:, :3]
            # gather per-point features at work res on GPU, then move to CPU
            feat = pred_work[j][:, ys, xs].permute(1, 0).contiguous().cpu().numpy().astype(np.float32)
            vox = np.floor(pts_w / args.voxel).astype(np.int64)
            # reduce this frame's points to unique voxels via fast 1D linear keys
            vmin_f = vox.min(0); span_f = np.maximum(vox.max(0) - vmin_f + 1, 1)
            lin_f = lin_keys(vox, vmin_f, span_f)
            uf, inv = np.unique(lin_f, return_inverse=True)
            Uf = uf.shape[0]
            fsum = np.zeros((Uf, feat.shape[1]), dtype=np.float32); np.add.at(fsum, inv, feat)
            fcnt = np.bincount(inv, minlength=Uf).astype(np.int32)
            uf_keys = np.stack([uf // (span_f[1] * span_f[2]) + vmin_f[0],
                                (uf % (span_f[1] * span_f[2])) // span_f[2] + vmin_f[1],
                                uf % span_f[2] + vmin_f[2]], axis=1).astype(np.int64)
            all_keys.append(uf_keys); all_sums.append(fsum); all_cnts.append(fcnt)
        if (i // args.n_views) % 5 == 0:
            print(f"    batch {i//args.n_views}/{(len(frames)+args.n_views-1)//args.n_views} "
                  f"frames_done={len(all_keys)}", flush=True)
    # ---- single global merge of all per-frame reduced voxels ----
    cat_keys = np.concatenate(all_keys, 0)
    cat_sums = np.concatenate(all_sums, 0).astype(np.float32)
    cat_cnts = np.concatenate(all_cnts, 0).astype(np.float32)
    vmin = cat_keys.min(0); span = np.maximum(cat_keys.max(0) - cat_keys.min(0) + 1, 1)
    g_lin = lin_keys(cat_keys, vmin, span)
    ug, inv_g = np.unique(g_lin, return_inverse=True)
    G = ug.shape[0]
    vox_keys = np.stack([ug // (span[1] * span[2]) + vmin[0],
                         (ug % (span[1] * span[2])) // span[2] + vmin[1],
                         ug % span[2] + vmin[2]], axis=1).astype(np.int64)
    C = cat_sums.shape[1]
    vox_sum = np.zeros((G, C), dtype=np.float32)
    for c in range(C):
        vox_sum[:, c] = np.bincount(inv_g, weights=cat_sums[:, c], minlength=G).astype(np.float32)
    vox_cnt = np.bincount(inv_g, minlength=G).astype(np.float32)
    vox_feat = vox_sum / np.maximum(vox_cnt[:, None], 1)
    vox_feat = torch.from_numpy(np.ascontiguousarray(vox_feat, dtype=np.float32)).to(device)
    vox_feat = F.normalize(vox_feat, dim=-1, eps=1e-6)
    vox_centers = vox_keys.astype(np.float64) * args.voxel + args.voxel / 2
    return vox_keys, vox_centers, vox_feat, dict(n_batches=n_inf_b, n_frames=n_inf_f, inf_time_s=inf_time)


@torch.no_grad()
def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    backbone = VGGTOmegaBackbone(ckpt_path=args.checkpoint, use_layers=(4, 11, 17, 23),
                                 amp_dtype=torch.bfloat16).to(device).eval()
    for p in backbone.parameters(): p.requires_grad = False
    teacher = DINOv3Talk2DinoTeacher(teacher_long=args.teacher_long, device=str(device),
                                     amp_dtype=torch.bfloat16, load_text=True).eval()
    ckpts = [args.ckpt] + list(args.ensemble_ckpts or [])
    heads = _load_heads(ckpts, device, args.feature_res, args.resolution)
    text = teacher.text_embed(SCANNETPP_EVAL).to(device).float()       # (20,768)
    raw2idx = build_raw_to_scannetpp().numpy()

    all_preds = []; all_gts = []
    tot_t = 0.0; tot_f = 0; tot_b = 0
    for sc in args.scenes:
        sd = Path(args.data_root) / sc
        if not (sd / "dslr" / "transforms.json").is_file():
            print(f"[{sc}] no transforms.json, skip"); continue
        frames, fx, fy, cx, cy, w, h = load_scene_frames(sd, args.frame_stride, args.max_frames)
        print(f"[{sc}] {len(frames)} frames")
        if not frames:
            print(f"[{sc}] no undistorted images/depth, skip"); continue
        vox_keys, vox_centers, vox_feat, st = run_scene(sd, frames, (fx, fy, cx, cy, w, h), args, device, backbone, heads)
        tot_t += st["inf_time_s"]; tot_f += st["n_frames"]; tot_b += st["n_batches"]
        # GT
        pth = torch.load(sd / f"{sc}.pth", map_location="cpu", weights_only=False)
        gt_xyz = np.asarray(pth["sampled_coords"], dtype=np.float64) @ R_Z_90  # -> COLMAP frame
        gt_lab = np.asarray(pth["sampled_labels"])
        gt_cls = raw2idx[np.clip(gt_lab, 0, raw2idx.shape[0] - 1)]
        gt_cls[gt_lab < 0] = -1
        # assign GT points to voxels
        gt_vox = np.floor(gt_xyz / args.voxel).astype(np.int64)
        vmin = np.minimum(vox_keys.min(0), gt_vox.min(0)); vmax = np.maximum(vox_keys.max(0), gt_vox.max(0))
        span = np.maximum(vmax - vmin + 1, 1)
        def lin(a): return ((a[:, 0] - vmin[0]) * span[1] * span[2] + (a[:, 1] - vmin[1]) * span[2] + (a[:, 2] - vmin[2]))
        g_lin = lin(vox_keys)
        g_order = np.argsort(g_lin, kind="stable"); g_sorted = g_lin[g_order]
        gt_lin = lin(gt_vox)
        pos = np.searchsorted(g_sorted, gt_lin); pos_c = np.clip(pos, 0, len(g_sorted) - 1)
        hit = g_sorted[pos_c] == gt_lin
        pt_vidx = -np.ones(len(gt_vox), dtype=np.int64)
        pt_vidx[hit] = g_order[pos_c[hit]]
        valid = pt_vidx >= 0
        print(f"[{sc}] GT={gt_xyz.shape[0]} covered={valid.sum()} ({valid.mean():.1%})")
        pred = -np.ones(len(gt_xyz), dtype=np.int64)
        if valid.any():
            vf = vox_feat[pt_vidx[valid]]
            sim = vf @ text.t()
            pred[valid] = sim.argmax(dim=-1).cpu().numpy()
        np.save(out / f"{sc}_pred.npy", np.stack([gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], pred, gt_cls], 1).astype(np.float32))
        m = (gt_cls >= 0) & valid
        all_preds.append(pred[m]); all_gts.append(gt_cls[m])

    print(f"\ninference: frames={tot_f} batches={tot_b} total={tot_t:.2f}s ms/frame={tot_t/max(tot_f,1)*1000:.1f}")
    (out / "inference_time.txt").write_text(f"frames={tot_f} batches={tot_b} total_s={tot_t:.3f} ms_per_frame={tot_t/max(tot_f,1)*1000:.3f}\n")
    if all_preds:
        preds = np.concatenate(all_preds); gts = np.concatenate(all_gts)
        miou, macc, nc = _metrics(preds, gts, len(SCANNETPP_EVAL))
        print(f"\n================ scannetpp 3D open-vocab eval ================")
        print(f"scenes={len(args.scenes)} eval_points={len(gts)} classes_evaluated={nc}")
        print(f"mIoU = {miou*100:.2f}   mACC = {macc*100:.2f}")
        (out / "result_scannetpp.txt").write_text(f"mIoU={miou*100:.2f} mACC={macc*100:.2f} classes={nc} points={len(gts)}\n")


if __name__ == "__main__":
    main()
