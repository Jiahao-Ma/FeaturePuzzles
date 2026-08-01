"""3D open-vocabulary semantic segmentation eval on ScanNet (EmbodiedSplat setup).

Pipeline (matches the user's plan + EmbodiedSplat Table 1 "2D search domain"):
    ScanNet test scene frames
        -> VGGT-Omega + RGBGuidedFeatureHead  -> per-view dense DINOv3 feature (768,128,128)
        -> upsample to depth resolution (480x640)
        -> GT depth + intrinsic_depth + cam->world extrinsics  -> unproject each pixel to world
        -> voxel-grid accumulate (feature sum + count)  -> per-voxel averaged feature
        -> for each GT point: take its voxel's feature -> cos(text) -> argmax -> nyu40 label
        -> map nyu40 -> scannet{20,15,10} -> mIoU / mACC

GT point cloud:  dataset/scannet/test/<scene>/<scene_id>.npy  (N,12): xyz=[:, :3], nyu40 label=[:, 10]
Frames:         dataset/scannet/test/<scene>/color/<i>.jpg  (1296x968)
                dataset/scannet/test/<scene>/depth/<i>.png  (480x640 uint16, MILLIMETERS)
                dataset/scannet/test/<scene>/intrinsic/intrinsic_{color,depth}.txt  (4x4)
                dataset/scannet/test/<scene>/extrinsics.npy  (nFrames,4,4) cam->world
                dataset/scannet/test/<scene>/valid_frames.txt  (scene fIdx ?)

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval_scannet_3d.py \
        --ckpt output/feat_dinov3/checkpoint-best.pth \
        --data_root /mnt/dataset/marko/puzzles/embodiedsplat_dl/dataset/scannet \
        --out   output/feat_dinov3/eval_scannet3d --dataset scannet20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJ = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ / "src"))

from models.vggt_omega_backbone import VGGTOmegaBackbone  # noqa: E402
from models.vggt_omega_feature_head import build_rgb_guided_feature_head  # noqa: E402
from models.dinov3_teacher import DINOv3Talk2DinoTeacher  # noqa: E402

# ---- class tables (from EmbodiedSplat src/evaluation/database/class_labels.py) ----
SCANNET20 = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
             'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
             'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub']
SCANNET15 = ["wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
             "window", "bookshelf", "counter", "desk", "curtain", "toilet", "sink"]
SCANNET10 = ["wall", "floor", "bed", "chair", "sofa", "table", "door",
             "window", "bookshelf", "toilet"]

# nyu40 id -> name (from scannet_label_database.yaml, validation subset)
NYU40_NAME = {
    1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair", 6: "sofa",
    7: "table", 8: "door", 9: "window", 10: "bookshelf", 11: "picture",
    12: "counter", 14: "desk", 16: "curtain", 24: "refrigerator",
    28: "shower curtain", 33: "toilet", 34: "sink", 36: "bathtub",
}


def class_set(name: str):
    if name == "scannet200":
        from utils.scannet200_labels import SCANNET200_EVAL
        return SCANNET200_EVAL
    return {"scannet20": SCANNET20, "scannet15": SCANNET15, "scannet10": SCANNET10}[name]


def build_nyu_to_clsidx(cls_labels):
    """nyu40 id -> index into cls_labels (or -1 if not in the set)."""
    name_to_idx = {n: i for i, n in enumerate(cls_labels)}
    mapping = -np.ones(41, dtype=np.int64)  # nyu40 ids 0..40
    for nid, nm in NYU40_NAME.items():
        if nm in name_to_idx:
            mapping[nid] = name_to_idx[nm]
    return mapping


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="output/feat_dinov3/checkpoint-best.pth")
    p.add_argument("--data_root",
                   default="/mnt/dataset/marko/puzzles/embodiedsplat_dl/dataset/scannet")
    p.add_argument("--out", default="output/feat_dinov3/eval_scannet3d")
    p.add_argument("--checkpoint", default="checkpoints/vggt_omega_1b_256_text.pt")
    p.add_argument("--dataset", default="scannet20", choices=["scannet20", "scannet15", "scannet10"])
    p.add_argument("--n_views", type=int, default=6, help="frames per VGGT batch")
    p.add_argument("--resolution", type=int, default=256, help="VGGT input side (must be 16x)")
    p.add_argument("--feature_res", type=int, default=128)
    p.add_argument("--teacher_long", type=int, default=2048)
    p.add_argument("--frame_stride", type=int, default=20,
                   help="sample every k-th valid frame (controls cost)")
    p.add_argument("--max_frames", type=int, default=200, help="cap frames per scene")
    p.add_argument("--voxel", type=float, default=0.05, help="voxel size (m) for lifting")
    p.add_argument("--scenes", nargs="*", default=None, help="subset of scene ids")
    p.add_argument("--no_eval", action="store_true",
                   help="only save per-scene _pred.npy (skip global mIoU/mACC) — for multi-GPU sharding")
    p.add_argument("--ensemble_ckpts", nargs="*", default=None,
                   help="extra feature-head ckpts; average normalized preds with --ckpt")
    p.add_argument("--tta_flip", action="store_true",
                   help="horizontal-flip TTA: average pred with flipped-input pred")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="evaluate multiple class sets in one pass (single 3D lift); "
                        "e.g. scannet10 scannet15 scannet20. Defaults to [args.dataset].")
    return p.parse_args()


def _metrics(preds, gts, K):
    ious, accs = [], []
    for c in range(K):
        m = gts == c
        if m.sum() == 0:
            continue
        p = (preds == c) & m
        u = (preds == c) | m
        ious.append(p.sum() / max(u.sum(), 1))
        accs.append((preds[m] == c).mean())
    miou = float(np.mean(ious)) if ious else 0.0
    macc = float(np.mean(accs)) if accs else 0.0
    return miou, macc, len(ious)


def _load_heads(ckpts, device, feature_res, resolution):
    """Load one or more RGBGuidedFeatureHead checkpoints."""
    heads = []
    for ckpt in ckpts:
        h = build_rgb_guided_feature_head(
            device=str(device), n_vggt_layers=4, work_dim=768,
            feature_out_dim=768, feature_res=feature_res,
            start_res=resolution // 16, use_cross_view=True)
        rm = torch.load(ckpt, map_location=device, weights_only=False)
        h.load_state_dict(rm["feature_head_state_dict"])
        h.eval()
        heads.append(h)
        print(f"  head: {ckpt} epoch={rm.get('epoch')} val_cos={rm.get('val_cos', -1):.4f}")
    return heads


def _predict_batch(backbone, heads, imgs, *, tta_flip, device):
    """Run backbone+heads on (N,3,H,W); return (N,C,Fr,Fr) L2-normalised, ensemble+TTA averaged."""
    # imgs: (N,3,r,r)
    preds = []
    views = [imgs]
    if tta_flip:
        views.append(torch.flip(imgs, dims=[-1]))
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for v in views:
            flip_back = v is not views[0]
            for h in heads:
                tokens_list, _ = backbone(v.unsqueeze(0))
                p = h(tokens_list, v.unsqueeze(0)).float()[0]   # (N,C,Fr,Fr)
                if flip_back:
                    p = torch.flip(p, dims=[-1])
                preds.append(p)
    out = torch.stack(preds, dim=0).mean(0)
    out = F.normalize(out, dim=1, eps=1e-6)
    return out


def load_intrinsic(path: Path):
    """4x4 intrinsic txt -> (3,3)."""
    K = np.loadtxt(path).reshape(4, 4)[:3, :3].astype(np.float64)
    return K


def select_frames(scene_dir: Path, stride, max_frames):
    vf = scene_dir / "valid_frames.txt"
    if vf.is_file():
        idxs = []
        for line in vf.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                idxs.append(int(parts[1]))
        idxs = sorted(set(idxs))
    else:
        idxs = sorted(int(p.stem) for p in (scene_dir / "color").glob("*.jpg"))
    sel = idxs[::max(1, stride)]
    if max_frames > 0:
        sel = sel[:max_frames]
    return sel


@torch.no_grad()
def run_scene(scene_dir: Path, scene_id: str, backbone, heads, text, args, device):
    Kd = load_intrinsic(scene_dir / "intrinsic" / "intrinsic_depth.txt")
    extrinsics = np.load(scene_dir / "extrinsics.npy")          # (nF,4,4) cam->world
    frame_idxs = select_frames(scene_dir, args.frame_stride, args.max_frames)
    print(f"  [{scene_id}] {len(frame_idxs)} frames selected (stride={args.frame_stride})")

    # inference-timing accumulators (input image -> output feature map)
    n_inf_batches = 0
    inf_time_s = 0.0          # wall time of backbone+head forward per batch
    n_inf_frames = 0

    # voxel accumulator over the whole scene (vectorised; per-frame merge via
    # searchsorted on sorted global keys — memory-bounded by #voxels, not #points)
    vox_keys = None      # (V,3) int64 sorted by linearised key
    vox_lin = None       # (V,)  int64 sorted linearised keys (ascending)
    vox_sum = None       # (V,768) float32
    vox_cnt = None       # (V,) int32

    r = args.resolution
    Fr = args.feature_res

    def lin_keys(keys, vmin, span):
        return ((keys[:, 0] - vmin[0]) * span[1] * span[2]
                + (keys[:, 1] - vmin[1]) * span[2]
                + (keys[:, 2] - vmin[2])).astype(np.int64)

    for i in range(0, len(frame_idxs), args.n_views):
        batch_idxs = frame_idxs[i:i + args.n_views]
        N = len(batch_idxs)
        imgs = torch.zeros(N, 3, r, r, device=device)
        depth_maps = []
        poses = []
        for j, fi in enumerate(batch_idxs):
            img = Image.open(scene_dir / "color" / f"{fi}.jpg").convert("RGB")
            img = img.resize((r, r), Image.LANCZOS)
            imgs[j] = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
            d = np.array(Image.open(scene_dir / "depth" / f"{fi}.png"))  # (480,640) uint16 mm
            depth_maps.append(d)
            poses.append(extrinsics[fi])
        poses = np.stack(poses)                                  # (N,4,4)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if device.type == "cuda":
                torch.cuda.synchronize()
            _t0 = time.time()
            pred = _predict_batch(backbone, heads, imgs, tta_flip=args.tta_flip, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inf_time_s += time.time() - _t0
            n_inf_batches += 1
            n_inf_frames += N
        # pred: (N,768,Fr,Fr)

        # upsample feature to depth resolution (480,640)
        Dh, Dw = depth_maps[0].shape
        pred_up = F.interpolate(pred, (Dh, Dw), mode="bilinear",
                                align_corners=False).cpu().numpy()  # (N,768,Dh,Dw)

        for j in range(N):
            d = depth_maps[j].astype(np.float32) / 1000.0        # meters
            valid = d > 0
            T = poses[j]                                          # cam->world
            fx, fy = Kd[0, 0], Kd[1, 1]
            cx, cy = Kd[0, 2], Kd[1, 2]
            ys, xs = np.nonzero(valid)
            z = d[ys, xs]
            x_c = (xs - cx) * z / fx
            y_c = (ys - cy) * z / fy
            pts_c = np.stack([x_c, y_c, z, np.ones_like(z)], axis=1)  # (P,4)
            pts_w = (T @ pts_c.T).T[:, :3]                         # (P,3) world
            feat = pred_up[j][:, ys, xs].T                        # (P,768)
            vox = np.floor(pts_w / args.voxel).astype(np.int64)   # (P,3)
            feat = feat.astype(np.float32)
            # unique-reduce within this frame first (cuts P by ~10x)
            vmin_f = vox.min(0)
            span_f = np.maximum(vox.max(0) - vmin_f + 1, 1)
            lin_f = lin_keys(vox, vmin_f, span_f)
            uf, inv_f = np.unique(lin_f, return_inverse=True)
            Uf = uf.shape[0]
            fsum = np.zeros((Uf, feat.shape[1]), dtype=np.float32)
            np.add.at(fsum, inv_f, feat)
            fcnt = np.bincount(inv_f, minlength=Uf).astype(np.int32)
            uf_keys = np.stack([
                uf // (span_f[1] * span_f[2]) + vmin_f[0],
                (uf % (span_f[1] * span_f[2])) // span_f[2] + vmin_f[1],
                uf % span_f[2] + vmin_f[2],
            ], axis=1).astype(np.int64)
            # merge into global accumulator
            if vox_keys is None:
                vox_keys = uf_keys.copy()
                vox_sum = fsum
                vox_cnt = fcnt
            else:
                # global vmin/span covering both sets
                vmin = np.minimum(vox_keys.min(0), uf_keys.min(0))
                vmax = np.maximum(vox_keys.max(0), uf_keys.max(0))
                span = np.maximum(vmax - vmin + 1, 1)
                g_lin = lin_keys(vox_keys, vmin, span)
                n_lin = lin_keys(uf_keys, vmin, span)
                # sort global (already ascending if we keep it sorted)
                order = np.argsort(g_lin, kind="stable")
                g_lin_sorted = g_lin[order]
                # find positions of new keys in sorted global
                pos = np.searchsorted(g_lin_sorted, n_lin)
                pos_clipped = np.clip(pos, 0, len(g_lin_sorted) - 1)
                hit = g_lin_sorted[pos_clipped] == n_lin
                # remap existing global indices back to original order
                g_idx_existing = order[pos_clipped[hit]]
                # new keys that don't exist yet
                new_mask = ~hit
                new_keys = uf_keys[new_mask]
                new_sum = fsum[new_mask]
                new_cnt = fcnt[new_mask]
                # accumulate hits
                if hit.any():
                    np.add.at(vox_sum, g_idx_existing, fsum[hit])
                    np.add.at(vox_cnt, g_idx_existing, fcnt[hit].astype(vox_cnt.dtype))
                # append new
                if new_keys.shape[0] > 0:
                    vox_keys = np.concatenate([vox_keys, new_keys], axis=0)
                    vox_sum = np.concatenate([vox_sum, new_sum], axis=0)
                    vox_cnt = np.concatenate([vox_cnt, new_cnt], axis=0)
        if (i // args.n_views) % 5 == 0:
            print(f"    batch {i//args.n_views}/{(len(frame_idxs)+args.n_views-1)//args.n_views} "
                  f"voxels={vox_keys.shape[0] if vox_keys is not None else 0}", flush=True)

    # build voxel arrays
    keys = [tuple(k) for k in vox_keys]
    vox_coords = vox_keys.astype(np.float64) * args.voxel + args.voxel / 2  # (V,3) centers
    vox_feat = vox_sum / np.maximum(vox_cnt[:, None], 1)                    # (V,768)
    vox_feat = torch.from_numpy(np.ascontiguousarray(vox_feat, dtype=np.float32)).to(device)
    vox_feat = F.normalize(vox_feat, dim=-1, eps=1e-6)
    key_to_idx = {k: i for i, k in enumerate(keys)}
    inf_stats = dict(n_batches=n_inf_batches, n_frames=n_inf_frames,
                     inf_time_s=inf_time_s,
                     ms_per_frame=(inf_time_s / max(n_inf_frames, 1)) * 1000.0)
    print(f"  [{scene_id}] inference: {n_inf_batches} batches, {n_inf_frames} frames, "
          f"{inf_time_s:.2f}s total, {inf_stats['ms_per_frame']:.1f} ms/frame (backbone+head)", flush=True)
    return vox_coords, vox_feat, key_to_idx, vox_keys, inf_stats


@torch.no_grad()
def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    backbone = VGGTOmegaBackbone(
        ckpt_path=args.checkpoint, use_layers=(4, 11, 17, 23),
        amp_dtype=torch.bfloat16).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    teacher_model = DINOv3Talk2DinoTeacher(
        teacher_long=args.teacher_long, device=str(device),
        amp_dtype=torch.bfloat16, load_text=True).eval()

    ckpt_list = [args.ckpt] + list(args.ensemble_ckpts or [])
    heads = _load_heads(ckpt_list, device, args.feature_res, args.resolution)
    print(f"loaded {len(heads)} head(s), tta_flip={args.tta_flip}")

    # support multiple class sets in one pass (single 3D lift per scene)
    ds_names = args.datasets if args.datasets else [args.dataset]
    # build per-dataset GT mappers: gt_fn(scene_dir, sid, gt_nyu) -> gt_cls (N,) int64 (-1 void)
    from utils.scannet200_labels import build_raw_to_scannet200
    raw2idx200 = build_raw_to_scannet200().numpy()
    ds_specs = []   # (name, cls_labels, text (K,768), gt_fn, out_subdir, all_preds, all_gts)
    for dn in ds_names:
        cl = class_set(dn)
        tx = teacher_model.text_embed(cl).to(device).float()
        if dn == "scannet200":
            def gt_fn(scene_dir, sid, gt_nyu, _m=raw2idx200):
                p = scene_dir / f"{sid}_scannet200.npy"
                if not p.is_file():
                    return -np.ones(len(gt_nyu), dtype=np.int64)   # scene has no scannet200 GT
                raw = np.load(p)[:, 10].astype(np.int64)
                raw = np.clip(raw, 0, _m.shape[0] - 1)
                return _m[raw]
        else:
            n2i = build_nyu_to_clsidx(cl)
            def gt_fn(scene_dir, sid, gt_nyu, _m=n2i):
                return _m[gt_nyu]
        sub = out_dir / dn
        sub.mkdir(parents=True, exist_ok=True)
        ds_specs.append([dn, cl, tx, gt_fn, sub, [], []])
        print(f"dataset={dn} classes={len(cl)} text={tuple(tx.shape)}")

    root = Path(args.data_root) / "test"
    scenes = sorted(p.name for p in root.iterdir() if p.is_dir())
    if args.scenes:
        scenes = [s for s in scenes if s in args.scenes]
    print(f"scenes: {scenes}")

    tot_inf_time = 0.0; tot_inf_frames = 0; tot_inf_batches = 0
    for sc in scenes:
        scene_dir = root / sc
        sid = sc.split("scene")[-1]                               # e.g. 0000_01
        gt_path = scene_dir / f"{sid}.npy"
        if not gt_path.is_file():
            print(f"  [{sc}] no GT npy, skip")
            continue
        gt = np.load(gt_path)
        gt_xyz = gt[:, :3]
        gt_nyu = gt[:, 10].astype(np.int64)

        # single 3D lift (text arg unused for lifting; pass first dataset's text)
        vox_coords, vox_feat, key_to_idx, vox_keys, inf_stats = run_scene(
            scene_dir, sc, backbone, heads, ds_specs[0][2], args, device)
        tot_inf_time += inf_stats["inf_time_s"]
        tot_inf_frames += inf_stats["n_frames"]
        tot_inf_batches += inf_stats["n_batches"]

        # assign each GT point to its voxel key (vectorised)
        gt_vox = np.floor(gt_xyz / args.voxel).astype(np.int64)
        vmin = np.minimum(vox_keys.min(0), gt_vox.min(0))
        vmax = np.maximum(vox_keys.max(0), gt_vox.max(0))
        span = np.maximum(vmax - vmin + 1, 1)
        def lin(a):
            return ((a[:, 0] - vmin[0]) * span[1] * span[2]
                    + (a[:, 1] - vmin[1]) * span[2]
                    + (a[:, 2] - vmin[2]))
        g_lin = lin(vox_keys)
        vmap = {int(lk): i for i, lk in enumerate(g_lin)}
        gt_lin = lin(gt_vox)
        pt_vidx = np.array([vmap.get(int(lk), -1) for lk in gt_lin], dtype=np.int64)
        valid = pt_vidx >= 0
        print(f"  [{sc}] GT={gt_xyz.shape[0]} pts, covered={valid.sum()} ({valid.mean():.1%})")

        # per-point feature for covered points (computed once, reused across class sets)
        vf = vox_feat[pt_vidx[valid]] if valid.any() else None

        for (dn, cl, tx, gt_fn, sub, apreds, agts) in ds_specs:
            pred_label = -np.ones(len(gt_xyz), dtype=np.int64)
            if valid.any():
                sim = vf @ tx.t()                              # (P,K)
                pred_label[valid] = sim.argmax(dim=-1).cpu().numpy()
            gt_cls = gt_fn(scene_dir, sid, gt_nyu)
            np.save(sub / f"{sc}_pred.npy",
                    np.stack([gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2],
                              pred_label, gt_cls], axis=1).astype(np.float32))
            eval_mask = (gt_cls >= 0) & valid
            apreds.append(pred_label[eval_mask])
            agts.append(gt_cls[eval_mask])
        print(f"  [{sc}] saved preds for {len(ds_specs)} dataset(s)")

    # inference timing summary (input image -> output feature map, backbone+head)
    print(f"\n================ inference timing (backbone+head) ================")
    print(f"frames={tot_inf_frames} batches={tot_inf_batches} "
          f"total={tot_inf_time:.2f}s  ms/frame={tot_inf_time/max(tot_inf_frames,1)*1000:.1f}  "
          f"ms/batch={tot_inf_time/max(tot_inf_batches,1)*1000:.1f}")
    (out_dir / "inference_time.txt").write_text(
        f"frames={tot_inf_frames} batches={tot_inf_batches} total_s={tot_inf_time:.3f} "
        f"ms_per_frame={tot_inf_time/max(tot_inf_frames,1)*1000:.3f} "
        f"ms_per_batch={tot_inf_time/max(tot_inf_batches,1)*1000:.3f}\n")

    if args.no_eval:
        print("--no_eval: skipping global mIoU/mACC (per-scene _pred.npy saved)")
        return

    for (dn, cl, tx, gt_fn, sub, apreds, agts) in ds_specs:
        preds = np.concatenate(apreds); gts = np.concatenate(agts)
        miou, macc, ncls = _metrics(preds, gts, len(cl))
        print(f"\n================ {dn} 3D open-vocab eval ================")
        print(f"scenes={len(scenes)} eval_points={len(gts)} classes_evaluated={ncls}")
        print(f"mIoU = {miou*100:.2f}   mACC = {macc*100:.2f}")
        (sub / f"result_{dn}.txt").write_text(
            f"{dn}: mIoU={miou*100:.2f} mACC={macc*100:.2f}\n")


if __name__ == "__main__":
    main()
