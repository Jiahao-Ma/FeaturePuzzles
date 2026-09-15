#!/usr/bin/env python3
"""ScanNet++ scene demo for the joint feature + promptable-segmentation model.

One forward pass of the shared VGGT-Omega backbone (with the jointly trained
LoRA) feeds two heads:

  * RGBGuidedFeatureHead  -> dense DINOv3-aligned features. Cosine against
    Talk2DINO text embeddings gives OPEN-VOCABULARY segmentation for arbitrary
    text prompts (here: sofa / table / cabinet / chair).
  * VGGTOmegaSegmentationHead -> PROMPTABLE segmentation. Text-derived blobs
    become box+point prompts; the head propagates each prompt to every view in
    the batch, which we then lift and associate in 3D into scene INSTANCES.

Geometry: the backbone's own depth+pose heads give a feed-forward
reconstruction (saved for visualisation); the metric scene point cloud and all
3D instances use the dataset's posed RGB-D so that numbers are comparable with
eval_scannetpp_3d.py.

Output: a single self-contained scene bundle (.npz) consumed by render.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from models.vggt_omega_backbone import VGGTOmegaBackbone            # noqa: E402
from models.vggt_omega_seg_head import VGGTOmegaSegmentationHead    # noqa: E402
from models.vggt_omega_feature_head import build_rgb_guided_feature_head  # noqa: E402
from models.dinov3_teacher import DINOv3Talk2DinoTeacher            # noqa: E402
from utils.scannetpp_labels import SCANNETPP_EVAL, build_raw_to_scannetpp  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import associate                                                    # noqa: E402

# ScanNet++ GT sampled_coords -> nerfstudio/COLMAP frame: swap X/Y and flip Z
# (proper rotation, det=+1). Found by scoring all 24 axis-aligned rotations against
# our depth-unprojected cloud: this one reaches 80-98% voxel coverage on all four
# scenes, whereas the diag(1,1,-1) used by eval_scannetpp_3d.py reaches only 20-29%.
R_GT_TO_NERF = np.array([[0.0, 1.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [0.0, 0.0, -1.0]], dtype=np.float64)

# The four requested text prompts.
TARGET_PROMPTS = ["sofa", "table", "cabinet", "chair"]
TARGET_ZH = {"sofa": "沙发", "table": "桌子", "cabinet": "柜子", "chair": "椅子"}


def build_vocab(targets):
    """Prompt vocabulary: the 4 requested targets, plus two background sets.

    Open-vocabulary segmentation is an argmax over prompts, so the background
    set decides precision: with only a handful of background prompts the targets
    win almost every pixel. Two settings are evaluated in the same pass because
    they differ only in which background prompt wins:

      bg20  the 16 non-target classes of the ScanNet++ 20-class benchmark
            (matches the background the repo's existing 3D eval sees)
      bg141 all 141 ScanNet++ labels, a much harder and more realistic set

    Labels that are synonyms of a target are excluded from both background sets,
    since those points should be won by the target prompt -- the same grouping
    the ground truth uses when scoring.
    """
    from utils.scannetpp_labels import SCANNETPP_CLASS_LABELS, SCANNETPP_EVAL as EV
    syn = {n for t in targets for n in associate.SYNONYMS.get(t, [t])}
    bg20 = [c for c in EV if c not in syn]
    bg141 = [c for c in SCANNETPP_CLASS_LABELS if c not in syn]
    # one flat prompt list: targets, then bg20, then the bg141-only remainder
    extra = [c for c in bg141 if c not in bg20]
    return list(targets) + bg20 + extra, bg20, extra


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def build_model(ckpt_path, device, load_text=True):
    rm = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = rm["args"]
    lora_cfg = dict(rank=a["backbone_lora_rank"], alpha=a["backbone_lora_alpha"],
                    start_layer=a["backbone_lora_start_layer"], inter_frame_multiplier=2)
    bb_ckpt = Path(a["checkpoint"])
    if not bb_ckpt.is_absolute():
        bb_ckpt = ROOT / bb_ckpt
    backbone = VGGTOmegaBackbone(ckpt_path=str(bb_ckpt), use_layers=tuple(a["vggt_layers"]),
                                 amp_dtype=torch.bfloat16, lora_config=lora_cfg)
    inc = backbone.load_state_dict(rm["backbone_lora_state_dict"], strict=False)
    missing_lora = [k for k in inc.missing_keys if "lora" in k]
    if missing_lora or inc.unexpected_keys:
        raise RuntimeError(f"backbone LoRA mismatch: missing={len(missing_lora)} "
                           f"unexpected={len(inc.unexpected_keys)}")
    backbone = backbone.to(device).eval()

    R = a["resolution"]
    feat_head = build_rgb_guided_feature_head(
        device=str(device), n_vggt_layers=len(a["vggt_layers"]), work_dim=a["work_dim"],
        feature_out_dim=a["feature_dim"], feature_res=a["feature_res"],
        start_res=R // 16, lora_rank=a["lora_rank"], lora_alpha=a["lora_alpha"],
        nhead=a["nhead"], use_cross_view=a["use_cross_view"])
    feat_head.load_state_dict(rm["feature_head_state_dict"])
    feat_head.eval()

    seg_head = VGGTOmegaSegmentationHead(
        token_dim=2048, n_vggt_layers=len(a["vggt_layers"]), work_dim=a["work_dim"],
        n_obj_tokens=a["n_obj_tokens"], n_twoway_layers=a["n_twoway_layers"],
        nhead=a["nhead"], pixel_out_dim=a["pixel_out_dim"], pixel_hidden_dims=(384, 192, 96),
        lora_rank=a["lora_rank"], lora_alpha=a["lora_alpha"],
        use_cross_view=a["use_cross_view"], use_geom_pe=a["use_geom_pe"])
    seg_head.load_state_dict(rm["head_state_dict"])
    seg_head = seg_head.to(device).eval()

    teacher = DINOv3Talk2DinoTeacher(teacher_long=a["teacher_long"], device=str(device),
                                     amp_dtype=torch.bfloat16, load_text=load_text).eval()
    meta = dict(epoch=rm.get("epoch"), val_iou=float(rm.get("val_iou", -1)),
                val_cos=float(rm.get("val_cos", -1)), args=a)
    return backbone, feat_head, seg_head, teacher, meta


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def load_scene_frames(scene_dir: Path, stride: int, max_frames: int):
    t = json.load(open(scene_dir / "dslr" / "nerfstudio" / "transforms_undistorted.json"))
    fx, fy, cx, cy = t["fl_x"], t["fl_y"], t["cx"], t["cy"]
    w, h = t["w"], t["h"]
    frames = [f for f in t["frames"] if not f.get("is_bad", False)]
    sel = frames[::max(1, stride)]
    if max_frames > 0:
        sel = sel[:max_frames]
    out = []
    for fr in sel:
        stem = Path(fr["file_path"]).stem
        img = scene_dir / "dslr" / "undistorted_images" / f"{stem}.JPG"
        dep = scene_dir / "dslr" / "undistorted_depths" / f"{stem}.png"
        if img.is_file() and dep.is_file():
            out.append(dict(stem=stem, img=img, depth=dep,
                            c2w=np.array(fr["transform_matrix"], dtype=np.float64)))
    return out, (fx, fy, cx, cy, w, h)


def load_depth(path: Path):
    d = np.array(Image.open(path))
    if np.issubdtype(d.dtype, np.integer):
        return d.astype(np.float32) / 1000.0        # mm -> m
    return d.astype(np.float32)


def unproject(depth, c2w, K, gl=True):
    """depth (h,w) metres -> (P,3) world points + the (ys,xs) pixels used."""
    fx, fy, cx, cy = K
    ys, xs = np.nonzero(depth > 0)
    z = depth[ys, xs]
    x_c = (xs - cx) * z / fx
    y_c = -(ys - cy) * z / fy                       # image v-down -> cam y-up
    z_c = -z if gl else z                           # OpenGL: forward = -z
    pts = np.stack([x_c, y_c, z_c, np.ones_like(z)], 0)
    return (c2w @ pts).T[:, :3], ys, xs


# ---------------------------------------------------------------------------
# voxel bookkeeping
# ---------------------------------------------------------------------------


class VoxelGrid:
    """Sparse voxel accumulator keyed by an int64 linear code."""

    def __init__(self, voxel):
        self.voxel = voxel

    def keys(self, xyz):
        return np.floor(np.asarray(xyz) / self.voxel).astype(np.int64)

    # 18-bit fields with a 2^17 offset: +-6.5 km at 5 cm voxels, no int64 overflow
    _BITS, _OFF = 18, 1 << 17

    @classmethod
    def encode(cls, vk):
        o = (vk + cls._OFF).astype(np.int64)
        if o.size and (o.min() < 0 or o.max() >= (1 << cls._BITS)):
            raise ValueError("voxel coordinate out of packing range")
        return (o[:, 0] << (2 * cls._BITS)) | (o[:, 1] << cls._BITS) | o[:, 2]

    @classmethod
    def decode(cls, code):
        m = (1 << cls._BITS) - 1
        o = np.stack([(code >> (2 * cls._BITS)) & m,
                      (code >> cls._BITS) & m, code & m], 1)
        return o - cls._OFF

    def centers(self, vk):
        return vk.astype(np.float64) * self.voxel + self.voxel / 2


# ---------------------------------------------------------------------------
# prompts derived from the open-vocabulary response
# ---------------------------------------------------------------------------


def seed_points(lab, fg, cls_id, min_area, spacing, max_seeds):
    """Sample one prompt point per object-sized lobe of the class region.

    A connected component of a *semantic* map routinely covers several adjacent
    objects (three chairs at one table touch in the image), so prompting with
    the component's box asks the head to segment all of them at once. Instead we
    drop single positive points inside the region, which is exactly the regime
    the head was trained on with SA-1B instance masks. Seeds are peaks of the
    distance transform with a minimum separation, so touching objects each get
    their own seed.
    """
    m = (lab == cls_id) & fg
    if m.sum() < min_area:
        return []
    m = ndimage.binary_opening(m, np.ones((3, 3)), iterations=1)
    cc, n = ndimage.label(m)
    seeds = []
    for i in range(1, n + 1):
        comp = cc == i
        if comp.sum() < min_area:
            continue
        dt = ndimage.distance_transform_edt(comp)
        work = dt.copy()
        yy, xx = np.mgrid[0:dt.shape[0], 0:dt.shape[1]]
        while work.max() > 0.9:
            idx = int(work.argmax())
            py, px = divmod(idx, work.shape[1])
            seeds.append(dict(cls=cls_id, xy=(px, py), dt=float(dt[py, px]),
                              area=int(comp.sum())))
            r = max(spacing, dt[py, px] * 1.1)
            work[(yy - py) ** 2 + (xx - px) ** 2 <= r * r] = 0
            if len(seeds) >= max_seeds:
                return seeds
    return seeds


def mask_nms(masks, scores, iou_thresh, max_frac, min_frac):
    """Greedy NMS over one view's predicted masks. Returns kept indices."""
    areas = masks.reshape(len(masks), -1).sum(1).astype(np.float64)
    total = masks.shape[-1] * masks.shape[-2]
    ok = (areas / total > min_frac) & (areas / total < max_frac)
    order = [i for i in np.argsort(-scores) if ok[i]]
    kept = []
    flat = masks.reshape(len(masks), -1)
    for i in order:
        dup = False
        for k in kept:
            inter = np.logical_and(flat[i], flat[k]).sum()
            if inter and inter / min(areas[i], areas[k]) > iou_thresh:
                dup = True
                break
        if not dup:
            kept.append(i)
    return kept


def largest_blob_3d(pts, voxel):
    """Keep only the largest 6-connected component of `pts` on a `voxel` grid.

    A single predicted mask can pick up scattered background fragments; without
    this the lifted detection is not a compact object and 3D association turns
    into a chain reaction across the room.
    """
    vk = np.floor(pts / voxel).astype(np.int64)
    lo = vk.min(0)
    shape = (vk.max(0) - lo + 1)
    if np.prod(shape.astype(np.float64)) > 4e7:
        return np.ones(len(pts), bool)
    occ = np.zeros(shape, bool)
    idx = tuple((vk - lo).T)
    occ[idx] = True
    cc, n = ndimage.label(occ)
    if n <= 1:
        return np.ones(len(pts), bool)
    sizes = np.bincount(cc.ravel())
    sizes[0] = 0
    keep_id = int(sizes.argmax())
    return cc[idx] == keep_id


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(ROOT / "output/joint_feat_seg/checkpoint-best.pth"))
    p.add_argument("--data_root", default="/mnt/dataset/marko/puzzles/embodiedsplat_dl/dataset/scannetpp/data")
    p.add_argument("--scene", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_views", type=int, default=6, help="frames per forward (training used 6)")
    p.add_argument("--frame_stride", type=int, default=12)
    p.add_argument("--max_frames", type=int, default=96)
    p.add_argument("--voxel", type=float, default=0.05)
    p.add_argument("--work_scale", type=float, default=0.25,
                   help="depth/intrinsics downsample for unprojection")
    p.add_argument("--max_prompts", type=int, default=16, help="prompts per forward (training: 16)")
    p.add_argument("--min_blob_area", type=int, default=40, help="in 128x128 feature pixels")
    p.add_argument("--seed_spacing", type=float, default=9.0,
                   help="min separation between prompt seeds, in feature pixels")
    p.add_argument("--max_seeds_per_class", type=int, default=6)
    p.add_argument("--mask_thresh", type=float, default=0.5)
    p.add_argument("--nms_iou", type=float, default=0.6,
                   help="per-view intersection-over-smaller above which two masks are duplicates")
    p.add_argument("--max_mask_frac", type=float, default=0.30,
                   help="reject masks covering more than this fraction of the view")
    p.add_argument("--min_view_area", type=float, default=0.004,
                   help="mask area fraction below which a view is treated as 'object absent'")
    p.add_argument("--min_iou_pred", type=float, default=0.40)
    p.add_argument("--max_det_diag", type=float, default=3.2,
                   help="reject a lifted detection whose 3D box diagonal exceeds this (m)")
    p.add_argument("--assoc_voxel_mult", type=float, default=2.0,
                   help="association grid = voxel * this (coarser => tolerant to partial views)")
    p.add_argument("--merge_iou", type=float, default=0.42, help="3D voxel IoU to fuse")
    p.add_argument("--merge_containment", type=float, default=0.75,
                   help="intersection-over-smaller to fuse (handles partial views)")
    p.add_argument("--cross_class_absorb", type=float, default=0.60,
                   help="drop an instance this contained in a larger one of another class")
    p.add_argument("--box_iou_gate", type=float, default=0.05,
                   help="min 3D box IoU before a containment link is accepted")
    p.add_argument("--min_dets", type=int, default=8,
                   help="an instance must be seen by at least this many per-view detections")
    p.add_argument("--min_instance_vox", type=int, default=150)
    p.add_argument("--showcase_batch", type=int, default=-1,
                   help="batch index whose 2D panels are saved (-1 = auto: most prompts)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def main():
    args = get_args()
    rng = np.random.RandomState(args.seed)
    device = torch.device("cuda")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    backbone, feat_head, seg_head, teacher, meta = build_model(args.ckpt, device)
    A = meta["args"]
    R, FR = A["resolution"], A["feature_res"]
    print(f"[model] epoch={meta['epoch']} val_iou={meta['val_iou']:.4f} "
          f"val_cos={meta['val_cos']:.4f} res={R} feat_res={FR}")

    vocab, bg20, extra = build_vocab(TARGET_PROMPTS)
    n_tgt = len(TARGET_PROMPTS)
    n_bg20 = n_tgt + len(bg20)                 # end of the bg20 slice
    text = teacher.text_embed(vocab).to(device).float()          # (K,768)
    text_eval = teacher.text_embed(SCANNETPP_EVAL).to(device).float()
    i_floor, i_ceil = vocab.index("floor"), vocab.index("ceiling")
    sim_cols = TARGET_PROMPTS + ["_bgmax20", "_bgmax141", "floor", "ceiling"]
    print(f"[text] {n_tgt} targets + bg20({len(bg20)}) + bg141 extra({len(extra)}) "
          f"= {len(vocab)} prompts; {len(SCANNETPP_EVAL)} eval classes")

    scene_dir = Path(args.data_root) / args.scene
    frames, (fx, fy, cx, cy, W_full, H_full) = load_scene_frames(
        scene_dir, args.frame_stride, args.max_frames)
    print(f"[data] {args.scene}: {len(frames)} frames  full={W_full}x{H_full}")
    if not frames:
        raise SystemExit("no usable frames")

    s = args.work_scale
    Wh, Ww = int(round(H_full * s)), int(round(W_full * s))
    Kw = (fx * s, fy * s, cx * s, cy * s)
    grid = VoxelGrid(args.voxel)
    # instances are associated on a coarser grid: two views of the same object see
    # different surface patches, so fine-voxel IoU underestimates the match badly
    coarse = VoxelGrid(args.voxel * args.assoc_voxel_mult)

    # --- global accumulators -------------------------------------------------
    acc_code, acc_sim, acc_rgb, acc_cnt = [], [], [], []
    acc_sim_eval = []
    instances = []            # fused 3D instances
    showcase = None
    cam_pos = np.stack([f["c2w"][:3, 3] for f in frames])
    n_batches = (len(frames) + args.n_views - 1) // args.n_views
    t_feat = t_seg = 0.0
    n_prompt_total = 0

    for bi in range(n_batches):
        batch = frames[bi * args.n_views:(bi + 1) * args.n_views]
        N = len(batch)
        imgs = torch.zeros(1, N, 3, R, R, device=device)
        rgb_small = []
        for j, fr in enumerate(batch):
            im = Image.open(fr["img"]).convert("RGB")
            imr = np.asarray(im.resize((R, R), Image.LANCZOS)).copy()
            rgb_small.append(imr)
            imgs[0, j] = torch.from_numpy(imr).permute(2, 0, 1).float() / 255.0

        torch.cuda.synchronize(); t0 = time.time()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            tokens, geom, _ = backbone.forward_split(imgs, return_geometry=True,
                                                     external_geometry=None)
            feat = feat_head(tokens, imgs).float()[0]            # (N,768,FR,FR)
        torch.cuda.synchronize(); t_feat += time.time() - t0

        featn = F.normalize(feat, dim=1)
        sim_all = torch.einsum("kc,nchw->nkhw", text, featn)       # (N,K,FR,FR)
        sim_eval = torch.einsum("kc,nchw->nkhw", text_eval, featn)
        # prompting uses the bg20 setting; both settings are scored in 3D
        tgt_sim = sim_all[:, :n_tgt]
        bg20_max = sim_all[:, n_tgt:n_bg20].amax(1, keepdim=True)
        bg141_max = sim_all[:, n_tgt:].amax(1, keepdim=True)
        lab2d = tgt_sim.argmax(1)                                 # (N,FR,FR)
        fg2d = tgt_sim.amax(1) > bg20_max[:, 0]
        # Only the target scores and the two background maxima are carried into
        # 3D: the open-vocab decision is max(target) vs max(background), so these
        # channels reproduce it exactly while keeping the accumulator small
        # enough to hold a whole scene. 'floor'/'ceiling' ride along so the
        # renderer can orient the scene from the model's own output.
        sim = torch.cat([tgt_sim, bg20_max, bg141_max,
                         sim_all[:, i_floor:i_floor + 1],
                         sim_all[:, i_ceil:i_ceil + 1]], 1)              # (N,8,FR,FR)

        # ---------------- text-derived prompts -> promptable seg -------------
        lab_np, fg_np = lab2d.cpu().numpy(), fg2d.cpu().numpy()
        cand = []
        for v in range(N):
            for c in range(n_tgt):
                for s in seed_points(lab_np[v], fg_np[v], c, args.min_blob_area,
                                     args.seed_spacing, args.max_seeds_per_class):
                    s["view"] = v
                    cand.append(s)
        M = len(cand)
        seg_masks = iou_pred = None
        if M:
            up = R / FR
            # the head takes M prompts per forward; chunk if there are more
            chunks = [cand[i:i + args.max_prompts] for i in range(0, M, args.max_prompts)]
            probs, ious = [], []
            for ch in chunks:
                Mc = len(ch)
                pm = torch.zeros(1, N, Mc, R, R, device=device)
                pv = torch.zeros(1, Mc, dtype=torch.long, device=device)
                pp = torch.zeros(1, N, Mc, 1, 2, device=device)
                pl = -torch.ones(1, N, Mc, 1, device=device)
                for m, s in enumerate(ch):
                    v = s["view"]
                    pv[0, m] = v
                    px, py = s["xy"]
                    pp[0, v, m, 0] = torch.tensor([(px + 0.5) * up, (py + 0.5) * up],
                                                 device=device)
                    pl[0, v, m, 0] = 1.0
                torch.cuda.synchronize(); t0 = time.time()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    so = seg_head(tokens, pm, pv, R, R, prompt_points=pp,
                                  prompt_point_labels=pl, geometry=None)
                torch.cuda.synchronize(); t_seg += time.time() - t0
                probs.append(so["segmentation_logits"].float()[0].sigmoid() > args.mask_thresh)
                ious.append(so["iou_pred"].float()[0])
            seg_masks = torch.cat(probs, 1).cpu().numpy()               # (N,M,R,R)
            iou_pred = torch.cat(ious, 1).cpu().numpy()                 # (N,M)
            n_prompt_total += M

            # per-view NMS: several seeds on one object return the same mask
            keep_mask = np.zeros((N, M), bool)
            for v in range(N):
                for i in mask_nms(seg_masks[v], iou_pred[v], args.nms_iou,
                                  args.max_mask_frac, args.min_view_area):
                    keep_mask[v, i] = True
            keep_mask &= iou_pred >= args.min_iou_pred

        # ---------------- lift this batch into 3D ---------------------------
        for j, fr in enumerate(batch):
            d_full = load_depth(fr["depth"])
            d = np.array(Image.fromarray(d_full).resize((Ww, Wh), Image.NEAREST))
            pts, ys, xs = unproject(d, fr["c2w"], Kw, gl=True)
            if len(pts) == 0:
                continue
            vk = grid.keys(pts)
            code = grid.encode(vk)

            # per-point class similarity + colour, sampled at work resolution
            u = torch.from_numpy(np.stack([(xs + 0.5) / Ww, (ys + 0.5) / Wh], 1)
                                 ).float().to(device)
            gsim = _sample(sim[j], u)                       # (P,K)
            gsim_eval = _sample(sim_eval[j], u)
            col = np.asarray(Image.fromarray(rgb_small[j]).resize((Ww, Wh), Image.BILINEAR))
            gcol = col[ys, xs].astype(np.float32)

            uc, inv = np.unique(code, return_inverse=True)
            cnt = np.bincount(inv, minlength=len(uc)).astype(np.float32)
            acc_code.append(uc)
            acc_cnt.append(cnt)
            acc_sim.append(_seg_sum(gsim.cpu().numpy(), inv, len(uc)))
            acc_sim_eval.append(_seg_sum(gsim_eval.cpu().numpy(), inv, len(uc)))
            acc_rgb.append(_seg_sum(gcol, inv, len(uc)))

            # instance masks for this view -> compact 3D voxel sets
            if seg_masks is not None:
                for m in range(M):
                    if not keep_mask[j, m]:
                        continue
                    mk_w = np.array(Image.fromarray(seg_masks[j, m].astype(np.uint8) * 255)
                                    .resize((Ww, Wh), Image.NEAREST)) > 127
                    sel = np.nonzero(mk_w[ys, xs])[0]
                    if len(sel) < 30:
                        continue
                    p = pts[sel]
                    p = p[largest_blob_3d(p, args.voxel * args.assoc_voxel_mult)]
                    if len(p) < 30:
                        continue
                    ext = p.max(0) - p.min(0)
                    if np.linalg.norm(ext) > args.max_det_diag:
                        continue                        # not a plausible single object
                    instances.append(dict(cls=cand[m]["cls"],
                                          codes=np.unique(grid.encode(grid.keys(p))),
                                          ccodes=np.unique(coarse.encode(coarse.keys(p))),
                                          score=float(iou_pred[j, m]),
                                          views=[(bi, j)]))

        if showcase is None or (args.showcase_batch < 0 and M > showcase["M"]) \
                or bi == args.showcase_batch:
            showcase = dict(batch=bi, M=M, rgb=np.stack(rgb_small),
                            lab=lab_np, fg=fg_np,
                            sim=sim.cpu().numpy().astype(np.float16),
                            masks=(seg_masks if seg_masks is not None
                                   else np.zeros((N, 0, R, R), bool)),
                            keep=(keep_mask if M else np.zeros((N, 0), bool)),
                            iou_pred=(iou_pred if M else np.zeros((N, 0), np.float32)),
                            cand=[dict(view=b["view"], cls=b["cls"], xy=b["xy"],
                                       area=b["area"]) for b in cand],
                            depth=geom["depth"].float()[0, :, :, :, 0].cpu().numpy(),
                            depth_conf=geom["depth_conf"].float()[0].cpu().numpy(),
                            c2w=np.stack([f["c2w"] for f in batch]),
                            stems=[f["stem"] for f in batch])
        print(f"  batch {bi+1}/{n_batches} views={N} prompts={M} "
              f"raw_inst={len(instances)}", flush=True)

    # --- merge voxels --------------------------------------------------------
    print("[merge] fusing voxels ...")
    code = np.concatenate(acc_code)
    uc, inv = np.unique(code, return_inverse=True)
    G = len(uc)
    cnt = np.bincount(inv, weights=np.concatenate(acc_cnt), minlength=G).astype(np.float32)
    vsim = _seg_sum(np.concatenate(acc_sim), inv, G) / np.maximum(cnt[:, None], 1)
    vsim_eval = _seg_sum(np.concatenate(acc_sim_eval), inv, G) / np.maximum(cnt[:, None], 1)
    vrgb = (_seg_sum(np.concatenate(acc_rgb), inv, G) / np.maximum(cnt[:, None], 1))
    vk = grid.decode(uc)
    centers = grid.centers(vk)
    # averaging cosine scores == argmax of the normalised mean feature.
    # vsim columns: [target_0..3, bgmax20, bgmax141, floor, ceiling]
    vlab = vsim[:, :n_tgt].argmax(1).astype(np.int16)
    vfg = vsim[:, :n_tgt].max(1) > vsim[:, n_tgt]           # bg20 setting
    vfg141 = vsim[:, :n_tgt].max(1) > vsim[:, n_tgt + 1]    # bg141 setting
    vlab_eval = vsim_eval.argmax(1).astype(np.int16)
    print(f"[merge] {G} voxels @ {args.voxel} m; target-class voxels: "
          f"bg20={int(vfg.sum())} ({vfg.mean()*100:.1f}%)  "
          f"bg141={int(vfg141.sum())} ({vfg141.mean()*100:.1f}%)")

    # --- fuse per-view detections into 3D instances --------------------------
    assoc_params = dict(merge_iou=args.merge_iou, merge_containment=args.merge_containment,
                        min_dets=args.min_dets, min_instance_vox=args.min_instance_vox,
                        cross_class_absorb=args.cross_class_absorb,
                        box_iou_gate=args.box_iou_gate)
    keep = associate.fuse(instances, decode=VoxelGrid.decode, voxel=args.voxel,
                          **assoc_params)
    keep = attach_voxels(keep, uc, centers)
    for c in range(n_tgt):
        print(f"        {TARGET_PROMPTS[c]:8s}: {sum(1 for f in keep if f['cls'] == c)}")

    # --- GT for evaluation / reference --------------------------------------
    gt, gt_inst, inst_metrics = {}, None, {}
    pth = scene_dir / f"{args.scene}.pth"
    if pth.is_file():
        p = torch.load(pth, map_location="cpu", weights_only=False)
        gxyz = np.asarray(p["sampled_coords"], np.float64) @ R_GT_TO_NERF
        raw2idx = build_raw_to_scannetpp().numpy()
        graw = np.asarray(p["sampled_labels"])
        gcls = raw2idx[np.clip(graw, 0, raw2idx.shape[0] - 1)]
        gcls[graw < 0] = -1
        gcode = grid.encode(grid.keys(gxyz))
        order = np.argsort(uc)
        pos = np.clip(np.searchsorted(uc[order], gcode), 0, G - 1)
        hit = uc[order][pos] == gcode
        vidx = -np.ones(len(gcode), np.int64)
        vidx[hit] = order[pos[hit]]
        print(f"[gt] {len(gxyz)} pts, covered={hit.sum()} ({hit.mean()*100:.1f}%)")
        # 4-class GT for the actual text prompts, with synonyms accepted
        raw2tgt = associate.build_raw_to_target(TARGET_PROMPTS)
        gtgt = raw2tgt[np.clip(graw, 0, raw2tgt.shape[0] - 1)]
        gtgt[graw < 0] = -1
        print(f"[gt] prompt-class GT points: " + " ".join(
            f"{TARGET_PROMPTS[c]}={int((gtgt == c).sum())}" for c in range(n_tgt)))
        gt = dict(xyz=gxyz.astype(np.float32), cls=gcls.astype(np.int16),
                  tgt=gtgt.astype(np.int16), vox_idx=vidx,
                  eval_names=np.array(SCANNETPP_EVAL),
                  synonyms=associate.SYNONYMS)

        gt_inst = associate.load_gt_instances(scene_dir, TARGET_PROMPTS, R_GT_TO_NERF,
                                             grid.encode, grid.keys)
        if gt_inst is not None:
            inst_metrics = associate.match_instances(keep, gt_inst["objs"], n_tgt)
            print(f"[inst-eval] pred={inst_metrics['n_pred']} gt={inst_metrics['n_gt']} "
                  f"mean_best_IoU={inst_metrics['mean_best_iou']:.3f}")
            for th in ("iou25", "iou50"):
                d = inst_metrics[th]
                print(f"            {th}: P={d['precision']*100:.1f} R={d['recall']*100:.1f} "
                      f"F1={d['f1']*100:.1f} (tp={d['tp']} fp={d['fp']} fn={d['fn']})")
            for c in range(n_tgt):
                d = inst_metrics["per_class"][c]
                print(f"            {TARGET_PROMPTS[c]:8s} gt={d['n_gt']} pred={d['n_pred']} "
                      f"best_iou={d['mean_best_iou']:.3f}")

    np.savez_compressed(
        out_dir / f"{args.scene}_bundle.npz",
        scene=args.scene, voxel=args.voxel,
        vox_keys=vk.astype(np.int32), vox_centers=centers.astype(np.float32),
        vox_rgb=vrgb.astype(np.uint8), vox_count=cnt.astype(np.int32),
        vox_sim=vsim.astype(np.float16), vox_label=vlab, vox_fg=vfg, vox_fg141=vfg141,
        vox_sim_eval=vsim_eval.astype(np.float16), vox_label_eval=vlab_eval,
        vocab=np.array(vocab), n_targets=n_tgt, sim_cols=np.array(sim_cols),
        instances=np.array(keep, dtype=object),
        raw_dets=np.array(instances, dtype=object),
        assoc_params=np.array([assoc_params], dtype=object),
        gt_instances=np.array([gt_inst], dtype=object),
        inst_metrics=np.array([inst_metrics], dtype=object),
        cam_pos=cam_pos.astype(np.float32),
        frame_stems=np.array([f["stem"] for f in frames]),
        showcase=np.array([showcase], dtype=object),
        timing=np.array([dict(t_feat=t_feat, t_seg=t_seg, n_frames=len(frames),
                              n_batches=n_batches, n_prompts=n_prompt_total)], dtype=object),
        model=np.array([meta], dtype=object),
        gt=np.array([gt], dtype=object),
    )
    print(f"[done] wrote {out_dir / f'{args.scene}_bundle.npz'}")
    print(f"[time] feature+backbone {t_feat:.1f}s ({t_feat/len(frames)*1000:.0f} ms/frame), "
          f"seg head {t_seg:.1f}s over {n_prompt_total} prompts")


def attach_voxels(keep, uc, centers):
    """Resolve each instance's voxel codes into indices of the global cloud and
    derive its axis-aligned box / centroid from the voxel centres."""
    order = np.argsort(uc)
    for f in keep:
        pos = np.clip(np.searchsorted(uc[order], f["codes"]), 0, len(uc) - 1)
        hit = uc[order][pos] == f["codes"]
        idx = order[pos[hit]]
        f["vox_idx"] = idx
        c = centers[idx]
        f["bmin"], f["bmax"] = c.min(0), c.max(0)
        f["centroid"] = c.mean(0)
    return keep


def _sample(fmap, uv):
    """fmap (C,H,W), uv (P,2) in [0,1] -> (P,C) bilinear."""
    g = (uv * 2 - 1).view(1, 1, -1, 2)
    return F.grid_sample(fmap[None], g, mode="bilinear", align_corners=False)[0, :, 0].T


def _seg_sum(vals, inv, G):
    """Sum rows of `vals` (P,C) into G groups. Column-wise bincount is ~100x
    faster than np.add.at for the millions of points involved here."""
    vals = np.ascontiguousarray(vals, dtype=np.float64)
    out = np.empty((G, vals.shape[1]), np.float32)
    for c in range(vals.shape[1]):
        out[:, c] = np.bincount(inv, weights=vals[:, c], minlength=G)
    return out


if __name__ == "__main__":
    main()
