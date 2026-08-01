"""Open-vocabulary segmentation eval for the trained student (VGGT-Omega + head).

Pipeline (matches JAFAR_Puzzles.md §7-8):
    sub-views -> frozen VGGT-Omega tokens -> RGBGuidedFeatureHead -> pred (N,768,128,128)
    text classnames -> OpenAI-CLIP(quickgelu) -> Talk2DINO proj -> DINOv3-768
    cosine(pred, text) -> per-class sim -> argmax segmentation

Reports label-free proxies:
    separability   : mean top1-top2 softmax margin  (higher = more discriminative)
    edge_alignment : label-boundary vs image-gradient corr (higher = sharper)
for the STUDENT and for the DINOv3-warp TEACHER (upper-bound reference).

Also saves qualitative panels: [image | student seg | teacher seg] per view,
and single-prompt cosine heatmaps.

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval_student_openvocab.py \
        --ckpt output/feat_dinov3/checkpoint-best.pth \
        --out  output/feat_dinov3/eval_openvocab --n_samples 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJ = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ / "src"))

from sa1b.dinov3_onthefly_dataset import DINOv3OnTheFlyDataset, dinov3_collate  # noqa: E402
from models.vggt_omega_backbone import VGGTOmegaBackbone  # noqa: E402
from models.vggt_omega_feature_head import build_rgb_guided_feature_head  # noqa: E402
from models.dinov3_teacher import DINOv3Talk2DinoTeacher, warp_parent_to_views  # noqa: E402


def separability(sim: torch.Tensor) -> float:
    """Mean top1-top2 softmax margin over pixels (higher = more discriminative)."""
    s = (sim * 50.0).softmax(dim=0)
    top2 = s.topk(2, dim=0).values
    return float((top2[0] - top2[1]).mean())


def edge_alignment(argmax: np.ndarray, gray: np.ndarray) -> float:
    """Correlation between label boundaries and image gradient magnitude."""
    from scipy import ndimage as ndi
    lab = argmax.astype(np.float32)
    bnd = (ndi.maximum_filter(lab, 3) != ndi.minimum_filter(lab, 3)).astype(np.float32)
    gm = np.hypot(*np.gradient(gray.astype(np.float32)))
    gm = gm / (gm.max() + 1e-6)
    if bnd.sum() < 1:
        return 0.0
    return float((gm * bnd).sum() / (bnd.sum() + 1e-6))

# A broad, generic open vocabulary (SA-1B has no semantic GT; these proxies are
# label-free, so the vocabulary only needs to span typical scene content).
VOCAB = [
    "sky", "tree", "grass", "plant", "building", "wall", "window", "door",
    "road", "ground", "floor", "stairs", "stone", "water", "person", "car",
    "furniture", "food", "sign", "fence",
]

# 20 visually-distinct colors for the argmax overlay.
_PALETTE = np.array([
    [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
    [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
    [210, 245, 60], [250, 190, 212], [0, 128, 128], [220, 190, 255],
    [170, 110, 40], [255, 250, 200], [128, 0, 0], [170, 255, 195],
    [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128],
], dtype=np.uint8)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="output/feat_dinov3/checkpoint-best.pth")
    p.add_argument("--out", type=str, default="output/feat_dinov3/eval_openvocab")
    p.add_argument("--sa1b_root", type=str, default="/mnt/dataset/marko/puzzles/sa1b/data")
    p.add_argument("--checkpoint", type=str, default="checkpoints/vggt_omega_1b_256_text.pt")
    p.add_argument("--n_samples", type=int, default=6)
    p.add_argument("--n_views", type=int, default=6)
    p.add_argument("--vis_views", type=int, default=3)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--teacher_long", type=int, default=2048)
    p.add_argument("--feature_res", type=int, default=128)
    p.add_argument("--start_idx", type=int, default=0)
    return p.parse_args()


def seg_from_feat(feat_chw, text_kc, out_hw):
    """feat (C,Fr,Fr) L2-normed, text (K,C) L2-normed -> (sim(K,H,W), argmax(H,W))."""
    sim = torch.einsum("kc,chw->khw", text_kc, feat_chw)          # (K,Fr,Fr)
    sim = F.interpolate(sim.unsqueeze(0), out_hw, mode="bilinear",
                        align_corners=False).squeeze(0)
    return sim, sim.argmax(0).cpu().numpy()


def overlay(img_hwc_u8, argmax_hw, alpha=0.55):
    color = _PALETTE[argmax_hw % len(_PALETTE)]
    return (img_hwc_u8 * (1 - alpha) + color * alpha).astype(np.uint8)


def gray_of(img_hwc_u8):
    return (img_hwc_u8.astype(np.float32) @ np.array([0.299, 0.587, 0.114],
                                                     dtype=np.float32))


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

    head = build_rgb_guided_feature_head(
        device=str(device), n_vggt_layers=4, work_dim=768,
        feature_out_dim=768, feature_res=args.feature_res,
        start_res=args.resolution // 16, use_cross_view=True)
    rm = torch.load(args.ckpt, map_location=device, weights_only=False)
    head.load_state_dict(rm["feature_head_state_dict"])
    head.eval()
    print(f"loaded ckpt epoch={rm.get('epoch')} val_cos={rm.get('val_cos'):.4f}")

    text = teacher_model.text_embed(VOCAB).to(device)            # (K,768)
    print(f"vocab={len(VOCAB)} text_embed={tuple(text.shape)}")

    ds = DINOv3OnTheFlyDataset(
        sa1b_root=args.sa1b_root, teacher_long=args.teacher_long,
        n_views=args.n_views, resolution=args.resolution, augment=False)

    agg = {"stu_sep": [], "stu_edge": [], "tea_sep": [], "tea_edge": []}
    n_done = 0
    idx = args.start_idx
    while n_done < args.n_samples and idx < len(ds):
        try:
            batch = dinov3_collate([ds[idx]])
        except Exception as e:  # noqa: BLE001
            print(f"skip idx {idx}: {e}")
            idx += 1
            continue
        idx += 1

        imgs = batch["imgs"].to(device)                          # (1,N,3,r,r)
        parent_img = batch["parent_img"].to(device)
        if not parent_img.is_floating_point():
            parent_img = parent_img.float().div_(255.0)
        boxes = batch["crop_boxes_parent"].to(device)
        psize = batch["parent_size"].to(device)
        N = imgs.shape[1]
        r = args.resolution

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            tokens_list, _ = backbone(imgs)
            pred = head(tokens_list, imgs)                        # (1,N,768,Fr,Fr)
        pred = pred.float()
        pf = teacher_model.encode_parent(parent_img)
        teacher = warp_parent_to_views(pf, boxes, psize, args.feature_res).float()

        rows = []
        for v in range(N):
            img_u8 = (imgs[0, v].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            g = gray_of(img_u8)

            s_sim, s_arg = seg_from_feat(pred[0, v], text, (r, r))
            t_sim, t_arg = seg_from_feat(teacher[0, v], text, (r, r))

            agg["stu_sep"].append(separability(s_sim.cpu()))
            agg["stu_edge"].append(edge_alignment(s_arg, g))
            agg["tea_sep"].append(separability(t_sim.cpu()))
            agg["tea_edge"].append(edge_alignment(t_arg, g))

            if v < args.vis_views:
                rows.append(np.concatenate(
                    [img_u8, overlay(img_u8, s_arg), overlay(img_u8, t_arg)], axis=1))
        if rows:
            panel = np.concatenate(rows, axis=0)
            Image.fromarray(panel).save(out_dir / f"sample{n_done}_seg_img_student_teacher.png")

        # single-prompt heatmap for the most-activated class on view 0 (student)
        s_sim0, _ = seg_from_feat(pred[0, 0], text, (r, r))
        top_k = int(s_sim0.mean(dim=(1, 2)).argmax())
        hm = s_sim0[top_k].clamp(0, 1).cpu().numpy()
        hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-6)
        img0 = (imgs[0, 0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        heat = np.stack([hm, np.zeros_like(hm), 1 - hm], axis=-1)
        blend = (img0 * 0.5 + heat * 255 * 0.5).astype(np.uint8)
        Image.fromarray(np.concatenate([img0, blend], axis=1)).save(
            out_dir / f"sample{n_done}_prompt_{VOCAB[top_k]}.png")

        print(f"sample {n_done} (idx {idx-1}): "
              f"student sep={np.mean(agg['stu_sep'][-N:]):.4f} edge={np.mean(agg['stu_edge'][-N:]):.4f} | "
              f"teacher sep={np.mean(agg['tea_sep'][-N:]):.4f} edge={np.mean(agg['tea_edge'][-N:]):.4f} "
              f"| top='{VOCAB[top_k]}'")
        n_done += 1

    print("\n================ open-vocab eval summary ================")
    print(f"samples={n_done}  views/sample={args.n_views}  vocab={len(VOCAB)}")
    print(f"{'':10s}{'separability':>16s}{'edge_alignment':>16s}   (higher is better)")
    print(f"{'student':10s}{np.mean(agg['stu_sep']):>16.4f}{np.mean(agg['stu_edge']):>16.4f}")
    print(f"{'teacher':10s}{np.mean(agg['tea_sep']):>16.4f}{np.mean(agg['tea_edge']):>16.4f}")
    print(f"student/teacher ratio: sep={np.mean(agg['stu_sep'])/max(np.mean(agg['tea_sep']),1e-6):.2%} "
          f"edge={np.mean(agg['stu_edge'])/max(np.mean(agg['tea_edge']),1e-6):.2%}")
    print(f"\nsaved panels to {out_dir}")


if __name__ == "__main__":
    main()
