"""Inference demo for the JAFAR-Puzzles feature branch.

Demonstrates that the trained FeatureSequenceHead produces CLIP-aligned dense
features and supports promptable segmentation, without needing a text encoder.

Modes:
  --mode reconstruct  (default)
      Verifies the Stage-1 feature head reconstructs JAFAR features.
      Input image -> frozen VGGT-Omega backbone -> FeatureSequenceHead ->
      predicted PCA64 feature map. Compared against the cached JAFAR teacher
      (PCA64) for the same image via cosine-similarity + PCA(3) visualisations.

  --mode promptable
      Promptable segmentation via feature matching. Given a box / point / mask
      prompt on the image, extract a prototype from the predicted feature map at
      the prompt region, then compute a cosine-similarity map over the whole
      image -> threshold + largest connected component -> mask overlay. This is
      the dense-feature analogue of SAM-style promptable segmentation, using the
      CLIP-aligned features so the prototype generalises across viewpoints.

  --text <prompt>   (optional, experimental)
      Best-effort open-vocabulary matching: a CLIP text embedding (open_clip
      laion2b_s34b_b88k ViT-B/16) is projected through the cached PCA matrix into
      the 64-d feature space and matched against the predicted features. Note:
      the text encoder is an approximation of the LAION2b_ft_in12k image encoder
      used for the cache, so text results are indicative, not exact.

Usage:
    python infer_jafar_feature_demo.py \
        --image /mnt/dataset/marko/puzzles/sa1b/data/sa_000047/sa_536209.jpg \
        --feature_head_ckpt output/feat_stage1/checkpoint-best.pth \
        --checkpoint /mnt/dataset/marko/puzzles/puzzles/checkpoints/vggt_omega_1b_256_text.pt \
        --mode reconstruct --out_dir repo/JAFAR/vis/output/infer
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.vggt_omega_backbone import VGGTOmegaBackbone  # noqa: E402
from models.vggt_omega_feature_head import build_feature_sequence_head  # noqa: E402


# ---------------------------------------------------------------------------
# Image preprocessing for the VGGT-Omega backbone (256x256, [0,1]).
# ---------------------------------------------------------------------------

def load_image_tensor(path: str, resolution: int = 256) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((resolution, resolution), Image.BILINEAR)
    t = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().div_(255.0)
    return t.unsqueeze(0).unsqueeze(0)        # (1, 1, 3, H, W)


def pca3(feat: torch.Tensor) -> torch.Tensor:
    """feat: (C, H, W) -> (3, H, W) via top-3 eigenvectors of the CxC covariance."""
    C, H, W = feat.shape
    x = feat.reshape(C, -1).float()
    x = x - x.mean(dim=1, keepdim=True)
    cov = x @ x.t() / max(x.shape[1], 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    proj = eigvecs[:, -3:].t() @ x
    proj = proj - proj.min(dim=1, keepdim=True).values
    proj = proj / proj.max(dim=1, keepdim=True).values.clamp_min(1e-6)
    return proj.reshape(3, H, W)


def largest_cc(mask: np.ndarray) -> np.ndarray:
    import cv2
    mask_u8 = (mask > 0).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n <= 1:
        return mask_u8.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = 1 + int(areas.argmax())
    return lbl == biggest


def overlay(img: torch.Tensor, mask: torch.Tensor, color=(0.3, 0.9, 0.4), alpha=0.5) -> torch.Tensor:
    """img: (3,H,W) [0,1], mask: (H,W) -> (3,H,W) with mask coloured overlay."""
    out = img.clone()
    m = (mask > 0.5).float().unsqueeze(0)                        # (1,H,W)
    cm = torch.tensor(color, dtype=img.dtype).view(3, 1, 1) * m  # (3,H,W)
    out = out * (1 - alpha * m) + cm * alpha
    return out.clamp(0, 1)


def save_grid(panels, path, nrow):
    import torchvision.utils as vutils
    g = vutils.make_grid(panels, nrow=nrow, padding=6, pad_value=0.2)
    vutils.save_image(g, path)
    print("saved:", path)


# ---------------------------------------------------------------------------
# Feature head inference
# ---------------------------------------------------------------------------

def predict_features(image_path, backbone, feature_head, device, resolution=256):
    img_t = load_image_tensor(image_path, resolution).to(device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        tokens_list, _ = backbone(img_t)
        feat = feature_head(tokens_list)        # (1, 1, C, Fr, Fr)
    img = img_t[0, 0].cpu().float()
    feat = feat[0, 0].float().cpu()
    return img, feat


def load_teacher(image_path, cache_root, feature_res, feature_dim, device):
    """Load the cached JAFAR PCA64 teacher for the image and upsample to Fr."""
    p = Path(image_path)
    shard = p.parent.name
    stem = p.stem
    npz = Path(cache_root) / shard / f"{stem}.npz"
    if not npz.is_file():
        return None
    z = np.load(npz)
    teacher = torch.from_numpy(np.ascontiguousarray(z["parent_feature"])).float()
    teacher = F.normalize(teacher.unsqueeze(0), dim=1).squeeze(0)   # (C, Hf, Wf)
    if teacher.shape[-1] != feature_res or teacher.shape[-2] != feature_res:
        teacher = F.interpolate(teacher.unsqueeze(0), (feature_res, feature_res),
                                mode="bilinear", align_corners=False).squeeze(0)
    return teacher


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def mode_reconstruct(args, backbone, feature_head, device):
    img, feat = predict_features(args.image, backbone, feature_head, device, args.resolution)
    teacher = load_teacher(args.image, args.jafar_cache_root, args.feature_res,
                           args.feature_dim, device)
    hw = (args.resolution, args.resolution)
    pred_pca = F.interpolate(pca3(feat).unsqueeze(0), hw, mode="bilinear",
                             align_corners=False).squeeze(0)
    panels = [img, pred_pca]
    if teacher is not None:
        cos = (feat * teacher).sum(0, keepdim=True).clamp(-1, 1)
        teach_pca = F.interpolate(pca3(teacher).unsqueeze(0), hw, mode="bilinear",
                                  align_corners=False).squeeze(0)
        cos_up = F.interpolate(cos.unsqueeze(0), hw, mode="bilinear",
                               align_corners=True).squeeze(0)
        cos_map = cos_up.repeat(3, 1, 1) * 0.5 + 0.5
        panels += [teach_pca, cos_map]
        mean_cos = (feat * teacher).sum(0).mean().item()
        print(f"mean cosine(pred, teacher) = {mean_cos:.4f}")
    else:
        print("no cached teacher found for this image; showing pred only.")
    save_grid(panels, str(Path(args.out_dir) / "reconstruct.png"), nrow=len(panels))


def mode_promptable(args, backbone, feature_head, device):
    img, feat = predict_features(args.image, backbone, feature_head, device, args.resolution)
    C, Fr, _ = feat.shape
    H = W = args.resolution
    # Build a binary prompt mask at feature resolution.
    prompt_mask = torch.zeros(Fr, Fr, dtype=torch.float32)
    if args.box is not None:
        x1, y1, x2, y2 = [float(v) for v in args.box]
        # image coords -> feature coords
        fx1 = int(max(0, min(Fr - 1, x1 / W * Fr)))
        fy1 = int(max(0, min(Fr - 1, y1 / H * Fr)))
        fx2 = int(max(0, min(Fr, x2 / W * Fr)))
        fy2 = int(max(0, min(Fr, y2 / H * Fr)))
        prompt_mask[fy1:fy2, fx1:fx2] = 1.0
        print(f"box prompt (image px) = ({x1},{y1},{x2},{y2}) -> feat region "
              f"({fx1},{fy1},{fx2},{fy2})")
    elif args.point is not None:
        px, py = [float(v) for v in args.point]
        fpx = int(min(Fr - 1, px / W * Fr))
        fpy = int(min(Fr - 1, py / H * Fr))
        r = max(1, Fr // 32)
        yy, xx = torch.meshgrid(torch.arange(Fr), torch.arange(Fr), indexing="ij")
        prompt_mask = (((xx - fpx).abs() <= r) & ((yy - fpy).abs() <= r)).float()
        print(f"point prompt (image px) = ({px},{py}) -> feat center ({fpx},{fpy}) r={r}")
    else:
        raise ValueError("promptable mode requires --box or --point")
    if prompt_mask.sum() < 1:
        raise ValueError("prompt region is empty")
    proto = (feat * prompt_mask.unsqueeze(0)).sum(dim=(1, 2)) / prompt_mask.sum().clamp_min(1.0)
    proto = F.normalize(proto, dim=0)
    sim = (feat * proto.view(C, 1, 1)).sum(0)        # (Fr, Fr) in [-1,1]
    sim_up = F.interpolate(sim.unsqueeze(0).unsqueeze(0), (H, W),
                           mode="bilinear", align_corners=True).squeeze().clamp(-1, 1)
    prob = (sim_up + 1) / 2
    thr = args.threshold
    mask = largest_cc((prob > thr).numpy())
    mask_t = torch.from_numpy(mask).float()
    panels = [
        img,
        overlay(img, mask_t),
        prob.unsqueeze(0).repeat(3, 1, 1),
        overlay(img, prompt_mask_interpolated(prompt_mask, H)),
    ]
    save_grid(panels, str(Path(args.out_dir) / "promptable.png"), nrow=4)


def prompt_mask_interpolated(prompt_mask, H):
    return F.interpolate(prompt_mask.unsqueeze(0).unsqueeze(0), (H, H),
                         mode="nearest").squeeze()


def text_openvocab(args, backbone, feature_head, device, pca_mean, pca_comp):
    """Best-effort open-vocab via CLIP text encoder + PCA projection."""
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k")
    model = model.to(device).eval()
    tok = open_clip.get_tokenizer("ViT-B-16")
    prompts = [f"a photo of a {t}" for t in args.text.split("|")]
    with torch.no_grad():
        emb = model.encode_text(tok(prompts).to(device))
        emb = emb / emb.norm(dim=-1, keepdim=True)               # (T, 768)
        # project through PCA (image-fit) -> (T, 64)
        proj = (emb - pca_mean) @ pca_comp.t()                   # (T, 64)
        proj = F.normalize(proj, dim=-1)
    img, feat = predict_features(args.image, backbone, feature_head, device, args.resolution)
    C, Fr, _ = feat.shape
    feat_dev = feat.to(device)
    sim = torch.einsum("chw,tc->thw", feat_dev, proj).clamp(-1, 1)   # (T, Fr, Fr)
    H = W = args.resolution
    panels = [img]
    for t in range(sim.shape[0]):
        s = sim[t].cpu()
        prob = ((s + 1) / 2)
        prob_up = F.interpolate(prob.unsqueeze(0).unsqueeze(0), (H, W),
                                mode="bilinear", align_corners=True).squeeze()
        mask = largest_cc((prob_up > args.threshold).numpy())
        panels.append(img)
        panels.append(overlay(img, torch.from_numpy(mask).float()))
        panels.append(prob_up.unsqueeze(0).repeat(3, 1, 1))
    save_grid(panels, str(Path(args.out_dir) / "openvocab_text.png"), nrow=3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser("JAFAR feature inference demo")
    p.add_argument("--image", required=True)
    p.add_argument("--feature_head_ckpt", required=True)
    p.add_argument("--checkpoint", default="/mnt/dataset/marko/puzzles/puzzles/checkpoints/vggt_omega_1b_256_text.pt")
    p.add_argument("--jafar_cache_root",
                   default="/mnt/dataset/marko/puzzles/sa1b/data/jafar_clip_cache/pca64_long192_ar")
    p.add_argument("--pca_matrix",
                   default="/mnt/dataset/marko/puzzles/sa1b/data/jafar_clip_cache/pca64_long192_ar/pca_matrix.pt")
    p.add_argument("--out_dir", default="repo/JAFAR/vis/output/infer")
    p.add_argument("--mode", choices=["reconstruct", "promptable"], default="reconstruct")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--feature_res", type=int, default=64)
    p.add_argument("--feature_dim", type=int, default=64)
    p.add_argument("--vggt_layers", type=int, nargs="+", default=[4, 11, 17, 23])
    p.add_argument("--work_dim", type=int, default=768)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=1.0)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--no_cross_view", action="store_true")
    p.add_argument("--box", nargs=4, type=float, default=None, help="x1 y1 x2 y2 (image px)")
    p.add_argument("--point", nargs=2, type=float, default=None, help="x y (image px)")
    p.add_argument("--text", type=str, default=None,
                   help="Experimental open-vocab text prompt(s), '|'-separated.")
    p.add_argument("--threshold", type=float, default=0.55)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    backbone = VGGTOmegaBackbone(
        ckpt_path=args.checkpoint, use_layers=tuple(args.vggt_layers),
        amp_dtype=torch.bfloat16,
    ).to(device); backbone.eval()
    feature_head = build_feature_sequence_head(
        device=str(device), n_vggt_layers=len(args.vggt_layers),
        work_dim=args.work_dim, feature_out_dim=args.feature_dim,
        feature_res=args.feature_res, lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha, nhead=args.nhead,
        use_cross_view=not args.no_cross_view,
    )
    ck = torch.load(args.feature_head_ckpt, map_location=device, weights_only=False)
    feature_head.load_state_dict(ck["feature_head_state_dict"], strict=False)
    feature_head.eval()
    print(f"loaded feature head from {args.feature_head_ckpt} (val_cos={ck.get('val_cos','?')})")

    if args.mode == "reconstruct":
        mode_reconstruct(args, backbone, feature_head, device)
    elif args.mode == "promptable":
        mode_promptable(args, backbone, feature_head, device)

    if args.text:
        obj = torch.load(args.pca_matrix, map_location=device, weights_only=False)
        text_openvocab(args, backbone, feature_head, device,
                       pca_mean=obj["mean"].float(), pca_comp=obj["components"].float())


if __name__ == "__main__":
    main()
