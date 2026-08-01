"""Visualise JAFAR-Puzzles Stage-1 features on SceneNN with the NEW
DINOv3-trained RGBGuidedFeatureHead (128x128x768, RGB-guided).

Same idea as infer_scenenn_feature_vis.py, but the head now needs the input
images (RGB guidance) and outputs 768-d DINOv3-aligned features at 128x128.

For a SceneNN scene, sample N nearby frames, run them TOGETHER through the
frozen VGGT-Omega backbone (cross-view attention) + RGBGuidedFeatureHead, then
render:
  row 1: original frames
  row 2: predicted dense feature maps (shared PCA-3 across all frames)

A single PCA is fit jointly over all N feature maps so the same semantics map
to the same colour across views -> row-2 colour consistency shows the predicted
DINOv3-aligned features are stable across viewpoints.

Usage:
    python infer_scenenn_feature_vis_dinov3.py \
        --scene 263 --n_views 6 \
        --feature_head_ckpt output/feat_dinov3/checkpoint-best.pth \
        --out repo/JAFAR/vis/output/scenenn_dinov3/scenenn_263_feature_vis.png
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
from models.vggt_omega_feature_head import build_rgb_guided_feature_head  # noqa: E402


def list_frames(scene_dir: Path) -> list[Path]:
    imgs = sorted((scene_dir / "image").glob("image*.png"))
    if not imgs:
        raise FileNotFoundError(f"no frames under {scene_dir/'image'}")
    return imgs


def pick_frames(frames: list[Path], n: int, start_frac: float, view_stride: int) -> list[Path]:
    """Pick a CONTIGUOUS window of nearby frames so viewpoints overlap heavily."""
    if len(frames) <= n:
        return frames
    span = n * view_stride
    start = int(round((len(frames) - span) * start_frac))
    start = max(0, min(start, len(frames) - span))
    sel = frames[start:start + span:view_stride]
    return sel[:n]


def load_image_tensor(path: Path, resolution: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((resolution, resolution), Image.BILINEAR)
    return torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().div_(255.0)


def shared_pca3(feats: torch.Tensor) -> torch.Tensor:
    """feats: (N, C, H, W) -> (N, 3, H, W). One PCA fit jointly over all frames."""
    N, C, H, W = feats.shape
    x = feats.permute(0, 2, 3, 1).reshape(-1, C).float()      # (N*H*W, C)
    mean = x.mean(dim=0, keepdim=True)
    xc = x - mean
    cov = (xc.t() @ xc) / max(xc.shape[0], 1)                 # (C, C)
    cov = torch.nan_to_num(cov)
    _, eigvecs = torch.linalg.eigh(cov)
    comp = eigvecs[:, -3:]                                    # (C, 3), top-3
    proj = xc @ comp                                          # (N*H*W, 3)
    lo = proj.min(dim=0, keepdim=True).values
    hi = proj.max(dim=0, keepdim=True).values
    proj = (proj - lo) / (hi - lo).clamp_min(1e-6)           # normalise jointly
    return proj.reshape(N, H, W, 3).permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def run(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    scene_dir = Path(args.scenenn_root) / args.scene
    frames = pick_frames(list_frames(scene_dir), args.n_views, args.start_frac, args.view_stride)
    print(f"scene {args.scene}: using {len(frames)} frames -> {[f.name for f in frames]}")

    imgs = torch.stack([load_image_tensor(f, args.resolution) for f in frames])   # (N,3,H,W)
    img_batch = imgs.unsqueeze(0).to(device)                                       # (1,N,3,H,W)

    backbone = VGGTOmegaBackbone(
        ckpt_path=args.checkpoint, use_layers=tuple(args.vggt_layers),
        amp_dtype=torch.bfloat16,
    ).to(device).eval()
    feature_head = build_rgb_guided_feature_head(
        device=str(device), n_vggt_layers=len(args.vggt_layers),
        work_dim=args.work_dim, feature_out_dim=args.feature_dim,
        feature_res=args.feature_res, start_res=args.resolution // 16,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, nhead=args.nhead,
        use_cross_view=not args.no_cross_view,
    )
    ck = torch.load(args.feature_head_ckpt, map_location=device, weights_only=False)
    feature_head.load_state_dict(ck["feature_head_state_dict"], strict=False)
    feature_head.eval()
    print(f"loaded feature head (val_cos={ck.get('val_cos','?')})")

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        tokens_list, _ = backbone(img_batch)
        feats = feature_head(tokens_list, img_batch)          # (1, N, C, Fr, Fr)
    feats = feats[0].float().cpu()                            # (N, C, Fr, Fr)

    pca = shared_pca3(feats)                                   # (N, 3, Fr, Fr)
    pca_up = F.interpolate(pca, (args.resolution, args.resolution),
                           mode="bilinear", align_corners=False)

    _render(imgs, pca_up, frames, args)


def _render(imgs, pca_up, frames, args):
    import matplotlib.pyplot as plt
    N = imgs.shape[0]
    fig, axes = plt.subplots(2, N, figsize=(2.2 * N, 4.8), dpi=150)
    if N == 1:
        axes = axes.reshape(2, 1)
    for j in range(N):
        axes[0, j].imshow(imgs[j].permute(1, 2, 0).numpy())
        axes[0, j].set_title(frames[j].name.replace("image", "").replace(".png", ""),
                             fontsize=8)
        axes[0, j].axis("off")
        axes[1, j].imshow(pca_up[j].permute(1, 2, 0).numpy())
        axes[1, j].axis("off")
    axes[0, 0].set_ylabel("RGB", fontsize=10)
    axes[1, 0].set_ylabel("Feature PCA3", fontsize=10)
    fig.suptitle(
        f"SceneNN {args.scene} - Stage-1 DINOv3-aligned features "
        f"(RGB-guided head, shared PCA-3, {N} views)", fontsize=11)
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out)


def get_args():
    p = argparse.ArgumentParser("SceneNN Stage-1 feature visualisation (DINOv3 head)")
    p.add_argument("--scene", default="263")
    p.add_argument("--scenenn_root",
                   default="/mnt/dataset/marko/puzzles/puzzles/ESAMBench/data/scenenn_raw/oni")
    p.add_argument("--feature_head_ckpt",
                   default="output/feat_dinov3/checkpoint-best.pth")
    p.add_argument("--checkpoint",
                   default="/mnt/dataset/marko/puzzles/puzzles/checkpoints/vggt_omega_1b_256_text.pt")
    p.add_argument("--out", default="repo/JAFAR/vis/output/scenenn_dinov3/scenenn_feature_vis.png")
    p.add_argument("--n_views", type=int, default=6)
    p.add_argument("--start_frac", type=float, default=0.25,
                   help="Where in the scene the contiguous window starts (0..1).")
    p.add_argument("--view_stride", type=int, default=1,
                   help="Index step in the on-disk frame list (1 = adjacent frames).")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--feature_res", type=int, default=128)
    p.add_argument("--feature_dim", type=int, default=768)
    p.add_argument("--vggt_layers", type=int, nargs="+", default=[4, 11, 17, 23])
    p.add_argument("--work_dim", type=int, default=768)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=1.0)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--no_cross_view", action="store_true")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


if __name__ == "__main__":
    run(get_args())
