"""Inference script for Pi3X multi-view consistent segmentation.

Given a set of multi-view images and a prompt mask/bbox on view 0,
this script propagates the segmentation across all views and saves
a visualisation grid.

Usage:
    python inference_segmentation.py \
        --images_dir examples/room/rgb \
        --prompt_mask examples/room/prompt_mask.png \
        --seg_checkpoint output/seg/checkpoint-best.pth \
        --output_dir output/seg/inference
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "repo" / "Pi3"))

from models.pi3x_segmentation import build_pi3x_segmentation  # noqa: E402
from models.query_segmentation_head import QuerySegmentationHead  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def load_images(
    source: str, *, resolution: int = 224, interval: int = 1
) -> Tuple[torch.Tensor, List[np.ndarray]]:
    """Load images from a directory or video file.

    Returns:
        imgs:     (1, N, 3, H, W) float32 tensor in [0, 1]
        originals: list of N original-size BGR numpy arrays
    """
    path = Path(source)
    raw_frames: List[np.ndarray] = []

    if path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        paths = sorted(p for p in path.iterdir() if p.suffix.lower() in exts)
        for p in paths[::interval]:
            raw_frames.append(cv2.imread(str(p)))
    elif path.suffix.lower() in {".mp4", ".avi", ".mkv", ".webm", ".mov"}:
        cap = cv2.VideoCapture(str(path))
        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                raw_frames.append(frame)
            idx += 1
        cap.release()
    else:
        raise ValueError(f"Unsupported source: {source}")

    if not raw_frames:
        raise FileNotFoundError(f"No frames found in {source}")

    tensors = []
    for f in raw_frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(resized).float() / 255.0
        tensors.append(t.permute(2, 0, 1))

    imgs = torch.stack(tensors).unsqueeze(0)
    return imgs, raw_frames


def load_prompt_mask(
    path: str, resolution: int = 224
) -> torch.Tensor:
    """Load a single binary mask image as (1, 1, 1, H, W) tensor."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read prompt mask: {path}")
    mask = cv2.resize(mask, (resolution, resolution), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.float32)
    return torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).unsqueeze(0)


def create_bbox_mask(
    bbox: Tuple[int, int, int, int],
    resolution: int = 224,
    img_h: int = 224,
    img_w: int = 224,
) -> torch.Tensor:
    """Create a filled rectangle mask from (x1, y1, x2, y2) in pixel coords."""
    x1, y1, x2, y2 = bbox
    mask = np.zeros((img_h, img_w), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    mask = cv2.resize(mask, (resolution, resolution), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer(
    model: torch.nn.Module,
    imgs: torch.Tensor,
    prompt_mask: torch.Tensor,
    device: torch.device,
    threshold: float = 0.5,
) -> np.ndarray:
    """Run segmentation inference.

    Args:
        model:        Pi3XSegmentation
        imgs:         (1, N, 3, H, W)
        prompt_mask:  (1, 1, 1, H, W) – mask on view 0

    Returns:
        pred_masks: (N, H, W) binary uint8 array
    """
    N = imgs.shape[1]
    _, _, _, H, W = imgs.shape

    # Place prompt only on view 0; all other views get zeros
    # (matching the single-view prompt paradigm used in training)
    sam_masks = torch.zeros(1, N, prompt_mask.shape[2], H, W)
    sam_masks[:, 0] = prompt_mask[:, 0]

    imgs = imgs.to(device)
    sam_masks = sam_masks.to(device)

    model.eval()
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.amp.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
        results = model(imgs, sam_masks)

    logits = results["segmentation_logits"]
    pred = (logits[0, :, 0].sigmoid() > threshold).cpu().numpy().astype(np.uint8) * 255
    return pred


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualise_predictions(
    originals: List[np.ndarray],
    pred_masks: np.ndarray,
    output_path: str,
    resolution: int = 224,
    alpha: float = 0.45,
) -> None:
    """Overlay predicted masks on original images and save a grid."""
    N = len(originals)
    cols = min(N, 4)
    rows = (N + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.atleast_2d(axes)

    overlay_color = np.array([0, 255, 100], dtype=np.uint8)

    for i in range(N):
        r, c = divmod(i, cols)
        ax = axes[r, c]

        orig = cv2.cvtColor(originals[i], cv2.COLOR_BGR2RGB)
        h, w = orig.shape[:2]

        mask_resized = cv2.resize(pred_masks[i], (w, h), interpolation=cv2.INTER_NEAREST)
        mask_bool = mask_resized > 127

        overlay = orig.copy()
        overlay[mask_bool] = (
            overlay[mask_bool].astype(np.float32) * (1 - alpha)
            + overlay_color.astype(np.float32) * alpha
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        ax.imshow(overlay)
        label = "View 0 (prompt)" if i == 0 else f"View {i}"
        ax.set_title(label, fontsize=12)
        ax.set_axis_off()

    for i in range(N, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].set_axis_off()

    fig.suptitle("Multi-View Consistent Segmentation", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved visualisation to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Pi3X Segmentation Inference")

    p.add_argument("--images_dir", type=str, required=True,
                   help="Directory of images or path to a video file")
    p.add_argument("--prompt_mask", type=str, default=None,
                   help="Path to a binary mask image for the prompt (view 0)")
    p.add_argument("--prompt_bbox", type=int, nargs=4, default=None,
                   metavar=("X1", "Y1", "X2", "Y2"),
                   help="Bounding box prompt on view 0 (pixel coords)")

    p.add_argument("--pi3x_checkpoint", type=str, default=None)
    p.add_argument("--seg_checkpoint", type=str, required=True,
                   help="Path to trained segmentation head checkpoint")

    p.add_argument("--output_dir", type=str, default="output/seg/inference")
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--interval", type=int, default=1)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--query_dim", type=int, default=256)
    p.add_argument("--num_decoder_layers", type=int, default=3)
    p.add_argument("--nhead", type=int, default=8)

    return p.parse_args()


def main() -> None:
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading images from %s", args.images_dir)
    imgs, originals = load_images(
        args.images_dir, resolution=args.resolution, interval=args.interval
    )

    if args.prompt_mask is not None:
        prompt = load_prompt_mask(args.prompt_mask, args.resolution)
    elif args.prompt_bbox is not None:
        prompt = create_bbox_mask(
            tuple(args.prompt_bbox), args.resolution, args.resolution, args.resolution
        )
    else:
        raise ValueError("Provide --prompt_mask or --prompt_bbox")

    logger.info("Building model ...")
    model = build_pi3x_segmentation(
        checkpoint_path=args.pi3x_checkpoint,
        device=str(device),
        query_dim=args.query_dim,
        num_decoder_layers=args.num_decoder_layers,
        nhead=args.nhead,
    )

    seg_ckpt = torch.load(args.seg_checkpoint, map_location=device, weights_only=True)
    model.seg_head.load_state_dict(seg_ckpt["seg_head_state_dict"])
    logger.info("Loaded segmentation head from %s", args.seg_checkpoint)

    logger.info("Running inference on %d views ...", imgs.shape[1])
    pred_masks = infer(model, imgs, prompt, device, threshold=args.threshold)

    save_path = str(out_dir / "segmentation_result.png")
    visualise_predictions(originals, pred_masks, save_path, args.resolution)

    for i in range(pred_masks.shape[0]):
        cv2.imwrite(str(out_dir / f"mask_view{i:02d}.png"), pred_masks[i])
    logger.info("Individual masks saved to %s", out_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
