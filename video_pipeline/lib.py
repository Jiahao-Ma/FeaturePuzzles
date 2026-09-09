"""Shared helpers for the long-video VGGT-Omega promptable-segmentation pipeline.

Keeps only the functions used by `video_omega_seg_pipeline.py` and
`video_promptable_pcd.py`: image padding, model construction (VGGT-Omega
backbone + LoRA + promptable seg head), the split forward pass, and 2-D mask
overlay. The vanilla-VGGT / viser / scene-loading helpers from the original
TrainingInspect lib have been removed.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# Release root = parent of video_pipeline/  -> add src/ so `models.*` resolves.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.vggt_omega_seg_head import VGGTOmegaSegmentationHead  # noqa: E402

logger = logging.getLogger("video_pipeline")


# =====================================================================
# Image preprocessing (pad to square `target` x `target`)
# =====================================================================
def pad_image_square(img_pil, target: int = 518):
    """Pad to square (black) and resize to (target, target).

    Returns:
        canvas: PIL.Image (target, target), RGB.
        coords:  (x1, y1, x2, y2) bbox of original content inside the canvas.
        new_size: (new_W, new_H) aspect-preserved size before pad.
    """
    from PIL import Image
    W, H = img_pil.size
    max_dim = max(W, H)
    scale = target / max_dim
    new_W, new_H = int(round(W * scale)), int(round(H * scale))
    img_resized = img_pil.resize((new_W, new_H), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (target, target), (0, 0, 0))
    left = (target - new_W) // 2
    top = (target - new_H) // 2
    canvas.paste(img_resized, (left, top))
    return canvas, (left, top, left + new_W, top + new_H), (new_W, new_H)


# =====================================================================
# Model construction
# =====================================================================
def build_omega_seg_head(seg_ckpt_path: Path, device: torch.device,
                         *, lora_config: Optional[Dict] = None):
    """Load a VGGTOmegaSegmentationHead from a training checkpoint.

    Head params are aligned with the seg_omega_lora_multiprompt args.json
    (work_dim=768, n_obj_tokens=4, n_twoway_layers=4, pixel_out_dim=96,
    lora_rank=16, lora_alpha=1.0, use_cross_view=True). use_geom_pe is
    auto-detected from the checkpoint (multiprompt has no geom PE -> False).

    Returns (head, backbone_lora_state_dict_or_None).
    """
    ckpt = torch.load(str(seg_ckpt_path), map_location="cpu", weights_only=False)
    head_sd = ckpt["head_state_dict"]
    use_geom_pe = any(k.startswith("geom_pe.") for k in head_sd)
    head = VGGTOmegaSegmentationHead(
        token_dim=2048,
        n_vggt_layers=4,
        work_dim=768,
        n_obj_tokens=4,
        n_twoway_layers=4,
        nhead=8,
        pixel_out_dim=96,
        pixel_hidden_dims=(384, 192, 96),
        lora_rank=16,
        lora_alpha=1.0,
        use_cross_view=True,
        use_geom_pe=use_geom_pe,
    )
    incompat = head.load_state_dict(head_sd, strict=False)
    if incompat.missing_keys or incompat.unexpected_keys:
        logger.info("seg head load: missing=%d unexpected=%d (non-strict)",
                    len(incompat.missing_keys), len(incompat.unexpected_keys))
    head.eval().to(device)
    for p in head.parameters():
        p.requires_grad = False
    backbone_lora_sd = ckpt.get("backbone_lora_state_dict")
    logger.info("Loaded VGGT-Omega seg head from %s (val_iou=%.4f, has_lora=%s)",
                seg_ckpt_path, ckpt.get("val_iou", float("nan")),
                backbone_lora_sd is not None)
    return head, backbone_lora_sd


def build_omega_full(backbone_ckpt: Path, seg_ckpt: Path, device: torch.device):
    """Build VGGT-Omega backbone + seg head, loading backbone LoRA if present."""
    from models.vggt_omega_backbone import VGGTOmegaBackbone
    head, lora_sd = build_omega_seg_head(seg_ckpt, device)

    lora_config = None
    if lora_sd is not None:
        # Infer the LoRA start layer from which aggregator blocks carry adapters.
        start_layer = 0
        block_ids = []
        for k in lora_sd:
            parts = k.split(".")
            for i, tok in enumerate(parts):
                if tok in ("frame_blocks", "inter_frame_blocks") and i + 1 < len(parts):
                    if parts[i + 1].isdigit():
                        block_ids.append(int(parts[i + 1]))
        if block_ids:
            start_layer = min(block_ids)
        lora_config = dict(rank=16, alpha=16.0, start_layer=start_layer,
                            inter_frame_multiplier=2)
        logger.info("Backbone LoRA start_layer inferred as %d", start_layer)

    backbone = VGGTOmegaBackbone(
        ckpt_path=str(backbone_ckpt),
        use_layers=(4, 11, 17, 23),
        amp_dtype=torch.bfloat16,
        lora_config=lora_config,
    )
    if lora_sd is not None:
        backbone.load_state_dict(lora_sd, strict=False)
        logger.info("Loaded backbone LoRA weights (%d tensors)", len(lora_sd))
    backbone.eval().to(device)
    return backbone, head


# =====================================================================
# Forward pass — split so the aggregator can be cached
# =====================================================================
@torch.no_grad()
def forward_omega_aggregator(backbone: torch.nn.Module, imgs: torch.Tensor) -> List[torch.Tensor]:
    """Run VGGTOmegaBackbone and return the token list (fp32)."""
    tokens_list, _meta = backbone(imgs)
    return tokens_list


@torch.no_grad()
def forward_omega_split(backbone: torch.nn.Module, imgs: torch.Tensor,
                        return_geometry: bool = True,
                        external_geometry: Optional[Dict] = None):
    """Split forward: returns (tokens_list, geometry).

    geometry is the clean-recon-tail depth + camera pose (or external GT).
    """
    tokens_list, geometry, _meta = backbone.forward_split(
        imgs, return_geometry=return_geometry, external_geometry=external_geometry,
    )
    return tokens_list, geometry


@torch.no_grad()
def forward_omega_seg(seg_head: VGGTOmegaSegmentationHead,
                      tokens_list: List[torch.Tensor],
                      prompt_masks: torch.Tensor,
                      prompt_views: torch.Tensor,
                      img_h: int, img_w: int,
                      return_pre_sim: bool = False,
                      prompt_points: Optional[torch.Tensor] = None,
                      prompt_point_labels: Optional[torch.Tensor] = None,
                      prompt_boxes: Optional[torch.Tensor] = None,
                      prompt_box_valid: Optional[torch.Tensor] = None,
                      geometry: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
    return seg_head(
        tokens_list, prompt_masks, prompt_views, img_h, img_w,
        return_pre_sim=return_pre_sim,
        prompt_points=prompt_points,
        prompt_point_labels=prompt_point_labels,
        prompt_boxes=prompt_boxes,
        prompt_box_valid=prompt_box_valid,
        geometry=geometry,
    )


# =====================================================================
# 2-D visualization
# =====================================================================
def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray,
                  fill_color, edge_color, alpha: float = 0.5,
                  contour_thickness: int = 3) -> np.ndarray:
    out = image_bgr.copy()
    binary = (mask > 0.5).astype(np.uint8)
    if binary.any():
        colored = np.zeros_like(out)
        colored[..., 0] = fill_color[0]
        colored[..., 1] = fill_color[1]
        colored[..., 2] = fill_color[2]
        m3 = binary[..., None].astype(bool).repeat(3, axis=2)
        out[m3] = (out[m3].astype(np.float32) * (1 - alpha)
                    + colored[m3].astype(np.float32) * alpha).astype(np.uint8)
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, edge_color, contour_thickness)
    return out
