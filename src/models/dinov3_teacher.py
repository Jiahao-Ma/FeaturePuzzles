"""On-the-fly DINOv3 + Talk2DINO teacher for the JAFAR-Puzzles feature branch.

This module produces the *teacher* target used to distil the VGGT-Omega
FeatureSequenceHead. Unlike Stage-0's pre-cached CLIP-PCA cache, the teacher is
generated **on the fly** during training:

    parent RGB (long side ~= teacher_long, e.g. 2048)
        -> DINOv3 ViT-B/16 dense patch tokens      (B, 768, gh, gw)   [text-agnostic but sharp]
        -> (per view) grid_sample warp into each Puzzles crop box      (B, N, 768, Fr, Fr)
        -> channel-L2-normalise                                        teacher target

Open-vocabulary at inference:
    text -> OpenAI-CLIP text tower -> Talk2DINO projection MLP -> DINOv3-768 space
    cosine( predicted_feature , projected_text )  ->  per-class score map

Only the visual DINOv3 tower is needed at training time (``encode_parent``);
the Talk2DINO text projection (``text_embed``) is used for eval / demo.

Everything here is frozen (no grad) — it only defines the target, never trains.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Resolve the Talk2DINO repo relative to this file (release root / repo / Talk2DINO),
# overridable via the TALK2DINO_REPO env var for portability.
TALK2DINO_REPO = os.environ.get(
    "TALK2DINO_REPO",
    str(Path(__file__).resolve().parents[2] / "repo" / "Talk2DINO"),
)

# ImageNet-style template ensemble for robust text embeddings.
_TEMPLATES = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a close-up photo of a {}.",
    "a cropped photo of a {}.",
    "a bright photo of a {}.",
    "a good photo of a {}.",
    "itap of a {}.",
    "a {} in a scene.",
]


class _Talk2DinoProj(nn.Module):
    """Talk2DINO CLIP-text -> DINO-space projection MLP
    (configs/vitb_mlp_infonce.yaml: tanh act, one hidden layer, 512->768->768)."""

    def __init__(self, clip_dim: int = 512, dino_dim: int = 768):
        super().__init__()
        self.linear_layer = nn.Linear(clip_dim, dino_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(dino_dim, dino_dim)])
        self.act = nn.Tanh()

    def project_clip_txt(self, text: torch.Tensor) -> torch.Tensor:
        x = self.linear_layer(text.float())
        for hidden in self.hidden_layers:
            x = hidden(self.act(x))
        return x


class DINOv3Talk2DinoTeacher(nn.Module):
    """Frozen DINOv3 dense teacher (+ Talk2DINO text head for open-vocab eval).

    Args:
        variant: timm model name for the DINOv3 backbone.
        proj_weights: Talk2DINO projection checkpoint (relative to weights/).
        teacher_long: square side the parent image is resized to before DINOv3.
            With patch 16, the feature grid is teacher_long // 16 per side
            (2048 -> 128, 1024 -> 64).
        device / dtype: placement + compute dtype for the DINOv3 forward.
    """

    OUT_DIM = 768
    PATCH = 16

    def __init__(
        self,
        variant: str = "vit_base_patch16_dinov3.lvd1689m",
        proj_weights: str = "dinov3_vitb_mlp_infonce.pth",
        teacher_long: int = 2048,
        device: str = "cuda",
        amp_dtype: torch.dtype = torch.bfloat16,
        load_text: bool = False,
    ):
        super().__init__()
        import timm

        self.device = torch.device(device)
        self.amp_dtype = amp_dtype
        self.teacher_long = int(teacher_long)
        self.variant = variant

        model = timm.create_model(
            variant, pretrained=True, num_classes=0, dynamic_img_size=True,
        ).eval()
        cfg = timm.data.resolve_model_data_config(model)
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1))
        self.backbone = model
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.to(self.device)

        # Text projection (only needed for open-vocab eval / demo).
        self.proj: Optional[_Talk2DinoProj] = None
        self.clip = None
        self.tokenizer = None
        self._proj_weights = proj_weights
        if load_text:
            self.load_text_head()

    # ------------------------------------------------------------------
    def load_text_head(self) -> None:
        if self.proj is not None:
            return
        import open_clip

        sd = torch.load(f"{TALK2DINO_REPO}/weights/{self._proj_weights}",
                        map_location="cpu")
        if "linear_layer2.weight" in sd:
            sd["hidden_layers.0.weight"] = sd.pop("linear_layer2.weight")
            sd["hidden_layers.0.bias"] = sd.pop("linear_layer2.bias")
        proj = _Talk2DinoProj()
        proj.load_state_dict(sd)
        self.proj = proj.to(self.device).eval()
        # Talk2DINO trained against OpenAI CLIP ViT-B/16 (QuickGELU) text tower.
        clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16-quickgelu", pretrained="openai",
        )
        self.clip = clip_model.to(self.device).eval().float()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-16-quickgelu")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_parent(self, parent_rgb01: torch.Tensor) -> torch.Tensor:
        """Dense DINOv3 features for parent images.

        Args:
            parent_rgb01: (B, 3, H, W) in [0, 1]; H, W multiples of 16.
        Returns:
            feats: (B, 768, gh, gw) — final-norm dense patch tokens (NOT yet
                   channel-L2-normalised; the caller normalises after warping).
        """
        x = (parent_rgb01.to(self.device) - self.mean) / self.std
        with torch.amp.autocast("cuda", dtype=self.amp_dtype,
                                enabled=self.device.type == "cuda"):
            out = self.backbone.forward_intermediates(
                x, 1, return_prefix_tokens=True, norm=True,
                output_fmt="NCHW", intermediates_only=True,
            )[0]
        feats = out[0] if isinstance(out, (tuple, list)) else out
        return feats.float()

    @torch.no_grad()
    def text_embed(self, classnames: Sequence[str]) -> torch.Tensor:
        """(K, 768) L2-normalised text embeddings in DINOv3 space."""
        if self.proj is None:
            self.load_text_head()
        embs = []
        for name in classnames:
            toks = self.tokenizer([t.format(name) for t in _TEMPLATES]).to(self.device)
            e = self.clip.encode_text(toks).float()          # (T, 512)
            e = self.proj.project_clip_txt(e)                # (T, 768) DINO space
            e = F.normalize(e, dim=-1).mean(dim=0)
            embs.append(F.normalize(e, dim=-1))
        return torch.stack(embs, dim=0)


@torch.no_grad()
def warp_parent_to_views(
    parent_feat: torch.Tensor,     # (B, C, gh, gw)
    boxes: torch.Tensor,           # (B, N, 4) [x1, y1, x2, y2] in parent px
    parent_size: torch.Tensor,     # (B, 2) [H, W] original image size
    feature_res: int,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Warp parent-image dense features into each Puzzles crop box.

    Because the parent feature grid spans the *whole* parent image, a view's
    crop box maps to normalised grid coordinates via (px / W, py / H); the AR of
    the square-resized parent does not matter.

    Returns:
        teacher: (B, N, C, Fr, Fr) — channel-L2-normalised if ``normalize``.
    """
    B, C, gh, gw = parent_feat.shape
    N = boxes.shape[1]
    Fr = feature_res
    device = parent_feat.device
    dtype = parent_feat.dtype

    H = parent_size[:, 0].to(device=device, dtype=dtype).view(B, 1, 1, 1)
    W = parent_size[:, 1].to(device=device, dtype=dtype).view(B, 1, 1, 1)
    box = boxes.to(device=device, dtype=dtype)

    lin = (torch.arange(Fr, device=device, dtype=dtype) + 0.5) / Fr
    gy, gx = torch.meshgrid(lin, lin, indexing="ij")          # (Fr, Fr)
    gx = gx.view(1, 1, Fr, Fr)
    gy = gy.view(1, 1, Fr, Fr)
    x1 = box[..., 0].view(B, N, 1, 1)
    y1 = box[..., 1].view(B, N, 1, 1)
    x2 = box[..., 2].view(B, N, 1, 1)
    y2 = box[..., 3].view(B, N, 1, 1)
    px = x1 + gx * (x2 - x1)                                  # (B, N, Fr, Fr)
    py = y1 + gy * (y2 - y1)
    grid_x = (px / W) * 2.0 - 1.0
    grid_y = (py / H) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)              # (B, N, Fr, Fr, 2)

    pf = parent_feat.unsqueeze(1).expand(B, N, C, gh, gw).reshape(B * N, C, gh, gw)
    grid = grid.reshape(B * N, Fr, Fr, 2)
    warped = F.grid_sample(
        pf, grid, mode="bilinear", padding_mode="border", align_corners=False,
    ).view(B, N, C, Fr, Fr)
    if normalize:
        warped = F.normalize(warped, dim=2, eps=1e-6)
    return warped.contiguous()
