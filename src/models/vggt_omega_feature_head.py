"""FeatureSequenceHead — predicts JAFAR-CLIP-aligned dense feature maps from
the frozen VGGT-Omega aggregator tokens (Stage-1 of the JAFAR-Puzzles plan).

Pipeline (Section 6.2 of JAFAR_Puzzles.md):
  1. LearnableLayerFusion (LoRA-style per-layer adapter + softmax-weighted
     fusion across the 4 cached blocks) — reused verbatim from the seg head so
     the feature branch starts from the same adapted representation.
  2. Project fused tokens to work_dim.
  3. CrossViewFeatureAttention: at each spatial patch, the N view-tokens attend
     each other so the per-view feature maps become cross-view consistent
     (the dense-feature analogue of the seg head's CrossViewAttention, but
     without the instance dimension M).
  4. FeaturePixelDecoder: per-layer projection + softmax-weighted fusion +
     ConvTranspose upsampling to ``feature_res``.
  5. ChannelL2Norm: unit-normalise along the channel dim so cosine similarity
     against the (similarly normalised) JAFAR teacher is a simple dot product.

Forward:
    tokens_list: List of L tensors, each (B, N, H_p, W_p, D=2048), float32.
Returns:
    features: (B, N, C_f, feature_res, feature_res), channel-L2-normalised.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vggt_omega_seg_head import LearnableLayerFusion, _ResBlock, _upsampler


# ---------------------------------------------------------------------------
# Cross-view attention without the instance dimension (dense feature variant).
# ---------------------------------------------------------------------------


class CrossViewFeatureAttention(nn.Module):
    """Per-patch attention across the N views for dense feature maps.

    Input:  (B, N, P, D) with P = H_p * W_p.
    Output: (B, N, P, D).  Attention runs across N for each (b, p).
    """

    def __init__(self, dim: int, nhead: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, nhead, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, P, D = x.shape
        h = x.permute(0, 2, 1, 3).reshape(B * P, N, D)
        q = self.norm_q(h)
        h = h + self.attn(q, q, q, need_weights=False)[0]
        h = h + self.mlp(self.norm_mlp(h))
        return h.view(B, P, N, D).permute(0, 2, 1, 3).contiguous()


# ---------------------------------------------------------------------------
# Pixel decoder for the feature branch.
# ---------------------------------------------------------------------------


class FeaturePixelDecoder(nn.Module):
    """Upsample a single fused feature tensor to ``target_res`` and emit
    ``out_dim`` channels.

    Unlike ``WeightedPixelDecoder`` (which fuses multiple layers internally),
    this consumes the already-fused representation produced by
    ``LearnableLayerFusion`` + ``CrossViewFeatureAttention``.
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dims: Tuple[int, ...] = (384, 192),
        out_dim: int = 64,
        target_res: int = 64,
    ):
        super().__init__()
        self.target_res = target_res
        dims = [in_dim] + list(hidden_dims)
        self.ups = nn.ModuleList(
            [
                nn.Sequential(
                    _upsampler(dims[i], dims[i + 1]),
                    _ResBlock(dims[i + 1]),
                    _ResBlock(dims[i + 1]),
                )
                for i in range(len(hidden_dims))
            ]
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], out_dim, 3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(1, out_dim), nn.ReLU(True),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, H_p, W_p, in_dim) -> (B, N, out_dim, Fr, Fr)."""
        B, N, H_p, W_p, _ = x.shape
        x = x.permute(0, 1, 4, 2, 3).contiguous()        # (B, N, C, H_p, W_p)
        BN = B * N
        x = x.view(BN, x.shape[2], H_p, W_p)
        for up in self.ups:
            x = up(x)
        x = F.interpolate(x, (self.target_res, self.target_res),
                          mode="bilinear", align_corners=False)
        x = self.out_proj(x)
        C = x.shape[1]
        return x.view(B, N, C, self.target_res, self.target_res)


# ---------------------------------------------------------------------------
# Channel L2 normalisation (so cosine = dot product).
# ---------------------------------------------------------------------------


def channel_l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.normalize(x, dim=1, eps=eps)


# ---------------------------------------------------------------------------
# Full head
# ---------------------------------------------------------------------------


class FeatureSequenceHead(nn.Module):
    """Predict JAFAR-CLIP-aligned dense features from frozen backbone tokens.

    Output:
        features: (B, N, C_f, feature_res, feature_res), channel-L2-normalised.
    """

    def __init__(
        self,
        token_dim: int = 2048,
        n_vggt_layers: int = 4,
        work_dim: int = 768,
        feature_out_dim: int = 64,
        feature_res: int = 64,
        lora_rank: int = 16,
        lora_alpha: float = 1.0,
        nhead: int = 8,
        use_cross_view: bool = True,
        hidden_dims: Tuple[int, ...] = (384, 192),
    ):
        super().__init__()
        self.token_dim = token_dim
        self.work_dim = work_dim
        self.feature_out_dim = feature_out_dim
        self.feature_res = feature_res

        self.fusion = LearnableLayerFusion(
            n_layers=n_vggt_layers, dim=token_dim,
            rank=lora_rank, alpha=lora_alpha,
        )
        self.tok_proj = nn.Linear(token_dim, work_dim)

        if use_cross_view:
            self.cross_view = CrossViewFeatureAttention(dim=work_dim, nhead=nhead)
            self.cv_backproj = nn.Linear(work_dim, work_dim)
            nn.init.zeros_(self.cv_backproj.weight)
            nn.init.zeros_(self.cv_backproj.bias)
        else:
            self.cross_view = None
            self.cv_backproj = None

        self.decoder = FeaturePixelDecoder(
            in_dim=work_dim,
            hidden_dims=hidden_dims,
            out_dim=feature_out_dim,
            target_res=feature_res,
        )

    def forward(
        self,
        tokens_list: List[torch.Tensor],   # L x (B, N, H_p, W_p, 2048)
    ) -> torch.Tensor:
        B, N, H_p, W_p, _ = tokens_list[-1].shape
        fused = self.fusion(tokens_list)                 # (B, N, H_p, W_p, D)
        proj = self.tok_proj(fused)                      # (B, N, H_p, W_p, work_dim)
        if self.cross_view is not None:
            P = H_p * W_p
            cv = proj.reshape(B, N, P, self.work_dim)
            cv = self.cross_view(cv).reshape(B, N, H_p, W_p, self.work_dim)
            # Zero-init residual so training starts identical to the no-cross-view
            # path and learns to inject cross-view info.
            proj = proj + self.cv_backproj(cv)
        feat = self.decoder(proj)                        # (B, N, C_f, Fr, Fr)
        return channel_l2norm(feat)


def build_feature_sequence_head(
    *,
    device: str = "cuda",
    token_dim: int = 2048,
    n_vggt_layers: int = 4,
    work_dim: int = 768,
    feature_out_dim: int = 64,
    feature_res: int = 64,
    lora_rank: int = 16,
    lora_alpha: float = 1.0,
    nhead: int = 8,
    use_cross_view: bool = True,
    hidden_dims: Tuple[int, ...] = (384, 192),
) -> FeatureSequenceHead:
    head = FeatureSequenceHead(
        token_dim=token_dim,
        n_vggt_layers=n_vggt_layers,
        work_dim=work_dim,
        feature_out_dim=feature_out_dim,
        feature_res=feature_res,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        nhead=nhead,
        use_cross_view=use_cross_view,
        hidden_dims=hidden_dims,
    ).to(device)
    return head


# ===========================================================================
# RGB-guided variant (DINOv3 teacher, 128x128x768, sharp features)
# ===========================================================================


class _RGBStem(nn.Module):
    """Shared lightweight RGB feature extractor applied at each decoder scale.

    Produces per-pixel guidance (edges / texture) that steers the token
    upsampling toward image-aligned, high-frequency boundaries — the JAFAR-style
    "image guides feature upsampling" idea, adapted to a conv decoder.
    """

    def __init__(self, out_ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, out_ch, 3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(8, out_ch), nn.SiLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(8, out_ch), nn.SiLU(True),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.net(rgb)


class RGBGuidedFeatureDecoder(nn.Module):
    """Upsample fused tokens (P x P) to ``target_res`` with per-scale RGB guidance.

    At every 2x upsampling stage the token features are concatenated with RGB
    guidance features (from ``_RGBStem`` on the image resized to that scale) and
    fused, so the final feature map inherits image-sharp boundaries.
    """

    def __init__(
        self,
        in_dim: int = 768,
        start_res: int = 16,
        target_res: int = 128,
        hidden_dims: Tuple[int, ...] = (384, 256, 192),
        out_dim: int = 768,
        rgb_ch: int = 32,
    ):
        super().__init__()
        self.start_res = start_res
        self.target_res = target_res
        n_stages = int(round(math.log2(target_res / start_res)))
        assert start_res * (2 ** n_stages) == target_res, \
            f"target_res {target_res} must be start_res {start_res} * 2^k"
        assert len(hidden_dims) == n_stages, \
            f"need {n_stages} hidden_dims for {start_res}->{target_res}"

        self.rgb_stem = _RGBStem(rgb_ch)
        dims = [in_dim] + list(hidden_dims)
        self.up = nn.ModuleList([_upsampler(dims[i], dims[i + 1]) for i in range(n_stages)])
        self.fuse = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dims[i + 1] + rgb_ch, dims[i + 1], 3, padding=1,
                          padding_mode="replicate"),
                nn.GroupNorm(8, dims[i + 1]), nn.SiLU(True),
                _ResBlock(dims[i + 1]),
            )
            for i in range(n_stages)
        ])
        self.out_proj = nn.Conv2d(dims[-1], out_dim, 3, padding=1,
                                  padding_mode="replicate")
        self.out_dim = out_dim

    def forward(self, tokens: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        """tokens: (BN, in_dim, P, P);  rgb: (BN, 3, R, R) in [0,1].
        Returns (BN, out_dim, target_res, target_res)."""
        x = tokens
        r = self.start_res
        for up, fuse in zip(self.up, self.fuse):
            x = up(x)
            r *= 2
            g = self.rgb_stem(F.interpolate(rgb, (r, r), mode="bilinear",
                                            align_corners=False))
            x = fuse(torch.cat([x, g], dim=1))
        return self.out_proj(x)


class RGBGuidedFeatureHead(nn.Module):
    """Predict DINOv3-aligned dense features (channel-L2-normalised) at high res.

    Output: (B, N, C_out, feature_res, feature_res), C_out = 768 (DINOv3 space).
    """

    def __init__(
        self,
        token_dim: int = 2048,
        n_vggt_layers: int = 4,
        work_dim: int = 768,
        feature_out_dim: int = 768,
        feature_res: int = 128,
        start_res: int = 16,
        lora_rank: int = 16,
        lora_alpha: float = 1.0,
        nhead: int = 8,
        use_cross_view: bool = True,
        hidden_dims: Tuple[int, ...] = (384, 256, 192),
        rgb_ch: int = 32,
    ):
        super().__init__()
        self.work_dim = work_dim
        self.feature_out_dim = feature_out_dim
        self.feature_res = feature_res

        self.fusion = LearnableLayerFusion(
            n_layers=n_vggt_layers, dim=token_dim, rank=lora_rank, alpha=lora_alpha,
        )
        self.tok_proj = nn.Linear(token_dim, work_dim)

        if use_cross_view:
            self.cross_view = CrossViewFeatureAttention(dim=work_dim, nhead=nhead)
            self.cv_backproj = nn.Linear(work_dim, work_dim)
            nn.init.zeros_(self.cv_backproj.weight)
            nn.init.zeros_(self.cv_backproj.bias)
        else:
            self.cross_view = None
            self.cv_backproj = None

        self.decoder = RGBGuidedFeatureDecoder(
            in_dim=work_dim, start_res=start_res, target_res=feature_res,
            hidden_dims=hidden_dims, out_dim=feature_out_dim, rgb_ch=rgb_ch,
        )

    def forward(
        self,
        tokens_list: List[torch.Tensor],   # L x (B, N, H_p, W_p, 2048)
        imgs: torch.Tensor,                # (B, N, 3, R, R) in [0,1]
    ) -> torch.Tensor:
        B, N, H_p, W_p, _ = tokens_list[-1].shape
        fused = self.fusion(tokens_list)                 # (B, N, H_p, W_p, D)
        proj = self.tok_proj(fused)                      # (B, N, H_p, W_p, work_dim)
        if self.cross_view is not None:
            P = H_p * W_p
            cv = proj.reshape(B, N, P, self.work_dim)
            cv = self.cross_view(cv).reshape(B, N, H_p, W_p, self.work_dim)
            proj = proj + self.cv_backproj(cv)
        tok = proj.permute(0, 1, 4, 2, 3).reshape(B * N, self.work_dim, H_p, W_p)
        rgb = imgs.reshape(B * N, *imgs.shape[2:])
        feat = self.decoder(tok, rgb)                    # (BN, C_out, Fr, Fr)
        feat = feat.view(B, N, self.feature_out_dim, self.feature_res, self.feature_res)
        # L2-normalise along the CHANNEL axis (dim=2) so cosine == dot product.
        return F.normalize(feat, dim=2, eps=1e-6)


def build_rgb_guided_feature_head(
    *,
    device: str = "cuda",
    token_dim: int = 2048,
    n_vggt_layers: int = 4,
    work_dim: int = 768,
    feature_out_dim: int = 768,
    feature_res: int = 128,
    start_res: int = 16,
    lora_rank: int = 16,
    lora_alpha: float = 1.0,
    nhead: int = 8,
    use_cross_view: bool = True,
    hidden_dims: Tuple[int, ...] = (384, 256, 192),
    rgb_ch: int = 32,
) -> RGBGuidedFeatureHead:
    return RGBGuidedFeatureHead(
        token_dim=token_dim, n_vggt_layers=n_vggt_layers, work_dim=work_dim,
        feature_out_dim=feature_out_dim, feature_res=feature_res, start_res=start_res,
        lora_rank=lora_rank, lora_alpha=lora_alpha, nhead=nhead,
        use_cross_view=use_cross_view, hidden_dims=hidden_dims, rgb_ch=rgb_ch,
    ).to(device)
