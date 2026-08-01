"""VGGT-Omega based multi-view consistent segmentation head.

Pipeline:
  1. LearnableLayerFusion (LoRA-style per-layer adapter + softmax-weighted
     fusion across 4 cached blocks): adapts the FROZEN backbone tokens with a
     small number of trainable parameters, so the effective representation
     becomes task-aware without back-propagating into the heavy backbone.
  2. FiLMPromptConditioning: inject prompt into reference-view tokens.
  3. Linear project to work_dim.
  4. CrossViewAttention: per spatial location, attend across the N views so
     non-prompt views directly receive prompt-conditioned signal (instead of
     relying on the K obj-token bottleneck only).
  5. K object tokens per (view, instance); shared identity init.
  6. TwoWayAttention x n_layers (deeper than VGGT baseline).
  7. PixelDecoder (multi-layer fusion + upsample) on unconditioned tokens.
  8. mask_tok -> dynamic 1x1 kernel; einsum with dense feature = mask logit.
  9. Aux heads (iou, occ).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Learnable layer fusion: LoRA-style adapter + softmax-weighted layer mix.
# At init: A=0 -> adapter contribution is exactly zero -> identical to
# Stage-1-style L2-normalized equal-weight fusion (since softmax(zeros) is
# uniform), so training starts from a faithful reproduction of the
# training-free baseline and then learns to deviate.
# ---------------------------------------------------------------------------


class LearnableLayerFusion(nn.Module):
    """LoRA per layer + softmax-weighted sum across cached VGGT-Omega blocks."""

    def __init__(
        self,
        n_layers: int = 4,
        dim: int = 2048,
        rank: int = 16,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.dim = dim
        self.rank = rank
        self.alpha = alpha

        # A initialized to zero, B initialized to small random -> initial
        # delta = (x @ 0) @ B = 0, so adapted == x.
        self.lora_A = nn.ParameterList(
            [nn.Parameter(torch.zeros(dim, rank)) for _ in range(n_layers)]
        )
        self.lora_B = nn.ParameterList(
            [nn.Parameter(torch.randn(rank, dim) * 0.01) for _ in range(n_layers)]
        )
        # Zero-init -> uniform softmax weights initially (equivalent to mean).
        self.layer_weights = nn.Parameter(torch.zeros(n_layers))

    def forward(self, tokens_list: List[torch.Tensor]) -> torch.Tensor:
        """Args:
            tokens_list: list of L tensors, each (B, N, H_p, W_p, D).
        Returns:
            fused: (B, N, H_p, W_p, D), unit-norm along the last dim.
        """
        assert len(tokens_list) == self.n_layers, (
            f"LearnableLayerFusion expected {self.n_layers} layers, got "
            f"{len(tokens_list)}"
        )
        w = F.softmax(self.layer_weights, dim=0)
        fused: Optional[torch.Tensor] = None
        for i, t in enumerate(tokens_list):
            delta = (t @ self.lora_A[i]) @ self.lora_B[i] * self.alpha
            adapted = F.normalize(t + delta, dim=-1)
            term = w[i] * adapted
            fused = term if fused is None else fused + term
        return F.normalize(fused, dim=-1)


# ---------------------------------------------------------------------------
# PixelDecoder with learnable softmax layer weighting (instead of plain mean).
# ---------------------------------------------------------------------------


class _ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(1, ch), nn.ReLU(True),
            nn.Conv2d(ch, ch, 3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(max(ch // 32, 1), ch), nn.ReLU(True),
            nn.Conv2d(ch, ch, 3, padding=1, padding_mode="replicate"),
        )

    def forward(self, x):
        return x + self.net(x)


def _upsampler(in_ch, out_ch):
    m = nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode="replicate"),
    )
    m[0].weight.data[:] = m[0].weight.data[:, :, :1, :1]
    return m


class WeightedPixelDecoder(nn.Module):
    """Per-layer projection + softmax-weighted fusion + upsample to (H, W).

    Same outer interface as the baseline `PixelDecoder` but the cross-layer
    combination is learned, matching the same softmax pattern used in
    `LearnableLayerFusion` so all layer-fusion knobs are co-trained.
    """

    def __init__(
        self,
        in_dim: int = 2048,
        n_layers: int = 4,
        proj_dim: int = 768,
        hidden_dims: Tuple[int, ...] = (384, 192, 96),
        out_dim: int = 96,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.layer_proj = nn.ModuleList(
            [nn.Linear(in_dim, proj_dim) for _ in range(n_layers)]
        )
        self.layer_weights = nn.Parameter(torch.zeros(n_layers))
        dims = [proj_dim] + list(hidden_dims)
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

    def forward(
        self,
        tokens_list: List[torch.Tensor],   # each (BN, H_p, W_p, D)
        H: int, W: int,
    ) -> torch.Tensor:
        assert len(tokens_list) == self.n_layers
        w = F.softmax(self.layer_weights, dim=0)
        fused: Optional[torch.Tensor] = None
        for i, tok in enumerate(tokens_list):
            x = self.layer_proj[i](tok)
            term = w[i] * x
            fused = term if fused is None else fused + term
        x = fused.permute(0, 3, 1, 2).contiguous()      # (BN, C, H_p, W_p)
        for up in self.ups:
            x = up(x)
        x = F.interpolate(x, (H, W), mode="bilinear", align_corners=False)
        return self.out_proj(x)


# ---------------------------------------------------------------------------
# FiLM Prompt Conditioning (unchanged from baseline).
# ---------------------------------------------------------------------------


class FiLMPromptConditioning(nn.Module):
    """Per-(view, instance) prompt coverage -> FiLM gamma/beta over tokens.

    Every (view, instance) pair that carries a prompt (``present``) is
    modulated by its own coverage; pairs without a prompt pass through
    identity. This generalises the original single-prompt-view conditioning to
    multi-view prompting.
    """

    def __init__(self, token_dim: int = 2048, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden), nn.GELU(),
            nn.Linear(hidden, 2 * token_dim),
        )
        self.token_dim = token_dim

    def forward(
        self,
        tokens_last: torch.Tensor,       # (B, N, H_p, W_p, D)
        coverage: torch.Tensor,          # (B, N, M, H_p, W_p)
        present: torch.Tensor,           # (B, N, M) bool
    ) -> torch.Tensor:
        B, N, H_p, W_p, D = tokens_last.shape
        M = coverage.shape[2]

        gb = self.mlp(coverage.unsqueeze(-1))                 # (B,N,M,H_p,W_p,2D)
        gamma, beta = gb.chunk(2, dim=-1)

        tok_exp = tokens_last.unsqueeze(2).expand(B, N, M, H_p, W_p, D)
        pres = present.view(B, N, M, 1, 1, 1).to(tok_exp.dtype)

        modulated = tok_exp * (1.0 + pres * gamma) + pres * beta
        return modulated


# ---------------------------------------------------------------------------
# Cross-view attention: at each spatial patch location, the N tokens (one per
# view) attend each other. Lets the prompt-conditioned signal in the prompt
# view propagate directly to every other view, instead of having to squeeze
# through the K obj-token bottleneck. Complexity: O(N^2) per patch -> tiny.
# ---------------------------------------------------------------------------


class CrossViewAttention(nn.Module):
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
        """Args:
            x: (B, N, M, P, D) with P = H_p * W_p.
        Returns:
            (B, N, M, P, D).
        """
        B, N, M, P, D = x.shape
        # Group so attention runs across N for each (b, m, p).
        h = x.permute(0, 2, 3, 1, 4).reshape(B * M * P, N, D)
        q = self.norm_q(h)
        h = h + self.attn(q, q, q, need_weights=False)[0]
        h = h + self.mlp(self.norm_mlp(h))
        return h.view(B, M, P, N, D).permute(0, 3, 1, 2, 4).contiguous()


# ---------------------------------------------------------------------------
# Two-Way Attention Block (unchanged structure; we just stack more of them).
# ---------------------------------------------------------------------------


class TwoWayAttentionBlock(nn.Module):
    """SAM-style: obj tokens attend image tokens, then image attends obj."""

    def __init__(self, dim: int = 768, nhead: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_q1 = nn.LayerNorm(dim)
        self.norm_k1 = nn.LayerNorm(dim)
        self.attn_obj = nn.MultiheadAttention(dim, nhead, batch_first=True)

        self.norm_q2 = nn.LayerNorm(dim)
        self.norm_k2 = nn.LayerNorm(dim)
        self.attn_img = nn.MultiheadAttention(dim, nhead, batch_first=True)

        self.norm_obj = nn.LayerNorm(dim)
        self.mlp_obj = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.norm_img = nn.LayerNorm(dim)
        self.mlp_img = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, obj: torch.Tensor, img: torch.Tensor,
                obj_key_padding_mask: Optional[torch.Tensor] = None):
        # obj_key_padding_mask: (BNM, T_obj) True = ignore as key (padded sparse).
        q, k = self.norm_q1(obj), self.norm_k1(img)
        obj = obj + self.attn_obj(q, k, k, need_weights=False)[0]
        q, k = self.norm_q2(img), self.norm_k2(obj)
        img = img + self.attn_img(
            q, k, k, need_weights=False, key_padding_mask=obj_key_padding_mask)[0]
        obj = obj + self.mlp_obj(self.norm_obj(obj))
        img = img + self.mlp_img(self.norm_img(img))
        return obj, img


# ---------------------------------------------------------------------------
# Full head
# ---------------------------------------------------------------------------


class VGGTOmegaSegmentationHead(nn.Module):
    """See module docstring. Output keys match VGGTSegmentationHead so the
    training loop / loss / vis utils need no changes.

    Output keys:
        segmentation_logits:   (B, N, M, H, W)
        iou_pred:              (B, N, M)  in [0, 1]
        occ_logits:            (B, N, M)  raw logits
        obj_tokens:            (B, N, M, K, work_dim)
        pre_conditioning_sim:  (B, N, M, H_p, W_p) only if return_pre_sim=True
    """
    MASK_IDX, IOU_IDX, OCC_IDX, REFINE_IDX = 0, 1, 2, 3

    def __init__(
        self,
        token_dim: int = 2048,
        n_vggt_layers: int = 4,
        work_dim: int = 768,
        n_obj_tokens: int = 4,
        n_twoway_layers: int = 4,
        nhead: int = 8,
        pixel_out_dim: int = 96,
        pixel_hidden_dims: Tuple[int, ...] = (384, 192, 96),
        lora_rank: int = 16,
        lora_alpha: float = 1.0,
        use_cross_view: bool = True,
        use_geom_pe: bool = True,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.work_dim = work_dim
        self.n_obj = n_obj_tokens
        self.use_cross_view = use_cross_view
        self.use_geom_pe = use_geom_pe

        self.fusion = LearnableLayerFusion(
            n_layers=n_vggt_layers, dim=token_dim,
            rank=lora_rank, alpha=lora_alpha,
        )

        self.tok_proj = nn.Linear(token_dim, work_dim)
        self.film = FiLMPromptConditioning(token_dim=token_dim, hidden=256)

        from .prompt_encoder import PromptEncoder
        self.prompt_encoder = PromptEncoder(work_dim=work_dim)

        if use_geom_pe:
            from .geometry_pe import Geometry3DPE
            self.geom_pe = Geometry3DPE(out_dim=work_dim)
        else:
            self.geom_pe = None

        self.obj_base = nn.Parameter(torch.randn(n_obj_tokens, work_dim) * 0.02)

        if use_cross_view:
            self.cross_view = CrossViewAttention(dim=work_dim, nhead=nhead)
        else:
            self.cross_view = None

        self.twoway = nn.ModuleList(
            [TwoWayAttentionBlock(dim=work_dim, nhead=nhead) for _ in range(n_twoway_layers)]
        )

        self.pixel_decoder = WeightedPixelDecoder(
            in_dim=token_dim,
            n_layers=n_vggt_layers,
            proj_dim=work_dim,
            hidden_dims=pixel_hidden_dims,
            out_dim=pixel_out_dim,
        )

        self.kernel_mlp = nn.Sequential(
            nn.Linear(work_dim, work_dim), nn.GELU(),
            nn.Linear(work_dim, pixel_out_dim),
        )

        self.iou_head = nn.Sequential(
            nn.Linear(work_dim, work_dim // 2), nn.GELU(),
            nn.Linear(work_dim // 2, 1),
        )
        self.occ_head = nn.Sequential(
            nn.Linear(work_dim, work_dim // 2), nn.GELU(),
            nn.Linear(work_dim // 2, 1),
        )

    @staticmethod
    def compute_pre_conditioning_sim(
        tokens_fused: torch.Tensor,      # (B, N, H_p, W_p, D), unit-norm
        coverage: torch.Tensor,          # (B, N, M, H_p, W_p)
    ) -> torch.Tensor:
        """Training-free cosine similarity between the (multi-view) soft-query
        and every patch token. The query aggregates prompt-covered tokens
        across ALL views that carry a prompt for the instance."""
        B, N, H_p, W_p, D = tokens_fused.shape
        t_norm = F.normalize(tokens_fused, dim=-1)

        # weighted aggregation across views n: q[b,m] = sum_n,h,w cov * tok
        num = torch.einsum("bnhwd,bnmhw->bmd", t_norm, coverage)
        den = coverage.sum(dim=(1, 3, 4)).unsqueeze(-1) + 1e-6   # (B,M,1)
        q = F.normalize(num / den, dim=-1)

        sim = torch.einsum("bnhwd,bmd->bnmhw", t_norm, q)
        return sim

    def _build_obj_tokens(
        self,
        tokens_proj: torch.Tensor,       # (B, N, H_p, W_p, work_dim)
        coverage: torch.Tensor,          # (B, N, M, H_p, W_p)
    ) -> torch.Tensor:
        """Returns obj_tokens: (B, N, M, K, work_dim). The mask-token query is
        seeded from prompt-covered tokens aggregated across ALL prompt views."""
        B, N, H_p, W_p, D = tokens_proj.shape
        M = coverage.shape[2]
        K = self.n_obj

        num = torch.einsum("bnhwd,bnmhw->bmd", tokens_proj, coverage)
        den = coverage.sum(dim=(1, 3, 4)).unsqueeze(-1) + 1e-6
        q = F.normalize(num / den, dim=-1)                       # (B, M, D)

        base = self.obj_base.view(1, 1, 1, K, D).expand(B, N, M, K, D).contiguous()
        base[..., self.MASK_IDX, :] = base[..., self.MASK_IDX, :] + q.view(B, 1, M, D)
        return base

    def forward(
        self,
        tokens_list: List[torch.Tensor],  # L x (B, N, H_p, W_p, 2048)
        prompt_masks: Optional[torch.Tensor],   # (B, N, M, H, W)
        prompt_views: Optional[torch.Tensor],   # (B, M) [legacy / fallback]
        H: int, W: int,
        return_pre_sim: bool = False,
        prompt_points: Optional[torch.Tensor] = None,        # (B,N,M,P,2)
        prompt_point_labels: Optional[torch.Tensor] = None,  # (B,N,M,P)
        prompt_boxes: Optional[torch.Tensor] = None,         # (B,N,M,4)
        prompt_box_valid: Optional[torch.Tensor] = None,     # (B,N,M)
        geometry: Optional[Dict[str, torch.Tensor]] = None,
        return_dense: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, N, H_p, W_p, D_vggt = tokens_list[-1].shape
        # M can come from any prompt modality.
        if prompt_masks is not None:
            M = prompt_masks.shape[2]
        elif prompt_points is not None:
            M = prompt_points.shape[2]
        elif prompt_boxes is not None:
            M = prompt_boxes.shape[2]
        else:
            raise ValueError("at least one prompt modality must be provided")
        K = self.n_obj
        work_dim = self.work_dim

        tokens_fused = self.fusion(tokens_list)             # (B, N, H_p, W_p, D)

        # ---- encode all prompt modalities into coverage + sparse tokens ----
        pe = self.prompt_encoder(
            H, W, H_p, W_p,
            prompt_masks=prompt_masks,
            prompt_points=prompt_points,
            prompt_point_labels=prompt_point_labels,
            prompt_boxes=prompt_boxes,
            prompt_box_valid=prompt_box_valid,
            ref_shape=(B, N, M),
        )
        coverage = pe["coverage"]                           # (B,N,M,H_p,W_p)
        present = pe["present"]                             # (B,N,M)
        # Legacy fallback: if only prompt_views given (mask present on one view)
        # the coverage already reflects it; nothing else needed.

        pre_sim = None
        if return_pre_sim:
            pre_sim = self.compute_pre_conditioning_sim(tokens_fused, coverage)

        modulated_last = self.film(tokens_fused, coverage, present)

        mod_proj = self.tok_proj(modulated_last)            # (B, N, M, H_p, W_p, work_dim)
        tokens_proj_ref = self.tok_proj(tokens_fused)       # (B, N, H_p, W_p, work_dim)

        # ---- 3D positional encoding (added to image tokens) ----
        if self.geom_pe is not None and geometry is not None:
            geom = self.geom_pe(geometry, H_p, W_p, (H, W))  # (B,N,H_p,W_p,work_dim) or None
            if geom is not None:
                mod_proj = mod_proj + geom.unsqueeze(2)      # broadcast over M

        obj = self._build_obj_tokens(tokens_proj_ref, coverage)

        if self.cross_view is not None:
            mod_proj_5d = mod_proj.reshape(B, N, M, H_p * W_p, work_dim)
            mod_proj_5d = self.cross_view(mod_proj_5d)
            mod_proj = mod_proj_5d.reshape(B, N, M, H_p, W_p, work_dim)

        # ---- append sparse prompt tokens to the obj/query set (SAM-style) ----
        sparse = pe["sparse"]                               # (B,N,M,S,work_dim)
        sparse_valid = pe["sparse_valid"]                   # (B,N,M,S)
        S = sparse.shape[3]

        BNM = B * N * M
        img_tok = mod_proj.reshape(BNM, H_p * W_p, work_dim)
        if S > 0:
            query = torch.cat([obj, sparse], dim=3)         # (B,N,M,K+S,work_dim)
            obj_tok = query.reshape(BNM, K + S, work_dim)
            valid_full = torch.cat(
                [torch.ones(B, N, M, K, dtype=torch.bool, device=obj.device),
                 sparse_valid], dim=3)
            key_pad = (~valid_full).reshape(BNM, K + S)
            # never mask the whole key set (avoid NaNs): force first K valid
            key_pad[:, :K] = False
        else:
            obj_tok = obj.reshape(BNM, K, work_dim)
            key_pad = None

        for blk in self.twoway:
            obj_tok, img_tok = blk(obj_tok, img_tok, obj_key_padding_mask=key_pad)
        obj_out = obj_tok.view(B, N, M, K + S, work_dim)[..., :K, :]

        BN = B * N
        tokens_dec = [t.reshape(BN, H_p, W_p, D_vggt) for t in tokens_list]
        dense = self.pixel_decoder(tokens_dec, H, W)        # (BN, C, H, W)
        C = dense.shape[1]
        dense = dense.view(B, N, C, H, W)

        mask_tok = obj_out[..., self.MASK_IDX, :]
        kernel = self.kernel_mlp(mask_tok)                  # (B, N, M, C)
        logits = torch.einsum("bnchw,bnmc->bnmhw", dense, kernel)

        iou_pred = self.iou_head(obj_out[..., self.IOU_IDX, :]).squeeze(-1).sigmoid()
        occ_logits = self.occ_head(obj_out[..., self.OCC_IDX, :]).squeeze(-1)

        out = {
            "segmentation_logits": logits,
            "iou_pred": iou_pred,
            "occ_logits": occ_logits,
            "obj_tokens": obj_out,
            "prompt_present": present,
        }
        if return_pre_sim:
            out["pre_conditioning_sim"] = pre_sim
        if return_dense:
            out["dense_features"] = dense
        return out
