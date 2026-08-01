"""Frozen lingbot-map (GCTStream) backbone wrapper for multi-view feature training.

Mirrors ``vggt_omega_backbone.VGGTOmegaBackbone`` so the existing
``RGBGuidedFeatureHead`` + DINOv3-teacher training pipeline can be reused with
no changes to the head / losses.

Key differences from VGGT-Omega:
  - patch_size = 14 (vs 16)  -> resolution must be a multiple of 14
  - num_register_tokens = 4 + camera(1) + scale(1) -> patch_start_idx = 6
  - Streaming *causal* aggregator (GCTStream). When the KV cache is empty the
    aggregator runs in batch mode: all N views are fed together and the first
    ``num_frame_for_scale`` views receive the scale token (bidirectional among
    themselves) while the remaining views attend causally to past views. This
    matches the streaming inference behaviour and is why we use MORE views per
    training sample than the VGGT-Omega branch (n_views default 16 vs 6).
  - Aggregator output per cached block: (B, S, P, 2*embed_dim=2048) where
    P = patch_start_idx + H_p * W_p. We drop the special tokens and reshape to
    (B, N, H_p, W_p, 2048) — identical contract to VGGTOmegaBackbone.

The entire GCTStream model is frozen; gradients never flow into the aggregator
or the reconstruction heads. Only the external feature head is trained.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def _ensure_lingbot_on_path() -> None:
    """Locate the ``lingbot_map`` package with the same fallback strategy as
    ``vggt_omega_backbone._ensure_vggt_omega_on_path``."""
    try:
        import lingbot_map  # noqa: F401
        return
    except ImportError:
        pass
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        project_root / "repo" / "lingbot-map",
        Path("/mnt/dataset/marko/puzzles/puzzles/repo/lingbot-map"),
    ]
    for cand in candidates:
        if (cand / "lingbot_map").is_dir():
            sys.path.insert(0, str(cand))
            try:
                import lingbot_map  # noqa: F401
                return
            except ImportError:
                sys.path.pop(0)
    raise ImportError(
        "Could not locate `lingbot_map` package. Clone "
        "https://github.com/robbyant/lingbot-map to puzzles/repo/lingbot-map "
        "and run `pip install -e .`."
    )


# GCTStream exposes the same cached blocks (depth=24) as VGGT-Omega.
_LINGBOT_BLOCKS: Tuple[int, ...] = (4, 11, 17, 23)


class LingBotBackbone(nn.Module):
    """Frozen lingbot-map (GCTStream) aggregator returning multi-layer patch tokens.

    Args:
        ckpt_path: Local .pt weights (the released lingbot-map checkpoint). If
            None or missing, the aggregator is instantiated with RANDOM weights
            (smoke testing only).
        use_layers: Subset of {4, 11, 17, 23}. Defaults to the full set.
        amp_dtype: bf16 on Ampere+, fp16 otherwise.
        use_sdpa: If True, use the SDPA backend (no FlashInfer dependency) for
            the streaming KV cache. Recommended during training for simplicity.
        num_frame_for_scale: Number of leading views that receive the scale
            token (bidirectional attention among themselves). The remaining views
            attend causally. Set to ``n_views`` for fully bidirectional, or a
            small value (e.g. 2) to stay faithful to streaming inference.

    Forward:
        imgs: (B, N, 3, H, W) float32 in [0, 1]; H, W must be multiples of 14.
    Returns:
        tokens_list: List of len(use_layers) tensors, each
            (B, N, H_p, W_p, D=2*embed_dim=2048), float32.
        meta: dict(H_p, W_p, patch_size, patch_start_idx, D).
    """

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        use_layers: Tuple[int, ...] = _LINGBOT_BLOCKS,
        amp_dtype: torch.dtype = torch.bfloat16,
        use_sdpa: bool = True,
        num_frame_for_scale: int = 2,
        enable_3d_rope: bool = False,
        max_frame_num: int = 1024,
    ):
        super().__init__()
        _ensure_lingbot_on_path()
        from lingbot_map.models.gct_stream import GCTStream

        for bn in use_layers:
            if bn not in _LINGBOT_BLOCKS:
                raise ValueError(
                    f"lingbot-map exposes blocks {_LINGBOT_BLOCKS}, got {bn}"
                )
        self.use_layers = tuple(use_layers)
        self.num_frame_for_scale = int(num_frame_for_scale)
        self.amp_dtype = amp_dtype

        # Build GCTStream with the SDPA backend (no FlashInfer needed for
        # training; the KV cache stays empty because we run in batch mode).
        model = GCTStream(
            img_size=518,
            patch_size=14,
            embed_dim=1024,
            patch_embed="dinov2_vitl14_reg",
            pretrained_path="",          # we load the full GCTStream ckpt below
            enable_camera=True,
            enable_depth=True,
            enable_point=False,
            enable_local_point=False,
            enable_track=False,
            enable_normalize=False,
            enable_3d_rope=enable_3d_rope,
            max_frame_num=max_frame_num,
            use_sdpa=use_sdpa,
            use_gradient_checkpoint=False,  # frozen, no grad -> no ckpt needed
            camera_num_iterations=1,         # we never run the camera head
        )

        if ckpt_path and Path(ckpt_path).is_file():
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            incompat = model.load_state_dict(state, strict=False)
            if incompat.missing_keys:
                print(
                    f"[LingBotBackbone] missing keys (first 5): "
                    f"{incompat.missing_keys[:5]} (total {len(incompat.missing_keys)})"
                )
            if incompat.unexpected_keys:
                print(
                    f"[LingBotBackbone] unexpected keys (first 5): "
                    f"{incompat.unexpected_keys[:5]} "
                    f"(total {len(incompat.unexpected_keys)})"
                )
        else:
            if ckpt_path:
                print(
                    f"[LingBotBackbone] ckpt {ckpt_path} not found; using RANDOM "
                    "weights. Only suitable for shape/gradient smoke tests."
                )
            else:
                print(
                    "[LingBotBackbone] no ckpt provided; using RANDOM weights. "
                    "Only suitable for shape/gradient smoke tests."
                )

        self.aggregator = model.aggregator
        # Keep the reconstruction heads (frozen, never used) so state_dict
        # loading is clean; we only ever call the aggregator.
        self.patch_size = int(self.aggregator.patch_size)
        self.patch_start_idx = int(self.aggregator.patch_start_idx)

        for p in self.parameters():
            p.requires_grad = False
        super().train(False)

    def train(self, mode: bool = True):
        """Always stay in eval mode."""
        super().train(False)
        return self

    def forward(
        self, imgs: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], dict]:
        with torch.no_grad():
            return self._forward_impl(imgs)

    def _forward_impl(
        self, imgs: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], dict]:
        B, N, C, H, W = imgs.shape
        assert C == 3, f"expected 3 channels, got {C}"
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"H={H}, W={W} must be multiples of {self.patch_size}"
        H_p, W_p = H // self.patch_size, W // self.patch_size

        use_amp = (imgs.device.type == "cuda")
        if use_amp:
            amp_ctx = torch.amp.autocast("cuda", dtype=self.amp_dtype)
        else:
            from contextlib import nullcontext
            amp_ctx = nullcontext()

        # Batch mode: empty KV cache -> all N views processed together. The
        # first ``num_frame_for_scale`` views get the scale token (bidirectional
        # among themselves); the rest attend causally. We reset the cache
        # before every forward so each training batch is independent.
        self.aggregator.clean_kv_cache()
        nfs = min(self.num_frame_for_scale, N)
        with amp_ctx:
            agg_list, patch_start_idx = self.aggregator(
                imgs,
                selected_idx=list(self.use_layers),
                num_frame_for_scale=nfs,
                num_frame_per_block=N,
            )

        # lingbot-map's aggregator returns ONLY the selected blocks (filtered,
        # in ascending block order), unlike VGGT-Omega which returns a full
        # depth-length list with None placeholders. So we index by position;
        # use_layers is kept sorted so position i == use_layers[i].
        assert len(agg_list) == len(self.use_layers), \
            f"aggregator returned {len(agg_list)} blocks, expected {len(self.use_layers)}"
        tokens_out: List[torch.Tensor] = []
        for i, bn in enumerate(self.use_layers):
            tok = agg_list[i]
            if tok is None:
                raise RuntimeError(
                    f"Aggregator did not cache block {bn} (position {i})."
                )
            # tok: (B, N, P, 2*embed_dim) where P = patch_start_idx + H_p*W_p
            tok = tok[:, :, patch_start_idx:, :]
            tok = tok.float().view(B, N, H_p, W_p, -1).contiguous()
            tokens_out.append(tok)

        # Reset cache again to bound memory between batches.
        self.aggregator.clean_kv_cache()

        meta = {
            "H_p": H_p, "W_p": W_p,
            "patch_size": self.patch_size,
            "patch_start_idx": int(patch_start_idx),
            "D": tokens_out[0].shape[-1],
        }
        return tokens_out, meta
