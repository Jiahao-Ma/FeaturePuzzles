"""Frozen VGGT-Omega backbone wrapper for multi-view segmentation training.

Runs VGGT-Omega's aggregator once per forward and returns a list of multi-layer
patch tokens. The entire aggregator is frozen; gradients never flow into it.

API mirrors `vggt_backbone.VGGTBackbone` so the rest of the pipeline can be kept
homogeneous. Key differences from VGGT:
  - patch_size = 16 (vs 14 in VGGT)
  - num_register_tokens = 16 -> patch_token_start = 17 (vs 5 in VGGT)
  - Aggregator returns a sparse list of length 24 where only cached layers
    (4, 11, 17, 23) are non-None; we must index by absolute block number, not
    by position in a sub-list.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def _ensure_vggt_omega_on_path() -> None:
    """Locate vggt_omega on sys.path with the same fallback strategy as
    `vggt_backbone._ensure_vggt_on_path`."""
    try:
        import vggt_omega  # noqa: F401
        return
    except ImportError:
        pass
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        project_root / "repo" / "vggt-omega",
        Path("/mnt/dataset/marko/puzzles/puzzles/repo/vggt-omega"),
    ]
    for cand in candidates:
        if (cand / "vggt_omega").is_dir():
            sys.path.insert(0, str(cand))
            try:
                import vggt_omega  # noqa: F401
                return
            except ImportError:
                sys.path.pop(0)
    raise ImportError(
        "Could not locate `vggt_omega` package. Clone "
        "https://github.com/facebookresearch/vggt-omega to puzzles/repo/vggt-omega "
        "and run `pip install -e .`."
    )


# Default cached blocks exposed by vggt_omega Aggregator (depth=24).
_VGGT_OMEGA_BLOCKS: Tuple[int, ...] = (4, 11, 17, 23)


class VGGTOmegaBackbone(nn.Module):
    """Frozen VGGT-Omega aggregator returning multi-layer patch tokens.

    Args:
        ckpt_path: Local .pt weights. If None or missing, the aggregator is
            instantiated with RANDOM weights (useful for smoke testing only).
        use_layers: Subset of {4, 11, 17, 23}. Defaults to the full set.
        amp_dtype: bf16 on Ampere+, fp16 otherwise.
        enable_text_alignment: If True, load the text-alignment-trained variant.
            We never use the text head; this flag only controls weight loading
            (the visual backbone weights are different across variants).

    Forward:
        imgs: (B, N, 3, H, W) float32 in [0, 1]; H, W must be multiples of 16.
    Returns:
        tokens_list: List of len(use_layers) tensors, each
            (B, N, H_p, W_p, D=2*embed_dim=2048), float32.
        meta: dict(H_p, W_p, patch_size, patch_start_idx, D).
    """

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        use_layers: Tuple[int, ...] = _VGGT_OMEGA_BLOCKS,
        amp_dtype: torch.dtype = torch.bfloat16,
        enable_text_alignment: bool = True,
        lora_config: Optional[Dict] = None,
    ):
        super().__init__()
        _ensure_vggt_omega_on_path()
        from vggt_omega.models import VGGTOmega

        for bn in use_layers:
            if bn not in _VGGT_OMEGA_BLOCKS:
                raise ValueError(
                    f"vggt-omega exposes blocks {_VGGT_OMEGA_BLOCKS}, got {bn}"
                )
        self.use_layers = tuple(use_layers)

        # We never run the heads; instantiate them lightly to make state_dict
        # loading happy when the checkpoint contains those keys, then discard.
        model = VGGTOmega(
            patch_size=16,
            embed_dim=1024,
            enable_camera=True,
            enable_depth=True,
            enable_alignment=enable_text_alignment,
        )

        if ckpt_path and Path(ckpt_path).is_file():
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            incompat = model.load_state_dict(state, strict=False)
            if incompat.missing_keys:
                print(
                    f"[VGGTOmegaBackbone] missing keys (first 5): "
                    f"{incompat.missing_keys[:5]} (total {len(incompat.missing_keys)})"
                )
            if incompat.unexpected_keys:
                print(
                    f"[VGGTOmegaBackbone] unexpected keys (first 5): "
                    f"{incompat.unexpected_keys[:5]} "
                    f"(total {len(incompat.unexpected_keys)})"
                )
        else:
            if ckpt_path:
                print(
                    f"[VGGTOmegaBackbone] ckpt {ckpt_path} not found; using RANDOM "
                    "weights. Only suitable for shape/gradient smoke tests."
                )
            else:
                print(
                    "[VGGTOmegaBackbone] no ckpt provided; using RANDOM weights. "
                    "Only suitable for shape/gradient smoke tests."
                )

        self.aggregator = model.aggregator
        # Keep the reconstruction heads (frozen, never LoRA'd) so the split
        # forward can produce clean depth + camera geometry for the seg head.
        self.depth_head = model.dense_head
        self.camera_head = model.camera_head
        self.patch_size = int(self.aggregator.patch_size)
        self.patch_start_idx = int(self.aggregator.patch_token_start)
        self.amp_dtype = amp_dtype

        for p in self.parameters():
            p.requires_grad = False
        super().train(False)

        # Optional LoRA fine-tuning of the (otherwise frozen) aggregator.
        self._lora_enabled = False
        self.lora_start_layer = 0
        if lora_config is not None:
            self.lora_start_layer = int(lora_config.get("start_layer", 0))
            self._apply_lora(lora_config)

    def _apply_lora(self, cfg: Dict) -> None:
        from .lora_utils import (
            DEFAULT_TARGETS,
            inject_lora_into_aggregator,
            make_rank_schedule,
            set_lora_trainable,
        )

        schedule = make_rank_schedule(
            base_rank=int(cfg.get("rank", 16)),
            base_alpha=float(cfg.get("alpha", 16.0)),
        )
        injected = inject_lora_into_aggregator(
            self.aggregator,
            schedule,
            targets=tuple(cfg.get("targets", DEFAULT_TARGETS)),
            start_layer=int(cfg.get("start_layer", 0)),
            inter_frame_multiplier=int(cfg.get("inter_frame_multiplier", 2)),
        )
        # LoRA modules were created on CPU; move to the aggregator's device.
        dev = next(self.aggregator.parameters()).device
        self.aggregator.to(dev)
        set_lora_trainable(self.aggregator, True)
        self._lora_enabled = True
        print(
            f"[VGGTOmegaBackbone] injected LoRA into {len(injected)} Linear layers "
            f"(start_layer={cfg.get('start_layer', 0)}, rank={cfg.get('rank', 16)})."
        )

    def train(self, mode: bool = True):
        """Always stay in eval mode."""
        super().train(False)
        return self

    def forward(
        self, imgs: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], dict]:
        # When LoRA is active and we're in a grad-enabled context (training),
        # run with autograd so the adapters receive gradients. Otherwise keep
        # the original cheap no_grad path (pure reconstruction / eval).
        if self._lora_enabled and torch.is_grad_enabled():
            return self._forward_impl(imgs)
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

        with amp_ctx:
            agg_list, patch_start_idx = self.aggregator(imgs)

        # agg_list has length aggregator.depth (=24). Only cached layers are
        # non-None; index by absolute block number.
        tokens_out: List[torch.Tensor] = []
        for bn in self.use_layers:
            tok = agg_list[bn]
            if tok is None:
                raise RuntimeError(
                    f"Aggregator did not cache block {bn}. cached_layer_indices "
                    f"was {sorted(self.aggregator.cached_layer_indices)}."
                )
            # tok: (B, N, P, 2*embed_dim) where P = patch_start_idx + H_p*W_p
            tok = tok[:, :, patch_start_idx:, :]              # drop special tokens
            tok = tok.float().view(B, N, H_p, W_p, -1).contiguous()
            tokens_out.append(tok)

        meta = {
            "H_p": H_p, "W_p": W_p,
            "patch_size": self.patch_size,
            "patch_start_idx": int(patch_start_idx),
            "D": tokens_out[0].shape[-1],
        }
        return tokens_out, meta

    # ------------------------------------------------------------------
    # Split forward: shared trunk (0..start) once, then a clean recon tail
    # (LoRA off) for depth/camera geometry and a LoRA seg tail (start..depth).
    # ------------------------------------------------------------------
    def _amp_ctx(self, imgs):
        if imgs.device.type == "cuda":
            return torch.amp.autocast("cuda", dtype=self.amp_dtype)
        from contextlib import nullcontext
        return nullcontext()

    def _assemble_full_list(self, cached_a: dict, cached_b: dict) -> List[Optional[torch.Tensor]]:
        depth = self.aggregator.depth
        full: List[Optional[torch.Tensor]] = [None] * depth
        for i in range(depth):
            if i in cached_a:
                full[i] = cached_a[i]
            elif i in cached_b:
                full[i] = cached_b[i]
        return full

    @torch.no_grad()
    def _run_recon_heads(self, full_list, imgs) -> Dict[str, torch.Tensor]:
        patch_start = self.patch_start_idx
        out: Dict[str, torch.Tensor] = {}
        if self.depth_head is not None:
            depth, depth_conf = self.depth_head(full_list, images=imgs, patch_token_start=patch_start)
            out["depth"] = depth                 # (B, N, H, W, 1)
            out["depth_conf"] = depth_conf       # (B, N, H, W)
        if self.camera_head is not None:
            out["pose_enc"] = self.camera_head(full_list, patch_token_start=patch_start)  # (B, N, 9)
        return out

    def _tokens_from_cache(self, merged: dict, B, N, H_p, W_p) -> List[torch.Tensor]:
        tokens_out: List[torch.Tensor] = []
        for bn in self.use_layers:
            tok = merged.get(bn)
            if tok is None:
                raise RuntimeError(
                    f"forward_split: missing cached block {bn}; have "
                    f"{sorted(merged.keys())}."
                )
            tok = tok[:, :, self.patch_start_idx:, :]
            tok = tok.float().view(B, N, H_p, W_p, -1).contiguous()
            tokens_out.append(tok)
        return tokens_out

    def forward_split(
        self,
        imgs: torch.Tensor,
        return_geometry: bool = True,
        external_geometry: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[List[torch.Tensor], Optional[Dict[str, torch.Tensor]], dict]:
        """Run the aggregator once on the shared trunk, then split into a clean
        recon tail (depth/camera) and a LoRA seg tail.

        Returns:
            tokens_list: seg features for ``use_layers`` (LoRA-adapted tail).
            geometry: {"depth", "depth_conf", "pose_enc"} from the clean tail,
                or ``external_geometry`` if provided, or None.
            meta: dict(H_p, W_p, patch_size, patch_start_idx, D).
        """
        from .lora_utils import set_lora_enabled

        B, N, C, H, W = imgs.shape
        assert C == 3, f"expected 3 channels, got {C}"
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"H={H}, W={W} must be multiples of {self.patch_size}"
        H_p, W_p = H // self.patch_size, W // self.patch_size
        depth_blocks = self.aggregator.depth
        split = self.lora_start_layer if self._lora_enabled else depth_blocks

        # ---- shared trunk [0, split): frozen, no LoRA in this range ----
        with torch.no_grad(), self._amp_ctx(imgs):
            tokens, frame_rope, dims = self.aggregator.embed_tokens(imgs)
            tokens_shared, cached_shared = self.aggregator.run_blocks(
                tokens, dims, frame_rope, 0, split,
            )

        # ---- clean recon tail (LoRA disabled) ----
        geometry = external_geometry
        if return_geometry and external_geometry is None:
            if self._lora_enabled:
                set_lora_enabled(self.aggregator, False)
            with torch.no_grad(), self._amp_ctx(imgs):
                _, cached_clean = self.aggregator.run_blocks(
                    tokens_shared, dims, frame_rope, split, depth_blocks,
                )
            full = self._assemble_full_list(cached_shared, cached_clean)
            geometry = self._run_recon_heads(full, imgs)

        # ---- seg tail (LoRA enabled) ----
        if self._lora_enabled:
            set_lora_enabled(self.aggregator, True)
            seg_grad = torch.is_grad_enabled()
            from contextlib import nullcontext
            grad_ctx = nullcontext() if seg_grad else torch.no_grad()
            with grad_ctx, self._amp_ctx(imgs):
                _, cached_seg = self.aggregator.run_blocks(
                    tokens_shared, dims, frame_rope, split, depth_blocks,
                )
            merged = {**cached_shared, **cached_seg}
        else:
            merged = cached_shared

        tokens_out = self._tokens_from_cache(merged, B, N, H_p, W_p)
        meta = {
            "H_p": H_p, "W_p": W_p,
            "patch_size": self.patch_size,
            "patch_start_idx": self.patch_start_idx,
            "D": tokens_out[0].shape[-1],
        }
        return tokens_out, geometry, meta
