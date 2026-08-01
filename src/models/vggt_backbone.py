"""Frozen VGGT backbone wrapper for multi-view segmentation training.

Runs VGGT's aggregator once per forward and returns a list of multi-layer
patch tokens. The entire VGGT model is frozen; gradients never flow into
it, so its original depth / point map / camera heads remain bit-exact.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


def _ensure_vggt_on_path() -> None:
    """Locate FastVGGT (or VGGT) on sys.path. Matches extractor.py fallbacks."""
    try:
        import vggt  # noqa: F401
        return
    except ImportError:
        pass
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        project_root / "repo" / "FastVGGT",
        project_root / "TrainingFreeVGGTMatching" / "FastVGGT",
        Path("/mnt/dataset/marko/code/puzzles/TrainingFreeVGGTMatching/FastVGGT"),
        Path("/mnt/dataset/marko/code/Depth/vggt"),
    ]
    for cand in candidates:
        if (cand / "vggt").is_dir():
            sys.path.insert(0, str(cand))
            try:
                import vggt  # noqa: F401
                return
            except ImportError:
                sys.path.pop(0)
    raise ImportError(
        "Could not locate `vggt` package. FastVGGT should be at "
        "TrainingFreeVGGTMatching/FastVGGT or repo/FastVGGT."
    )


_FASTVGGT_BLOCKS = [4, 11, 17, 23]


class VGGTBackbone(nn.Module):
    """Frozen VGGT aggregator returning multi-layer patch tokens.

    Args:
        ckpt_path: Local .pt weights (e.g. checkpoints/model.pt). If None →
            fallback to VGGT.from_pretrained("facebook/VGGT-1B").
        use_layers: Subset of {4, 11, 17, 23}. Default is (17, 23) to match
            TrainingFreeVGGTMatching/prompt_matching (Stage-1 baseline).
        amp_dtype: bf16 on Ampere+, fp16 otherwise.

    Forward:
        imgs: (B, N, 3, H, W) float32 in [0, 1]
    Returns:
        tokens_list: List of len(use_layers) tensors, each
            (B, N, H_p, W_p, D=2048), float32.
        meta: dict(H_p, W_p, patch_size, patch_start_idx, D).
    """

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        use_layers: Tuple[int, ...] = (17, 23),
        amp_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        _ensure_vggt_on_path()
        from vggt.models.vggt import VGGT
        import inspect

        sig = inspect.signature(VGGT.__init__)
        kw = {}
        if "merging" in sig.parameters:
            # CRITICAL: disable token merging for grid alignment.
            # FastVGGT triggers merge whenever block_num >= self.merging,
            # so 0 still merges everywhere; only None fully disables.
            kw["merging"] = None

        if ckpt_path and Path(ckpt_path).is_file():
            model = VGGT(**kw)
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            incompat = model.load_state_dict(state, strict=False)
            if incompat.missing_keys:
                print(f"[VGGTBackbone] missing keys (first 5): {incompat.missing_keys[:5]}")
            if incompat.unexpected_keys:
                print(f"[VGGTBackbone] unexpected keys (first 5): {incompat.unexpected_keys[:5]}")
        else:
            if ckpt_path:
                print(f"[VGGTBackbone] ckpt {ckpt_path} not found, falling back to HF hub.")
            try:
                try:
                    model = VGGT.from_pretrained("facebook/VGGT-1B", **kw)
                except TypeError:
                    model = VGGT.from_pretrained("facebook/VGGT-1B")
            except Exception as e:  # network / cache miss → random init (test mode)
                print(
                    f"[VGGTBackbone] from_pretrained failed ({type(e).__name__}: {e}); "
                    "instantiating VGGT with RANDOM weights. This is only suitable for "
                    "shape/gradient smoke tests, NOT real training/inference."
                )
                model = VGGT(**kw)

        resolved = []
        for bn in use_layers:
            if bn not in _FASTVGGT_BLOCKS:
                raise ValueError(
                    f"FastVGGT only exposes blocks {_FASTVGGT_BLOCKS}, got {bn}"
                )
            resolved.append(_FASTVGGT_BLOCKS.index(bn))
        self.register_buffer(
            "_resolved_layers", torch.tensor(resolved, dtype=torch.long),
            persistent=False,
        )
        self.use_layers = tuple(use_layers)

        self.aggregator = model.aggregator
        self.patch_size = int(getattr(self.aggregator, "patch_size", 14))
        self.amp_dtype = amp_dtype

        for p in self.parameters():
            p.requires_grad = False
        super().train(False)

    def train(self, mode: bool = True):
        """Always stay in eval mode — never learn anything in this module."""
        super().train(False)
        return self

    @torch.no_grad()
    def forward(
        self, imgs: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], dict]:
        B, N, C, H, W = imgs.shape
        assert C == 3, f"expected 3 channels, got {C}"
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"H={H}, W={W} must be multiples of {self.patch_size}"
        H_p, W_p = H // self.patch_size, W // self.patch_size

        if hasattr(self.aggregator, "update_patch_dimensions"):
            self.aggregator.update_patch_dimensions(W_p, H_p)

        use_amp = (imgs.device.type == "cuda")
        if use_amp:
            amp_ctx = torch.amp.autocast("cuda", dtype=self.amp_dtype)
        else:
            from contextlib import nullcontext
            amp_ctx = nullcontext()

        with amp_ctx:
            agg_list, patch_start_idx = self.aggregator(imgs)

        tokens_out: List[torch.Tensor] = []
        for idx in self._resolved_layers.tolist():
            tok = agg_list[idx]                          # (B, N, P, D)
            tok = tok[:, :, patch_start_idx:, :]         # drop special tokens → (B, N, H_p*W_p, D)
            tok = tok.float().view(B, N, H_p, W_p, -1).contiguous()
            tokens_out.append(tok)

        meta = {
            "H_p": H_p, "W_p": W_p,
            "patch_size": self.patch_size,
            "patch_start_idx": int(patch_start_idx),
            "D": tokens_out[0].shape[-1],
        }
        return tokens_out, meta
