# FeaturePuzzles → VGGT Segmentation 迁移计划

> Target repo: [`FeaturePuzzles`](https://github.com/Jiahao-Ma/FeaturePuzzles)
> Branch: `feat/vggt-segmentation`
> Conda env: `puzzles`
> Status: cursor-ready, 按 Task 顺序执行

---

## 0. 先决条件 (已确认)

基于 `files.md` + 现有 `TrainingFreeVGGTMatching/prompt_matching/extractor.py` 的 fallback 路径，**FastVGGT 已经存在于机器本地**：

```
/mnt/dataset/marko/code/puzzles/TrainingFreeVGGTMatching/FastVGGT/
```

这是 `extractor.py:64` 硬编码的 fallback 路径之一。因此：

- ❌ **不需要** `scripts/setup_vggt.sh` 或新 submodule
- ✅ 复用现有的 `_ensure_vggt_on_path()` 发现逻辑
- ✅ 直接 `from vggt.models.vggt import VGGT` 即可

**Python 环境**：`conda activate puzzles`

**Checkpoint 路径 (已确认)**：
- VGGT-1B 权重已存在于本地：
  ```
  /mnt/dataset/marko/code/Depth/vggt/checkpoints/vggt.pt   (~5 GB, 1797 keys, has aggregator.*)
  ```
  这就是训练 / 推理时 `--checkpoint` 应该传入的值。
- ⚠️ **不要**用 `puzzles/checkpoints/model.safetensors`：那是 **Pi3X** 权重 (`camera_decoder.*`, `pose_inject_blk.*`, `register_token` 等 Pi3X-only keys)，不是 VGGT。
- 备选：`python scripts/download_weights.py --model vggt` 会下到 `checkpoints/model.pt`。但本地已有可直接复用的 `vggt.pt`，**不需要重新下载**。
- 加载 `vggt.pt` 时会出现 `unexpected keys: point_head.*` 警告 — 这是因为我们用 `enable_point=False` 实例化 VGGT (只用 aggregator)，可以**安全忽略**。

**现有 forward/loss/dataset 三个契约不变**，drop-in 替换：
```python
# 契约 A (model)
model(imgs, prompt_masks) → {"segmentation_logits": (B, N, M, H, W), ...}

# 契约 B (loss)
segmentation_loss(logits, gt, prompt_views=pv, lambda_bce=5.0, lambda_dice=5.0, extras=...)

# 契约 C (dataset)
{"imgs": (N, 3, H, W), "prompt_masks": (N, M, H, W),
 "gt_consistent_masks": (N, M, H, W), "prompt_views": (M,)}
```

**VGGT 关键维度**：
- `token_dim = 2 * embed_dim = 2048` (和 Pi3X 一样)
- `patch_size = 14`, `patch_start_idx = 5` (1 cam + 4 reg)
- 训练 `resolution=224` → `H_p=16, W_p=16`
- FastVGGT aggregator 返回 4-element list，对应 `block_num ∈ {4, 11, 17, 23}`

---

## 1. Task 清单 (按依赖顺序执行)

| # | Task | 路径 | 类型 |
|---|---|---|---|
| 1 | 复用/扩展 checkpoint 下载 | `scripts/download_weights.py` | 修改 |
| 2 | VGGT backbone wrapper | `src/models/vggt_backbone.py` | 新增 |
| 3 | VGGT 分割 head (含 Prompt Conditioning + Mask DPT) | `src/models/vggt_seg_head.py` | 新增 |
| 4 | 顶层模型 factory | `src/models/vggt_segmentation.py` | 新增 |
| 5 | loss 加一致性正则 + 辅助 head | `src/utils/loss_utils.py` | 修改 |
| 6 | 训练脚本切换 | `train_segmentation.py` | 修改 (5 处) |
| 7 | 可视化工具 (sim + pred 叠加) | `src/utils/vis_utils.py` | 新增 |
| 8 | Smoke test | `tests/test_vggt_seg.py` | 新增 |
| 9 | 保留 Pi3X 旧代码作为 ablation | — | 不动 |

---

## 2. Task 1 — `scripts/download_weights.py` (修改)

目标：支持下载 VGGT-1B 权重。保留原 Pi3X 下载能力。

**整文件替换**：

```python
"""Download pretrained weights from Hugging Face Hub.

Examples:
    # Pi3X (default, backward-compatible)
    python scripts/download_weights.py

    # VGGT-1B
    python scripts/download_weights.py --model vggt

    # Custom repo
    python scripts/download_weights.py --repo-id facebook/VGGT-1B --filename model.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path


PRESETS = {
    "pi3x": {"repo_id": "yyfz233/Pi3X",       "filename": "model.safetensors"},
    "vggt": {"repo_id": "facebook/VGGT-1B",   "filename": "model.pt"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download model weights")
    parser.add_argument("--model", choices=list(PRESETS), default="pi3x",
                        help="Preset: pi3x or vggt")
    parser.add_argument("--repo-id", default=None,
                        help="Override Hugging Face repo ID")
    parser.add_argument("--filename", default=None,
                        help="Override weight file name")
    parser.add_argument(
        "--local-dir",
        default=str(Path(__file__).resolve().parents[1] / "checkpoints"),
        help="Local directory to save the weights",
    )
    args = parser.parse_args()

    preset = PRESETS[args.model]
    repo_id = args.repo_id or preset["repo_id"]
    filename = args.filename or preset["filename"]

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit(
            "huggingface-hub is not installed. Run: pip install huggingface-hub[torch]"
        )

    print(f"Downloading {filename} from {repo_id} -> {args.local_dir}")
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=args.local_dir)
    print(f"Saved to: {path}")


if __name__ == "__main__":
    main()
```

**Usage**:
```bash
conda activate puzzles
python scripts/download_weights.py --model vggt
# → checkpoints/model.pt
```

**验收**：`checkpoints/model.pt` 存在 (~3.5 GB)。

**或直接跳过此步**：本机已有可用 VGGT 权重
`/mnt/dataset/marko/code/Depth/vggt/checkpoints/vggt.pt`，所有训练 / smoke 命令直接传这个路径即可。最后 fallback：
`VGGTBackbone(checkpoint_path=None)` 会调用 `VGGT.from_pretrained("facebook/VGGT-1B")` 自动缓存到 HF hub。

---

## 3. Task 2 — `src/models/vggt_backbone.py` (新增)

目标：把 `extractor.py` 的加载逻辑剥离成**训练友好**的 frozen `nn.Module`。

**完整代码**：

```python
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


# FastVGGT aggregator stores intermediates ONLY at these block indices.
_FASTVGGT_BLOCKS = [4, 11, 17, 23]


class VGGTBackbone(nn.Module):
    """Frozen VGGT aggregator returning multi-layer patch tokens.

    Args:
        ckpt_path: Local .pt weights (e.g. checkpoints/model.pt). If None →
            fallback to VGGT.from_pretrained("facebook/VGGT-1B").
        use_layers: Subset of {4, 11, 17, 23}. Default uses all 4 for
            DPT-style multi-scale fusion.
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
        use_layers: Tuple[int, ...] = (4, 11, 17, 23),
        amp_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        _ensure_vggt_on_path()
        from vggt.models.vggt import VGGT
        import inspect

        sig = inspect.signature(VGGT.__init__)
        kw = {}
        if "merging" in sig.parameters:
            kw["merging"] = 0  # CRITICAL: disable token merging for grid alignment

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
                model = VGGT.from_pretrained("facebook/VGGT-1B", **kw)
            except TypeError:
                model = VGGT.from_pretrained("facebook/VGGT-1B")

        # Resolve which index into aggregated_tokens_list each requested block_num maps to
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

        # Freeze EVERYTHING
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
```

**验收**：
```python
# under conda env `puzzles`
import torch
from src.models.vggt_backbone import VGGTBackbone
b = VGGTBackbone(use_layers=(17, 23)).cuda()
imgs = torch.rand(1, 4, 3, 224, 224, device="cuda")
toks, meta = b(imgs)
assert len(toks) == 2 and toks[0].shape == (1, 4, 16, 16, 2048)
assert all(not p.requires_grad for p in b.parameters())
```

---

## 4. Task 3 — `src/models/vggt_seg_head.py` (新增)

目标：替代 `query_segmentation_head.py`。保留旧 head 的 PixelDecoder + einsum 设计；加上：
1. **`FiLMPromptConditioning`** — pixel prompt mask → patch coverage → FiLM 调制 ref-view tokens
2. **Learnable K object tokens**，初始化时 mask_tok 从 soft query 种下 (跨 view 共享 identity)
3. **`TwoWayAttentionBlock`** — obj ↔ image 双向 attention
4. **IoU / Occlusion heads**
5. **Per-view dynamic kernel · dense feature einsum** → mask logit

**完整代码**：

```python
"""VGGT-based multi-view consistent segmentation head.

Pipeline:
  1. FiLMPromptConditioning: inject prompt into reference-view tokens
     (per-instance, patch-grid level).
  2. Build K object tokens per (view, instance). mask_tok init ← soft
     query from ref view, BROADCAST across all N views → shared identity.
  3. Two-way attention × n_layers: obj ↔ image, per-(view, instance) scope.
  4. PixelDecoder on unconditioned multi-layer tokens → dense (BN, C, H, W).
  5. mask_tok → dynamic 1×1 kernel. einsum with dense feature = mask logit.
  6. iou_tok, occ_tok → scalar heads (auxiliary supervision).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# PixelDecoder (multi-layer FPN-like fusion + upsample)
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
    def forward(self, x): return x + self.net(x)


def _upsampler(in_ch, out_ch):
    m = nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode="replicate"),
    )
    m[0].weight.data[:] = m[0].weight.data[:, :, :1, :1]
    return m


class PixelDecoder(nn.Module):
    """Project+fuse L multi-layer token tensors, then upsample to (H, W)."""

    def __init__(
        self,
        in_dim: int = 2048,
        n_layers: int = 4,
        proj_dim: int = 512,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        out_dim: int = 64,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.layer_proj = nn.ModuleList([
            nn.Linear(in_dim, proj_dim) for _ in range(n_layers)
        ])
        dims = [proj_dim] + list(hidden_dims)
        self.ups = nn.ModuleList([
            nn.Sequential(
                _upsampler(dims[i], dims[i+1]),
                _ResBlock(dims[i+1]),
                _ResBlock(dims[i+1]),
            )
            for i in range(len(hidden_dims))
        ])
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
        fused = None
        for proj, tok in zip(self.layer_proj, tokens_list):
            x = proj(tok)
            fused = x if fused is None else fused + x
        fused = fused / self.n_layers
        x = fused.permute(0, 3, 1, 2).contiguous()      # (BN, C, H_p, W_p)
        for up in self.ups:
            x = up(x)
        x = F.interpolate(x, (H, W), mode="bilinear", align_corners=False)
        return self.out_proj(x)                          # (BN, out_dim, H, W)


# ---------------------------------------------------------------------------
# FiLM Prompt Conditioning
# ---------------------------------------------------------------------------

class FiLMPromptConditioning(nn.Module):
    """Pixel prompt mask → patch coverage → FiLM gamma/beta over tokens.

    Only ref-view tokens are modulated; other views pass through identity.
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
        prompt_masks: torch.Tensor,      # (B, N, M, H, W)
        prompt_views: torch.Tensor,      # (B, M) int64
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns:
            modulated: (B, N, M, H_p, W_p, D)
            pv_coverage: (B, M, H_p, W_p) — coverage on prompt_view per instance
        """
        B, N, H_p, W_p, D = tokens_last.shape
        M = prompt_masks.shape[2]

        # Downsample prompt_masks to patch grid via area pooling → fractional coverage
        pm = prompt_masks.reshape(B * N * M, 1, prompt_masks.shape[-2], prompt_masks.shape[-1])
        cov = F.adaptive_avg_pool2d(pm, (H_p, W_p))          # (B*N*M, 1, H_p, W_p)
        cov = cov.reshape(B, N, M, H_p, W_p).clamp(0, 1)

        # Gather coverage on the prompt_view only
        idx = prompt_views.view(B, 1, M, 1, 1).expand(B, 1, M, H_p, W_p)
        pv_coverage = torch.gather(cov, dim=1, index=idx).squeeze(1)   # (B, M, H_p, W_p)

        # FiLM parameters from coverage
        gb = self.mlp(pv_coverage.unsqueeze(-1))               # (B, M, H_p, W_p, 2D)
        gamma, beta = gb.chunk(2, dim=-1)                      # each (B, M, H_p, W_p, D)

        # Per-instance modulated token stack
        tok_exp = tokens_last.unsqueeze(2).expand(B, N, M, H_p, W_p, D)

        # view-mask: 1.0 where v == prompt_views[b, m], else 0
        view_idx = torch.arange(N, device=tokens_last.device).view(1, N, 1)
        is_pv = (view_idx == prompt_views.view(B, 1, M)).float()        # (B, N, M)
        mask = is_pv.view(B, N, M, 1, 1, 1)

        gamma_e = gamma.unsqueeze(1)                             # (B, 1, M, H_p, W_p, D)
        beta_e = beta.unsqueeze(1)

        modulated = tok_exp * (1.0 + mask * gamma_e) + mask * beta_e
        return modulated, pv_coverage


# ---------------------------------------------------------------------------
# Two-Way Attention Block
# ---------------------------------------------------------------------------

class TwoWayAttentionBlock(nn.Module):
    """SAM-style: obj tokens attend image tokens, then image attends obj."""
    def __init__(self, dim: int = 512, nhead: int = 8, mlp_ratio: float = 4.0):
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

    def forward(self, obj: torch.Tensor, img: torch.Tensor):
        q, k = self.norm_q1(obj), self.norm_k1(img)
        obj = obj + self.attn_obj(q, k, k, need_weights=False)[0]
        q, k = self.norm_q2(img), self.norm_k2(obj)
        img = img + self.attn_img(q, k, k, need_weights=False)[0]
        obj = obj + self.mlp_obj(self.norm_obj(obj))
        img = img + self.mlp_img(self.norm_img(img))
        return obj, img


# ---------------------------------------------------------------------------
# Full head
# ---------------------------------------------------------------------------

class VGGTSegmentationHead(nn.Module):
    """See module docstring. Output keys:
        segmentation_logits: (B, N, M, H, W)
        iou_pred:            (B, N, M)  in [0,1] after sigmoid
        occ_logits:          (B, N, M)  raw logits
        obj_tokens:          (B, N, M, K, work_dim)  — for consistency loss / viz
        pre_conditioning_sim: (B, N, M, H_p, W_p)    — cosine sim BEFORE head
                                                       (for visualization)
    """
    MASK_IDX, IOU_IDX, OCC_IDX, REFINE_IDX = 0, 1, 2, 3

    def __init__(
        self,
        token_dim: int = 2048,
        n_vggt_layers: int = 4,
        work_dim: int = 512,
        n_obj_tokens: int = 4,
        n_twoway_layers: int = 2,
        nhead: int = 8,
        pixel_out_dim: int = 64,
        pixel_hidden_dims: Tuple[int, ...] = (256, 128, 64),
    ):
        super().__init__()
        self.token_dim = token_dim
        self.work_dim = work_dim
        self.n_obj = n_obj_tokens

        self.tok_proj = nn.Linear(token_dim, work_dim)
        self.film = FiLMPromptConditioning(token_dim=token_dim, hidden=256)

        self.obj_base = nn.Parameter(torch.randn(n_obj_tokens, work_dim) * 0.02)

        self.twoway = nn.ModuleList([
            TwoWayAttentionBlock(dim=work_dim, nhead=nhead)
            for _ in range(n_twoway_layers)
        ])

        self.pixel_decoder = PixelDecoder(
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

    # --------------------------------------------------------
    # Pre-conditioning similarity (for visualization / debug)
    # --------------------------------------------------------

    @staticmethod
    def compute_pre_conditioning_sim(
        tokens_last: torch.Tensor,       # (B, N, H_p, W_p, D)
        prompt_masks: torch.Tensor,      # (B, N, M, H, W)
        prompt_views: torch.Tensor,      # (B, M) int64
    ) -> torch.Tensor:
        """Training-free cosine similarity between soft-query (ref view) and
        every patch token, pre-head. Mirrors Stage-1 pipeline exactly.

        Returns: (B, N, M, H_p, W_p) cosine similarity in [-1, 1].
        """
        B, N, H_p, W_p, D = tokens_last.shape
        M = prompt_masks.shape[2]

        # Coverage on prompt view
        pm = prompt_masks.reshape(B * N * M, 1, prompt_masks.shape[-2], prompt_masks.shape[-1])
        cov = F.adaptive_avg_pool2d(pm, (H_p, W_p)).reshape(B, N, M, H_p, W_p).clamp(0, 1)
        idx = prompt_views.view(B, 1, M, 1, 1).expand(B, 1, M, H_p, W_p)
        pv_cov = torch.gather(cov, dim=1, index=idx).squeeze(1)   # (B, M, H_p, W_p)

        # L2-normalized tokens (matches Stage-1 extractor.py)
        t_norm = F.normalize(tokens_last, dim=-1)                  # (B, N, H_p, W_p, D)

        # Gather ref-view tokens per instance: (B, M, H_p, W_p, D)
        idx2 = prompt_views.view(B, 1, M, 1, 1, 1).expand(B, 1, M, H_p, W_p, D)
        ref = torch.gather(
            t_norm.unsqueeze(2).expand(B, N, M, H_p, W_p, D),
            dim=1, index=idx2,
        ).squeeze(1)

        # Soft query = coverage-weighted avg of ref-view tokens, then re-L2
        w = pv_cov.unsqueeze(-1)                                   # (B, M, H_p, W_p, 1)
        q = (ref * w).sum(dim=(2, 3)) / (w.sum(dim=(2, 3)) + 1e-6) # (B, M, D)
        q = F.normalize(q, dim=-1)

        # Cosine sim: (B, N, H_p, W_p, D) · (B, M, D) → (B, N, M, H_p, W_p)
        sim = torch.einsum("bnhwd,bmd->bnmhw", t_norm, q)
        return sim

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def _build_obj_tokens(
        self,
        tokens_proj: torch.Tensor,       # (B, N, H_p, W_p, work_dim)
        pv_cov: torch.Tensor,            # (B, M, H_p, W_p)
        prompt_views: torch.Tensor,      # (B, M)
    ) -> torch.Tensor:
        """Returns obj_tokens: (B, N, M, K, work_dim). Shared identity init."""
        B, N, H_p, W_p, D = tokens_proj.shape
        M = prompt_views.shape[1]
        K = self.n_obj

        # Gather ref-view projected tokens
        idx = prompt_views.view(B, 1, M, 1, 1, 1).expand(B, 1, M, H_p, W_p, D)
        ref = torch.gather(
            tokens_proj.unsqueeze(2).expand(B, N, M, H_p, W_p, D),
            dim=1, index=idx,
        ).squeeze(1)                      # (B, M, H_p, W_p, D)

        # Soft query from coverage, L2-normalized
        w = pv_cov.unsqueeze(-1)
        q = (ref * w).sum(dim=(2, 3)) / (w.sum(dim=(2, 3)) + 1e-6)
        q = F.normalize(q, dim=-1)        # (B, M, D)

        base = self.obj_base.view(1, 1, 1, K, D).expand(B, N, M, K, D).contiguous()
        # Inject soft query into mask_tok, broadcast across N → identity shared
        base[..., self.MASK_IDX, :] = base[..., self.MASK_IDX, :] + q.view(B, 1, M, D)
        return base

    def forward(
        self,
        tokens_list: List[torch.Tensor],  # L × (B, N, H_p, W_p, 2048)
        prompt_masks: torch.Tensor,       # (B, N, M, H, W)
        prompt_views: torch.Tensor,       # (B, M)
        H: int, W: int,
        return_pre_sim: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, N, H_p, W_p, D_vggt = tokens_list[-1].shape
        M = prompt_masks.shape[2]
        K = self.n_obj
        work_dim = self.work_dim

        # Optional: pre-head similarity for visualization
        pre_sim = None
        if return_pre_sim:
            pre_sim = self.compute_pre_conditioning_sim(
                tokens_list[-1], prompt_masks, prompt_views,
            )

        # Step 1: FiLM modulation on last-layer tokens
        modulated_last, pv_cov = self.film(tokens_list[-1], prompt_masks, prompt_views)

        # Project to work_dim
        mod_proj = self.tok_proj(modulated_last)        # (B, N, M, H_p, W_p, work_dim)
        tokens_proj_ref = self.tok_proj(tokens_list[-1])# (B, N, H_p, W_p, work_dim)

        # Step 2: Build obj tokens (shared identity)
        obj = self._build_obj_tokens(tokens_proj_ref, pv_cov, prompt_views)

        # Step 3: Two-way attention, per-(b, v, m)
        BNM = B * N * M
        img_tok = mod_proj.reshape(BNM, H_p * W_p, work_dim)
        obj_tok = obj.reshape(BNM, K, work_dim)
        for blk in self.twoway:
            obj_tok, img_tok = blk(obj_tok, img_tok)
        obj_out = obj_tok.view(B, N, M, K, work_dim)

        # Step 4: PixelDecoder on UNCONDITIONED multi-layer tokens (per-view, not per-instance)
        BN = B * N
        tokens_dec = [t.reshape(BN, H_p, W_p, D_vggt) for t in tokens_list]
        dense = self.pixel_decoder(tokens_dec, H, W)    # (BN, C, H, W)
        C = dense.shape[1]
        dense = dense.view(B, N, C, H, W)

        # Step 5: Dynamic kernel · dense feature
        mask_tok = obj_out[..., self.MASK_IDX, :]       # (B, N, M, work_dim)
        kernel = self.kernel_mlp(mask_tok)              # (B, N, M, C)
        logits = torch.einsum("bnchw,bnmc->bnmhw", dense, kernel)

        # Step 6: Aux heads
        iou_pred = self.iou_head(obj_out[..., self.IOU_IDX, :]).squeeze(-1).sigmoid()
        occ_logits = self.occ_head(obj_out[..., self.OCC_IDX, :]).squeeze(-1)

        out = {
            "segmentation_logits": logits,
            "iou_pred": iou_pred,
            "occ_logits": occ_logits,
            "obj_tokens": obj_out,
        }
        if return_pre_sim:
            out["pre_conditioning_sim"] = pre_sim
        return out
```

**验收**：
```python
head = VGGTSegmentationHead()
B, N, M = 2, 4, 3
tokens_list = [torch.randn(B, N, 16, 16, 2048) for _ in range(4)]
pm = torch.rand(B, N, M, 224, 224)
pv = torch.randint(0, N, (B, M))
out = head(tokens_list, pm, pv, 224, 224, return_pre_sim=True)
assert out["segmentation_logits"].shape == (B, N, M, 224, 224)
assert out["iou_pred"].shape == (B, N, M)
assert out["occ_logits"].shape == (B, N, M)
assert out["obj_tokens"].shape == (B, N, M, 4, 512)
assert out["pre_conditioning_sim"].shape == (B, N, M, 16, 16)
```

---

## 5. Task 4 — `src/models/vggt_segmentation.py` (新增)

```python
"""VGGTSegmentation = frozen VGGTBackbone + trainable VGGTSegmentationHead."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .vggt_backbone import VGGTBackbone
from .vggt_seg_head import VGGTSegmentationHead


class VGGTSegmentation(nn.Module):
    """Frozen VGGT aggregator + trainable seg head.

    Forward signature matches Pi3XSegmentation so train_segmentation.py
    drops in with minimal changes.
    """
    def __init__(self, backbone: VGGTBackbone, head: VGGTSegmentationHead):
        super().__init__()
        self.backbone = backbone
        self.head = head
        for p in self.backbone.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(
        self,
        imgs: torch.Tensor,                         # (B, N, 3, H, W)
        prompt_masks: torch.Tensor,                 # (B, N, M, H, W)
        prompt_views: Optional[torch.Tensor] = None,
        return_pre_sim: bool = False,
    ) -> Dict[str, Any]:
        B, N, _, H, W = imgs.shape
        if prompt_views is None:
            view_mass = prompt_masks.sum(dim=(-2, -1))
            prompt_views = view_mass.argmax(dim=1)
        tokens_list, _ = self.backbone(imgs)
        return self.head(
            tokens_list, prompt_masks, prompt_views, H, W,
            return_pre_sim=return_pre_sim,
        )


def build_vggt_segmentation(
    checkpoint_path: Optional[str] = None,
    *,
    device: str = "cuda",
    work_dim: int = 512,
    n_obj_tokens: int = 4,
    n_twoway_layers: int = 2,
    nhead: int = 8,
    pixel_out_dim: int = 64,
    use_layers: Tuple[int, ...] = (4, 11, 17, 23),
    amp_dtype: torch.dtype = torch.bfloat16,
) -> VGGTSegmentation:
    backbone = VGGTBackbone(
        ckpt_path=checkpoint_path,
        use_layers=use_layers,
        amp_dtype=amp_dtype,
    ).to(device)
    head = VGGTSegmentationHead(
        token_dim=2048,
        n_vggt_layers=len(use_layers),
        work_dim=work_dim,
        n_obj_tokens=n_obj_tokens,
        n_twoway_layers=n_twoway_layers,
        nhead=nhead,
        pixel_out_dim=pixel_out_dim,
    ).to(device)
    return VGGTSegmentation(backbone, head)
```

**验收**：
```python
m = build_vggt_segmentation(checkpoint_path="checkpoints/model.pt").cuda()
imgs = torch.rand(2, 4, 3, 224, 224, device="cuda")
pm = torch.rand(2, 4, 3, 224, 224, device="cuda")
out = m(imgs, pm)
assert out["segmentation_logits"].shape == (2, 4, 3, 224, 224)
# Gradient check
loss = out["segmentation_logits"].mean()
loss.backward()
assert all(p.grad is None for p in m.backbone.parameters())
assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.head.parameters())
```

---

## 6. Task 5 — 修改 `src/utils/loss_utils.py`

目标：新增 **3 个辅助 loss** (向后兼容)：
- `lambda_iou`: IoU prediction L1
- `lambda_occ`: occlusion BCE
- `lambda_consist`: **跨 view mask_tok 一致性**（你问题 2 的直接对应）

**整文件替换**：

```python
"""Segmentation loss with direct 1:1 prompt-GT correspondence.

Additions for VGGT:
  - lambda_iou: regress predicted IoU scalar to real IoU (MAE)
  - lambda_occ: occlusion BCE (is GT mask non-empty on this view?)
  - lambda_consist: cross-view mask_tok L2 variance — same prompt on
    different views should yield similar mask_tok after two-way attention.
    This is the regularizer for Question 2.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _dice_score(pred, target, smooth=1.0):
    intersection = (pred * target).sum(dim=-1)
    return (2.0 * intersection + smooth) / (
        pred.sum(dim=-1) + target.sum(dim=-1) + smooth
    )


def dice_loss(pred, target):
    return (1.0 - _dice_score(pred.sigmoid().flatten(-2), target.flatten(-2))).mean()


@torch.no_grad()
def _compute_real_iou(logits, gt, thresh=0.5):
    pred = (logits.sigmoid() > thresh).float()
    intersection = (pred * gt).sum(dim=(-2, -1))
    union = ((pred + gt) > 0).float().sum(dim=(-2, -1))
    return intersection / union.clamp(min=1.0)


def mask_tok_consistency_loss(obj_tokens: torch.Tensor) -> torch.Tensor:
    """Cross-view L2 variance of mask_tok for the same (batch, instance).

    obj_tokens: (B, N, M, K, D).  We regularize index 0 (mask_tok).
    After FiLM + two-way attention, the N views have each specialized to
    their own patches — but they all describe the SAME physical instance,
    so their representations should remain close.

    Returns: scalar — mean squared deviation from cross-view centroid.
    """
    mask_tok = obj_tokens[..., 0, :]                     # (B, N, M, D)
    centroid = mask_tok.mean(dim=1, keepdim=True)        # (B, 1, M, D)
    diff = mask_tok - centroid
    return (diff ** 2).mean()


def segmentation_loss(
    pred_logits: torch.Tensor,                          # (B, N, M, H, W)
    gt_masks: torch.Tensor,
    *,
    prompt_views: Optional[torch.Tensor] = None,
    # Pi3X-compatible knobs
    lambda_bce: float = 5.0,
    lambda_dice: float = 5.0,
    # VGGT-specific knobs (default 0 → backward-compatible)
    extras: Optional[Dict[str, torch.Tensor]] = None,
    lambda_iou: float = 1.0,
    lambda_occ: float = 0.2,
    lambda_consist: float = 0.01,
) -> torch.Tensor:
    B, N, M, H, W = pred_logits.shape

    pred_flat = pred_logits.reshape(B * N * M, H * W)
    gt_flat = gt_masks.reshape(B * N * M, H * W)

    bce = F.binary_cross_entropy_with_logits(pred_flat, gt_flat, reduction="mean")
    dice = (1.0 - _dice_score(pred_flat.sigmoid(), gt_flat)).mean()
    total = lambda_bce * bce + lambda_dice * dice

    if extras is None:
        return total

    if "iou_pred" in extras and lambda_iou > 0:
        real_iou = _compute_real_iou(pred_logits, gt_masks)
        total = total + lambda_iou * F.l1_loss(extras["iou_pred"], real_iou)

    if "occ_logits" in extras and lambda_occ > 0:
        with torch.no_grad():
            occ_gt = (gt_masks.sum(dim=(-2, -1)) > 0).float()
        total = total + lambda_occ * F.binary_cross_entropy_with_logits(
            extras["occ_logits"], occ_gt,
        )

    if "obj_tokens" in extras and lambda_consist > 0:
        total = total + lambda_consist * mask_tok_consistency_loss(extras["obj_tokens"])

    return total
```

---

## 7. Task 6 — 修改 `train_segmentation.py` (共 5 处)

### 6.1 — Import (行 45 附近)
```python
# OLD
from models.pi3x_segmentation import build_pi3x_segmentation  # noqa: E402
# NEW
from models.vggt_segmentation import build_vggt_segmentation  # noqa: E402
```

### 6.2 — Args (加在 `query_dim` 附近, 行 99 附近)
```python
p.add_argument("--backbone", choices=["vggt", "pi3x"], default="vggt")
p.add_argument("--work_dim", type=int, default=512)
p.add_argument("--n_obj_tokens", type=int, default=4)
p.add_argument("--n_twoway_layers", type=int, default=2)
p.add_argument("--vggt_layers", type=int, nargs="+", default=[4, 11, 17, 23])

p.add_argument("--lambda_iou", type=float, default=1.0)
p.add_argument("--lambda_occ", type=float, default=0.2)
p.add_argument("--lambda_consist", type=float, default=0.01,
               help="Cross-view mask_tok consistency (Question 2 regularizer)")
```
同时把 `--checkpoint` 的 help 改成 `"VGGT .pt (HF hub fallback)"`。

### 6.3 — Build model (行 392 附近)
```python
# OLD
model = build_pi3x_segmentation(
    checkpoint_path=args.checkpoint,
    device=str(device),
    query_dim=args.query_dim,
    num_decoder_layers=args.num_decoder_layers,
    nhead=args.nhead,
)
# NEW
if args.backbone == "vggt":
    model = build_vggt_segmentation(
        checkpoint_path=args.checkpoint,
        device=str(device),
        work_dim=args.work_dim,
        n_obj_tokens=args.n_obj_tokens,
        n_twoway_layers=args.n_twoway_layers,
        nhead=args.nhead,
        use_layers=tuple(args.vggt_layers),
    )
else:
    from models.pi3x_segmentation import build_pi3x_segmentation
    model = build_pi3x_segmentation(
        checkpoint_path=args.checkpoint, device=str(device),
        query_dim=args.query_dim,
        num_decoder_layers=args.num_decoder_layers, nhead=args.nhead,
    )
```

### 6.4 — Loss call (2 处: train loop ~行 462, val loop ~行 524)
```python
# OLD
results = model(imgs, pm)
logits = results["segmentation_logits"]
loss = segmentation_loss(
    logits, gt, prompt_views=pv,
    lambda_bce=args.lambda_bce, lambda_dice=args.lambda_dice,
)
# NEW — two changes: pass prompt_views in, pass extras to loss
results = model(imgs, pm, prompt_views=pv,
                return_pre_sim=(global_step % args.vis_freq == 0))  # only when visualizing
logits = results["segmentation_logits"]
extras = {k: v for k, v in results.items()
          if k in ("iou_pred", "occ_logits", "obj_tokens")}
loss = segmentation_loss(
    logits, gt, prompt_views=pv,
    lambda_bce=args.lambda_bce, lambda_dice=args.lambda_dice,
    extras=extras,
    lambda_iou=args.lambda_iou,
    lambda_occ=args.lambda_occ,
    lambda_consist=args.lambda_consist,
)
```
Val loop: same structure but `return_pre_sim=not vis_logged` (只在要可视化时取).

### 6.5 — Checkpoint save (2 处: last + best, 行 560, 571)
```python
# OLD
"seg_head_state_dict": raw_model.seg_head.state_dict(),
# NEW (handles both backbones)
head_attr = "head" if args.backbone == "vggt" else "seg_head"
# ...
"backbone_type": args.backbone,
"head_state_dict": getattr(raw_model, head_attr).state_dict(),
```

### 6.6 — 可视化钩子 (替换 `log_vis_to_tb` 调用，行 492 和 539)

在 train loop 和 val loop 里都改用新的 `log_vis_extended`（下一个 Task 定义）：
```python
# OLD
if global_step % args.vis_freq == 0:
    log_vis_to_tb(writer, "train", imgs, pm, gt, logits, global_step)
# NEW
if global_step % args.vis_freq == 0:
    from utils.vis_utils import log_vis_extended
    log_vis_extended(
        writer, "train", imgs, pm, gt, logits,
        pre_sim=results.get("pre_conditioning_sim"),
        obj_tokens=results.get("obj_tokens"),
        global_step=global_step,
    )
```

---

## 8. Task 7 — `src/utils/vis_utils.py` (新增)

目标：**直接回应你的问题 1** — 把 TB 可视化扩展到三块：
- 原图 + prompt + GT + pred (已有)
- **Pre-conditioning similarity**: Stage-1 风格的 cosine heatmap，每个 (view, instance) 一张
- **Obj-token cross-view similarity matrix**: (N, N) 的 cosine 热图，证明"跨视角一致"

```python
"""TensorBoard visualization helpers for VGGT segmentation.

Extends the original log_vis_to_tb with:
  - pre-conditioning cosine similarity heatmaps (patch-level, resized to img)
  - cross-view mask_tok similarity matrix (sanity check for Question 2)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


_TURBO = torch.tensor([
    [0.19, 0.07, 0.23], [0.27, 0.17, 0.59], [0.18, 0.42, 0.84],
    [0.13, 0.65, 0.73], [0.29, 0.82, 0.42], [0.56, 0.93, 0.25],
    [0.84, 0.88, 0.12], [0.97, 0.65, 0.11], [0.94, 0.37, 0.07],
    [0.74, 0.16, 0.02],
])


def _turbo_colorize(prob: torch.Tensor) -> torch.Tensor:
    """(H, W) in [0,1] → (3, H, W) turbo heatmap."""
    idx = (prob.clamp(0, 1) * (_TURBO.shape[0] - 1)).long().cpu()
    return _TURBO[idx].permute(2, 0, 1)


def _overlay(img: torch.Tensor, heatmap: torch.Tensor, alpha: float = 0.55) -> torch.Tensor:
    """Blend turbo heatmap over RGB image."""
    return (img * (1 - alpha) + heatmap.to(img.device) * alpha).clamp(0, 1)


@torch.no_grad()
def log_vis_extended(
    writer: SummaryWriter,
    tag: str,
    imgs: torch.Tensor,             # (B, N, 3, H, W)
    prompt_masks: torch.Tensor,     # (B, N, M, H, W)
    gt_masks: torch.Tensor,         # (B, N, M, H, W)
    pred_logits: torch.Tensor,      # (B, N, M, H, W)
    pre_sim: Optional[torch.Tensor] = None,    # (B, N, M, H_p, W_p)
    obj_tokens: Optional[torch.Tensor] = None, # (B, N, M, K, D)
    global_step: int = 0,
    max_views: int = 4,
    max_inst: int = 2,
    batch_idx: int = 0,
):
    """Log three kinds of panels:
      1. {tag}/panel — RGB | prompt overlay | GT | pred (original style)
      2. {tag}/pre_sim — cosine sim BEFORE conditioning (proves Stage-1 baseline)
      3. {tag}/post_pred — predicted mask AFTER head (proves Stage-2 lift)
      4. {tag}/obj_consistency — (N, N) cosine matrix of mask_tok across views
    """
    B, N, _, H, W = imgs.shape
    M = prompt_masks.shape[2]
    b = batch_idx

    # ---- Panel 1: standard overlay ----
    nv = min(N, max_views)
    ni = min(M, max_inst)
    rows = []
    for v in range(nv):
        img_v = imgs[b, v].cpu()                          # (3, H, W)
        for i in range(ni):
            prompt_hm = _turbo_colorize(prompt_masks[b, v, i])
            gt_hm = _turbo_colorize(gt_masks[b, v, i])
            pred_prob = pred_logits[b, v, i].sigmoid().cpu()
            pred_hm = _turbo_colorize(pred_prob)
            row = torch.cat([img_v, _overlay(img_v, prompt_hm),
                             _overlay(img_v, gt_hm), _overlay(img_v, pred_hm)], dim=-1)
            rows.append(row)
    panel = torch.cat(rows, dim=-2)
    writer.add_image(f"{tag}/panel_rgb_prompt_gt_pred", panel, global_step)

    # ---- Panel 2: pre-conditioning cosine similarity ----
    if pre_sim is not None:
        # pre_sim: (B, N, M, H_p, W_p) in [-1, 1] → rescale to [0, 1]
        sim = pre_sim[b].clamp(-1, 1).add(1).div(2)        # (N, M, H_p, W_p)
        sim_rs = F.interpolate(
            sim.reshape(N * M, 1, sim.shape[-2], sim.shape[-1]),
            (H, W), mode="bilinear", align_corners=False,
        ).reshape(N, M, H, W)
        sim_rows = []
        for v in range(nv):
            img_v = imgs[b, v].cpu()
            for i in range(ni):
                hm = _turbo_colorize(sim_rs[v, i].cpu())
                sim_rows.append(torch.cat([img_v, _overlay(img_v, hm)], dim=-1))
        writer.add_image(f"{tag}/pre_conditioning_sim",
                         torch.cat(sim_rows, dim=-2), global_step)

    # ---- Panel 3: cross-view obj_token similarity matrix ----
    if obj_tokens is not None:
        mask_tok = obj_tokens[b, :, :, 0, :]               # (N, M, D)
        sim_matrices = []
        for i in range(ni):
            t = mask_tok[:, i]                             # (N, D)
            sim_mat = F.cosine_similarity(
                t[:, None, :], t[None, :, :], dim=-1,
            )                                              # (N, N) in [-1, 1]
            vis_mat = sim_mat.add(1).div(2).cpu()          # (N, N) in [0, 1]
            # enlarge to ~100 px per cell for visibility
            cell = 40
            vis_mat = vis_mat.repeat_interleave(cell, 0).repeat_interleave(cell, 1)
            sim_matrices.append(_turbo_colorize(vis_mat))
        writer.add_image(
            f"{tag}/obj_tok_cross_view_cosine",
            torch.cat(sim_matrices, dim=-1),
            global_step,
        )
        # Also log scalar: off-diagonal mean (should settle around 0.7-0.9)
        n = N
        off_diag_mask = ~torch.eye(n, dtype=torch.bool, device=mask_tok.device)
        for i in range(M):
            t = mask_tok[:, i]
            sim_mat = F.cosine_similarity(t[:, None], t[None, :], dim=-1)
            writer.add_scalar(
                f"{tag}/obj_tok_off_diag_cos_inst{i}",
                sim_mat[off_diag_mask].mean().item(),
                global_step,
            )
```

**说明**：
- **Panel 1** 和原 `log_vis_to_tb` 格式一致，cursor 可以删掉 `train_segmentation.py` 里的 `log_vis_to_tb` 或保留（新旧并存）。
- **Panel 2** `pre_conditioning_sim` 是 stage-1 training-free 的 sim map，看这个能确认 **"VGGT token 本身已经能对齐到正确区域"**。
- **Panel 3** 是 (N, N) 的 cosine matrix。训练早期所有元素应该 ≈ 1.0 (shared init)。训练稳定后 off-diag 应该在 [0.7, 0.9] —— 作为 Question 2 的直接验证工具。
- **scalar log**: `obj_tok_off_diag_cos_inst{i}` 在 TB 里会画曲线；训练中 **不应该** 掉到 0.5 以下（identity 保住）也 **不应该** 保持 1.0（per-view 特化应该发生）。

---

## 9. Task 8 — `tests/test_vggt_seg.py` (新增)

```python
"""Smoke tests. Run: pytest tests/test_vggt_seg.py -v"""
import torch
import torch.nn.functional as F


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_backbone_frozen_and_shapes():
    from src.models.vggt_backbone import VGGTBackbone
    b = VGGTBackbone(use_layers=(17, 23)).to(_device())
    imgs = torch.rand(1, 2, 3, 224, 224, device=_device())
    toks, meta = b(imgs)
    assert len(toks) == 2 and toks[0].shape == (1, 2, 16, 16, 2048)
    assert all(not p.requires_grad for p in b.parameters())
    assert meta["H_p"] == 16 and meta["W_p"] == 16


def test_head_forward_shapes():
    from src.models.vggt_seg_head import VGGTSegmentationHead
    dev = _device()
    head = VGGTSegmentationHead().to(dev)
    B, N, M, H, W = 2, 3, 2, 224, 224
    tokens_list = [torch.randn(B, N, 16, 16, 2048, device=dev) for _ in range(4)]
    pm = torch.zeros(B, N, M, H, W, device=dev)
    pv = torch.tensor([[0, 1], [2, 0]], device=dev)
    for b_ in range(B):
        for m in range(M):
            pm[b_, pv[b_, m], m, 50:150, 50:150] = 1.0
    out = head(tokens_list, pm, pv, H, W, return_pre_sim=True)
    assert out["segmentation_logits"].shape == (B, N, M, H, W)
    assert out["pre_conditioning_sim"].shape == (B, N, M, 16, 16)
    assert out["obj_tokens"].shape == (B, N, M, 4, 512)


def test_end_to_end_backward():
    from src.models.vggt_segmentation import build_vggt_segmentation
    from src.utils.loss_utils import segmentation_loss
    dev = _device()
    m = build_vggt_segmentation().to(dev)
    B, N, M = 1, 2, 2
    imgs = torch.rand(B, N, 3, 224, 224, device=dev)
    pm = torch.rand(B, N, M, 224, 224, device=dev)
    gt = (torch.rand(B, N, M, 224, 224, device=dev) > 0.5).float()
    pv = torch.tensor([[0, 1]], device=dev)
    out = m(imgs, pm, prompt_views=pv)
    extras = {k: v for k, v in out.items() if k in ("iou_pred", "occ_logits", "obj_tokens")}
    loss = segmentation_loss(out["segmentation_logits"], gt, extras=extras)
    loss.backward()
    assert all(p.grad is None for p in m.backbone.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.head.parameters())


def test_consistency_init_and_loss_behavior():
    """At init, mask_tok is shared across N views → sim = 1.0.
       After one forward pass through random two-way attn, sim should still be
       high (> 0.5) but may not be exactly 1."""
    from src.models.vggt_segmentation import build_vggt_segmentation
    dev = _device()
    m = build_vggt_segmentation().to(dev)
    m.eval()
    B, N, M = 1, 3, 1
    imgs = torch.rand(B, N, 3, 224, 224, device=dev)
    pm = torch.zeros(B, N, M, 224, 224, device=dev)
    pm[0, 0, 0, 50:150, 50:150] = 1.0
    pv = torch.tensor([[0]], device=dev)
    with torch.no_grad():
        out = m(imgs, pm, prompt_views=pv)
    mask_tok = out["obj_tokens"][0, :, 0, 0, :]    # (N, D)
    sim = F.cosine_similarity(mask_tok[:, None], mask_tok[None, :], dim=-1)
    off_diag = sim[~torch.eye(N, dtype=torch.bool, device=dev)].mean().item()
    assert 0.3 < off_diag < 1.0, f"off-diag={off_diag}: should share identity but not be identical"
```

**Cursor 命令**：
```bash
conda activate puzzles
cd /mnt/dataset/marko/code/puzzles
pytest tests/test_vggt_seg.py -v
# Expect: 4 tests pass
```

---

## 10. Task 9 — 保留 Pi3X 旧代码

不动：
- `src/models/pi3x_segmentation.py`
- `src/models/query_segmentation_head.py`
- `src/models/curope` symlink
- `repo/Pi3/` (如存在)

用于 `--backbone pi3x` ablation 对比。

---

## 11. 执行顺序 (给 cursor 的 task queue)

```
Task  1: 修改 scripts/download_weights.py (加 --model vggt)
Task  2: 新增 src/models/vggt_backbone.py
         smoke: python -c "import sys; sys.path.insert(0,'src'); from models.vggt_backbone import VGGTBackbone; import torch; print(VGGTBackbone(use_layers=(17,23)).cuda()(torch.rand(1,2,3,224,224).cuda())[1])"
Task  3: 新增 src/models/vggt_seg_head.py
Task  4: 新增 src/models/vggt_segmentation.py
Task  5: 整文件替换 src/utils/loss_utils.py
Task  6: 编辑 train_segmentation.py 共 5 处
Task  7: 新增 src/utils/vis_utils.py
Task  8: 新增 tests/test_vggt_seg.py
         run: pytest tests/test_vggt_seg.py -v
Task  9: Smoke training run (1 epoch, batch_size=1)
         python train_segmentation.py \
           --sa1b_root /mnt/dataset/marko/data/sa1b/data \
           --checkpoint /mnt/dataset/marko/code/Depth/vggt/checkpoints/vggt.pt \
           --backbone vggt \
           --epochs 1 --batch_size 1 --num_workers 2 \
           --vis_freq 5 --print_freq 2 \
           --output_dir output/smoke_vggt
         Expect: loss decreases, TB shows panel+pre_sim+obj_consistency
Task 10: Full training (50 epochs)
```

每个 Task 做完 commit 一次。

---

## 12. 推荐超参 (起手)

```bash
conda activate puzzles
torchrun --nproc_per_node=4 train_segmentation.py \
  --sa1b_root /mnt/dataset/marko/data/sa1b/data \
  --checkpoint /mnt/dataset/marko/code/Depth/vggt/checkpoints/vggt.pt \
  --backbone vggt \
  --output_dir output/vggt_seg_v1 \
  --n_views 6 --resolution 224 --max_prompts 4 \
  --epochs 50 --batch_size 1 --accum_steps 8 \
  --lr 2e-4 --weight_decay 0.05 --warmup_epochs 3 \
  --work_dim 512 --n_obj_tokens 4 --n_twoway_layers 2 --nhead 8 \
  --vggt_layers 4 11 17 23 \
  --lambda_bce 5.0 --lambda_dice 5.0 \
  --lambda_iou 1.0 --lambda_occ 0.2 --lambda_consist 0.01 \
  --curriculum --vis_freq 200 --print_freq 50 \
  --val_split 0.05 --num_workers 4
```

---

## 13. Ablation 验证 (训完跑这组)

| # | Flag 变化 | 验证问题 |
|---|---|---|
| V1 | `--backbone pi3x` | Pi3X 基线 |
| V2 | `--backbone vggt` (全配) | VGGT 比 Pi3X 强多少 |
| V3 | `--backbone vggt --lambda_consist 0.0` | consistency loss 是否必要 (看 obj_tok off-diag) |
| V4 | `--backbone vggt --n_twoway_layers 0` | two-way attention 是否必要 |
| V5 | `--backbone vggt --vggt_layers 17 23` | 2 层 vs 4 层 (省显存时) |

---

## 14. TensorBoard 面板一览 (训练开起来后你会看到)

- `train/loss_step`、`train/iou_step`、`train/lr`
- `train/panel_rgb_prompt_gt_pred` — 4 列: RGB / prompt / GT / pred
- `train/pre_conditioning_sim` — 2 列: RGB / 相似度叠加 **(问题 1 第一部分)**
- `train/obj_tok_cross_view_cosine` — (N, N) 矩阵热图 **(问题 2 证明图)**
- `train/obj_tok_off_diag_cos_inst{i}` — 曲线，追踪 identity consistency
- `val/*` — 同样结构
- `curriculum/min_overlap` 等

---

## 15. DONE definition (S2.0 完成标准)

1. ✅ `pytest tests/test_vggt_seg.py -v` 全绿
2. ✅ 10-epoch smoke training 完成，`train/loss` 下降，`val/iou` > 0.3
3. ✅ 50-epoch 完整训练，`val/iou` 超过 Pi3X baseline
4. ✅ `train/obj_tok_off_diag_cos_inst0` 曲线稳定在 [0.7, 0.9]
5. ✅ `model.backbone.aggregator` 权重 bit-exact 等于加载值（可用一个参数做 `torch.equal` 校验）—— 原 VGGT 重建能力必然保留
6. ✅ TB 里 `pre_conditioning_sim` 和 `panel_rgb_prompt_gt_pred` 都能清楚看到 — 前者应该是 Stage-1 风格的粗糙 heatmap，后者应该是 head 预测的精细 pixel mask

---

## 16. 常见问题

**Q: VGGT 要求 image normalization 吗？**
A: VGGT 内部 DINOv2 patch_embed 会做 ImageNet 标准化。训练脚本已经把 imgs 归一化到 `[0,1]` — 实测 extractor.py 用 `[0,1]` 输入能工作 → 不需要额外 normalize。如果 smoke test 全 NaN 再加。

**Q: 显存不够？**
按优先级砍：`--vggt_layers 17 23` → `--work_dim 256` → `--n_views 4` → `--resolution 196`。

**Q: `prompt_views` shape？**
Dataset 输出 `(M,)` per-sample; DataLoader collate 后 `(B, M)`。代码按 `(B, M)` 处理。

**Q: 为什么 PixelDecoder 接 unconditioned tokens 而不是 FiLM-modulated？**
Per-instance FiLM 过 decoder 会让 decoder 跑 B×N×M 次而不是 B×N 次 (M=4 时多 4× 开销)。per-instance 特化靠 **dynamic kernel from mask_tok** 实现，不需要也不应该让 decoder 重复跑。

**Q: `return_pre_sim=True` 每个 step 都开吗？**
不。只在 `global_step % vis_freq == 0` 开 — 避免多一次 L2-norm + einsum 的开销。

**Q: Tensorboard 可视化指令**
```
tensorboard \
    --logdir /mnt/dataset/marko/puzzles/puzzles/output/seg_ddp_2gpu/tb \
    --port 6006 \
    --host 0.0.0.0 \
    --samples_per_plugin "images=50"
```


# Training scripts:
```
cd /mnt/dataset/marko/puzzles/puzzles
nohup /opt/conda/envs/puzzles/bin/torchrun --nproc_per_node=8 --master_port=29912 \
  train_segmentation.py \
  --sa1b_root /mnt/dataset/marko/puzzles/sa1b/data \
  --checkpoint /mnt/dataset/marko/puzzles/puzzles/checkpoints/vggt.pt \
  --output_dir output/seg_ddp_8gpu \
  --batch_size 10 \
  --accum_steps 1 \
  --num_workers 4 \
  --prefetch_factor 2 \
  --vis_freq 50 \
  --print_freq 10 \
  > output/seg_ddp_8gpu/train.log 2>&1 &
echo $!   # 记 PID

```

Resume Training scripts:
```
cd /mnt/dataset/marko/puzzles/puzzles
nohup /opt/conda/envs/puzzles/bin/torchrun --nproc_per_node=8 --master_port=29912 \
  train_segmentation.py \
  --sa1b_root /mnt/dataset/marko/puzzles/sa1b/data \
  --checkpoint /mnt/dataset/marko/puzzles/puzzles/checkpoints/vggt.pt \
  --resume output/seg_ddp_8gpu/checkpoint-last.pth \
  --output_dir output/seg_ddp_8gpu \
  --batch_size 10 \
  --accum_steps 1 \
  --num_workers 4 \
  --prefetch_factor 2 \
  --vis_freq 50 \
  --print_freq 10 \
  > output/seg_ddp_8gpu/train.log 2>&1 &
echo $!   # 记 PID

```

# Canvas 文件
Cursor Canvas
```
/root/.cursor/projects/mnt-dataset-marko-puzzles/canvases/vggt-omega-lora-review.canvas.tsx
```

HTML
```
/mnt/dataset/marko/puzzles/puzzles/TrainingInspect/vggt_omega_review.html
```

点云数据
```
/mnt/dataset/marko/puzzles/puzzles/TrainingInspect/compare_thumbs/pcd_interactive_v2.json
```

```
┌─ Round 0 ─────────────────────────────────┐
│  原始 prompt (mask/box/point) on V0       │
│  → 网络传播到所有视角 → masks_0           │
└───────────────────────────────────────────┘
          ↓
┌─ Geo Projection ──────────────────────────┐
│  V0 的 prompt 像素 → 深度反投影到 3D      │
│  → 投影到每个视角 (z-buffer 遮挡剔除)     │
│  → geo[v] 几何锚点                        │
└───────────────────────────────────────────┘
          ↓
┌─ Round 1 ─────────────────────────────────┐
│  选非 ref 视角中 geo 面积最大的 → geo→box │
│  → 用 box prompt 从该视角传播 → masks_1   │
└───────────────────────────────────────────┘
          ↓
┌─ Fusion ──────────────────────────────────┐
│  每视角：geo门控保留 masks_0 和 masks_1    │
│  中与 geo 有重叠的整个连通块 (union)      │
│  IoU 取贡献更大那轮的 iou_pred            │
└───────────────────────────────────────────┘

```
