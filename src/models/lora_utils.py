"""LoRA adapters for the frozen VGGT-Omega aggregator.

Injects low-rank adapters into the attention / MLP `nn.Linear` layers of the
otherwise-frozen reconstruction backbone, so the token representation can be
fine-tuned for segmentation WITHOUT modifying the original pretrained weights.

Design guarantees:
  * `lora_B` is zero-initialised  -> the adapter delta is exactly 0 at the
    start of fine-tuning, so the backbone reproduces its pristine behaviour.
  * Setting every adapter's `enabled = False` (or loading a checkpoint without
    the LoRA tensors) restores the exact reconstruction backbone -- this is
    what makes the "two-pass inference" (reconstruction vs. segmentation)
    safe: reconstruction simply runs with LoRA disabled.

The original `nn.Linear` (which may be a `LinearKMaskedBias` subclass for the
packed QKV projection) is wrapped, never mutated: we always call its forward,
then add the low-rank delta.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_LORA_KEYS = ("lora_A", "lora_B")

# Default Linear sub-modules inside each SelfAttentionBlock that we adapt.
DEFAULT_TARGETS: Tuple[str, ...] = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")


class LoRALinear(nn.Module):
    """Wrap a frozen ``nn.Linear`` with a trainable low-rank adapter.

    out = original(x) + scaling * (x @ A^T @ B^T)
        where A: (rank, in), B: (out, rank), scaling = alpha / rank.
    """

    def __init__(self, original: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.enabled = True

        # Keep (frozen) original module so masked-bias / dtype behaviour is
        # preserved exactly.
        self.original = original
        for p in self.original.parameters():
            p.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B stays zero -> zero initial delta.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.original(x)
        if not self.enabled:
            return out
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return out + self.scaling * delta.to(out.dtype)


RankSchedule = Callable[[int], Tuple[int, float]]


def make_rank_schedule(
    base_rank: int = 16,
    base_alpha: float = 16.0,
    *,
    mid_start: int = 12,
    late_start: int = 18,
    mid_mult: int = 2,
    late_mult: int = 4,
) -> RankSchedule:
    """Progressive rank: deeper (more task-specific) layers get more capacity.

    rank/alpha are scaled together so the effective scaling (alpha/rank) is
    constant across tiers.
    """

    def sched(layer_idx: int) -> Tuple[int, float]:
        if layer_idx >= late_start:
            return base_rank * late_mult, base_alpha * late_mult
        if layer_idx >= mid_start:
            return base_rank * mid_mult, base_alpha * mid_mult
        return base_rank, base_alpha

    return sched


def inject_lora_into_aggregator(
    aggregator: nn.Module,
    schedule: RankSchedule,
    targets: Sequence[str] = DEFAULT_TARGETS,
    *,
    start_layer: int = 0,
    inter_frame_multiplier: int = 2,
) -> List[str]:
    """Replace target ``nn.Linear`` layers in the aggregator with ``LoRALinear``.

    Args:
        aggregator: VGGT-Omega ``Aggregator`` (has ``depth``, ``frame_blocks``,
            ``inter_frame_blocks``).
        schedule: maps layer index -> (rank, alpha).
        targets: dotted module paths inside each block to adapt.
        start_layer: only inject into blocks with ``layer_idx >= start_layer``.
            Restricting to deeper layers shortens the backward graph (frozen
            earlier layers need no stored activations), saving memory.
        inter_frame_multiplier: inter-frame (cross-view) blocks carry the
            cross-view matching signal that this task hinges on, so give them
            extra rank.

    Returns:
        List of injected module paths (for logging).
    """
    injected: List[str] = []
    for layer_idx in range(aggregator.depth):
        if layer_idx < start_layer:
            continue
        for block_name in ("frame_blocks", "inter_frame_blocks"):
            block = getattr(aggregator, block_name)[layer_idx]
            rank, alpha = schedule(layer_idx)
            if block_name == "inter_frame_blocks":
                rank *= inter_frame_multiplier
            for target in targets:
                parent = block
                parts = target.split(".")
                ok = True
                for p in parts[:-1]:
                    if not hasattr(parent, p):
                        ok = False
                        break
                    parent = getattr(parent, p)
                if not ok:
                    continue
                leaf = parts[-1]
                lin = getattr(parent, leaf, None)
                if not isinstance(lin, nn.Linear):
                    continue
                setattr(parent, leaf, LoRALinear(lin, rank, alpha))
                injected.append(f"{block_name}.{layer_idx}.{target}")
    return injected


def is_lora_param_name(name: str) -> bool:
    return any(k in name for k in _LORA_KEYS)


def lora_parameters(module: nn.Module) -> List[nn.Parameter]:
    return [p for n, p in module.named_parameters() if is_lora_param_name(n)]


def set_lora_trainable(module: nn.Module, trainable: bool = True) -> int:
    """Set requires_grad on LoRA params; return the count of params toggled."""
    n = 0
    for name, p in module.named_parameters():
        if is_lora_param_name(name):
            p.requires_grad = trainable
            n += p.numel()
    return n


def lora_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract only the LoRA tensors (cpu clones) from ``module``."""
    return {
        name: p.detach().cpu().clone()
        for name, p in module.named_parameters()
        if is_lora_param_name(name)
    }


def set_lora_enabled(module: nn.Module, enabled: bool) -> None:
    """Toggle every ``LoRALinear`` adapter (e.g. disable for reconstruction)."""
    for m in module.modules():
        if isinstance(m, LoRALinear):
            m.enabled = enabled


def count_lora_params(module: nn.Module) -> int:
    return sum(p.numel() for p in lora_parameters(module))
