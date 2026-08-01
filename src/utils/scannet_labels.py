"""ScanNet label tables for DINOv3 fine-tuning.

NYU40_NAMES[i] = name of nyu40 id (i+1), i = 0..39  (id 0 = void).
SCANNET20 = the 19-class benchmark vocabulary (matches eval_scannet_3d.py).
build_nyu2scannet20() -> (41,) long: nyu40 id (0..40) -> scannet20 idx or -1.
"""
from __future__ import annotations

import numpy as np
import torch

# Full NYU40 label names (ids 1..40), standard ScanNet convention.
NYU40_NAMES = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "blinds", "desk", "shelves",
    "curtain", "dresser", "pillow", "mirror", "floor mat", "clothes", "ceiling",
    "books", "refrigerator", "television", "paper", "towel", "shower curtain",
    "box", "whiteboard", "person", "nightstand", "toilet", "sink", "lamp",
    "bathtub", "bag", "otherstructure", "otherfurniture", "otherprop",
]
assert len(NYU40_NAMES) == 40

# scannet20 benchmark vocabulary (same order as eval_scannet_3d.SCANNET20).
SCANNET20 = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
             'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
             'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub']
assert len(SCANNET20) == 19


def build_nyu2scannet20() -> torch.Tensor:
    """(41,) int64: nyu40 id (0..40) -> scannet20 idx (0..18) or -1.

    id 0 (void) -> -1; ids not in the scannet20 subset -> -1.
    """
    name_to_idx = {n: i for i, n in enumerate(SCANNET20)}
    mapping = -np.ones(41, dtype=np.int64)
    for nid in range(1, 41):
        nm = NYU40_NAMES[nid - 1]
        if nm in name_to_idx:
            mapping[nid] = name_to_idx[nm]
    return torch.from_numpy(mapping)
