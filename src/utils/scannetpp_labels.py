"""ScanNet++ label tables for EmbodiedSplat-style 3D eval (mirrors repo database/class_labels.py)."""
from __future__ import annotations
import numpy as np
import torch

SCANNETPP_CLASS_LABELS = ['wall', 'ceiling', 'floor', 'door', 'table', 'cabinet', 'ceiling lamp', 'curtain', 'chair', 'blinds', 'bookshelf', 'office chair', 'window', 'whiteboard', 'monitor', 'shelf', 'object', 'window frame', 'structure', 'box', 'heater', 'kitchen cabinet', 'sofa', 'bed', 'doorframe', 'plant', 'blanket', 'machine', 'tv', 'kitchen counter', 'trash can', 'jacket', 'sink', 'carpet', 'bag', 'counter', 'refrigerator', 'picture', 'pillow', 'towel', 'stool', 'suitcase', 'backpack', 'keyboard', 'poster', 'book', 'paper', 'microwave', 'bottle', 'sofa chair', 'oven', 'cushion', 'power socket', 'shoes', 'shoe rack', 'objects', 'coffee table', 'bin', 'stove', 'pc', 'closet', 'cardboard', 'plant pot', 'false ceiling', 'comforter', 'nightstand', 'dining chair', 'light switch', 'floor lamp', 'fan', 'dining table', 'speaker', 'light', 'lamp', 'kettle', 'shoe', 'bicycle', 'surfboard', 'mug', 'mouse', 'smoke detector', 'cup', 'cutting board', 'paper towel', 'ceiling vent', 'fitted wardrobe', 'clock', 'tap', 'wall lamp', 'plate', 'folding sofa', 'ottoman chair', 'jar', 'notebook', 'bedframe', 'whiteboard eraser', 'ventilator', 'shoes holder', 'slipper', 'toaster', 'spotlight', 'mouse pad', 'headphones', 'bedside cabinet', 'helmet', 'headphone', 'scarf', 'hat', 'wall board', 'footstool', 'water kettle', 'monitor stand', 'led ceiling fan', 'switch', 'boot', 'kitchen utensils', 'cap', 'sit-up pillow', 'sponge', 'kitchen hood', 'bananas', 'package', 'stick', 'postcards', 'pumper', 'coaster', 'knife', 'flip flop', 'knife holder', 'chessboard', 'cleaning brush', 'remote controller', 'kitchen glove', 'apple', 'scissor', 'stationery', 'ceiling speaker', 'post-it', 'pliers', 'headphone holder', 'door stopper']
assert len(SCANNETPP_CLASS_LABELS) == 141

SCANNETPP_IGNORE = ['ceiling', 'ceiling lamp', 'blinds', 'office chair', 'shelf', 'object', 'window frame', 'structure', 'box', 'heater', 'kitchen cabinet', 'doorframe', 'blanket', 'machine', 'kitchen counter', 'trash can', 'jacket', 'carpet', 'bag', 'pillow', 'towel', 'suitcase', 'backpack', 'keyboard', 'poster', 'book', 'paper', 'microwave', 'bottle', 'sofa chair', 'oven', 'cushion', 'power socket', 'shoes', 'shoe rack', 'objects', 'coffee table', 'bin', 'stove', 'pc', 'closet', 'cardboard', 'plant pot', 'false ceiling', 'comforter', 'nightstand', 'dining chair', 'light switch', 'floor lamp', 'fan', 'dining table', 'speaker', 'light', 'lamp', 'kettle', 'shoe', 'bicycle', 'surfboard', 'mug', 'mouse', 'smoke detector', 'cup', 'cutting board', 'paper towel', 'ceiling vent', 'fitted wardrobe', 'clock', 'tap', 'wall lamp', 'plate', 'folding sofa', 'ottoman chair', 'jar', 'notebook', 'bedframe', 'whiteboard eraser', 'ventilator', 'shoes holder', 'slipper', 'toaster', 'spotlight', 'mouse pad', 'headphones', 'bedside cabinet', 'helmet', 'headphone', 'scarf', 'hat', 'wall board', 'footstool', 'water kettle', 'monitor stand', 'led ceiling fan', 'switch', 'boot', 'kitchen utensils', 'cap', 'sit-up pillow', 'sponge', 'kitchen hood', 'bananas', 'package', 'stick', 'postcards', 'pumper', 'coaster', 'knife', 'flip flop', 'knife holder', 'chessboard', 'cleaning brush', 'remote controller', 'kitchen glove', 'apple', 'scissor', 'stationery', 'ceiling speaker', 'post-it', 'pliers', 'headphone holder', 'door stopper']
assert len(SCANNETPP_IGNORE) == 121

SCANNETPP_EVAL = [c for c in SCANNETPP_CLASS_LABELS if c not in SCANNETPP_IGNORE]
assert len(SCANNETPP_EVAL) == 20

# raw label id -> index into SCANNETPP_CLASS_LABELS (141)
SCANNETPP_CLASS_ID = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 11: 10, 12: 11, 13: 12, 14: 13, 17: 14, 18: 15, 19: 16, 20: 17, 22: 18, 23: 19, 24: 20, 25: 21, 27: 22, 28: 23, 30: 24, 35: 25, 36: 26, 37: 27, 42: 28, 46: 29, 47: 30, 49: 31, 54: 32, 55: 33, 56: 34, 57: 35, 58: 36, 59: 37, 60: 38, 63: 39, 66: 40, 67: 41, 68: 42, 71: 43, 83: 44, 88: 45, 91: 46, 96: 47, 106: 48, 108: 49, 115: 50, 117: 51, 118: 52, 122: 53, 129: 54, 137: 55, 142: 56, 144: 57, 147: 58, 150: 59, 153: 60, 159: 61, 173: 62, 176: 63, 177: 64, 192: 65, 200: 66, 204: 67, 208: 68, 209: 69, 213: 70, 214: 71, 216: 72, 243: 73, 253: 74, 255: 75, 268: 76, 271: 77, 284: 78, 296: 79, 299: 80, 300: 81, 301: 82, 312: 83, 334: 84, 349: 85, 358: 86, 368: 87, 369: 88, 370: 89, 381: 90, 386: 91, 387: 92, 389: 93, 406: 94, 414: 95, 429: 96, 433: 97, 463: 98, 471: 99, 472: 100, 513: 101, 515: 102, 521: 103, 536: 104, 543: 105, 549: 106, 551: 107, 560: 108, 565: 109, 581: 110, 615: 111, 638: 112, 664: 113, 702: 114, 707: 115, 764: 116, 783: 117, 851: 118, 855: 119, 957: 120, 1000: 121, 1006: 122, 1088: 123, 1135: 124, 1155: 125, 1169: 126, 1182: 127, 1214: 128, 1225: 129, 1318: 130, 1337: 131, 1353: 132, 1386: 133, 1477: 134, 1553: 135, 1554: 136, 1568: 137, 1584: 138, 1590: 139, 1611: 140}


def build_raw_to_scannetpp() -> torch.Tensor:
    """(4096,) int64: OFFICIAL raw label id -> SCANNETPP_EVAL index (0..19) or -1 (void/ignore).

    Uses the OFFICIAL ScanNet++ metadata/semantic_classes.txt (raw id i -> class name,
    2878 fine-grained classes), the authoritative mapping for the official semantic meshes.
    The previous SCANNETPP_CLASS_ID table came from a local HF-mirror that RE-NUMBERED the
    raw ids, so it does NOT apply to the official meshes (verified: same xyz, 62% different
    raw labels vs official; projection check confirms official raw 0=wall, raw 1=floor land
    correctly on walls/floor).
    """
    import os
    name_to_evalidx = {n: i for i, n in enumerate(SCANNETPP_EVAL)}
    here = os.path.dirname(os.path.abspath(__file__))
    sc_path = os.path.join(here, "scannetpp_semantic_classes_official.txt")
    names = open(sc_path).read().splitlines()
    mapping = -np.ones(4096, dtype=np.int64)
    for raw, nm in enumerate(names):
        if nm in name_to_evalidx:
            mapping[raw] = name_to_evalidx[nm]
    return torch.from_numpy(mapping)
