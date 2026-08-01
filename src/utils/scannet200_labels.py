"""ScanNet200 label tables for EmbodiedSplat-style 3D eval.

SCANNET200_CLASS_LABELS : 200 fine-grained classes (paper order).
SCANNET200_IGNORE       : classes excluded from eval (void).
SCANNET200_EVAL         : [c for c in SCANNET200_CLASS_LABELS if c not in IGNORE]  (eval index order).
RAWID_TO_NAME           : raw label id -> name (from scannet200_label_database.yaml).
build_raw_to_scannet200() -> (N,) int64: raw id -> eval idx or -1.
build_nyu40_to_scannet200_eval() -> (41,) int64: nyu40 id (0..40) -> SN200 eval idx or -1 (coarse, by name).
"""
from __future__ import annotations
import numpy as np
import torch

from .scannet_labels import NYU40_NAMES  # 40 nyu40 class names (id-1 index)

SCANNET200_CLASS_LABELS = ['wall', 'chair', 'floor', 'table', 'door', 'couch', 'cabinet', 'shelf', 'desk', 'office chair', 'bed', 'pillow', 'sink', 'picture', 'window', 'toilet', 'bookshelf', 'monitor', 'curtain', 'book', 'armchair', 'coffee table', 'box', 'refrigerator', 'lamp', 'kitchen cabinet', 'towel', 'clothes', 'tv', 'nightstand', 'counter', 'dresser', 'stool', 'cushion', 'plant', 'ceiling', 'bathtub', 'end table', 'dining table', 'keyboard', 'bag', 'backpack', 'toilet paper', 'printer', 'tv stand', 'whiteboard', 'blanket', 'shower curtain', 'trash can', 'closet', 'stairs', 'microwave', 'stove', 'shoe', 'computer tower', 'bottle', 'bin', 'ottoman', 'bench', 'board', 'washing machine', 'mirror', 'copier', 'basket', 'sofa chair', 'file cabinet', 'fan', 'laptop', 'shower', 'paper', 'person', 'paper towel dispenser', 'oven', 'blinds', 'rack', 'plate', 'blackboard', 'piano', 'suitcase', 'rail', 'radiator', 'recycling bin', 'container', 'wardrobe', 'soap dispenser', 'telephone', 'bucket', 'clock', 'stand', 'light', 'laundry basket', 'pipe', 'clothes dryer', 'guitar', 'toilet paper holder', 'seat', 'speaker', 'column', 'bicycle', 'ladder', 'bathroom stall', 'shower wall', 'cup', 'jacket', 'storage bin', 'coffee maker', 'dishwasher', 'paper towel roll', 'machine', 'mat', 'windowsill', 'bar', 'toaster', 'bulletin board', 'ironing board', 'fireplace', 'soap dish', 'kitchen counter', 'doorframe', 'toilet paper dispenser', 'mini fridge', 'fire extinguisher', 'ball', 'hat', 'shower curtain rod', 'water cooler', 'paper cutter', 'tray', 'shower door', 'pillar', 'ledge', 'toaster oven', 'mouse', 'toilet seat cover dispenser', 'furniture', 'cart', 'storage container', 'scale', 'tissue box', 'light switch', 'crate', 'power outlet', 'decoration', 'sign', 'projector', 'closet door', 'vacuum cleaner', 'candle', 'plunger', 'stuffed animal', 'headphones', 'dish rack', 'broom', 'guitar case', 'range hood', 'dustpan', 'hair dryer', 'water bottle', 'handicap bar', 'purse', 'vent', 'shower floor', 'water pitcher', 'mailbox', 'bowl', 'paper bag', 'alarm clock', 'music stand', 'projector screen', 'divider', 'laundry detergent', 'bathroom counter', 'object', 'bathroom vanity', 'closet wall', 'laundry hamper', 'bathroom stall door', 'ceiling light', 'trash bin', 'dumbbell', 'stair rail', 'tube', 'bathroom cabinet', 'cd case', 'closet rod', 'coffee kettle', 'structure', 'shower head', 'keyboard piano', 'case of water bottles', 'coat rack', 'storage organizer', 'folded chair', 'fire alarm', 'power strip', 'calendar', 'poster', 'potted plant', 'luggage', 'mattress']
assert len(SCANNET200_CLASS_LABELS) == 200

SCANNET200_IGNORE = ["office chair", "coffee table", "cushion", "plant", "end table", "dining table", "keyboard", "printer", "tv stand", "whiteboard", "blanket", "closet", "stairs", "computer tower", "bottle", "bin", "bench", "board", "washing machine", "basket", "sofa chair", "file cabinet", "fan", "paper", "person", "oven", "blinds", "rack", "plate", "blackboard", "piano", "rail", "recycling bin", "container", "telephone", "bucket", "stand", "pipe", "clothes dryer", "toilet paper holder", "seat", "speaker", "column", "ladder", "bathroom stall", "cup", "storage bin", "coffee maker", "dishwasher", "paper towel roll", "bulletin board", "soap dish", "ball", "shower curtain rod", "water cooler", "paper cutter", "tray", "shower door", "pillar", "mouse", "toilet seat cover dispenser", "furniture", "cart", "storage container", "crate", "power outlet", "decoration", "sign", "projector", "closet door", "candle", "plunger", "stuffed animal", "headphones", "broom", "guitar case", "dustpan", "hair dryer", "water bottle", "handicap bar", "purse", "mailbox", "bowl", "alarm clock", "music stand", "projector screen", "divider", "laundry detergent", "bathroom counter", "closet wall", "bathroom stall door", "ceiling light", "dumbbell", "stair rail", "tube", "bathroom cabinet", "cd case", "closet rod", "coffee kettle", "structure", "shower head", "keyboard piano", "case of water bottles", "coat rack", "storage organizer", "folded chair", "fire alarm", "power strip", "calendar", "poster", "potted plant", "luggage", "mattress", "ironing board", "paper bag", "range hood", "vent", "shower floor", "water pitcher", "object", "shower wall", "pillow", "machine", "bar", "mini fridge", "mat", "windowsill", "box", "hat", "ledge"]

SCANNET200_EVAL = [c for c in SCANNET200_CLASS_LABELS if c not in SCANNET200_IGNORE]

# raw label id -> name  (from scannet200_label_database.yaml)
RAWID_TO_NAME = {1: 'wall', 2: 'chair', 3: 'floor', 4: 'table', 5: 'door', 6: 'couch', 7: 'cabinet', 8: 'shelf', 9: 'desk', 10: 'office chair', 11: 'bed', 13: 'pillow', 14: 'sink', 15: 'picture', 16: 'window', 17: 'toilet', 18: 'bookshelf', 19: 'monitor', 21: 'curtain', 22: 'book', 23: 'armchair', 24: 'coffee table', 26: 'box', 27: 'refrigerator', 28: 'lamp', 29: 'kitchen cabinet', 31: 'towel', 32: 'clothes', 33: 'tv', 34: 'nightstand', 35: 'counter', 36: 'dresser', 38: 'stool', 39: 'cushion', 40: 'plant', 41: 'ceiling', 42: 'bathtub', 44: 'end table', 45: 'dining table', 46: 'keyboard', 47: 'bag', 48: 'backpack', 49: 'toilet paper', 50: 'printer', 51: 'tv stand', 52: 'whiteboard', 54: 'blanket', 55: 'shower curtain', 56: 'trash can', 57: 'closet', 58: 'stairs', 59: 'microwave', 62: 'stove', 63: 'shoe', 64: 'computer tower', 65: 'bottle', 66: 'bin', 67: 'ottoman', 68: 'bench', 69: 'board', 70: 'washing machine', 71: 'mirror', 72: 'copier', 73: 'basket', 74: 'sofa chair', 75: 'file cabinet', 76: 'fan', 77: 'laptop', 78: 'shower', 79: 'paper', 80: 'person', 82: 'paper towel dispenser', 84: 'oven', 86: 'blinds', 87: 'rack', 88: 'plate', 89: 'blackboard', 90: 'piano', 93: 'suitcase', 95: 'rail', 96: 'radiator', 97: 'recycling bin', 98: 'container', 99: 'wardrobe', 100: 'soap dispenser', 101: 'telephone', 102: 'bucket', 103: 'clock', 104: 'stand', 105: 'light', 106: 'laundry basket', 107: 'pipe', 110: 'clothes dryer', 112: 'guitar', 115: 'toilet paper holder', 116: 'seat', 118: 'speaker', 120: 'column', 121: 'bicycle', 122: 'ladder', 125: 'bathroom stall', 128: 'shower wall', 130: 'cup', 131: 'jacket', 132: 'storage bin', 134: 'coffee maker', 136: 'dishwasher', 138: 'paper towel roll', 139: 'machine', 140: 'mat', 141: 'windowsill', 145: 'bar', 148: 'toaster', 154: 'bulletin board', 155: 'ironing board', 156: 'fireplace', 157: 'soap dish', 159: 'kitchen counter', 161: 'doorframe', 163: 'toilet paper dispenser', 165: 'mini fridge', 166: 'fire extinguisher', 168: 'ball', 169: 'hat', 170: 'shower curtain rod', 177: 'water cooler', 180: 'paper cutter', 185: 'tray', 188: 'shower door', 191: 'pillar', 193: 'ledge', 195: 'toaster oven', 202: 'mouse', 208: 'toilet seat cover dispenser', 213: 'furniture', 214: 'cart', 221: 'storage container', 229: 'scale', 230: 'tissue box', 232: 'light switch', 233: 'crate', 242: 'power outlet', 250: 'decoration', 261: 'sign', 264: 'projector', 276: 'closet door', 283: 'vacuum cleaner', 286: 'candle', 300: 'plunger', 304: 'stuffed animal', 312: 'headphones', 323: 'dish rack', 325: 'broom', 331: 'guitar case', 342: 'range hood', 356: 'dustpan', 370: 'hair dryer', 392: 'water bottle', 395: 'handicap bar', 399: 'purse', 408: 'vent', 417: 'shower floor', 488: 'water pitcher', 540: 'mailbox', 562: 'bowl', 570: 'paper bag', 572: 'alarm clock', 581: 'music stand', 609: 'projector screen', 748: 'divider', 776: 'laundry detergent', 1156: 'bathroom counter', 1163: 'object', 1164: 'bathroom vanity', 1165: 'closet wall', 1166: 'laundry hamper', 1167: 'bathroom stall door', 1168: 'ceiling light', 1169: 'trash bin', 1170: 'dumbbell', 1171: 'stair rail', 1172: 'tube', 1173: 'bathroom cabinet', 1174: 'cd case', 1175: 'closet rod', 1176: 'coffee kettle', 1178: 'structure', 1179: 'shower head', 1180: 'keyboard piano', 1181: 'case of water bottles', 1182: 'coat rack', 1183: 'storage organizer', 1184: 'folded chair', 1185: 'fire alarm', 1186: 'power strip', 1187: 'calendar', 1188: 'poster', 1189: 'potted plant', 1190: 'luggage', 1191: 'mattress'}


def build_raw_to_scannet200() -> torch.Tensor:
    """(max_raw+1,) int64: raw label id -> SCANNET200_EVAL index or -1.

    Padded to 2048 to safely cover the full ScanNet raw nyu40-remapped id space
    (observed up to ~1256); unmapped ids -> -1 (void).
    """
    name_to_idx = {n: i for i, n in enumerate(SCANNET200_EVAL)}
    max_id = max(max(RAWID_TO_NAME.keys()), 2047)
    mapping = -np.ones(max_id + 1, dtype=np.int64)
    for rid, nm in RAWID_TO_NAME.items():
        if nm in name_to_idx:
            mapping[rid] = name_to_idx[nm]
    return torch.from_numpy(mapping)


def build_nyu40_to_scannet200_eval() -> torch.Tensor:
    """(41,) int64: nyu40 id (0..40) -> SCANNET200_EVAL index (0..69) or -1.

    Coarse mapping by exact class-name match between NYU40_NAMES and
    SCANNET200_EVAL (covers 26/70 eval classes). id 0 (void) -> -1; nyu40
    classes whose name is not an eval class -> -1. Used to add a coarse
    ScanNet200 text-CE term during fine-tuning using existing nyu40 labels
    (no extra 3D-annotation download required).
    """
    name_to_idx = {n: i for i, n in enumerate(SCANNET200_EVAL)}
    mapping = -np.ones(41, dtype=np.int64)
    for nid in range(1, 41):
        nm = NYU40_NAMES[nid - 1]
        if nm in name_to_idx:
            mapping[nid] = name_to_idx[nm]
    return torch.from_numpy(mapping)
