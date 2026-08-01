"""ScanNet Puzzles dataset for DINOv3-teacher fine-tuning on EmbodiedSplat data.

Single-image multi-crop (Puzzles) training, but the parent image is a ScanNet
colour frame and supervision adds a per-pixel text-aligned CE loss using the
dense nyu40 labels (precomputed by preprocess_scannet_labels.py).

Each item returns the SAME fields the DINOv3 training loop expects
(imgs / parent_img / crop_boxes_parent / parent_size) PLUS a dense parent label
map (nyu40, -1 void) which the training loop warps into per-view label maps with
the SAME crop-box grid as the teacher (nearest sampling) for the text-CE loss.

Geometry: colour (1296x968) and depth (480x640) share FOV + centre, so the colour
frame is resized to (480,640) to align with the precomputed depth-res labels.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .fast_box_gen import generate_overlapping_boxes as _gen_boxes

logger = logging.getLogger(__name__)


class ScanNetPuzzlesDataset(Dataset):
    """Per-frame ScanNet Puzzles crops + dense nyu40 parent label map."""

    def __init__(
        self,
        data_root: str,
        *,
        split: str = "train",
        layout: str = "embodiedsplat",   # or "frames25k" (official scannet_frames_25k)
        scenes: Optional[Sequence[str]] = None,
        n_views: int = 6,
        resolution: int = 256,
        teacher_long: int = 2048,
        feature_res: int = 128,
        frame_stride: int = 20,
        max_frames: int = 60,
        min_box_size_scale: float = 0.5,
        max_box_size_scale: float = 0.9,
        min_overlap: float = 0.5,
        max_overlap: float = 0.9,
        min_label_coverage: float = 0.4,
        fill_holes: bool = True,
        augment: bool = True,
        seed: int = 42,
        load_sn200_2d: bool = False,   # oracle: load projected SN200 2D labels (labels25k_sn200/<fi>.npy, raw id, -1 void)
    ) -> None:
        self.layout = layout
        self.load_sn200_2d = load_sn200_2d
        # embodiedsplat: <root>/<split>/<scene>; frames25k: <root>/<scene>;
        # gen3df: <root>/<scene> (HF Gen3DF/scannet extracted zips, labels25k)
        # snpp: <root>/<scene> (ScanNet++ DSLR frames + projected labels2d_snpp)
        if layout == "embodiedsplat":
            self.root = Path(data_root) / split
        else:
            self.root = Path(data_root)
        self.n_views = n_views
        self.resolution = resolution
        self.teacher_long = teacher_long
        self.feature_res = feature_res
        self.min_label_coverage = min_label_coverage
        self.fill_holes = fill_holes
        self.augment = augment
        self.seed = seed
        self._box_kwargs = dict(
            n_boxes=n_views,
            min_box_size_scale=min_box_size_scale,
            max_box_size_scale=max_box_size_scale,
            min_overlap=min_overlap,
            max_overlap=max_overlap,
            minTrue=min_label_coverage,
        )

        scene_dirs = sorted(p for p in self.root.iterdir() if p.is_dir())
        if scenes is not None:
            scene_dirs = [p for p in scene_dirs if p.name in scenes]
        self._index: List[Tuple[str, int]] = []
        for sd in scene_dirs:
            if self.layout == "frames25k":
                lab_dir = sd / "label"
                if not lab_dir.is_dir():
                    continue
                for f in sorted(lab_dir.glob("*.png")):
                    self._index.append((sd.name, int(f.stem)))
            elif self.layout == "snpp":
                lab_dir = sd / "labels2d_snpp"
                if not lab_dir.is_dir():
                    continue
                for f in sorted(lab_dir.glob("*.npy")):
                    self._index.append((sd.name, f.stem))   # string stem (DSC...)
            else:
                lab_dir = sd / "labels25k"
                if not lab_dir.is_dir():
                    continue
                for f in sorted(lab_dir.glob("*.npy")):
                    self._index.append((sd.name, int(f.stem)))
        logger.info("ScanNetPuzzlesDataset: %d frames across %d scenes (split=%s layout=%s)",
                    len(self._index), len(scene_dirs), split, layout)

    def __len__(self) -> int:
        return len(self._index)

    _MAX_RETRIES = 8

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        for attempt in range(self._MAX_RETRIES):
            try:
                return self._load((idx + attempt) % len(self._index))
            except (ValueError, RuntimeError, OSError, KeyError) as exc:
                if attempt == self._MAX_RETRIES - 1:
                    logger.warning("scanitem %d failed: %s", idx, exc)
                    return self._blank()
        return self._blank()

    def _blank(self) -> Dict[str, torch.Tensor]:
        N, R, S = self.n_views, self.resolution, self.teacher_long
        return {
            "imgs": torch.zeros(N, 3, R, R),
            "parent_img": torch.zeros(3, S, S, dtype=torch.uint8),
            "crop_boxes_parent": torch.zeros(N, 4),
            "parent_size": torch.tensor([480, 640], dtype=torch.float32),
            "parent_label": torch.full((1, 480, 640), -1, dtype=torch.int16),
            "parent_label_sn200": torch.full((1, 480, 640), -1, dtype=torch.int32),
            "parent_label_snpp": torch.full((1, 480, 640), -1, dtype=torch.int32),
        }

    def _load(self, idx: int) -> Dict[str, torch.Tensor]:
        scene, fi = self._index[idx]
        sd = self.root / scene
        if self.layout == "frames25k":
            color_path = sd / "color" / f"{fi:06d}.jpg"
        elif self.layout == "snpp":
            color_path = sd / "dslr" / "undistorted_images" / f"{fi}.JPG"
        else:
            color_path = sd / "color" / f"{fi}.jpg"
        color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        if color is None:
            raise OSError(f"color {scene}/{fi}")
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)               # (968,1296,3)
        # align colour to depth resolution (same FOV + centre)
        parent = cv2.resize(color, (640, 480), interpolation=cv2.INTER_AREA)  # (480,640,3)
        if self.layout == "snpp":
            # projected SN++ 2D labels (raw class id, -1 void) at DSLR res -> 480x640
            lbl = np.load(sd / "labels2d_snpp" / f"{fi}.npy")
            if lbl.shape[:2] != (480, 640):
                lbl = cv2.resize(lbl, (640, 480), interpolation=cv2.INTER_NEAREST)
            label = lbl.astype(np.int32)  # raw SN++ id, -1 void (id 0 = wall is VALID)
            valid = label >= 0
        elif self.layout == "frames25k":
            # official per-frame nyu40 label png (1296x968 or 640x480), 0 = void
            lbl = cv2.imread(str(sd / "label" / f"{fi:06d}.png"), cv2.IMREAD_UNCHANGED)
            if lbl is None:
                raise OSError(f"label {scene}/{fi}")
            if lbl.shape[:2] != (480, 640):
                lbl = cv2.resize(lbl, (640, 480), interpolation=cv2.INTER_NEAREST)
            label = lbl.astype(np.int32)                                     # 0 void
            valid = label > 0
        else:
            label = np.load(sd / "labels25k" / f"{fi}.npy").astype(np.int32)  # -1 void
            valid = label > 0
        if self.fill_holes:
            valid = self._close_mask(valid)
        if not valid.any():
            raise ValueError("empty label")

        rng = np.random.default_rng(self.seed + idx)
        boxes, _ = _gen_boxes(image=parent, mask=valid, rng=rng, **self._box_kwargs)
        if not boxes:
            # fall back to a centred box covering the labelled region
            ys, xs = np.nonzero(valid)
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            boxes = [(int(x1), int(y1), int(x2), int(y2))] * self.n_views
        # pad / trim to n_views
        if len(boxes) < self.n_views:
            boxes = list(boxes) + [boxes[-1]] * (self.n_views - len(boxes))
        elif len(boxes) > self.n_views:
            boxes = list(boxes)[:self.n_views]

        R = self.resolution
        crops = []
        for (x1, y1, x2, y2) in boxes:
            c = parent[y1:y2, x1:x2].copy()
            c = cv2.resize(c, (R, R), interpolation=cv2.INTER_LINEAR)
            crops.append(c)
        if self.augment:
            crops = self._photometric(crops, rng)
        imgs = torch.from_numpy(np.stack(crops, 0)).permute(0, 3, 1, 2).float().div_(255.0)

        S = self.teacher_long
        parent_sq = cv2.resize(parent, (S, S), interpolation=cv2.INTER_AREA)
        parent_t = torch.from_numpy(np.ascontiguousarray(parent_sq)).permute(2, 0, 1)  # uint8 (3,S,S)

        boxes_t = torch.tensor(boxes, dtype=torch.float32)            # (N,4)
        # parent_label is the NYU40 label used by the ce/ce20 loss. For snpp layout there
        # are no nyu40 labels, so emit void (-1) to keep ce/ce20 from firing on SN++ samples
        # (which would otherwise misread raw SN++ ids as nyu40 ids in mixed training).
        # Box generation above already used the real SN++ `valid` mask; ce_snpp uses
        # parent_label_snpp (the real SN++ id), not parent_label.
        if self.layout == "snpp":
            parent_nyu = np.full((480, 640), -1, dtype=np.int32)   # void
        else:
            parent_nyu = label                                       # nyu40 id (0 or -1 void)
        label_t = torch.from_numpy(parent_nyu).unsqueeze(0).to(torch.int16)  # (1,480,640) -1 void
        out = {
            "imgs": imgs.contiguous(),
            "parent_img": parent_t.contiguous(),
            "crop_boxes_parent": boxes_t,
            "parent_size": torch.tensor([480, 640], dtype=torch.float32),
            "parent_label": label_t.contiguous(),
        }
        # always provide parent_label_sn200 (default -1 void) so collate is uniform
        if self.load_sn200_2d:
            sn200_path = sd / "labels25k_sn200" / f"{fi}.npy"
            if sn200_path.exists():
                sn200 = np.load(sn200_path).astype(np.int32)  # (480,640) raw id, -1 void
            else:
                sn200 = np.full((480, 640), -1, dtype=np.int32)
        else:
            sn200 = np.full((480, 640), -1, dtype=np.int32)
        out["parent_label_sn200"] = torch.from_numpy(sn200).unsqueeze(0).contiguous()  # int32 (1,480,640)
        # always provide parent_label_snpp (default -1 void) so collate is uniform.
        # For snpp layout, parent_label IS the raw SN++ id -> mirror it into parent_label_snpp.
        if self.layout == "snpp":
            snpp = label.astype(np.int32)  # raw SN++ id, -1 void (id 0 = wall is VALID)
        else:
            snpp = np.full((480, 640), -1, dtype=np.int32)
        out["parent_label_snpp"] = torch.from_numpy(snpp).unsqueeze(0).contiguous()  # int32 (1,480,640)
        return out

    def load_label_only(self, idx: int) -> np.ndarray:
        """Return the raw nyu40 label map (H,W) for class-frequency stats.
        0 = void (frames25k) / -1 = void (embodiedsplat). Kept as-is.
        """
        scene, fi = self._index[idx]
        sd = self.root / scene
        if self.layout == "frames25k":
            lbl = cv2.imread(str(sd / "label" / f"{fi:06d}.png"), cv2.IMREAD_UNCHANGED)
            if lbl is None:
                raise OSError(f"label {scene}/{fi}")
            if lbl.shape[:2] != (480, 640):
                lbl = cv2.resize(lbl, (640, 480), interpolation=cv2.INTER_NEAREST)
            return lbl.astype(np.int32)
        if self.layout == "snpp":
            return np.load(sd / "labels2d_snpp" / f"{fi}.npy").astype(np.int32)  # raw SN++ id, -1 void
        return np.load(sd / "labels25k" / f"{fi}.npy").astype(np.int32)

    @staticmethod
    def _close_mask(mask: np.ndarray) -> np.ndarray:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        m = mask.astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        return m.astype(bool)

    @staticmethod
    def _photometric(crops: List[np.ndarray], rng) -> List[np.ndarray]:
        out = []
        for c in crops:
            img = c.astype(np.float32)
            # brightness
            img += rng.uniform(-20, 20)
            # contrast
            img = img * rng.uniform(0.9, 1.1)
            # slight colour jitter
            for ch in range(3):
                img[..., ch] += rng.uniform(-10, 10)
            img = np.clip(img, 0, 255).astype(np.uint8)
            out.append(img)
        return out


def scannet_collate(batch):
    return {
        "imgs": torch.stack([b["imgs"] for b in batch]),                     # (B,N,3,R,R)
        "parent_img": torch.stack([b["parent_img"] for b in batch]),         # (B,3,S,S)
        "crop_boxes_parent": torch.stack([b["crop_boxes_parent"] for b in batch]),  # (B,N,4)
        "parent_size": torch.stack([b["parent_size"] for b in batch]),       # (B,2)
        "parent_label": torch.stack([b["parent_label"] for b in batch]),     # (B,1,480,640)
        "parent_label_sn200": torch.stack([b["parent_label_sn200"] for b in batch]),  # (B,1,480,640) int32, -1 void
        "parent_label_snpp": torch.stack([b["parent_label_snpp"] for b in batch]),    # (B,1,480,640) int32, -1 void
    }
