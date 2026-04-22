"""Automatic prompt-mask generator backed by YOLOv8-seg with a FastSAM fallback.

Pipeline per image:

1. Run **YOLOv8-seg** (COCO-80 classes) and keep detections that pass a
   confidence / size / dedup filter. These masks come with semantic class
   labels we reuse for file names.
2. If fewer than ``k`` labelled masks were produced (scenes outside COCO's
   category set, e.g. bathrooms, minimalist kitchens), we call
   **FastSAM** for class-agnostic auto-segmentation and fill the remainder
   with labelled ``region_XX`` records, deduplicated against the YOLO
   masks by bbox IoU.

All masks are written as binary PNGs at the **original image resolution**
so ``prompt_matching.prompt.encode_mask_prompt`` can consume them directly.
The metadata (class, confidence, area, bbox, source) lands in
``masks_meta.json`` for downstream reporting.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import cv2
import numpy as np


@dataclass
class MaskRecord:
    mask_path: str
    class_name: str
    class_id: int
    confidence: float
    area_px: int
    area_ratio: float
    bbox_xyxy: List[int]
    score: float
    source: str  # "yolo" or "fastsam"


def _weights_path(name: str) -> str:
    if os.path.isfile(name):
        return name
    local = Path(__file__).resolve().parent / "weights" / name
    return str(local) if local.is_file() else name


def _load_yolo(model_name: str, device: str):
    from ultralytics import YOLO
    m = YOLO(_weights_path(model_name))
    try:
        m.to(device)
    except Exception:
        pass
    return m


def _load_fastsam(model_name: str, device: str):
    from ultralytics import FastSAM
    m = FastSAM(_weights_path(model_name))
    try:
        m.to(device)
    except Exception:
        pass
    return m


def _bbox_iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


def _bbox_from_mask(m_bin: np.ndarray) -> List[int]:
    ys, xs = np.where(m_bin > 0)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _to_candidates_from_yolo(result, H: int, W: int,
                             min_area_ratio: float, max_area_ratio: float):
    cands = []
    if result is None or result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
        return cands
    masks_np = result.masks.data.detach().float().cpu().numpy()
    confs = result.boxes.conf.detach().float().cpu().numpy()
    cls_ids = result.boxes.cls.detach().long().cpu().numpy()
    xyxy = result.boxes.xyxy.detach().float().cpu().numpy()
    names = result.names
    img_area = float(H * W)
    for i in range(masks_np.shape[0]):
        m = masks_np[i]
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        m_bin = (m >= 0.5).astype(np.uint8)
        area_px = int(m_bin.sum())
        ratio = area_px / img_area
        if ratio < min_area_ratio or ratio > max_area_ratio:
            continue
        cands.append({
            "mask": m_bin,
            "class_id": int(cls_ids[i]),
            "class_name": names[int(cls_ids[i])],
            "confidence": float(confs[i]),
            "area_px": area_px,
            "area_ratio": ratio,
            "bbox_xyxy": [int(x) for x in xyxy[i].tolist()],
            "score": float(confs[i] * np.sqrt(ratio)),
            "source": "yolo",
        })
    return cands


def _to_candidates_from_fastsam(result, H: int, W: int,
                                min_area_ratio: float, max_area_ratio: float):
    cands = []
    if result is None or result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
        return cands
    masks_np = result.masks.data.detach().float().cpu().numpy()
    confs_src = result.boxes.conf if result.boxes is not None else None
    img_area = float(H * W)
    for i in range(masks_np.shape[0]):
        m = masks_np[i]
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        m_bin = (m >= 0.5).astype(np.uint8)
        area_px = int(m_bin.sum())
        ratio = area_px / img_area
        if ratio < min_area_ratio or ratio > max_area_ratio:
            continue
        conf_i = float(confs_src[i].item()) if confs_src is not None and i < len(confs_src) else 1.0
        bbox = _bbox_from_mask(m_bin)
        cands.append({
            "mask": m_bin,
            "class_id": -1,
            "class_name": "region",
            "confidence": conf_i,
            "area_px": area_px,
            "area_ratio": ratio,
            "bbox_xyxy": bbox,
            "score": float(conf_i * np.sqrt(ratio)),
            "source": "fastsam",
        })
    return cands


def _dedup_and_rank(candidates, k: int, dedup_iou: float):
    candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)
    kept = []
    for c in candidates:
        dup = False
        for q in kept:
            if _bbox_iou(c["bbox_xyxy"], q["bbox_xyxy"]) > dedup_iou:
                dup = True
                break
        if not dup:
            kept.append(c)
        if len(kept) >= k:
            break
    return kept


def generate_prompt_masks(
    image_path: str,
    out_dir: str,
    k: int = 5,
    yolo_model: str = "yolov8x-seg.pt",
    fastsam_model: str = "FastSAM-x.pt",
    device: str = "cuda",
    imgsz: int = 1024,
    min_conf: float = 0.25,
    min_area_ratio: float = 0.003,
    max_area_ratio: float = 0.6,
    iou_suppress: float = 0.5,
    dedup_iou: float = 0.6,
    enable_fastsam_fallback: bool = True,
) -> List[MaskRecord]:
    """Run YOLOv8-seg → FastSAM fallback → persist top-``k`` masks."""

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(image_path)
    H, W = img_bgr.shape[:2]

    yolo = _load_yolo(yolo_model, device)
    y_res = yolo.predict(
        source=image_path,
        imgsz=imgsz,
        conf=min_conf,
        iou=iou_suppress,
        device=device,
        verbose=False,
        retina_masks=True,
    )[0]
    yolo_cands = _to_candidates_from_yolo(y_res, H, W, min_area_ratio, max_area_ratio)
    kept = _dedup_and_rank(yolo_cands, k, dedup_iou)

    if len(kept) < k and enable_fastsam_fallback:
        print(f"[generate_masks] YOLO produced {len(kept)} mask(s), running FastSAM fallback ...")
        fast = _load_fastsam(fastsam_model, device)
        f_res = fast.predict(
            source=image_path,
            imgsz=imgsz,
            conf=max(min_conf, 0.3),
            iou=iou_suppress,
            device=device,
            verbose=False,
            retina_masks=True,
        )[0]
        fastsam_cands = _to_candidates_from_fastsam(f_res, H, W, min_area_ratio, max_area_ratio)
        remaining = k - len(kept)
        sam_kept = _dedup_and_rank(
            [c for c in fastsam_cands
             if all(_bbox_iou(c["bbox_xyxy"], q["bbox_xyxy"]) <= dedup_iou for q in kept)],
            remaining,
            dedup_iou,
        )
        region_idx = 0
        for c in sam_kept:
            c["class_name"] = f"region_{region_idx:02d}"
            region_idx += 1
        kept.extend(sam_kept)

    records: List[MaskRecord] = []
    seen_classes: dict[str, int] = {}
    for c in kept:
        cls = c["class_name"].replace(" ", "_")
        idx = seen_classes.get(cls, 0)
        seen_classes[cls] = idx + 1
        suffix = f"_{idx}" if idx > 0 else ""
        fname = f"mask_{len(records):02d}_{cls}{suffix}.png"
        fpath = out_dir_p / fname
        cv2.imwrite(str(fpath), c["mask"] * 255)
        records.append(MaskRecord(
            mask_path=str(fpath),
            class_name=c["class_name"],
            class_id=c["class_id"],
            confidence=c["confidence"],
            area_px=c["area_px"],
            area_ratio=c["area_ratio"],
            bbox_xyxy=c["bbox_xyxy"],
            score=c["score"],
            source=c["source"],
        ))

    with open(out_dir_p / "masks_meta.json", "w") as f:
        json.dump(
            {
                "source_image": image_path,
                "image_HxW": [H, W],
                "yolo_model": yolo_model,
                "fastsam_model": fastsam_model,
                "imgsz": imgsz,
                "min_conf": min_conf,
                "min_area_ratio": min_area_ratio,
                "max_area_ratio": max_area_ratio,
                "records": [asdict(r) for r in records],
            },
            f,
            indent=2,
        )
    print(
        f"[generate_masks] wrote {len(records)} masks to {out_dir} "
        f"(classes: {[r.class_name for r in records]}, "
        f"sources: {[r.source for r in records]})"
    )
    return records


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--yolo", default="yolov8x-seg.pt")
    parser.add_argument("--fastsam", default="FastSAM-x.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_fastsam", action="store_true")
    args = parser.parse_args()

    recs = generate_prompt_masks(
        args.image, args.out_dir, k=args.k,
        yolo_model=args.yolo, fastsam_model=args.fastsam, device=args.device,
        enable_fastsam_fallback=not args.no_fastsam,
    )
    for r in recs:
        print(
            f"  {r.mask_path}  [{r.source:7s}]  class={r.class_name:15s}  "
            f"conf={r.confidence:.3f}  area_ratio={r.area_ratio:.3f}  score={r.score:.3f}"
        )
