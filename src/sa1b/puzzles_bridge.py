from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from .dataset import SA1BAnnotation, SA1BSample
from .visualization import decode_segmentation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PUZZLES_ROOT = PROJECT_ROOT / "puzzles"
if str(PUZZLES_ROOT) not in sys.path:
    sys.path.insert(0, str(PUZZLES_ROOT))

generate_overlapping_boxes = importlib.import_module("Puzzles.puzzles_utils2d").generate_overlapping_boxes


Box = Tuple[int, int, int, int]


@dataclass(frozen=True)
class SA1BPuzzleResult:
    sample_stem: str
    image: np.ndarray
    foreground_mask: np.ndarray
    boxes: List[Box]
    template_box: Box
    crops: List[np.ndarray]
    resized_crops: List[np.ndarray]
    kept_annotation_ids: List[int]


def build_foreground_mask(
    sample: SA1BSample,
    *,
    max_annotations: Optional[int] = None,
    min_area: float = 0.0,
    min_predicted_iou: Optional[float] = None,
    min_stability_score: Optional[float] = None,
) -> Tuple[np.ndarray, List[SA1BAnnotation]]:
    height = sample.image_info.height
    width = sample.image_info.width
    annotations = sample.annotations[:max_annotations] if max_annotations is not None else sample.annotations

    foreground_mask = np.zeros((height, width), dtype=bool)
    kept_annotations: List[SA1BAnnotation] = []
    for annotation in annotations:
        if annotation.area < min_area:
            continue
        if min_predicted_iou is not None:
            if annotation.predicted_iou is None or annotation.predicted_iou < min_predicted_iou:
                continue
        if min_stability_score is not None:
            if annotation.stability_score is None or annotation.stability_score < min_stability_score:
                continue

        foreground_mask |= decode_segmentation(annotation.segmentation, as_bool=True)
        kept_annotations.append(annotation)

    return foreground_mask, kept_annotations


def crop_boxes(image: np.ndarray, boxes: Sequence[Box]) -> List[np.ndarray]:
    return [image[y1:y2, x1:x2].copy() for x1, y1, x2, y2 in boxes]


def resize_crops_to_template(crops: Sequence[np.ndarray], template_box: Box) -> List[np.ndarray]:
    template_width = template_box[2] - template_box[0]
    template_height = template_box[3] - template_box[1]
    return [cv2.resize(crop, (template_width, template_height), interpolation=cv2.INTER_LINEAR) for crop in crops]


def generate_puzzle_crops(
    sample: SA1BSample,
    *,
    image: Optional[Image.Image] = None,
    n_boxes: int = 6,
    min_box_size_scale: float = 0.2,
    max_box_size_scale: float = 0.5,
    min_overlap: float = 0.1,
    max_overlap: float = 0.3,
    min_mask_coverage: float = 0.5,
    max_annotations: Optional[int] = None,
    min_area: float = 0.0,
    min_predicted_iou: Optional[float] = None,
    min_stability_score: Optional[float] = None,
    seed: int = 0,
) -> SA1BPuzzleResult:
    if image is None:
        with Image.open(sample.image_path) as opened_image:
            image = opened_image.convert("RGB")

    image_array = np.asarray(image)
    foreground_mask, kept_annotations = build_foreground_mask(
        sample,
        max_annotations=max_annotations,
        min_area=min_area,
        min_predicted_iou=min_predicted_iou,
        min_stability_score=min_stability_score,
    )
    if not foreground_mask.any():
        raise ValueError("No valid foreground mask could be built from the selected annotations")

    rng = np.random.default_rng(seed)
    boxes, template_box = generate_overlapping_boxes(
        image=image_array,
        mask=foreground_mask,
        n_boxes=n_boxes,
        min_box_size_scale=min_box_size_scale,
        max_box_size_scale=max_box_size_scale,
        min_overlap=min_overlap,
        max_overlap=max_overlap,
        minTrue=min_mask_coverage,
        rng=rng,
    )
    if not boxes:
        raise ValueError("Puzzles did not return any valid crop boxes for this sample")

    crops = crop_boxes(image_array, boxes)
    resized_crops = resize_crops_to_template(crops, template_box)
    kept_annotation_ids = [annotation.annotation_id for annotation in kept_annotations]

    return SA1BPuzzleResult(
        sample_stem=sample.sample_stem,
        image=image_array,
        foreground_mask=foreground_mask,
        boxes=list(boxes),
        template_box=template_box,
        crops=crops,
        resized_crops=resized_crops,
        kept_annotation_ids=kept_annotation_ids,
    )
