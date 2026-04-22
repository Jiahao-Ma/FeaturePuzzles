from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from PIL import Image

from .dataset import SA1BReader, SA1BSample


Color = Tuple[float, float, float]


def decode_segmentation(
    segmentation: Dict[str, object],
    *,
    as_bool: bool = True,
) -> np.ndarray:
    size = segmentation.get("size")
    counts = segmentation.get("counts")
    if not isinstance(size, Sequence) or len(size) != 2:
        raise ValueError("segmentation.size must be a sequence of [height, width]")
    height, width = int(size[0]), int(size[1])

    if isinstance(counts, str):
        runs = _decode_compressed_rle(counts)
    elif isinstance(counts, Sequence):
        runs = [int(value) for value in counts]
    else:
        raise TypeError("segmentation.counts must be a compressed string or a list of run lengths")

    mask = _decode_runs_to_mask(runs, height, width)
    return mask.astype(bool) if as_bool else mask


def plot_sample(
    sample: SA1BSample,
    *,
    image: Optional[Image.Image] = None,
    max_annotations: Optional[int] = None,
    show_bboxes: bool = True,
    show_points: bool = True,
    alpha: float = 0.4,
    seed: int = 0,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
) -> Axes:
    if image is None:
        with Image.open(sample.image_path) as opened_image:
            image = opened_image.convert("RGB")

    image_array = np.asarray(image)
    annotations = sample.annotations[:max_annotations] if max_annotations is not None else sample.annotations
    overlay = _compose_overlay(image_array, annotations, alpha=alpha, seed=seed)

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))

    ax.imshow(overlay)
    if show_bboxes:
        _draw_bboxes(ax, annotations, seed=seed)
    if show_points:
        _draw_points(ax, annotations)
    ax.set_axis_off()
    ax.set_title(title or f"{sample.sample_stem} | annotations={len(sample.annotations)}")
    return ax


def plot_sample_grid(
    reader: SA1BReader,
    shard_name: str,
    sample_stems: Sequence[Union[str, int]],
    *,
    max_annotations: Optional[int] = 20,
    columns: int = 2,
    figsize: Tuple[int, int] = (16, 12),
    seed: int = 0,
) -> Tuple[plt.Figure, np.ndarray]:
    sample_stems = list(sample_stems)
    rows = int(np.ceil(len(sample_stems) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=figsize)
    axes_array = np.atleast_1d(axes).reshape(rows, columns)

    for index, sample_stem in enumerate(sample_stems):
        row, column = divmod(index, columns)
        sample = reader.load_sample(shard_name, sample_stem)
        image = reader.load_image(shard_name, sample_stem)
        plot_sample(
            sample,
            image=image,
            max_annotations=max_annotations,
            show_bboxes=True,
            show_points=False,
            seed=seed + index,
            ax=axes_array[row, column],
            title=sample.sample_stem,
        )

    for index in range(len(sample_stems), rows * columns):
        row, column = divmod(index, columns)
        axes_array[row, column].set_axis_off()

    figure.tight_layout()
    return figure, axes_array


def _decode_compressed_rle(counts_text: str) -> List[int]:
    counts: List[int] = []
    current_index = 0
    run_index = 0

    while current_index < len(counts_text):
        shift = 0
        value = 0

        while True:
            byte_value = ord(counts_text[current_index]) - 48
            current_index += 1
            value |= (byte_value & 0x1F) << shift
            shift += 5
            if (byte_value & 0x20) == 0:
                if byte_value & 0x10:
                    value |= -1 << shift
                break

        if run_index > 2:
            value += counts[run_index - 2]

        counts.append(value)
        run_index += 1

    return counts


def _decode_runs_to_mask(runs: Sequence[int], height: int, width: int) -> np.ndarray:
    total_size = height * width
    flat_mask = np.zeros(total_size, dtype=np.uint8)
    cursor = 0
    value = 0

    for run in runs:
        if run < 0:
            raise ValueError("RLE run lengths must be non-negative after decoding")
        next_cursor = cursor + run
        if next_cursor > total_size:
            raise ValueError("Decoded RLE exceeds image size")
        if value == 1:
            flat_mask[cursor:next_cursor] = 1
        cursor = next_cursor
        value = 1 - value

    if cursor != total_size:
        raise ValueError("Decoded RLE does not fill the full image size")

    return flat_mask.reshape((width, height)).T


def _compose_overlay(
    image_array: np.ndarray,
    annotations: Sequence,
    *,
    alpha: float,
    seed: int,
) -> np.ndarray:
    overlay = image_array.astype(np.float32).copy()
    rng = np.random.default_rng(seed)

    for annotation in annotations:
        mask = decode_segmentation(annotation.segmentation, as_bool=True)
        color = rng.uniform(0.1, 0.95, size=3)
        overlay[mask] = overlay[mask] * (1.0 - alpha) + color * 255.0 * alpha

    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)


def _draw_bboxes(ax: Axes, annotations: Sequence, *, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for annotation in annotations:
        if len(annotation.bbox) != 4:
            continue
        x, y, width, height = annotation.bbox
        color = tuple(rng.uniform(0.1, 0.95, size=3))
        ax.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                fill=False,
                linewidth=1.5,
                edgecolor=color,
            )
        )


def _draw_points(ax: Axes, annotations: Sequence) -> None:
    for annotation in annotations:
        for point in annotation.point_coords:
            if len(point) != 2:
                continue
            ax.scatter(point[0], point[1], s=18, c="white", edgecolors="black", linewidths=0.5)