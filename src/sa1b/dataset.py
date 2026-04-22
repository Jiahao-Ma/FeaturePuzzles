from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

from PIL import Image


PathLike = Union[str, Path]


@dataclass(frozen=True)
class SA1BImageInfo:
    image_id: int
    width: int
    height: int
    file_name: str
    image_path: Path


@dataclass(frozen=True)
class SA1BAnnotation:
    annotation_id: int
    bbox: List[float]
    area: float
    segmentation: Dict[str, object]
    predicted_iou: Optional[float]
    point_coords: List[List[float]]
    crop_box: List[float]
    stability_score: Optional[float]
    raw: Dict[str, object]


@dataclass(frozen=True)
class SA1BSample:
    shard_name: str
    sample_stem: str
    image_info: SA1BImageInfo
    annotations: List[SA1BAnnotation]
    annotation_path: Path
    raw_annotation: Dict[str, object]

    @property
    def image_path(self) -> Path:
        return self.image_info.image_path


class SA1BReader:
    def __init__(self, root_dir: PathLike):
        self.root_dir = Path(root_dir).expanduser().resolve()
        if not self.root_dir.exists():
            raise FileNotFoundError(f"SA-1B root directory does not exist: {self.root_dir}")

    def list_shards(self) -> List[str]:
        return sorted(
            path.name
            for path in self.root_dir.iterdir()
            if path.is_dir() and path.name.startswith("sa_")
        )

    def shard_path(self, shard_name: str) -> Path:
        shard_path = self.root_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard directory does not exist: {shard_path}")
        return shard_path

    def list_sample_stems(self, shard_name: str) -> List[str]:
        shard_path = self.shard_path(shard_name)
        stems = {
            path.stem
            for path in shard_path.glob("sa_*.json")
            if (shard_path / f"{path.stem}.jpg").exists()
        }
        return sorted(stems, key=self._sample_sort_key)

    def load_image(self, shard_name: str, sample: Union[int, str]) -> Image.Image:
        image_path = self.image_path(shard_name, sample)
        with Image.open(image_path) as image:
            return image.convert("RGB")

    def image_path(self, shard_name: str, sample: Union[int, str]) -> Path:
        sample_stem = self._normalize_sample_stem(sample)
        image_path = self.shard_path(shard_name) / f"{sample_stem}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")
        return image_path

    def annotation_path(self, shard_name: str, sample: Union[int, str]) -> Path:
        sample_stem = self._normalize_sample_stem(sample)
        annotation_path = self.shard_path(shard_name) / f"{sample_stem}.json"
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file does not exist: {annotation_path}")
        return annotation_path

    def load_annotation_dict(self, shard_name: str, sample: Union[int, str]) -> Dict[str, object]:
        annotation_path = self.annotation_path(shard_name, sample)
        return json.loads(annotation_path.read_text(encoding="utf-8"))

    def load_sample(self, shard_name: str, sample: Union[int, str]) -> SA1BSample:
        sample_stem = self._normalize_sample_stem(sample)
        raw_annotation = self.load_annotation_dict(shard_name, sample_stem)
        image_info = self._build_image_info(shard_name, raw_annotation["image"])
        annotations = [self._build_annotation(item) for item in raw_annotation.get("annotations", [])]
        return SA1BSample(
            shard_name=shard_name,
            sample_stem=sample_stem,
            image_info=image_info,
            annotations=annotations,
            annotation_path=self.annotation_path(shard_name, sample_stem),
            raw_annotation=raw_annotation,
        )

    def iter_samples(
        self,
        shard_name: str,
        *,
        limit: Optional[int] = None,
    ) -> Iterator[SA1BSample]:
        sample_stems = self.list_sample_stems(shard_name)
        if limit is not None:
            sample_stems = sample_stems[:limit]
        for sample_stem in sample_stems:
            yield self.load_sample(shard_name, sample_stem)

    @staticmethod
    def _normalize_sample_stem(sample: Union[int, str]) -> str:
        if isinstance(sample, int):
            return f"sa_{sample}"
        sample_str = str(sample).strip()
        return sample_str if sample_str.startswith("sa_") else f"sa_{sample_str}"

    @staticmethod
    def _sample_sort_key(sample_stem: str) -> tuple[int, str]:
        numeric_part = sample_stem.replace("sa_", "", 1)
        if numeric_part.isdigit():
            return int(numeric_part), sample_stem
        return 0, sample_stem

    def _build_image_info(self, shard_name: str, raw_image: Dict[str, object]) -> SA1BImageInfo:
        image_id = int(raw_image["image_id"])
        file_name = str(raw_image["file_name"])
        image_path = self.shard_path(shard_name) / file_name
        return SA1BImageInfo(
            image_id=image_id,
            width=int(raw_image["width"]),
            height=int(raw_image["height"]),
            file_name=file_name,
            image_path=image_path,
        )

    @staticmethod
    def _build_annotation(raw_annotation: Dict[str, object]) -> SA1BAnnotation:
        return SA1BAnnotation(
            annotation_id=int(raw_annotation["id"]),
            bbox=[float(value) for value in raw_annotation.get("bbox", [])],
            area=float(raw_annotation.get("area", 0.0)),
            segmentation=dict(raw_annotation.get("segmentation", {})),
            predicted_iou=_optional_float(raw_annotation.get("predicted_iou")),
            point_coords=[list(map(float, point)) for point in raw_annotation.get("point_coords", [])],
            crop_box=[float(value) for value in raw_annotation.get("crop_box", [])],
            stability_score=_optional_float(raw_annotation.get("stability_score")),
            raw=raw_annotation,
        )


def _optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    return float(value)