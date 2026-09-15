"""Manifest record + cache helpers."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from pipeline_utils import write_json


@dataclass
class Manifest:
    run_dir: Path
    input_image: str
    input_sha1: str
    seed: int
    config: Dict[str, Any] = field(default_factory=dict)
    stages: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add(self, stage: str, data: Dict[str, Any]) -> None:
        self.stages.setdefault(stage, {}).update(data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_image": self.input_image,
            "input_sha1": self.input_sha1,
            "seed": self.seed,
            "config": self.config,
            "stages": self.stages,
        }

    def save(self) -> None:
        write_json(self.to_dict(), self.run_dir / "manifest.json")

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        with open(path, "r") as f:
            d = json.load(f)
        m = cls(run_dir=Path(path).parent, input_image=d["input_image"],
                input_sha1=d["input_sha1"], seed=d["seed"], config=d.get("config", {}),
                stages=d.get("stages", {}))
        return m


def stage_cache_key(stage: str, cfg: Dict[str, Any], input_sha1: str) -> str:
    """A stable key for a stage; changes when input/config change."""
    import hashlib
    blob = json.dumps({"stage": stage, "cfg": cfg, "sha1": input_sha1},
                      sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def stage_done(run_dir: Path, stage: str, key: str) -> bool:
    marker = run_dir / f".cache_{stage}_{key}.done"
    return marker.exists()


def mark_stage_done(run_dir: Path, stage: str, key: str) -> None:
    (run_dir / f".cache_{stage}_{key}.done").touch()
